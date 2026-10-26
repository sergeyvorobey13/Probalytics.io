"""
Copy-trading backtest on Polymarket, using Probalytics data
===========================================================
Claim under test: "copying the buys of successful Polymarket traders is profitable."

We split that claim into three hypotheses, tested out-of-sample on several
walk-forward folds, at ONE pre-registered spec (PRIMARY) chosen before looking:
  H1  copying the top wallets earns > 0 after fees, latency and slippage
  H2  the top wallets beat random wallets of similar activity (skill, not market drift)
  H3  the top wallets beat the bottom wallets (the ranking means something)
p-values are Holm-adjusted across H1-H3. Every other lag / slippage / exit rule
is reported as sensitivity, not as extra chances to "find" a profit.

The paper behind the idea
  Akey, Grégoire, Harvie & Martineau (2026), "Who Wins and Who Loses in
  Prediction Markets? Evidence from Polymarket", CEPR DP21615.
  - p.19-20: the best wallets buy contracts that resolve better than their
    price implied ("excess hit rate"). Our skill score is a t-stat of that.
  - p.28 (Table 6): last month's winners had about a 60% chance of a profit
    next month. Better than a coin flip, but only modest.
  - p.4 and p.9: this may be survivors rather than skill.
  - p.12: Polymarket has charged fees on all markets since 30 Mar 2026.

Probalytics docs used (docs.probalytics.io)
  - docs/datasets/fills    -> taker_id, taker_side, price, size, fee
  - docs/sql-guide/tables  -> column types, ORDER BY keys
  Wallet IDs are only filled in for Polymarket, so this test is Polymarket-only.

What makes it robust (and what each part guards against)
  - Walk-forward folds: pick on [form], copy on the next [test], roll back in
    time. One lucky period cannot carry the result.
  - No look-ahead: formation only scores markets that had RESOLVED before the
    test window starts.
  - Skill score = t-stat of per-market edge, not raw edge, so a wallet with
    one huge lucky bet does not rank first.
  - Execution: you enter at the next real taker BUY after `lag` seconds (the
    ask someone actually paid), pay that fill's fee, plus extra slippage for
    your own size. No fill within MAX_WAIT = you missed the trade.
  - Exits: "hold" keeps to resolution; "mirror" sells when the leader sells,
    at the next real taker SELL (the bid). Leaders who scalp are not credited
    with a resolution payoff they never held for.
  - Unresolved positions are marked to the last traded price, not dropped
    (dropping them biases the sample toward short markets).
  - SPLIT resolutions use their real payout (e.g. 0.5), not "lost".
  - Equal dollars per copied trade, so returns are what a copy bot earns.
  - Bootstrap resamples (fold, market) clusters: trades in one market are not
    independent.
  - Lag 0 = the leader's own price. Comparing it to lag > 0 splits the result
    into "does skill persist?" vs "can a copier capture it?".
  - Concentration check: the result must survive dropping the 5 wallets that
    contributed most.
  - `--selftest` runs the whole core on synthetic data with and without
    planted skill, so we know the pipeline finds skill when it exists and does
    not invent it when it does not.

Known approximations
  - A leader's first SELL after a buy is treated as a full exit.
  - Pool prices are bucketed to the second (always rounds against the copier).
  - Formation n_markets slightly overcounts markets that span two slices.
  - Groups are bootstrapped independently although they share markets; this
    makes the H2/H3 intervals a bit wider (conservative).

Usage
  ../venv/bin/python test_copy_trading.py             # full run
  ../venv/bin/python test_copy_trading.py --quick     # 1 small fold, to check the plumbing
  ../venv/bin/python test_copy_trading.py --selftest  # no database needed
"""

import argparse
import hashlib
import os
import pickle
import time
import numpy as np
import pandas as pd

# ── SETTINGS ────────────────────────────────────────────────────────────────
FORM_DAYS = 60           # days used to PICK the leaders
TEST_DAYS = 30           # later days used to COPY them
N_FOLDS = 6              # walk-forward folds, newest first, test windows do not overlap
BUFFER_DAYS = 14         # gap between the newest test window and "now", so most markets resolve
CHUNK_DAYS = 5           # size of each query slice; lower it if you hit a memory error
TOP_N = 200              # wallets per group (top / random / bottom)
MIN_FILLS = 100          # ignore wallets with fewer trades (one lucky bet is not skill)
MIN_MARKETS = 20
SE_FLOOR = 0.01          # minimum standard error of the edge per share (one tick)
SCORE = "tstat"          # "tstat" (default), "ehr" (paper's excess hit rate) or "pnl"
LAGS = [0, 1, 10, 30, 60, 300]     # seconds you are late vs the leader; 0 = leader's own price
SLIPPAGES = [0.0, 0.005, 0.01, 0.02]   # extra $ per share on top of the real fill price
EXITS = ["hold", "mirror"]
MAX_WAIT = pd.Timedelta("1h")      # no fill within 1 hour = you never got in (or out)
N_BOOT = 2000                      # bootstrap resamples for CIs and p-values
ALPHA = 0.05
PRIMARY = dict(lag=30, slip=0.005, exit="mirror")   # fixed BEFORE looking at results
ANCHOR = None            # "now" for the backtest; None = today 00:00 UTC (keeps the cache valid all day)
CACHE_DIR = "cache_copy_trading"   # query results are cached here; delete it to refetch

KEY = ["market_id", "outcome_id"]
GROUPS = ["top", "random", "bottom"]


# ── SQL ─────────────────────────────────────────────────────────────────────
# toFloat64() everywhere: multiplying two Decimal128 columns can overflow.
# `anchor` is "now", fixed once at the start, so every slice uses the same clock.
def at(anchor, days_ago):
    return (anchor - pd.Timedelta(days=days_ago)).strftime("'%Y-%m-%d %H:%M:%S'")


def payouts_table(anchor, days_ago, extra=""):
    """(market, outcome, payout) for binary markets resolved `days_ago` days before
    the anchor. STANDARD pays 1/0, SPLIT pays e.g. 0.5/0.5."""
    return f"""(SELECT id AS mid, p.1 AS oid, toFloat64(p.2) AS payout{extra}
                FROM markets FINAL ARRAY JOIN resolution_outcome_payouts AS p
                WHERE platform = 'POLYMARKET' AND market_type = 'BINARY'
                  AND status = 'RESOLVED' AND resolution_type IN ('STANDARD', 'SPLIT')
                  AND resolution_resolved_at <= {at(anchor, days_ago)})"""


def formation_sql(anchor, start_days, end_days, cutoff_days):
    # edge per fill = size * side * (payout - price); side = +1 for BUY, -1 for SELL.
    # Aggregated per (wallet, market) first so we can get a per-market variance.
    return f"""
SELECT wallet, sum(n) AS n_fills, count() AS n_markets,
       sum(s) AS shares, sum(e) AS edge, sum(e * e) AS e2, sum(s * s) AS s2, sum(e * s) AS es
FROM (
    SELECT f.taker_id AS wallet, f.market_id AS mkt, count() AS n,
           sum(toFloat64(f.size)) AS s,
           sum(toFloat64(f.size) * if(f.taker_side = 'BUY', 1, -1)
               * (m.payout - toFloat64(f.price))) AS e
    FROM fills AS f
    INNER JOIN {payouts_table(anchor, cutoff_days)} AS m
        ON f.market_id = m.mid AND f.outcome.id = m.oid
    WHERE f.platform = 'POLYMARKET' AND f.taker_id IS NOT NULL
      AND f.timestamp >= {at(anchor, start_days)}
      AND f.timestamp <  {at(anchor, end_days)}
    GROUP BY wallet, mkt)
GROUP BY wallet
"""


def signals_sql(anchor, start_days, end_days):
    # One signal = one order. A single order can create many fill rows, so we
    # group the fills of one transaction into one signal. BUYs are only needed
    # inside test windows; SELLs are needed up to now (they are the exits).
    return f"""
SELECT f.taker_id AS wallet, toString(f.market_id) AS mkt, toString(f.outcome.id) AS outc,
       toString(f.taker_side) AS side, f.source_tx_hash AS tx, min(f.timestamp) AS ts,
       sum(toFloat64(f.size) * toFloat64(f.price)) / sum(toFloat64(f.size)) AS px,
       sum(toFloat64(f.fee)) / sum(toFloat64(f.size)) AS fee_ps
FROM fills AS f
WHERE f.platform = 'POLYMARKET' AND f.size > 0
  AND f.timestamp >= {at(anchor, start_days)}
  AND f.timestamp <  {at(anchor, end_days)}
  AND (f.taker_side = 'SELL' OR f.timestamp < {at(anchor, BUFFER_DAYS)})
  AND f.taker_id IN %(wallets)s
GROUP BY wallet, mkt, outc, side, tx
"""


def payouts_sql(anchor):
    return f"""
SELECT toString(mid) AS mkt, toString(oid) AS outc, payout, resolved_at
FROM {payouts_table(anchor, 0, extra=", resolution_resolved_at AS resolved_at")}
WHERE mid IN %(mkts)s
"""


def pool_sql(anchor, start_days):
    # Every real taker fill in those markets = the prices a copier could have got.
    # BUY fills are asks (entries), SELL fills are bids (exits). Bucketed per
    # second to keep busy markets small; the bucket time is never later than the
    # real fill, so the forward lookup can only be pessimistic.
    return f"""
SELECT toString(f.market_id) AS mkt, toString(f.outcome.id) AS outc,
       toString(f.taker_side) AS side, toDateTime(f.timestamp) AS ts,
       argMin(toFloat64(f.price), f.timestamp) AS price,
       avg(toFloat64(f.fee) / toFloat64(f.size)) AS fee_ps
FROM fills AS f
WHERE f.platform = 'POLYMARKET' AND f.size > 0
  AND f.timestamp >= {at(anchor, start_days)}
  AND f.timestamp <  {at(anchor, 0)}
  AND f.market_id IN %(mkts)s
GROUP BY mkt, outc, side, ts
"""


def ch_query(client, sql, params=None, cache=True):
    """Run a query, caching the result on disk (same SQL + params = same file)."""
    params = params or {}
    key = hashlib.sha1((sql + repr(sorted(params.items()))).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key + ".pkl")
    if cache and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    rows, cols = client.execute(sql, params, with_column_types=True)
    df = pd.DataFrame(rows, columns=[c[0] for c in cols])
    if not cache:
        return df
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(df, fh)
    return df


def run_chunked(client, make_sql, first_day, last_day, params=None, post=None):
    """Run one query over [first_day .. last_day] days ago, CHUNK_DAYS at a time.
    `post` shrinks each slice before the next one is fetched."""
    parts, a = [], first_day
    while a > last_day:
        b = max(a - CHUNK_DAYS, last_day)
        t = time.time()
        part = ch_query(client, make_sql(a, b), params)
        if post is not None:
            part = post(part)
        print(f"   days {a} -> {b} ago: {len(part):,} rows in {time.time() - t:.0f}s")
        parts.append(part)
        a = b
    return pd.concat(parts, ignore_index=True)


def fetch_by_markets(client, sql, mkts, label, chunk=500):
    """Run a per-market query in chunks of markets so one query is never huge."""
    mkts = sorted(mkts)
    parts = []
    for i in range(0, len(mkts), chunk):
        t = time.time()
        parts.append(ch_query(client, sql, {"mkts": tuple(mkts[i:i + chunk])}))
        print(f"   {label}, markets {i}-{i + chunk} of {len(mkts)}: "
              f"{len(parts[-1]):,} rows in {time.time() - t:.0f}s")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


class Codes:
    """Maps long id strings (wallets, market and outcome UUIDs) to small ints, so
    tens of millions of rows fit in memory. One map per kind of id."""
    def __init__(self):
        self.maps, self.rev = {}, {}

    def encode(self, kind, values):
        m, r = self.maps.setdefault(kind, {}), self.rev.setdefault(kind, [])
        u = pd.unique(np.asarray(values, dtype=object))
        for x in u:
            if x not in m:
                m[x] = len(r)
                r.append(x)
        codes_u = np.array([m[x] for x in u], dtype=np.int32)
        return codes_u[pd.Index(u).get_indexer(values)]

    def decode(self, kind, codes):
        r = self.rev[kind]
        return [r[c] for c in codes]


def compact(df, codes, ts="ts"):
    """Encode ids, make side a 2-value category, and drop the tx hash."""
    df = df.drop(columns=[c for c in ("tx",) if c in df.columns])
    for c, kind in (("wallet", "wallet"), ("mkt", "mkt"), ("outc", "outc")):
        if c in df.columns:
            df[c] = codes.encode(kind, df[c])
    if "side" in df.columns:
        df["side"] = pd.Categorical(df["side"], categories=["BUY", "SELL"])
    if ts in df.columns:
        df[ts] = pd.to_datetime(df[ts]).astype("datetime64[ns]")
    return df


def _first_fills(book, targets):
    """The first fill in `book` at or after each target time `t`, within MAX_WAIT."""
    if book.empty or targets.empty:
        return book.iloc[:0]
    m = pd.merge_asof(targets.sort_values("t"), book.rename(columns={"ts": "fill_ts"}),
                      left_on="t", right_on="fill_ts", by=["mkt", "outc"],
                      direction="forward", tolerance=MAX_WAIT)
    m = m.dropna(subset=["price"])
    return (m[["mkt", "outc", "side", "fill_ts", "price", "fee_ps"]]
            .rename(columns={"fill_ts": "ts"}).drop_duplicates())


def fetch_pool(client, anchor, start_days, sig, codes, chunk=500):
    """Price pool, reduced chunk by chunk to exactly the fills the simulation can use:
    for every signal and every lag, the first taker BUY after the copy time (entry)
    and the first taker SELL after the leader's sell, or after our entry (exit).
    The result is identical to using the full pool, at a fraction of the memory.
    The last price per outcome is kept from the full chunk, to mark open positions.
    Reduced chunks are cached, so a crashed run resumes. Returns (pool, last_price)."""
    lags = [pd.Timedelta(seconds=l) for l in LAGS if l > 0]
    one_s = pd.Timedelta(seconds=1)
    sig = sig[["mkt", "outc", "side", "ts"]]
    code_sorted = np.unique(sig.loc[sig["side"] == "BUY", "mkt"])
    mkts = codes.decode("mkt", code_sorted)
    sig = sig[sig["mkt"].isin(code_sorted)]          # sells in markets nobody bought don't matter
    block = np.searchsorted(code_sorted, sig["mkt"].to_numpy()) // chunk
    by_block = dict(tuple(sig.groupby(block)))
    sql = pool_sql(anchor, start_days)
    parts, lasts, raw_rows = [], [], 0
    for j, i in enumerate(range(0, len(mkts), chunk)):
        t = time.time()
        names = tuple(mkts[i:i + chunk])
        key = hashlib.sha1(("pool-first-fills-v1" + sql + repr(names) + repr(LAGS)
                            + str(MAX_WAIT)).encode()).hexdigest()
        path = os.path.join(CACHE_DIR, key + ".pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                kept, last = pickle.load(fh)
            kept = compact(kept, codes)
            last.index = pd.MultiIndex.from_arrays([codes.encode("mkt", last.index.get_level_values(0)),
                                                    codes.encode("outc", last.index.get_level_values(1))],
                                                   names=["mkt", "outc"])
        else:
            raw = compact(ch_query(client, sql, {"mkts": names}, cache=False), codes).sort_values("ts")
            raw_rows += len(raw)
            s = by_block.get(j, sig.iloc[:0])
            buys, sells = s[s["side"] == "BUY"], s[s["side"] == "SELL"]
            entries = _first_fills(raw[raw["side"] == "BUY"],
                                   pd.concat([buys[["mkt", "outc"]].assign(t=buys["ts"] + L) for L in lags]))
            exits = _first_fills(raw[raw["side"] == "SELL"],
                                 pd.concat([sells[["mkt", "outc"]].assign(t=sells["ts"] + L) for L in lags]
                                           + [entries[["mkt", "outc"]].assign(t=entries["ts"] + one_s)]))
            kept = pd.concat([entries, exits], ignore_index=True)
            last = raw.groupby(["mkt", "outc"])["price"].last()
            # cache with the original string ids (int codes are only valid within one run)
            k2 = kept.assign(mkt=codes.decode("mkt", kept["mkt"]), outc=codes.decode("outc", kept["outc"]),
                             side=kept["side"].astype(str))
            l2 = last.copy()
            l2.index = pd.MultiIndex.from_arrays([codes.decode("mkt", last.index.get_level_values(0)),
                                                  codes.decode("outc", last.index.get_level_values(1))])
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump((k2, l2), fh)
        parts.append(kept)
        lasts.append(last)
        print(f"   price pool, markets {i}-{i + chunk} of {len(mkts)}: kept {len(kept):,} fills "
              f"in {time.time() - t:.0f}s")
    pool = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    last = pd.concat(lasts) if lasts else pd.Series(dtype=float)
    print(f"   kept {len(pool):,} fills (of {raw_rows:,} fetched this run)")
    return pool, last


# ── CORE LOGIC (plain pandas, no database needed) ───────────────────────────
def fold_windows(n_folds=N_FOLDS):
    """Day offsets (days ago) for each fold. Fold 0 is the most recent."""
    out = []
    for k in range(n_folds):
        test_end = BUFFER_DAYS + k * TEST_DAYS
        test_start = test_end + TEST_DAYS
        out.append(dict(fold=k, form_start=test_start + FORM_DAYS,
                        test_start=test_start, test_end=test_end))
    return out


def combine_formation(parts):
    """Add the slices together per wallet, apply the activity filter, and score."""
    cols = ["n_fills", "n_markets", "shares", "edge", "e2", "s2", "es"]
    g = parts.groupby("wallet", as_index=False)[cols].sum()
    g = g[(g["n_fills"] >= MIN_FILLS) & (g["n_markets"] >= MIN_MARKETS) & (g["shares"] > 0)].copy()
    g["ehr"] = g["edge"] / g["shares"]
    # Ratio-estimator standard error of ehr, treating each market as one draw:
    # residual_m = edge_m - ehr * shares_m.
    rss = (g["e2"] - 2 * g["ehr"] * g["es"] + g["ehr"] ** 2 * g["s2"]).clip(lower=0)
    n = g["n_markets"]
    se = np.sqrt(rss * n / (n - 1)) / g["shares"]
    # Floor at one tick: an edge smaller than 1c/share cannot be copied anyway, and
    # near-certain 99.9c bets would otherwise get near-zero se and infinite t.
    g["tstat"] = g["ehr"] / se.clip(lower=SE_FLOOR)
    g["pnl"] = g["edge"]
    return g


def pick_groups(form, top_n=None, score=None, seed=42):
    """Top / bottom by score, and random wallets from the middle, all drawn from
    the SAME activity pool. Requiring above-median shares keeps tiny wallets out,
    so any top-vs-random gap reflects skill rather than activity level."""
    top_n, score = top_n or TOP_N, score or SCORE
    pool = form[form["shares"] >= form["shares"].quantile(0.5)]
    top = pool.nlargest(top_n, score)
    bottom = pool.drop(top.index).nsmallest(top_n, score)
    rest = pool.drop(top.index).drop(bottom.index)
    rand = rest.sample(min(top_n, len(rest)), random_state=seed)
    return {"top": top, "random": rand, "bottom": bottom}


def prep_signals(sig, payouts):
    s = sig.rename(columns={"mkt": "market_id", "outc": "outcome_id"}).copy()
    s["ts"] = pd.to_datetime(s["ts"]).astype("datetime64[ns]")
    p = payouts.rename(columns={"mkt": "market_id", "outc": "outcome_id"}).drop_duplicates(KEY)
    cols = KEY + ["payout"] + (["resolved_at"] if "resolved_at" in p.columns else [])
    s = s.merge(p[cols], on=KEY, how="left")
    if "resolved_at" in s.columns:
        s["resolved_at"] = pd.to_datetime(s["resolved_at"]).astype("datetime64[ns]")
    return s


def prep_pool(pool, last=None):
    """`last` = last traded price per outcome; computed from `pool` if not given."""
    p = pool.rename(columns={"mkt": "market_id", "outc": "outcome_id", "ts": "fill_ts"}).copy()
    p["fill_ts"] = pd.to_datetime(p["fill_ts"]).astype("datetime64[ns]")
    p = p.sort_values("fill_ts")
    cols = KEY + ["fill_ts", "price", "fee_ps"]
    if last is None:
        last = p.groupby(KEY)["price"].last()
    last.index.names = KEY
    return {"BUY": p[p["side"] == "BUY"][cols].reset_index(drop=True),
            "SELL": p[p["side"] == "SELL"][cols].reset_index(drop=True),
            "last": last}


def _next_fill(left, on, book, ts_name, px_name, fee_name):
    """For each row of `left`, the first real fill in `book` at or after `on`."""
    b = book.rename(columns={"fill_ts": ts_name, "price": px_name, "fee_ps": fee_name})
    return pd.merge_asof(left.sort_values(on), b, left_on=on, right_on=ts_name,
                         by=KEY, direction="forward", tolerance=MAX_WAIT)


def simulate(sig, P, lag_s, exit_mode):
    """Copy every BUY signal `lag_s` seconds late. Returns (n_signals, trades),
    one row per trade we got into, with entry cost and payoff per share before
    extra slippage. Payoff = exit price (mirror), else payout, else last price."""
    b = sig[sig["side"] == "BUY"].copy()
    n_signals = len(b)
    if b.empty:
        return 0, pd.DataFrame()
    lag = pd.Timedelta(seconds=lag_s)
    if lag_s == 0:                      # the leader's own fill: an upper bound, not achievable
        b["entry_ts"] = b["ts"]
        b["entry_px"] = b["px"] + b["fee_ps"].fillna(0)
    else:
        b["t_entry"] = b["ts"] + lag
        b = _next_fill(b, "t_entry", P["BUY"], "entry_ts", "in_px", "in_fee")
        b = b.dropna(subset=["in_px"])
        b["entry_px"] = b["in_px"] + b["in_fee"].fillna(0)
    b = b.reset_index(drop=True)

    mtm = P["last"].reindex(pd.MultiIndex.from_frame(b[KEY])).to_numpy()
    b["payoff"] = np.where(b["payout"].notna(), b["payout"], mtm)
    b["status"] = np.where(b["payout"].notna(), "resolved", "mtm")
    b["exited"] = False
    # when the money is free again: exit fill, else resolution; NaT = still open at the end
    b["exit_ts"] = b["resolved_at"] if "resolved_at" in b.columns else pd.NaT

    if exit_mode == "mirror":
        sells = (sig[sig["side"] == "SELL"][["wallet"] + KEY + ["ts", "px", "fee_ps"]]
                 .rename(columns={"ts": "sell_ts", "px": "sell_px", "fee_ps": "sell_fee"})
                 .sort_values("sell_ts"))
        b = pd.merge_asof(b.sort_values("ts"), sells, left_on="ts", right_on="sell_ts",
                          by=["wallet"] + KEY, direction="forward",
                          allow_exact_matches=False).reset_index(drop=True)
        has = b["sell_ts"].notna()
        if lag_s == 0:
            exit_px = (b["sell_px"] - b["sell_fee"].fillna(0)).where(has)
        else:
            e = b.loc[has, KEY + ["sell_ts", "entry_ts"]].copy()
            e["t_exit"] = np.maximum(e["sell_ts"] + lag, e["entry_ts"] + pd.Timedelta(seconds=1))
            e["row"] = e.index
            e = _next_fill(e, "t_exit", P["SELL"], "out_ts", "out_px", "out_fee")
            exit_px = pd.Series(np.nan, index=b.index)
            out_ts = pd.Series(pd.NaT, index=b.index, dtype="datetime64[ns]")
            ok = e["out_px"].notna()
            exit_px.loc[e.loc[ok, "row"]] = (e.loc[ok, "out_px"] - e.loc[ok, "out_fee"].fillna(0)).to_numpy()
            out_ts.loc[e.loc[ok, "row"]] = e.loc[ok, "out_ts"].to_numpy()
        got = exit_px.notna()          # leader sold but no bid within MAX_WAIT -> we hold
        b.loc[got, "payoff"] = exit_px[got]
        b.loc[got, "exit_ts"] = (b["sell_ts"] if lag_s == 0 else out_ts)[got]
        b.loc[got, "status"] = "exited"
        b["exited"] = got

    b = b.dropna(subset=["payoff"])
    return n_signals, b[["wallet"] + KEY + ["entry_ts", "exit_ts", "entry_px", "payoff", "exited", "status"]]


def trade_returns(t, slip):
    """Return per $1 copied. Slippage is paid on entry, and on exit if we sold."""
    cost = t["entry_px"].to_numpy() + slip
    payoff = t["payoff"].to_numpy() - slip * t["exited"].to_numpy()
    return payoff / cost - 1


def _units(t, slip):
    """Collapse to (fold, market) clusters: the independent units for the bootstrap."""
    r = pd.Series(trade_returns(t, slip), index=t.index)
    g = r.groupby([t["fold"], t["market_id"]]).agg(["sum", "count"])
    return g["sum"].to_numpy(), g["count"].to_numpy()


def _boot(sums, counts, n_boot, seed, batch=200):
    rng = np.random.default_rng(seed)
    out, n = [], len(sums)
    for i in range(0, n_boot, batch):
        idx = rng.integers(0, n, size=(min(batch, n_boot - i), n))
        out.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    return np.concatenate(out)


def cell_stats(t, slip, n_boot=None, seed=0):
    """Mean return per $ copied, 95% cluster-bootstrap CI, one-sided p for H0: ret <= 0."""
    if t is None or t.empty:
        return dict(trades=0, markets=0, ret=np.nan, lo=np.nan, hi=np.nan, p_gt0=np.nan)
    sums, counts = _units(t, slip)
    boots = _boot(sums, counts, n_boot or N_BOOT, seed)
    return dict(trades=len(t), markets=len(sums), ret=sums.sum() / counts.sum(),
                lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                p_gt0=float((boots <= 0).mean()))


def compare(ta, tb, slip, n_boot=None, seed=0):
    """Bootstrap the A-minus-B return gap; one-sided p for H0: A <= B."""
    if ta.empty or tb.empty:
        return dict(diff=np.nan, lo=np.nan, hi=np.nan, pval=np.nan)
    sa, ca = _units(ta, slip)
    sb, cb = _units(tb, slip)
    d = _boot(sa, ca, n_boot or N_BOOT, seed) - _boot(sb, cb, n_boot or N_BOOT, seed + 1)
    return dict(diff=sa.sum() / ca.sum() - sb.sum() / cb.sum(),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                pval=float((d <= 0).mean()))


def holm(pvals):
    """Holm step-down adjustment (controls family-wise error across H1-H3)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def breakeven_slippage(t, lo=0.0, hi=0.25, iters=30):
    """Extra cost per share at which copying stops paying (0 if it never pays)."""
    f = lambda s: trade_returns(t, s).mean()
    if t.empty or f(lo) <= 0:
        return 0.0
    if f(hi) > 0:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return lo


def evaluate(folds, sig, P):
    """Simulate every (fold, group, lag, exit). Returns {(group, lag, exit): trades}
    with a `fold` column, plus signal counts for fill rates."""
    trades, n_sig = {}, {}
    for f in folds:
        in_test = (sig["ts"] >= f["test_start_ts"]) & (sig["ts"] < f["test_end_ts"])
        after = sig["ts"] >= f["test_start_ts"]
        for g in GROUPS:
            mine = sig["wallet"].isin(set(f["groups"][g]["wallet"]))
            s = sig[mine & (((sig["side"] == "BUY") & in_test) | ((sig["side"] == "SELL") & after))]
            for lag in LAGS:
                for ex in EXITS:
                    n, t = simulate(s, P, lag, ex)
                    if not t.empty:
                        t = t.assign(fold=f["fold"])
                    k = (g, lag, ex)
                    trades.setdefault(k, []).append(t)
                    n_sig[k] = n_sig.get(k, 0) + n
    trades = {k: pd.concat(v, ignore_index=True) for k, v in trades.items()}
    return trades, n_sig


def report(trades, n_sig, out_prefix="copy_trading"):
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 200)
    E = pd.DataFrame()

    # 1) Full sensitivity grid
    rows = []
    for (g, lag, ex), t in trades.items():
        for slip in SLIPPAGES:
            st = cell_stats(t, slip)
            rows.append(dict(group=g, lag_s=lag, exit=ex, slip=slip, signals=n_sig[(g, lag, ex)],
                             fill_rate=st["trades"] / n_sig[(g, lag, ex)] if n_sig[(g, lag, ex)] else np.nan,
                             **st))
    grid = pd.DataFrame(rows).sort_values(["exit", "slip", "group", "lag_s"])
    grid.to_csv(f"{out_prefix}_results.csv", index=False)
    print("\nSENSITIVITY GRID (return per $ copied; lag 0 = leader's own price, not achievable)")
    show = grid[grid["slip"].isin([0.0, PRIMARY["slip"]])]
    print(show[["exit", "slip", "group", "lag_s", "fill_rate", "trades", "markets",
                "ret", "lo", "hi", "p_gt0"]].to_string(index=False))

    # 2) Pre-registered hypotheses at the primary spec
    lag, slip, ex = PRIMARY["lag"], PRIMARY["slip"], PRIMARY["exit"]
    top, rnd, bot = (trades.get((g, lag, ex), E) for g in GROUPS)
    h1 = cell_stats(top, slip)
    h2 = compare(top, rnd, slip)
    h3 = compare(top, bot, slip)
    p_adj = holm([h1["p_gt0"], h2["pval"], h3["pval"]])
    hyp = pd.DataFrame([
        dict(h="H1 top copy return > 0", est=h1["ret"], lo=h1["lo"], hi=h1["hi"], p=h1["p_gt0"]),
        dict(h="H2 top > random", est=h2["diff"], lo=h2["lo"], hi=h2["hi"], p=h2["pval"]),
        dict(h="H3 top > bottom", est=h3["diff"], lo=h3["lo"], hi=h3["hi"], p=h3["pval"]),
    ])
    hyp["p_holm"] = p_adj
    hyp["verdict"] = np.where(hyp["p_holm"] < ALPHA, "YES", "no")
    hyp.to_csv(f"{out_prefix}_significance.csv", index=False)
    print(f"\nHYPOTHESES at primary spec: lag={lag}s, slippage={slip}, exit={ex} (Holm-adjusted)")
    print(hyp.to_string(index=False))

    # 3) Per-fold stability at the primary spec
    fold_rows = []
    for k in sorted(set(top["fold"]) if not top.empty else []):
        r = {g: trades_ret(trades.get((g, lag, ex), E), slip, fold=k) for g in GROUPS}
        fold_rows.append(dict(fold=k, **r, top_minus_random=r["top"] - r["random"]))
    folds = pd.DataFrame(fold_rows)
    if not folds.empty:
        folds.to_csv(f"{out_prefix}_folds.csv", index=False)
        print("\nPER FOLD at primary spec (fold 0 = most recent)")
        print(folds.to_string(index=False))

    # 4) Robustness of H1
    if top.empty:
        print("\nno top-wallet trades at the primary spec; nothing more to check")
        return hyp
    r = pd.Series(trade_returns(top, slip), index=top.index)
    by_wallet = r.groupby(top["wallet"]).sum().sort_values(ascending=False)
    worst5 = set(by_wallet.index[:5])
    no_top5 = cell_stats(top[~top["wallet"].isin(worst5)], slip)
    share_top5 = by_wallet.iloc[:5].sum() / by_wallet.sum() if by_wallet.sum() != 0 else np.nan
    be = breakeven_slippage(top)
    lag0 = trades.get(("top", 0, ex), E)
    print("\nROBUSTNESS (top wallets, primary spec)")
    print(f"  status of copied trades: {top['status'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  PnL share from the 5 biggest contributors: {share_top5:.1%}; "
          f"return without them: {no_top5['ret']:.4f} [{no_top5['lo']:.4f}, {no_top5['hi']:.4f}]")
    print(f"  break-even extra slippage per share: {be:.4f}")
    if not lag0.empty:
        print(f"  leader's own return (lag 0, persistence of skill): {cell_stats(lag0, 0.0)['ret']:.4f} "
              f"vs copier at {lag}s: {h1['ret']:.4f}  -> latency/execution cost "
              f"{cell_stats(lag0, 0.0)['ret'] - h1['ret']:.4f}")
    hold = (top["exit_ts"] - top["entry_ts"]).dt.total_seconds() / 60
    fmt = lambda h: f"median {h.median():8.1f}   mean {h.mean():8.1f}   90th pct {h.quantile(0.9):8.1f}"
    print("  holding time in MINUTES, by how the position ended:")
    for st, h in hold.groupby(top["status"]):
        print(f"     {st:10s} {len(h):>8,} trades   " + (fmt(h) if h.notna().any()
                                                         else "still open at the end (no end time)"))
    print(f"     all closed {hold.notna().sum():>8,} trades   {fmt(hold.dropna())}")
    pos_folds = int((folds["top"] > 0).sum()) if not folds.empty else 0
    beat_folds = int((folds["top_minus_random"] > 0).sum()) if not folds.empty else 0
    print(f"  folds with positive copy return: {pos_folds}/{len(folds)}; "
          f"folds where top beat random: {beat_folds}/{len(folds)}")

    # 5) Verdict: all of these must hold to call the claim supported
    checks = {
        "H1 significant after Holm": p_adj[0] < ALPHA,
        "H2 significant after Holm (skill, not drift)": p_adj[1] < ALPHA,
        "still > 0 without the top-5 wallets": no_top5["ret"] > 0,
        "positive in a majority of folds": pos_folds > len(folds) / 2,
    }
    print("\nVERDICT")
    for name, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {name}")
    supported = all(checks.values())
    print("  => copy trading top Polymarket wallets IS profitable under realistic execution"
          if supported else
          "  => the claim is NOT supported by this data at the pre-registered spec")
    return hyp


def save_trades(trades, path):
    """Every copied trade at the primary lag/exit, plus the leaders' own trades
    (lag 0), with its return at the primary slippage. Used for the profit curve."""
    out = []
    for g in GROUPS:
        for lag in sorted({0, PRIMARY["lag"]}):
            t = trades.get((g, lag, PRIMARY["exit"]))
            if t is not None and not t.empty:
                out.append(t.assign(group=g, lag_s=lag,
                                    ret=trade_returns(t, PRIMARY["slip"] if lag else 0.0)))
    if out:
        pd.concat(out, ignore_index=True).to_csv(path, index=False)
        print(f"trades saved to {path}")


def trades_ret(t, slip, fold=None):
    if t.empty:
        return np.nan
    if fold is not None:
        t = t[t["fold"] == fold]
    return float(trade_returns(t, slip).mean()) if len(t) else np.nan


# ── PRE-FLIGHT: quick checks on one day of data before the heavy scans ──────
def preflight(client):
    d = client.execute("""SELECT count() AS n, uniqExact(id) AS uniq FROM fills
                          WHERE platform = 'POLYMARKET'
                            AND timestamp >= now() - INTERVAL 1 DAY""")[0]
    print(f"1) last 24h: rows={d[0]:,} unique ids={d[1]:,} -> duplicate rows {1 - d[1] / d[0]:.3%} "
          f"(signals are grouped per tx; formation sums are inflated by about this much)")
    w = client.execute("""SELECT countIf(taker_id IS NOT NULL) / count() FROM fills
                          WHERE platform = 'POLYMARKET' AND timestamp >= now() - INTERVAL 1 DAY""")[0][0]
    print(f"2) share of fills with a taker wallet: {w:.1%}")
    fee = client.execute("""SELECT round(toFloat64(fee) / (toFloat64(size) * toFloat64(price)), 4) AS pct,
                                   count() AS fills
                            FROM fills WHERE platform = 'POLYMARKET' AND size > 0 AND price > 0
                              AND timestamp >= now() - INTERVAL 1 DAY
                            GROUP BY pct ORDER BY fills DESC LIMIT 8""")
    print("3) fee as a share of trade value (check this matches Polymarket's real fees):")
    for pct, n in fee:
        print(f"     {pct:.4f}  {n:,}")


def main(quick=False):
    global FORM_DAYS, TEST_DAYS, N_FOLDS, TOP_N, BUFFER_DAYS
    if quick:
        FORM_DAYS, TEST_DAYS, N_FOLDS, TOP_N, BUFFER_DAYS = 10, 5, 1, 50, 5
    from dotenv import load_dotenv
    from clickhouse_driver import Client
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    client = Client(host=os.environ["PROBALYTICS_CLICKHOUSE_HOST"],
                    user=os.environ["PROBALYTICS_CLICKHOUSE_USERNAME"],
                    password=os.environ["PROBALYTICS_CLICKHOUSE_PASSWORD"],
                    database="probalytics", secure=True, port=9440)
    anchor = pd.Timestamp(ANCHOR) if ANCHOR else pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
    t0 = time.time()
    preflight(client)
    folds = fold_windows(N_FOLDS)
    print(f"anchor {anchor}; {N_FOLDS} folds of {FORM_DAYS}d formation + {TEST_DAYS}d test; "
          f"history used: {folds[-1]['form_start']} days")

    # STEP 1: rank wallets in each fold using ONLY that fold's formation window
    for f in folds:
        print(f"STEP 1 fold {f['fold']}: scoring wallets, days {f['form_start']} -> {f['test_start']} ago")
        parts = run_chunked(client, lambda a, b: formation_sql(anchor, a, b, f["test_start"]),
                            f["form_start"], f["test_start"])
        form = combine_formation(parts)
        f["groups"] = pick_groups(form, seed=42 + f["fold"])
        f["test_start_ts"] = anchor - pd.Timedelta(days=f["test_start"])
        f["test_end_ts"] = anchor - pd.Timedelta(days=f["test_end"])
        best = f["groups"]["top"]
        print(f"   {len(form):,} eligible wallets; top-{TOP_N} {SCORE} from {best[SCORE].min():.2f} "
              f"to {best[SCORE].max():.2f}, median ehr {best['ehr'].median():.3f}")

    # STEP 2: what those wallets did later (one scan for all folds and groups)
    wallets = tuple(sorted({w for f in folds for g in GROUPS for w in f["groups"][g]["wallet"]}))
    oldest = folds[-1]["test_start"]
    print(f"STEP 2: collecting later orders of {len(wallets):,} wallets")
    codes = Codes()
    sig = run_chunked(client, lambda a, b: signals_sql(anchor, a, b), oldest, 0, {"wallets": wallets},
                      post=lambda d: compact(d, codes))
    for f in folds:
        for g in GROUPS:
            f["groups"][g] = f["groups"][g].assign(wallet=codes.encode("wallet", f["groups"][g]["wallet"]))

    # STEP 3: payouts and the real price pool for every market they bought
    mkts = codes.decode("mkt", np.unique(sig.loc[sig["side"] == "BUY", "mkt"]))
    print(f"STEP 3: payouts and prices for {len(mkts):,} markets")
    payouts = compact(fetch_by_markets(client, payouts_sql(anchor), mkts, "payouts", chunk=5000), codes)
    pool, last = fetch_pool(client, anchor, oldest, sig, codes)
    sig = prep_signals(sig, payouts)
    P = prep_pool(pool, last)
    del pool
    print(f"   signals={len(sig):,} | payouts={len(payouts):,} | price buckets kept={len(P['BUY']) + len(P['SELL']):,}")

    # STEP 4: copy them with delays, real prices, real fees; then test
    print("STEP 4: simulating the copies")
    trades, n_sig = evaluate(folds, sig, P)
    prefix = "copy_trading_quick" if quick else "copy_trading"
    report(trades, n_sig, out_prefix=prefix)
    save_trades(trades, f"{prefix}_trades.csv")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


# ── SELF-TEST: synthetic market with/without planted skill ──────────────────
def synthetic(skill, seed=0, n_mkts=2500, n_wallets=1200, n_trades=300, skilled_share=0.2):
    """Each market has a price p and a true probability q = p + delta, so buying at
    random is fair before costs (E[payout / price] = 1). Skilled wallets know the
    sign of delta and buy the underpriced side; everyone else picks at random.
    Returns (orders, pool, payouts, t0) in the same shapes as the SQL."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-01-01")
    start = rng.uniform(0, 100, n_mkts)
    life = rng.uniform(0.5, 10, n_mkts)
    p_yes = rng.uniform(0.1, 0.9, n_mkts)
    delta = rng.normal(0, 0.10, n_mkts)
    q = np.clip(p_yes + delta, 0.01, 0.99)
    yes = rng.random(n_mkts) < q
    mid = np.array([f"m{i}" for i in range(n_mkts)])
    days = lambda d: t0 + pd.to_timedelta(d, unit="D")

    def outc(m, is_yes):
        return np.where(is_yes, np.char.add(mid[m], "-Y"), np.char.add(mid[m], "-N"))

    def px(m, is_yes):
        return np.where(is_yes, p_yes[m], 1 - p_yes[m])

    # price pool: dense taker fills both sides; buys pay +1c, sells get -1c
    k = 600
    pm = np.repeat(np.arange(n_mkts), k)
    pt = start[pm] + rng.random(len(pm)) * life[pm]
    py = rng.random(len(pm)) < 0.5
    pb = rng.random(len(pm)) < 0.5
    price = px(pm, py) + np.where(pb, 0.01, -0.01)
    pool = pd.DataFrame(dict(mkt=mid[pm], outc=outc(pm, py), side=np.where(pb, "BUY", "SELL"),
                             ts=days(pt), price=price, fee_ps=0.002))

    # wallet orders
    wm = rng.integers(0, n_mkts, (n_wallets, n_trades))
    skilled = np.arange(n_wallets) < (skilled_share * n_wallets if skill else 0)
    wy = np.where(skilled[:, None], delta[wm] > 0, rng.random(wm.shape) < 0.5)
    wt = start[wm] + rng.random(wm.shape) * life[wm]
    wallet = np.repeat(np.array([f"w{i}" for i in range(n_wallets)]), n_trades)
    wm, wy, wt = wm.ravel(), wy.ravel(), wt.ravel()
    buys = pd.DataFrame(dict(wallet=wallet, mkt=mid[wm], outc=outc(wm, wy), side="BUY",
                             ts_d=wt, px=px(wm, wy) + 0.01, fee_ps=0.002))
    sold = rng.random(len(buys)) < 0.3
    sell_t = wt + rng.random(len(wt)) * (start[wm] + life[wm] - wt)
    sells = buys[sold].assign(side="SELL", ts_d=sell_t[sold], px=px(wm, wy)[sold] - 0.01)
    orders = pd.concat([buys, sells], ignore_index=True)
    orders["tx"] = np.arange(len(orders)).astype(str)
    orders["ts"] = days(orders["ts_d"])

    payouts = pd.DataFrame(dict(mkt=np.concatenate([mid, mid]),
                                outc=np.concatenate([np.char.add(mid, "-Y"), np.char.add(mid, "-N")]),
                                payout=np.concatenate([yes, ~yes]).astype(float),
                                resolved_d=np.concatenate([start + life, start + life])))
    return orders, pool, payouts, t0


def formation_from_orders(orders, payouts, form_lo, form_hi, cutoff):
    """Pandas mirror of formation_sql (used by the self-test only)."""
    pay = payouts[payouts["resolved_d"] <= cutoff]
    d = orders[(orders["ts_d"] >= form_lo) & (orders["ts_d"] < form_hi)].merge(
        pay[["mkt", "outc", "payout"]], on=["mkt", "outc"])
    d["e"] = np.where(d["side"] == "BUY", 1, -1) * (d["payout"] - d["px"])
    g = d.groupby(["wallet", "mkt"]).agg(n=("e", "size"), e=("e", "sum")).reset_index()
    g["s"] = g["n"].astype(float)
    g = g.assign(e2=g["e"] ** 2, s2=g["s"] ** 2, es=g["e"] * g["s"])
    return g.groupby("wallet").agg(n_fills=("n", "sum"), n_markets=("mkt", "size"), shares=("s", "sum"),
                                   edge=("e", "sum"), e2=("e2", "sum"), s2=("s2", "sum"),
                                   es=("es", "sum")).reset_index()


def _synthetic_run(skill, seed):
    """One synthetic backtest at the primary spec; returns (H1 p_holm, H2 p_holm, trades, n_sig)."""
    orders, pool, payouts, t0 = synthetic(skill, seed=seed)
    anchor_d = 108                          # markets ending after this are unresolved -> MTM
    form = combine_formation(formation_from_orders(orders, payouts, 0, 60, cutoff=60))
    folds = [dict(fold=0, groups=pick_groups(form, top_n=TOP_N, seed=seed),
                  test_start_ts=t0 + pd.Timedelta(days=60), test_end_ts=t0 + pd.Timedelta(days=90))]
    sig = prep_signals(orders.drop(columns="ts_d"), payouts[payouts["resolved_d"] <= anchor_d])
    P = prep_pool(pool[pool["ts"] <= t0 + pd.Timedelta(days=anchor_d)])
    trades, n_sig = evaluate(folds, sig, P)
    k = lambda g: trades[(g, PRIMARY["lag"], PRIMARY["exit"])]
    h1 = cell_stats(k("top"), PRIMARY["slip"], seed=seed)
    h2 = compare(k("top"), k("random"), PRIMARY["slip"], seed=seed)
    h3 = compare(k("top"), k("bottom"), PRIMARY["slip"], seed=seed)
    p = holm([h1["p_gt0"], h2["pval"], h3["pval"]])
    return p[0], p[1], h1["ret"], h2["diff"], trades, n_sig


def selftest(n_seeds=10):
    """Calibration on synthetic data. With no skill, H1/H2 must rarely fire (false
    positives); with planted skill, H2 must usually fire (power). One seed is not
    enough to judge a statistical test, so we count rejections over many."""
    global MIN_FILLS, MIN_MARKETS, TOP_N, LAGS, EXITS, SLIPPAGES, N_BOOT
    MIN_FILLS, MIN_MARKETS, TOP_N, N_BOOT = 20, 10, 100, 500
    LAGS, EXITS, SLIPPAGES = [0, PRIMARY["lag"]], [PRIMARY["exit"]], [0.0, PRIMARY["slip"]]
    rows = []
    for skill in (False, True):
        for seed in range(n_seeds):
            p1, p2, ret, diff, trades, n_sig = _synthetic_run(skill, seed)
            rows.append(dict(skill=skill, seed=seed, top_ret=ret, top_minus_random=diff,
                             p_h1=p1, p_h2=p2, h1=p1 < ALPHA, h2=p2 < ALPHA))
            print(f"  skill={skill!s:5} seed={seed}: top ret {ret:+.4f}, top-random {diff:+.4f}, "
                  f"p_h1 {p1:.3f}, p_h2 {p2:.3f}")
    r = pd.DataFrame(rows)
    null, alt = r[~r["skill"]], r[r["skill"]]
    fp_h1, fp_h2, power = null["h1"].mean(), null["h2"].mean(), alt["h2"].mean()
    checks = {f"no skill: H1 false-positive rate {fp_h1:.0%} <= 20%": fp_h1 <= 0.2,
              f"no skill: H2 false-positive rate {fp_h2:.0%} <= 20%": fp_h2 <= 0.2,
              f"planted skill: H2 detected in {power:.0%} >= 70% of runs": power >= 0.7,
              "planted skill: top beats random on average": alt["top_minus_random"].mean() > 0}
    print("\nSELF-TEST")
    for name, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {name}")
    print("\nexample full report (planted skill, seed 0):")
    *_, trades, n_sig = _synthetic_run(True, 0)
    report(trades, n_sig, out_prefix=os.path.join(os.environ.get("TMPDIR", "/tmp"), "selftest"))
    ok = all(checks.values())
    print("\nALL SELF-TESTS PASSED" if ok else "\nSELF-TEST FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data, no database")
    ap.add_argument("--quick", action="store_true", help="one small fold, to check the plumbing")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    main(quick=args.quick)
