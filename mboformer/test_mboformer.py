"""
Order-book event-history transformer (MBOFormer) on Polymarket, using Probalytics data
======================================================================================
Claim under test (LinkedIn post -> paper): a TINY transformer (7,203 parameters) that
reads the raw stream of order-book EVENTS predicts short-term price moves better
than much bigger models that only look at sampled order-book SNAPSHOTS.

The paper
  Schaurecker, Strand, O'Sullivan & Jakob (2026), "Event History Over Scale: Compact
  Transformers for Low-Latency Limit Order Book Forecasting", arXiv:2610.02917.
  - Data: market-by-order (L3) messages on EPEX power, NASDAQ AAPL, NYSE BAC.
  - One decision point after every 10th book-changing message.
  - Input: the last T = 128 events, 7 features each (type, side, depth from best,
    size, elapsed time, immediate mid change, at-best flag).
  - Label: up / flat / down move of the SMOOTHED mid h = 10/20/50/100 steps ahead.
  - Result: best macro-F1 in 11 of 12 settings, sub-millisecond on one CPU thread.
  - What it did NOT do: no trading P&L, no costs, no significance tests, and its
    label threshold is computed on the test set (look-ahead, the authors say so).

What changes on Polymarket (and why this is still a fair test of the idea)
  Probalytics does not ship single orders (L3). It ships a COMPLETE book every
  time the book changes (docs.probalytics.io/docs/datasets/orderbook):
  `orderbook_snapshots` rows, ordered by (timestamp, path_index). Diffing two
  consecutive books gives every price level that changed = one "event". That is
  market-by-PRICE, the finest event stream Polymarket exposes. So we compare:
      EVENT models    : MBOFormer, MBOFusion   (read every level change)
      SNAPSHOT models : L2-Former              (same backbone, reads sampled books)
      SIMPLE models   : logistic regression on classic book features, always-flat
  Same labels, same decision points, same splits. The only difference between
  MBOFormer and L2-Former is the INPUT, which is exactly the paper's claim.

What we add on top of the paper
  1. Label threshold alpha_h comes from the TRAINING contracts only (no look-ahead).
  2. Train / validation / test are different contracts, in time order.
  3. Paired block-bootstrap confidence intervals for "event minus snapshot" F1.
  4. A trading test: cross the spread at the signal, exit h steps later, with
     Polymarket's taker fee, plus a 1-step-late version and a perfect-foresight
     "oracle" that shows the best any classifier could do.
  5. Latency: model time next to Polymarket's own feed delay.

Probalytics docs used
  - docs/datasets/orderbook    -> bids/asks arrays, state, continuity (RESET), path_index
  - docs/sql-guide/tables      -> ORDER BY (market_id, outcome.id, timestamp, ...)
  - docs/datasets/fills        -> normalized_price (price in outcome-0 terms) to tag executions

Known approximations
  - Only the top LEVELS of each side are downloaded. A change is only counted
    if the price is inside the visible part of BOTH books (no fake events when a
    level slides in or out of the top 10).
  - Event type "execution" = a size decrease at a price where a fill printed
    within EXEC_WINDOW seconds (fills carry block time, a few seconds after the
    book). Polymarket's websocket sends trades in real time, so a live system has
    this information; we only approximate it.
  - One snapshot change can move several levels; each level is one event.
  - A RESET row starts a fresh book: no event is made from the transition into
    it (docs: "do not calculate a change between the two rows").

Usage
  ../venv/bin/python test_mboformer.py            # full run (downloads ~15M book rows once, then cached)
  ../venv/bin/python test_mboformer.py --family "BTC 15m"   # one family; run the three side by side to save time
  ../venv/bin/python test_mboformer.py --quick    # 1 family, few contracts, 1 seed: checks the plumbing
  ../venv/bin/python test_mboformer.py --plots    # redraw charts from saved results only
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import time
import numpy as np
import pandas as pd

# ── SETTINGS ────────────────────────────────────────────────────────────────
PERIOD = ("2026-10-08", "2026-10-23")   # contracts closing in this range (high-quality books start 2026-02-17)
N_CONTRACTS = 20          # per family
SPLIT = (12, 3, 5)        # train / validation / test contracts, in time order
WINDOW_MIN = 15           # minutes of book used per contract
LEVELS = 10               # book levels per side (the paper's L2 baselines use 10)
SAMPLE_EVERY = 10         # one decision point per 10 book changes (paper)
T_EVENTS = 128            # events the event models see (paper)
T_SNAPS = 16              # sampled snapshots the L2 model sees (paper's TLOB 7.2k: 16)
SMOOTH = 10               # labels use a 10-step trailing mean of the mid (paper)
HORIZONS = [10, 20, 50, 100]
SEEDS = [0, 1, 2]         # paper: three seeds
SEEDS_FUSION = [0]        # MBOFusion is the paper's add-on; one seed keeps the CPU budget sane
THREADS = 3               # torch threads per process (one process per family, run side by side)
TICK = 0.01
EXEC_WINDOW = (-6, 2)     # a book decrease at t is an execution if a fill printed in [t-2s, t+6s]
MAX_EPOCHS = 12
TRAIN_PER_EPOCH = 60_000   # random training decision points per epoch (CPU budget)
BATCH = 512
LR = 2e-3
EARLY_STOP = 5e-4         # stop when val loss fails to drop by this, two epochs running (paper)
N_BOOT = 1000
BLOCK_S = 60              # bootstrap blocks = (contract, minute): neighbouring samples are not independent
# Polymarket taker fee per share on fee-enabled markets (docs.polymarket.com fee table:
# 100 shares at $0.45 -> $0.69):  fee(p) = 0.25 * p * (p * (1 - p))^2   (max 0.78c at p = 0.5)
FEE_RATE, FEE_EXP = 0.25, 2
CACHE_DIR = "cache_mboformer"
KEEP_BOOK = False         # retest_mboformer.py turns this on to keep raw levels per decision point
OUT = "mboformer"         # prefix for csv / png outputs
PARTS = "mboformer_results"  # per-family result files

FAMILIES = {
    "BTC 15m": dict(kind="crypto", pattern=r"^Bitcoin Up or Down - "),
    "ETH 15m": dict(kind="crypto", pattern=r"^Ethereum Up or Down - "),
    "Sports & esports": dict(kind="match"),
}
CLASSES = ["down", "flat", "up"]


# ── SQL ─────────────────────────────────────────────────────────────────────
def crypto_markets_sql():
    return f"""
SELECT toString(id) AS mid, title, closes_at
FROM markets FINAL
WHERE platform = 'POLYMARKET' AND market_type = 'BINARY'
  AND closes_at >= '{PERIOD[0]}' AND closes_at < '{PERIOD[1]}'
  AND match(title, '^(Bitcoin|Ethereum) Up or Down - [A-Za-z]+ [0-9]+, [0-9:]+[AP]M-[0-9:]+[AP]M ET$')
"""


def match_markets_sql():
    # Head-to-head match winners only (no map / game / handicap / totals side markets),
    # ranked by number of trades: activity is chosen without looking at price direction.
    return f"""
SELECT toString(f.market_id) AS mid, any(m.title) AS title, any(m.closes_at) AS closes_at, count() AS n
FROM fills AS f
INNER JOIN (SELECT id, title, closes_at FROM markets FINAL
            WHERE platform = 'POLYMARKET' AND market_type = 'BINARY' AND title ILIKE '% vs%'
              AND NOT match(title, '(?i)(map |game |handicap|o/u|spread|total|set |exact|both teams|1st|first|draw|half)')
              AND closes_at >= '{PERIOD[0]}' AND closes_at < '{PERIOD[1]}') AS m
    ON f.market_id = m.id
WHERE f.platform = 'POLYMARKET' AND f.timestamp >= '{PERIOD[0]}' AND f.timestamp < '{PERIOD[1]}'
GROUP BY f.market_id ORDER BY n DESC LIMIT {N_CONTRACTS}
"""


def busiest_window_sql(mid):
    # The 15 minutes with the most book changes = the live part of the match.
    return f"""
SELECT toStartOfInterval(timestamp, INTERVAL {WINDOW_MIN} MINUTE) AS w, count() AS n
FROM orderbook_snapshots
WHERE market_id = '{mid}' AND outcome.index = 0
  AND timestamp >= '{PERIOD[0]}' AND timestamp < '{PERIOD[1]}'
GROUP BY w ORDER BY n DESC LIMIT 1
"""


def book_sql(mid, start, end):
    # Outcome index 0 ("Up" / first team) only; the other outcome's book mirrors it.
    # Prices as integer thousandths so level comparisons are exact.
    side = lambda arr, f: f"arrayMap(x -> {f}, arraySlice({arr}, 1, {LEVELS}))"
    return f"""
SELECT toUnixTimestamp64Milli(timestamp) AS ts, path_index, hash, toUInt8(continuity = 'RESET') AS reset,
       {side('bids', 'toInt32(round(toFloat64(x.price) * 1000))')} AS bp,
       {side('bids', 'toFloat32(x.size)')} AS bs,
       {side('asks', 'toInt32(round(toFloat64(x.price) * 1000))')} AS ap,
       {side('asks', 'toFloat32(x.size)')} AS asz,
       toUnixTimestamp64Milli(indexed_at) AS ix
FROM orderbook_snapshots
WHERE market_id = '{mid}' AND outcome.index = 0
  AND timestamp >= '{start}' AND timestamp < '{end}'
ORDER BY timestamp, path_index, indexed_at
"""


def fills_sql(mid, start, end):
    return f"""
SELECT toUnixTimestamp(timestamp) AS sec, toInt32(round(toFloat64(normalized_price) * 1000)) AS pm
FROM fills
WHERE platform = 'POLYMARKET' AND market_id = '{mid}'
  AND timestamp >= toDateTime('{start}') - INTERVAL 30 SECOND
  AND timestamp <  toDateTime('{end}') + INTERVAL 30 SECOND
"""


def ch_query(client, sql):
    """Run a query, caching the result on disk (same SQL = same file)."""
    path = os.path.join(CACHE_DIR, hashlib.sha1(sql.encode()).hexdigest() + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    rows, cols = client.execute(sql, with_column_types=True)
    df = pd.DataFrame(rows, columns=[c[0] for c in cols])
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(df, fh)
    return df


# ── CONTRACT SELECTION ──────────────────────────────────────────────────────
def window_minutes(title):
    """'..., 8:00PM-8:15PM ET' -> 15."""
    a, b = re.search(r"(\d+:\d\d[AP]M)-(\d+:\d\d[AP]M) ET$", title).groups()
    t = [pd.Timestamp("2000-01-01 " + x) for x in (a, b)]
    return ((t[1] - t[0]).total_seconds() / 60) % (24 * 60)


def pick_contracts(client, quick):
    """Evenly spaced in time, so contracts cover different days and hours."""
    n = 6 if quick else N_CONTRACTS
    out = {}
    crypto = ch_query(client, crypto_markets_sql())
    crypto = crypto[crypto["title"].map(window_minutes) == WINDOW_MIN].sort_values("closes_at")
    for fam, spec in FAMILIES.items():
        if quick and fam != "BTC 15m":
            continue
        if spec["kind"] == "crypto":
            c = crypto[crypto["title"].str.match(spec["pattern"])].reset_index(drop=True)
            idx = np.linspace(0, len(c) - 1, n).round().astype(int)
            c = c.iloc[idx].copy()
            c["end"] = pd.to_datetime(c["closes_at"])
            c["start"] = c["end"] - pd.Timedelta(minutes=WINDOW_MIN)
        else:
            c = ch_query(client, match_markets_sql()).head(n).copy()
            w = [ch_query(client, busiest_window_sql(m))["w"].iloc[0] for m in c["mid"]]
            c["start"] = pd.to_datetime(w)
            c["end"] = c["start"] + pd.Timedelta(minutes=WINDOW_MIN)
        c = c.sort_values("start").reset_index(drop=True)
        out[fam] = c[["mid", "title", "start", "end"]]
    return out


# ── FROM SNAPSHOTS TO EVENTS (plain numpy, no database needed) ──────────────
def clean_books(df):
    """Docs rules: order by (timestamp, path_index); if the same row identity was
    written twice keep the newest indexed_at copy."""
    df = df.drop_duplicates(["ts", "path_index", "hash", "reset"], keep="last")
    return df.sort_values(["ts", "path_index"], kind="stable").reset_index(drop=True)


def to_arrays(df):
    """Pad the level arrays to LEVELS columns. Missing levels: price 0, size 0."""
    n = len(df)
    out = {}
    for c in ["bp", "bs", "ap", "asz"]:
        a = np.zeros((n, LEVELS), dtype=np.float64)
        for i, row in enumerate(df[c].values):
            a[i, :len(row)] = row
        out[c] = a
    out["nb"] = (out["bs"] > 0).sum(1)
    out["na"] = (out["asz"] > 0).sum(1)
    out["ts"] = df["ts"].values.astype(np.int64)
    out["reset"] = df["reset"].values.astype(bool)
    out["delay"] = (df["ix"].values - df["ts"].values).astype(np.int64)
    return out


def mid_spread(b):
    """mid = (best bid + best ask) / 2,  spread = best ask - best bid   (in $)."""
    bb = np.where(b["nb"] > 0, b["bp"][:, 0], np.nan) / 1000
    ba = np.where(b["na"] > 0, b["ap"][:, 0], np.nan) / 1000
    return (bb + ba) / 2, ba - bb, bb, ba


def diff_side(p0, s0, n0, p1, s1, n1, bids):
    """Level changes between two books on one side, inside the range both books show.
    If a side has fewer than LEVELS levels we see all of it (no bound)."""
    d0 = dict(zip(p0[:n0], s0[:n0]))
    d1 = dict(zip(p1[:n1], s1[:n1]))
    if bids:   # bids are sorted high -> low: the deepest visible price is the last one
        lo = max(p0[n0 - 1] if n0 == LEVELS else -1, p1[n1 - 1] if n1 == LEVELS else -1)
        keep = lambda p: p >= lo
    else:
        hi = min(p0[n0 - 1] if n0 == LEVELS else 10**9, p1[n1 - 1] if n1 == LEVELS else 10**9)
        keep = lambda p: p <= hi
    out = []
    for p in set(d0) | set(d1):
        dq = d1.get(p, 0.0) - d0.get(p, 0.0)
        if abs(dq) > 1e-6 and keep(p):
            out.append((p, dq))
    return out


def build_events(b, fills):
    """One event per changed price level.
    Columns: snap index, ts(ms), type (0 add / 1 cancel / 2 execution), side (+1 bid / -1 ask),
             price (1/1000 $), |size change|, depth from best (ticks), at-best flag, mid change (ticks)."""
    mid, _, _, _ = mid_spread(b)
    # (second, price) keys where a trade printed, widened by EXEC_WINDOW
    keys = set()
    for s, p in zip(fills["sec"].values, fills["pm"].values):
        for k in range(EXEC_WINDOW[0], EXEC_WINDOW[1] + 1):
            keys.add((int(s) + k, int(p)))
    rows = []
    bp, bs, ap, asz, nb, na, ts = b["bp"], b["bs"], b["ap"], b["asz"], b["nb"], b["na"], b["ts"]
    for i in range(1, len(ts)):
        if b["reset"][i]:          # fresh book: its link to the previous one is not proven
            continue
        dmid = (mid[i] - mid[i - 1]) / TICK if np.isfinite(mid[i] - mid[i - 1]) else 0.0
        sec = int(ts[i] // 1000)
        for side, P0, S0, N0, P1, S1, N1 in ((1, bp[i - 1], bs[i - 1], nb[i - 1], bp[i], bs[i], nb[i]),
                                             (-1, ap[i - 1], asz[i - 1], na[i - 1], ap[i], asz[i], na[i])):
            best = P0[0] if N0 > 0 else (P1[0] if N1 > 0 else np.nan)
            best1 = P1[0] if N1 > 0 else np.nan
            for p, dq in diff_side(P0, S0, N0, P1, S1, N1, side == 1):
                if dq > 0:
                    typ = 0
                else:
                    typ = 2 if (sec, int(p)) in keys else 1
                depth = abs(p - best) / (TICK * 1000) if np.isfinite(best) else 0.0
                at_best = float(p == best or p == best1)
                rows.append((i, ts[i], typ, side, p, abs(dq), depth, at_best, dmid))
    ev = np.array(rows, dtype=np.float64).reshape(-1, 9)
    return ev


def event_features(ev):
    """The paper's 7 per-event features:
       type                         (categorical: add / cancel / execution)
       side                         (+1 bid, -1 ask)
       log(1 + ticks from best)     how far from the top of the book
       log(1 + size)                size of the change in shares
       log(1 + ms since last event) how fast events arrive
       mid change in ticks          did this change move the mid right away
       at-best flag                 1 if the level is the best bid/ask
    Returns (types int, 6 continuous features)."""
    dt = np.diff(ev[:, 1], prepend=ev[0, 1] if len(ev) else 0)
    cont = np.column_stack([ev[:, 3], np.log1p(ev[:, 6]), np.log1p(ev[:, 5]),
                            np.log1p(np.maximum(dt, 0)), np.clip(ev[:, 8], -10, 10), ev[:, 7]])
    return ev[:, 2].astype(np.int64), cont.astype(np.float32)


# ── DECISION POINTS, LABELS, FEATURES ───────────────────────────────────────
def decisions(b, ev, end_ms, kind):
    """One decision point after every SAMPLE_EVERY-th book change.
    Everything a model sees at a decision point is from that moment or earlier."""
    n = len(b["ts"])
    idx = np.arange(SAMPLE_EVERY - 1, n, SAMPLE_EVERY)
    mid, spread, bb, ba = mid_spread(b)
    d = pd.DataFrame({"snap": idx, "ts": b["ts"][idx], "mid": mid[idx], "spread": spread[idx],
                      "bid": bb[idx], "ask": ba[idx]})
    # Events visible at this decision point: all with snap index <= idx
    d["ev_end"] = np.searchsorted(ev[:, 0], idx, side="right") if len(ev) else 0

    # Snapshot (L2) features: for each of LEVELS levels per side
    #   price offset from mid in ticks,  log(1 + size)
    m = mid[idx][:, None]
    def offs(p, nlev):
        o = (p[idx] / 1000 - m) / TICK
        return np.where(np.arange(LEVELS)[None, :] < nlev[idx][:, None], o, 0.0)
    l2 = np.concatenate([offs(b["bp"], b["nb"]), np.log1p(b["bs"][idx]),
                         offs(b["ap"], b["na"]), np.log1p(b["asz"][idx])], axis=1)
    l2 = np.nan_to_num(np.clip(l2, -50, 50)).astype(np.float32)

    # Classic book features (for logistic regression and MBOFusion's context branch)
    bd, ad = b["bs"][idx].sum(1), b["asz"][idx].sum(1)
    b1, a1 = b["bs"][idx, 0], b["asz"][idx, 0]
    d["imb_top"] = (b1 - a1) / (b1 + a1 + 1e-9)          # top-of-book imbalance, -1..+1
    d["imb_depth"] = (bd - ad) / (bd + ad + 1e-9)        # 10-level depth imbalance
    d["log_bid_depth"], d["log_ask_depth"] = np.log1p(bd), np.log1p(ad)
    # microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size); gap to mid in ticks
    d["micro_gap"] = ((d["ask"] * b1 + d["bid"] * a1) / (b1 + a1 + 1e-9) - d["mid"]) / TICK
    d["spread_ticks"] = d["spread"] / TICK
    # Order-flow imbalance between decision points (Cont, Kukanov & Stoikov 2014):
    #   OFI = [bid size if bid moved up, -old bid size if it moved down, else change in size]
    #       - [same for the ask, mirrored]
    pb, pa = pd.Series(d["bid"]).shift(), pd.Series(d["ask"]).shift()
    qb, qa = pd.Series(b1).shift(), pd.Series(a1).shift()
    e_b = np.where(d["bid"] > pb, b1, np.where(d["bid"] < pb, -qb, b1 - qb))
    e_a = np.where(d["ask"] < pa, a1, np.where(d["ask"] > pa, -qa, a1 - qa))
    d["ofi1"] = np.nan_to_num(np.sign(e_b - e_a) * np.log1p(np.abs(e_b - e_a)))
    d["ofi5"] = d["ofi1"].rolling(5, min_periods=1).sum()
    d["mid"] = d["mid"].ffill()                           # one-sided book: keep last mid
    for k in (1, 5, 20):
        d[f"ret{k}"] = d["mid"].diff(k).fillna(0) / TICK
    d["vol16"] = d["ret1"].rolling(16, min_periods=2).std().fillna(0)
    # event mix since the previous decision point
    ev_start = np.r_[0, d["ev_end"].values[:-1]]
    typ = ev[:, 2] if len(ev) else np.array([])
    cnt = lambda t: np.array([(typ[a:e] == t).sum() for a, e in zip(ev_start, d["ev_end"].values)])
    adds, cancels, execs = cnt(0), cnt(1), cnt(2)
    secs = np.maximum(d["ts"].diff().fillna(1000).values / 1000, 1e-3)
    d["log_ev_rate"] = np.log1p((adds + cancels + execs) / secs)
    d["log_exec_rate"] = np.log1p(execs / secs)
    d["add_cancel"] = (adds - cancels) / (adds + cancels + 1e-9)
    d["log_step_s"] = np.log1p(secs)
    d["min_to_end"] = (end_ms - d["ts"]) / 60_000 if kind == "crypto" else 0.0
    if KEEP_BOOK:   # raw book levels at each decision point, to simulate fills bigger than 1 share
        for nm, arr in (("bpx", b["bp"] / 1000), ("bsx", b["bs"]), ("apx", b["ap"] / 1000), ("asx", b["asz"])):
            for k in range(LEVELS):
                d[f"{nm}{k}"] = arr[idx, k].astype(np.float32)
    d = d[np.isfinite(d["mid"])].copy()
    feats = [c for c in d.columns if c not in ("snap", "ts", "mid", "spread", "bid", "ask", "ev_end")]
    d[feats] = d[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    return d.reset_index(drop=True), l2[d.index.values] if len(d) else l2[:0]


def hidden_activity(b, ev, snaps):
    """What a snapshot pipeline misses. Between two sampled books:
         gross = sum of |size change| over every event in between
         net   = sum of |size change| per price level from one sampled book to the next
       An order added and cancelled in between counts in gross but not in net.
       hidden share = 1 - net / gross."""
    gross = ev[(ev[:, 0] > snaps[0]) & (ev[:, 0] <= snaps[-1]), 5].sum()
    net = 0.0
    for i, j in zip(snaps[:-1], snaps[1:]):
        if b["reset"][i + 1:j + 1].any():
            continue
        for bid, P, S, N in ((True, b["bp"], b["bs"], b["nb"]), (False, b["ap"], b["asz"], b["na"])):
            net += sum(abs(dq) for _, dq in diff_side(P[i], S[i], N[i], P[j], S[j], N[j], bid))
    return float(gross), float(net)


CONTEXT = ["ret1", "vol16", "spread_ticks", "log_bid_depth", "log_ask_depth", "imb_depth",
           "log_ev_rate", "log_exec_rate", "add_cancel", "min_to_end", "log_step_s", "imb_top"]
CLASSIC = ["imb_top", "imb_depth", "micro_gap", "spread_ticks", "ofi1", "ofi5",
           "ret1", "ret5", "ret20", "vol16", "log_ev_rate", "add_cancel"]


def smoothed_change(mid, h):
    """Paper's label input:
         m_bar(t)   = (1/10) * sum_{j=0..9} mid(t - j)        trailing 10-step mean
         delta(t,h) = m_bar(t + h) - m_bar(t)                 smoothed move h steps ahead
       NaN where t < 9 or t + h is past the end of the contract."""
    mbar = pd.Series(mid).rolling(SMOOTH).mean().values
    out = np.full(len(mid), np.nan)
    out[:len(mid) - h] = mbar[h:] - mbar[:len(mid) - h]
    return out


def label(delta, alpha):
    """up if delta > alpha, down if delta < -alpha, flat otherwise.
       alpha_h = (1 / 2N) * sum |delta(t,h)|  =  half the average absolute move,
       computed on TRAINING contracts only (the paper used the test sequence)."""
    y = np.ones(len(delta), dtype=np.int64)
    y[delta > alpha] = 2
    y[delta < -alpha] = 0
    return y


# ── DATASET ─────────────────────────────────────────────────────────────────
def load_family(client, fam, contracts):
    """Download (or load from cache) every contract of one family and build
    events + decision points. Returns one dict of concatenated arrays."""
    kind = FAMILIES[fam]["kind"]
    parts, ev_types, ev_cont, l2s, ev_off = [], [], [], [], 0
    stats = []
    for k, c in contracts.iterrows():
        s, e = c["start"].strftime("%Y-%m-%d %H:%M:%S"), c["end"].strftime("%Y-%m-%d %H:%M:%S")
        t = time.time()
        raw = clean_books(ch_query(client, book_sql(c["mid"], s, e)))
        fills = ch_query(client, fills_sql(c["mid"], s, e))
        if len(raw) < SAMPLE_EVERY * (max(HORIZONS) + SMOOTH + 50):
            print(f"   skip {c['title'][:60]}: only {len(raw)} book rows")
            continue
        b = to_arrays(raw)
        ev = build_events(b, fills)
        d, l2 = decisions(b, ev, int(c["end"].value // 10**6), kind)
        if len(ev) == 0 or len(d) == 0:
            continue
        typ, cont = event_features(ev)
        d["ev_end"] += ev_off
        d["ev_start"] = ev_off                       # windows never reach into another contract
        d["contract"] = k
        d["block"] = k * 10_000 + (d["ts"] - d["ts"].iloc[0]) // (BLOCK_S * 1000)
        for h in HORIZONS:
            d[f"delta{h}"] = smoothed_change(d["mid"].values, h)
        gross, net = hidden_activity(b, ev, d["snap"].values)
        stats.append(dict(contract=k, title=c["title"], rows=len(raw), events=len(ev), decisions=len(d),
                          resets=int(b["reset"].sum()), exec_share=float((typ == 2).mean()),
                          cancel_share=float((typ == 1).mean()), add_share=float((typ == 0).mean()),
                          fills=len(fills), sec=(b["ts"][-1] - b["ts"][0]) / 1000,
                          delay_p50=float(np.median(b["delay"])), delay_p99=float(np.percentile(b["delay"], 99)),
                          gross_change=gross, net_change=net))
        parts.append(d)
        ev_types.append(typ)
        ev_cont.append(cont)
        l2s.append(l2)
        ev_off += len(ev)
        print(f"   [{fam}] {k + 1}/{len(contracts)} {c['title'][:55]:55s} "
              f"{len(raw):>8,} books {len(ev):>8,} events {len(d):>7,} decisions  {time.time() - t:4.0f}s")
    d = pd.concat(parts, ignore_index=True)
    # Split by contract, in time order
    order = sorted(d["contract"].unique())
    # Fewer contracts than planned (e.g. one had no book): keep the same 60/15/25 proportions.
    # (Before this fix the fallback gave ETH 16/1/2, so its first-run test set was only 2 contracts.)
    n = len(order)
    ntr, nva = SPLIT[:2] if n >= sum(SPLIT) else (round(n * 0.6), max(1, round(n * 0.15)))
    d["split"] = np.where(d["contract"].isin(order[:ntr]), "train",
                          np.where(d["contract"].isin(order[ntr:ntr + nva]), "val", "test"))
    return dict(dec=d, ev_type=np.concatenate(ev_types), ev_cont=np.concatenate(ev_cont),
                l2=np.concatenate(l2s), stats=pd.DataFrame(stats), contracts=contracts)


def l2_windows(data):
    """Index of the last T_SNAPS decision points of the same contract (-1 = padding)."""
    d = data["dec"]
    pos = np.arange(len(d))
    first = pos - d.groupby("contract").cumcount().values
    w = pos[:, None] - np.arange(T_SNAPS - 1, -1, -1)[None, :]
    return np.where(w >= first[:, None], w, -1)


# ── MODELS (PyTorch) ────────────────────────────────────────────────────────
def build_models():
    import torch
    import torch.nn as nn

    D, HEADS, FF = 16, 4, 32

    class Block(nn.Module):
        """x = x + Attention(LayerNorm(x));  x = x + FeedForward(x)
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head) + causal mask) V, 4 heads of size 4.
        Causal: each position only looks at itself and the past."""
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(D)
            self.qkv = nn.Linear(D, 3 * D)
            self.out = nn.Linear(D, D)
            self.ff = nn.Sequential(nn.Linear(D, FF), nn.GELU(), nn.Linear(FF, D))

        def forward(self, x):
            B, T, _ = x.shape
            q, k, v = self.qkv(self.ln(x)).view(B, T, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.out(a.transpose(1, 2).reshape(B, T, D))
            return x + self.ff(x)

    class Encoder(nn.Module):
        """Shared backbone: tokens -> 2 causal blocks -> [newest token ; mean of valid tokens]."""
        def __init__(self, T):
            super().__init__()
            self.pos = nn.Parameter(torch.zeros(T, D))
            nn.init.normal_(self.pos, std=0.02)
            self.blocks = nn.ModuleList([Block(), Block()])
            self.ln = nn.LayerNorm(D)

        def forward(self, tok, valid):
            # Every window is full (rows without a full history are dropped), so the
            # fast causal attention kernel can be used with no padding mask.
            x = tok + self.pos
            for blk in self.blocks:
                x = blk(x)
            x = self.ln(x)
            return torch.cat([x[:, -1], x.mean(1)], 1)        # dual readout (paper): newest + mean, 2*D

    def head(n_in, hidden):
        return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, 3))

    class MBOFormer(nn.Module):
        """Event history in, 3 classes out. Token = embed(type) + Linear(6 numbers)."""
        def __init__(self):
            super().__init__()
            self.typ = nn.Embedding(3, D)
            self.lin = nn.Linear(6, D)
            self.enc = Encoder(T_EVENTS)
            self.head = head(2 * D, D)

        def summary(self, b):
            return self.enc(self.typ(b["ev_type"]) + self.lin(b["ev_cont"]), b["ev_valid"])

        def forward(self, b):
            return self.head(self.summary(b))

    class MBOFusion(MBOFormer):
        """MBOFormer + a slow context branch: 12 book statistics over the last 16
        decision points -> Linear -> causal conv -> [last ; mean ; max]."""
        def __init__(self):
            super().__init__()
            self.ctx_in = nn.Linear(len(CONTEXT), D)
            self.conv = nn.Conv1d(D, D, kernel_size=4)
            self.head = head(2 * D + 3 * D, 2 * D)

        def forward(self, b):
            c = self.ctx_in(b["ctx"]) * b["l2_valid"][..., None]          # (B, 16, D)
            c = torch.nn.functional.pad(c.transpose(1, 2), (3, 0))       # causal: pad the past only
            c = torch.relu(self.conv(c)).transpose(1, 2)
            cs = torch.cat([c[:, -1], c.mean(1), c.max(1).values], 1)
            return self.head(torch.cat([self.summary(b), cs], 1))

    class L2Former(nn.Module):
        """Same backbone, but tokens are the last 16 sampled book snapshots (40 numbers each)."""
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4 * LEVELS, D)
            self.enc = Encoder(T_SNAPS)
            self.head = head(2 * D, D)

        def forward(self, b):
            return self.head(self.enc(self.lin(b["l2"]), b["l2_valid"]))

    return dict(MBOFormer=MBOFormer, MBOFusion=MBOFusion, L2Former=L2Former)


class Batcher:
    """Builds model inputs for a set of decision points by gathering windows
    out of the flat event / snapshot arrays (nothing is copied per sample in advance)."""
    def __init__(self, data, norm):
        import torch
        self.torch = torch
        d = data["dec"]
        self.ev_end = d["ev_end"].values
        self.ev_start = d["ev_start"].values
        self.ev_type = data["ev_type"]
        self.ev_cont = (data["ev_cont"] - norm["ev_mu"]) / norm["ev_sd"]
        self.l2 = (data["l2"] - norm["l2_mu"]) / norm["l2_sd"]
        self.ctx = ((d[CONTEXT].values - norm["ctx_mu"]) / norm["ctx_sd"]).astype(np.float32)
        self.l2w = data["l2w"]

    def __call__(self, rows):
        t = self.torch
        e = self.ev_end[rows][:, None] - np.arange(T_EVENTS, 0, -1)[None, :]      # right-aligned window
        valid = e >= self.ev_start[rows][:, None]
        e = np.where(valid, e, 0)
        w = self.l2w[rows]
        lv = w >= 0
        w = np.where(lv, w, 0)
        return dict(ev_type=t.from_numpy(np.where(valid, self.ev_type[e], 0)),
                    ev_cont=t.from_numpy(np.where(valid[..., None], self.ev_cont[e], 0).astype(np.float32)),
                    ev_valid=t.from_numpy(valid),
                    l2=t.from_numpy(np.where(lv[..., None], self.l2[w], 0).astype(np.float32)),
                    ctx=t.from_numpy(np.where(lv[..., None], self.ctx[w], 0).astype(np.float32)),
                    l2_valid=t.from_numpy(lv))


def norm_stats(data, train_rows):
    d = data["dec"]
    tr_contracts = d.loc[train_rows, "contract"].unique()
    ev_mask = np.zeros(len(data["ev_type"]), bool)
    for c in tr_contracts:
        g = d[d["contract"] == c]
        ev_mask[g["ev_start"].iloc[0]:g["ev_end"].max()] = True
    ev = data["ev_cont"][ev_mask]
    ctx = d.loc[train_rows, CONTEXT].values
    l2 = data["l2"][train_rows]
    sd = lambda a: np.where(a.std(0) > 1e-6, a.std(0), 1.0)
    out = dict(ev_mu=ev.mean(0), ev_sd=sd(ev), l2_mu=l2.mean(0), l2_sd=sd(l2),
               ctx_mu=ctx.mean(0), ctx_sd=sd(ctx))
    out["ev_mu"][[0, 5]], out["ev_sd"][[0, 5]] = 0, 1          # keep side and at-best flags as they are
    return {k: v.astype(np.float32) for k, v in out.items()}


def macro_f1(y, p):
    """macro-F1 = average over the 3 classes of  F1_c = 2 * precision_c * recall_c / (precision_c + recall_c).
    Every class counts equally, so predicting "flat" all the time scores badly."""
    f = []
    for c in range(3):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def train_one(name, data, batcher, y, rows, seed):
    """AdamW + cross-entropy; keep the checkpoint with the lowest validation loss;
    stop when validation loss fails to improve by EARLY_STOP for 2 epochs (paper)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_models()[name]()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    tr, va = rows["train"], rows["val"]
    va = va if len(va) <= 20_000 else rng.choice(va, 20_000, replace=False)
    yt = torch.from_numpy(y)

    def val_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(va), 2048):
                r = va[i:i + 2048]
                tot += lossf(model(batcher(r)), yt[r]).item() * len(r)
        return tot / len(va)

    best, best_state, bad = np.inf, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        ep_rows = rng.choice(tr, min(TRAIN_PER_EPOCH, len(tr)), replace=False)
        for i in range(0, len(ep_rows), BATCH):
            r = ep_rows[i:i + BATCH]
            loss = lossf(model(batcher(r)), yt[r])
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = val_loss()
        print(f"      {name} seed {seed} epoch {ep + 1}: val loss {vl:.4f}", flush=True)
        if vl < best - EARLY_STOP:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, ep + 1


def predict(model, batcher, rows):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 4096):
            out.append(torch.softmax(model(batcher(rows[i:i + 4096])), 1).numpy())
    return np.concatenate(out)


def logistic(data, y, rows):
    """Multinomial logistic regression on 12 classic book features (CLASSIC)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = data["dec"][CLASSIC].values
    sc = StandardScaler().fit(X[rows["train"]])
    m = LogisticRegression(max_iter=500).fit(sc.transform(X[rows["train"]]), y[rows["train"]])
    return m.predict_proba(sc.transform(X[rows["test"]]))


def latency(n=1000):
    """Batch-of-one, single CPU thread, like the paper's Table 2."""
    import torch
    torch.set_num_threads(1)
    rows = []
    for name, cls in build_models().items():
        m = cls().eval()
        b = dict(ev_type=torch.zeros(1, T_EVENTS, dtype=torch.long), ev_cont=torch.randn(1, T_EVENTS, 6),
                 ev_valid=torch.ones(1, T_EVENTS, dtype=torch.bool), l2=torch.randn(1, T_SNAPS, 4 * LEVELS),
                 ctx=torch.randn(1, T_SNAPS, len(CONTEXT)), l2_valid=torch.ones(1, T_SNAPS, dtype=torch.bool))
        ts = []
        with torch.no_grad():
            for i in range(n + 100):
                t = time.perf_counter()
                m(b)
                if i >= 100:
                    ts.append((time.perf_counter() - t) * 1000)
        rows.append(dict(model=name, params=sum(p.numel() for p in m.parameters()),
                         p50_ms=np.percentile(ts, 50), p95_ms=np.percentile(ts, 95), p99_ms=np.percentile(ts, 99)))
    return pd.DataFrame(rows)


# ── TRADING TEST (the step the paper left out) ──────────────────────────────
def fee(p):
    """Polymarket taker fee per share: fee(p) = 0.25 * p * (p * (1 - p))^2."""
    return FEE_RATE * p * (p * (1 - p)) ** FEE_EXP


def trade_pnl(d, direction, h, lag=0):
    """Profit per share, in cents, of a taker who acts on every non-flat signal:
         long  (signal up):   sell at bid(t+lag+h)  - buy at ask(t+lag)  - fees
         short (signal down): sell at bid(t+lag)    - buy at ask(t+lag+h) - fees
       "mid" version ignores the spread:  direction * (mid(t+lag+h) - mid(t+lag)).
       Positions never cross into another contract."""
    n = len(d)
    i0 = np.arange(n) + lag
    i1 = i0 + h
    ok = (i1 < n) & (direction != 0)
    i0c, i1c = np.minimum(i0, n - 1), np.minimum(i1, n - 1)
    same = d["contract"].values[i0c] == d["contract"].values
    same &= d["contract"].values[i1c] == d["contract"].values
    ok &= same
    bid, ask, mid = d["bid"].values, d["ask"].values, d["mid"].values
    long_ = direction == 1
    entry = np.where(long_, ask[i0c], bid[i0c])
    exit_ = np.where(long_, bid[i1c], ask[i1c])
    gross = np.where(long_, exit_ - entry, entry - exit_)
    ok &= np.isfinite(gross)
    fees = fee(entry) + fee(exit_)
    midpnl = direction * (mid[i1c] - mid[i0c])
    return pd.DataFrame({"mid": midpnl * 100, "spread": gross * 100, "fees": (gross - fees) * 100,
                         "block": d["block"].values, "ts": d["ts"].values, "contract": d["contract"].values})[ok]


def block_boot(values, blocks, n=N_BOOT, seed=0):
    """Mean and 95% CI, resampling whole (contract, minute) blocks."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": values, "b": blocks})
    g = df.groupby("b")["v"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    if len(g) < 2:
        return values.mean() if len(values) else np.nan, np.nan, np.nan
    idx = rng.integers(0, len(g), (n, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def f1_diff_boot(y, pa, pb, blocks, n=N_BOOT, seed=0):
    """Paired bootstrap of macro-F1(A) - macro-F1(B): same resampled blocks for both."""
    rng = np.random.default_rng(seed)
    ub, inv = np.unique(blocks, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(ub))]
    diffs = []
    for _ in range(n):
        pick = np.concatenate([groups[k] for k in rng.integers(0, len(ub), len(ub))])
        diffs.append(macro_f1(y[pick], pa[pick]) - macro_f1(y[pick], pb[pick]))
    return macro_f1(y, pa) - macro_f1(y, pb), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


# ── EXPERIMENT ──────────────────────────────────────────────────────────────
def run_family(fam, data, seeds):
    import torch
    torch.set_num_threads(THREADS)
    d = data["dec"]
    data["l2w"] = l2_windows(data)
    f1_rows, trade_rows, curves = [], [], []
    for h in HORIZONS:
        delta = d[f"delta{h}"].values
        # a full event window AND a full snapshot window, so all models see the same rows
        has = np.isfinite(delta) & (d["ev_end"].values - d["ev_start"].values >= T_EVENTS) & (data["l2w"][:, 0] >= 0)
        tr = d["split"].values == "train"
        alpha = 0.5 * np.mean(np.abs(delta[has & tr]))           # alpha_h from TRAIN only
        y = label(np.nan_to_num(delta), alpha)
        rows = {s: np.where(has & (d["split"].values == s))[0] for s in ("train", "val", "test")}
        te = rows["test"]
        yte = y[te]
        blocks = d["block"].values[te]
        shares = np.bincount(yte, minlength=3) / len(yte)
        print(f"\n [{fam}] h={h}: alpha={alpha * 100:.3f}c  train/val/test={len(rows['train']):,}/"
              f"{len(rows['val']):,}/{len(te):,}  test classes down/flat/up="
              f"{shares[0]:.2f}/{shares[1]:.2f}/{shares[2]:.2f}")
        norm = norm_stats(data, rows["train"])
        batcher = Batcher(data, norm)
        probs = {}
        # baselines
        probs["Always flat"] = [np.tile([0, 1, 0], (len(te), 1)).astype(float)]
        probs["Logistic (12 features)"] = [logistic(data, y, rows)]
        for name in ["MBOFormer", "MBOFusion", "L2Former"]:
            probs[name] = []
            for seed in (SEEDS_FUSION if name == "MBOFusion" else seeds):
                t = time.time()
                model, ep = train_one(name, data, batcher, y, rows, seed)
                probs[name].append(predict(model, batcher, te))
                print(f"   {name:10s} seed {seed}: macro-F1 {macro_f1(yte, probs[name][-1].argmax(1)):.3f} "
                      f"({ep} epochs, {time.time() - t:.0f}s)")
        for name, ps in probs.items():
            f1s = [macro_f1(yte, p.argmax(1)) for p in ps]
            f1_rows.append(dict(family=fam, h=h, model=name, f1_mean=np.mean(f1s), f1_sd=np.std(f1s),
                                n_seeds=len(ps), n_test=len(te), alpha_c=alpha * 100,
                                share_down=shares[0], share_flat=shares[1], share_up=shares[2]))
        # event vs snapshot, paired bootstrap on seed-averaged probabilities
        ens = {k: np.mean(v, 0) for k, v in probs.items()}
        for a, bname in [("MBOFormer", "L2Former"), ("MBOFusion", "L2Former"), ("MBOFormer", "Logistic (12 features)")]:
            diff, lo, hi = f1_diff_boot(yte, ens[a].argmax(1), ens[bname].argmax(1), blocks)
            f1_rows.append(dict(family=fam, h=h, model=f"{a} - {bname}", f1_mean=diff, ci_lo=lo, ci_hi=hi))
        # trading: signal = argmax of seed-averaged probabilities; flat = no trade
        dte = d.iloc[te].reset_index(drop=True)
        dte["contract"] = d["contract"].values[te]
        signals = {k: ens[k].argmax(1) - 1 for k in ("MBOFormer", "MBOFusion", "L2Former", "Logistic (12 features)")}
        signals["Oracle (knows the label)"] = yte - 1
        conf = ens["MBOFormer"].max(1)
        top = conf >= np.quantile(conf, 0.9)
        signals["MBOFormer, top-10% confidence"] = np.where(top, ens["MBOFormer"].argmax(1) - 1, 0)
        for name, sig in signals.items():
            for lag in (0, 1):
                pnl = trade_pnl(dte, sig, h, lag)
                row = dict(family=fam, h=h, signal=name, lag=lag, n_trades=len(pnl),
                           hit_rate=float((pnl["mid"] > 0).mean()) if len(pnl) else np.nan)
                for col in ("mid", "spread", "fees"):
                    m, lo, hi = block_boot(pnl[col].values, pnl["block"].values)
                    row.update({f"{col}_c": m, f"{col}_lo": lo, f"{col}_hi": hi})
                trade_rows.append(row)
                if lag == 0 and name in ("MBOFormer", "Oracle (knows the label)") and h == 10:
                    curves.append(pnl.assign(family=fam, signal=name))
    return pd.DataFrame(f1_rows), pd.DataFrame(trade_rows), pd.concat(curves) if curves else pd.DataFrame()


def slug(fam):
    return re.sub(r"[^a-z0-9]+", "_", fam.lower()).strip("_")


def main(quick=False, families=None):
    """Each family writes its own files under PARTS, so families can run as
    separate processes side by side:  --family "BTC 15m"  etc."""
    global SEEDS, MAX_EPOCHS, TRAIN_PER_EPOCH, HORIZONS
    if quick:
        SEEDS, MAX_EPOCHS, TRAIN_PER_EPOCH, HORIZONS = [0], 2, 20_000, [10, 50]
    from dotenv import load_dotenv
    from clickhouse_driver import Client
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    client = Client(host=os.environ["PROBALYTICS_CLICKHOUSE_HOST"],
                    user=os.environ["PROBALYTICS_CLICKHOUSE_USERNAME"],
                    password=os.environ["PROBALYTICS_CLICKHOUSE_PASSWORD"],
                    database="probalytics", secure=True, port=9440)
    os.makedirs(PARTS, exist_ok=True)
    t0 = time.time()
    print("1) picking contracts", flush=True)
    contracts = pick_contracts(client, quick)
    for fam, c in contracts.items():
        if families and fam not in families:
            continue
        print(f"\n2) [{fam}] downloading books + building events ({len(c)} contracts)", flush=True)
        data = load_family(client, fam, c)
        p = os.path.join(PARTS, slug(fam))
        data["stats"].assign(family=fam).to_csv(f"{p}_data_stats.csv", index=False)
        c.assign(family=fam).to_csv(f"{p}_contracts.csv", index=False)
        print(f"\n3) [{fam}] training", flush=True)
        f, tr, cu = run_family(fam, data, SEEDS)
        f.to_csv(f"{p}_f1.csv", index=False)
        tr.to_csv(f"{p}_trading.csv", index=False)
        cu.to_csv(f"{p}_curves.csv", index=False)
    print(f"\ndone in {(time.time() - t0) / 60:.0f} min", flush=True)
    if not families:          # with --family, run --plots once every family has finished
        plots()


def read_parts(kind):
    """Concatenate one kind of per-family file, in FAMILIES order."""
    fs = [os.path.join(PARTS, f"{slug(f)}_{kind}.csv") for f in FAMILIES]
    fs = [f for f in fs if os.path.exists(f)]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


# ── CHARTS ──────────────────────────────────────────────────────────────────
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
MODEL_COLORS = {"MBOFormer": BLUE, "MBOFusion": ORANGE, "L2Former": AQUA,
                "Logistic (12 features)": YELLOW, "Always flat": INK2}
MODEL_LABELS = {"MBOFormer": "MBOFormer (events)", "MBOFusion": "MBOFusion (events + context)",
                "L2Former": "L2-Former (snapshots)", "Logistic (12 features)": "Logistic regression",
                "Always flat": "Always 'flat'"}


def style(ax, title, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    if not os.path.exists(f"{OUT}_latency.csv"):   # timed alone, with nothing else running
        print("timing models (latency)...", flush=True)
        latency().to_csv(f"{OUT}_latency.csv", index=False)
    f1 = read_parts("f1")
    tr = read_parts("trading")
    fams = list(dict.fromkeys(f1["family"]))

    # 1) Macro-F1 by horizon, the paper's Table 1 as a chart
    fig, axes = plt.subplots(1, len(fams), figsize=(max(4.6 * len(fams), 8), 4.2), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        g = f1[(f1["family"] == fam) & f1["model"].isin(MODEL_COLORS)]
        for m, col in MODEL_COLORS.items():
            s = g[g["model"] == m].sort_values("h")
            if s.empty:
                continue
            ls = "--" if m == "Always flat" else "-"
            ax.plot(s["h"], s["f1_mean"], color=col, lw=2, ls=ls, marker="o", ms=5,
                    label=MODEL_LABELS[m], markeredgecolor=SURFACE, markeredgewidth=1.5)
            if s["f1_sd"].fillna(0).max() > 0:
                ax.fill_between(s["h"], s["f1_mean"] - s["f1_sd"], s["f1_mean"] + s["f1_sd"], color=col, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(sorted(g["h"].unique()))
        ax.set_xticklabels(sorted(g["h"].unique()))
        style(ax, fam, "test macro-F1" if ax is axes[0][0] else None, "horizon h (decision steps ahead)")
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("How well each model predicts up / flat / down\nhigher is better; shaded band = spread over seeds",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}_1_macro_f1.png", dpi=150)
    plt.close(fig)

    # 2) Event minus snapshot, with paired bootstrap 95% CI
    pairs = [("MBOFormer - L2Former", BLUE, "MBOFormer − L2-Former"),
             ("MBOFusion - L2Former", ORANGE, "MBOFusion − L2-Former"),
             ("MBOFormer - Logistic (12 features)", YELLOW, "MBOFormer − Logistic")]
    fig, axes = plt.subplots(1, len(fams), figsize=(max(4.6 * len(fams), 8), 3.8), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        hs = sorted(f1[f1["family"] == fam]["h"].unique())
        x = np.arange(len(hs))
        for k, (m, col, lab) in enumerate(pairs):
            s = f1[(f1["family"] == fam) & (f1["model"] == m)].set_index("h").reindex(hs)
            xs = x + (k - 1) * 0.22
            ax.errorbar(xs, s["f1_mean"], yerr=[s["f1_mean"] - s["ci_lo"], s["ci_hi"] - s["f1_mean"]],
                        fmt="o", color=col, ms=6, lw=2, capsize=0, label=lab, markeredgecolor=SURFACE)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in hs])
        style(ax, fam, "macro-F1 difference" if ax is axes[0][0] else None)
    axes[0][0].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle("Does reading every event beat reading snapshots?\nabove 0 = first model better; bars = 95% block-bootstrap CI",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}_2_event_vs_snapshot.png", dpi=150)
    plt.close(fig)

    # 3) Trading: profit per trade at mid, after spread, after spread + fees (MBOFormer and oracle)
    fig, axes = plt.subplots(1, len(fams), figsize=(max(4.6 * len(fams), 8), 4.0), sharey=False, squeeze=False)
    steps = [("mid_c", "mid_lo", "mid_hi", BLUE, "at mid (no costs)"),
             ("spread_c", "spread_lo", "spread_hi", ORANGE, "after crossing the spread"),
             ("fees_c", "fees_lo", "fees_hi", MAGENTA, "after spread + taker fee")]
    for ax, fam in zip(axes[0], fams):
        hs = sorted(tr[tr["family"] == fam]["h"].unique())
        x = np.arange(len(hs))
        for sig, mk, off in [("MBOFormer", "o", -0.2), ("Oracle (knows the label)", "D", 0.2)]:
            s = tr[(tr["family"] == fam) & (tr["signal"] == sig) & (tr["lag"] == 0)].set_index("h").reindex(hs)
            for k, (c, lo, hi, col, lab) in enumerate(steps):
                xs = x + off + (k - 1) * 0.08
                ax.errorbar(xs, s[c], yerr=[s[c] - s[lo], s[hi] - s[c]], fmt=mk, color=col, ms=6, lw=1.5,
                            markeredgecolor=SURFACE, label=f"{'model' if sig == 'MBOFormer' else 'oracle'}: {lab}")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in hs])
        style(ax, fam, "profit per share per trade, cents" if ax is axes[0][0] else None)
        ax.text(0.01, 0.98, "circles = MBOFormer   diamonds = knows the label", transform=ax.transAxes,
                va="top", fontsize=8, color=INK2)
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_[:3], [l.split(": ")[1] for l in l_[:3]], frameon=False, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Can the signal be traded? Buy/sell at the signal as a taker, close h steps later\ncents per share per trade, 95% CI",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT}_3_trading.png", dpi=150)
    plt.close(fig)

    # 4) Cumulative profit of MBOFormer at h=10 on the test contracts (1 share per signal)
    cu = read_parts("curves")
    if len(cu):
        fig, axes = plt.subplots(1, len(fams), figsize=(max(4.6 * len(fams), 8), 3.8), squeeze=False)
        for ax, fam in zip(axes[0], fams):
            s = cu[(cu["family"] == fam) & (cu["signal"] == "MBOFormer")].sort_values(["contract", "ts"])
            n = np.arange(1, len(s) + 1)
            for col, c, lab in [("mid", BLUE, "at mid"), ("spread", ORANGE, "after spread"),
                                ("fees", MAGENTA, "after spread + fee")]:
                y = s[col].cumsum().values / 100
                ax.plot(n, y, color=c, lw=2, label=lab)
                ax.annotate((f"−${-y[-1]:,.0f}" if y[-1] < 0 else f"${y[-1]:,.0f}"), (n[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK)
            ax.axhline(0, color=INK2, lw=1)
            style(ax, fam, "cumulative profit, $" if ax is axes[0][0] else None,
                  "trades, in time order")
        axes[0][0].legend(frameon=False, fontsize=8.5, loc="lower left")
        fig.suptitle("MBOFormer at h=10 on the unseen test contracts\n1 share per signal", x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{OUT}_4_equity.png", dpi=150)
        plt.close(fig)

    # 5) Latency: model vs Polymarket's own feed delay
    st = read_parts("data_stats")
    if os.path.exists(f"{OUT}_latency.csv") and len(st):
        lat = pd.read_csv(f"{OUT}_latency.csv")
        names = [f"{r.model} ({r.params:,} params)" for r in lat.itertuples()]
        vals = list(lat["p50_ms"])
        cols = [MODEL_COLORS.get(m, BLUE) for m in lat["model"]]
        gap = (st["sec"] * 1000 / st["decisions"]).median()
        names += ["time between decision points (median)", "Probalytics delivery delay (median)"]
        vals += [gap, st["delay_p50"].median()]
        cols += [INK2, INK2]
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=cols, height=0.6)
        for yy, v in zip(y, vals):
            ax.text(v * 1.15, yy, f"{v:,.2f} ms", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xscale("log")
        style(ax, "Model speed (1 CPU thread, batch of 1, median)\nvs how fast the market itself moves", None, "milliseconds (log scale)")
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(f"{OUT}_5_latency.png", dpi=150)
        plt.close(fig)

    # 6) What snapshots hide: event mix and share of order-flow invisible between sampled books
    if len(st):
        g = st.groupby("family").agg(add=("add_share", "mean"), cancel=("cancel_share", "mean"),
                                     execu=("exec_share", "mean"), gross=("gross_change", "sum"),
                                     net=("net_change", "sum")).reindex(fams)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
        left = np.zeros(len(g))
        y = np.arange(len(g))[::-1]
        for col, c, lab in [("add", BLUE, "orders added"), ("cancel", ORANGE, "orders cancelled"),
                            ("execu", AQUA, "trades (executions)")]:
            a1.barh(y, g[col], left=left, color=c, height=0.6, label=lab, edgecolor=SURFACE, linewidth=2)
            for yy, l, v in zip(y, left, g[col]):
                if v > 0.04:
                    a1.text(l + v / 2, yy, f"{v:.0%}", ha="center", va="center", fontsize=8.5, color="white")
            left += g[col].values
        a1.set_yticks(y)
        a1.set_yticklabels(g.index)
        a1.set_xlim(0, 1)
        style(a1, "What the order-book events are")
        a1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
        hidden = 1 - g["net"] / g["gross"]
        a2.barh(y, hidden, color=BLUE, height=0.6)
        for yy, v in zip(y, hidden):
            a2.text(v + 0.01, yy, f"{v:.0%}", va="center", fontsize=8.5, color=INK)
        a2.set_yticks(y)
        a2.set_yticklabels(g.index)
        a2.set_xlim(0, 1)
        style(a2, "Order activity never seen by a\nsnapshot-every-10-changes view")
        a2.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(f"{OUT}_6_what_snapshots_miss.png", dpi=150)
        plt.close(fig)
    print("charts written:", sorted(f for f in os.listdir(".") if f.startswith(OUT) and f.endswith(".png")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--family", action="append", help="run only this family (repeatable)")
    a = ap.parse_args()
    plots() if a.plots else main(quick=a.quick, families=a.family)


# ── FINDINGS (October 2026 run; full write-up in MBOFORMER_FINDINGS.md) ─────
# Data: 59 Polymarket contracts (BTC 15m, ETH 15m, sports/esports), 11.7M book
# changes -> 9.1M events -> 1.17M decision points. Test = last 5 contracts per group.
#
# 1. The paper's claim holds: an EVENT model had the best macro-F1 in 12 of 12
#    settings (paper: 11 of 12). Same backbone, events vs snapshots: MBOFormer
#    is clearly better in 9 of 12 (+0.03 to +0.11), clearly worse in 1
#    (BTC h=10, -0.018), and too close to call in 2. It beats logistic
#    regression on classic book signals in all 12.
# 2. Why: 13-21% of order activity is placed-and-pulled between two snapshots,
#    so a snapshot model never sees it.
# 3. Prediction is not profit: crossing the spread on every signal loses
#    0.6-3.0c per share after fees in all 12 settings. Even a perfect-foresight
#    oracle loses at h=10 and h=20 (moves are smaller than the spread).
# 4. Lead, not a result: the top-10% most confident signals at h=50-100
#    (3-6 seconds) made +1.4 to +4.7c after costs in BTC/ETH. But the cutoff
#    came from the test set, there are only 5 test contracts, trades overlap,
#    and latency is optimistic. Needs a clean re-test before any money goes in.
# 5. Speed: MBOFormer 1.2 ms, L2-Former 0.7 ms per prediction on 1 CPU thread,
#    both far below Polymarket's 30-70 ms between decisions and ~45 ms+ feed
#    delay, so model latency is not the bottleneck here.
