"""
MBOFormer on Polymarket, v3: leak-free events, real 2026 costs, and a search for a strategy that pays
=====================================================================================================
What was wrong in v1/v2 (test_mboformer.py, retest_mboformer.py)
  1. LEAK. An event was typed "execution" if a fill printed at that price up to 6 s LATER
     (fills carry on-chain block time, 1-3 s after the match). The model saw the future.
  2. FEES. We used Polymarket's January 2026 crypto fee, 0.25*p*(p(1-p))^2 (max 0.78c/share).
     Fee Structure V2 (30 Mar 2026, docs.polymarket.com) is  fee = 0.07 * p * (1-p)  per share,
     max 1.75c at p = 0.5, charged on every taker fill. Makers pay 0 and get a 20% rebate pool.
  3. LATENCY. Crypto markets have a 150 ms taker delay (since 4 Sep 2026) + ~45 ms feed delay
     + network. We use 0.25 s from signal to taker fill (user's figure: ~200 ms + feed).
  4. TARGET. Models predicted the smoothed mid from the signal moment, with a threshold far
     below trading costs. Nothing in training knew about latency, spread or fees.

PAPER-FAITHFUL PART (science)
  Same as the paper and v1: one decision per 10 book changes, last 128 events, 7 features,
  smoothed-mid up/flat/down labels at h = 10/20/50/100 steps, alpha from TRAIN only, 3 seeds,
  MBOFormer (7,203 params) vs L2-Former (same backbone on 16 snapshots) vs logistic regression.
  ONLY change: event types are now knowable at their timestamp:
      0 add       a level's size went up
      1 decrease  a level's size went down (cancel or trade: the book alone cannot tell)
      2 trade     a fill as published on-chain, placed at the END of its block second,
                  side = taker direction in "Up"-token terms
  Execution-matched decreases (the old window rule) are kept ONLY as ground truth for
  simulating whether our own resting maker orders would have filled. Never a model input.

PRE-REGISTERED TRADING SPEC (fixed before any v3 number existed)
  Data      : BTC and ETH 15-minute Up/Down. Train 24 + validation 10 contracts (8-23 Oct,
              time order), TEST 40 fresh contracts closing 23 Oct 00:00 - 25 Oct 00:00 UTC.
  Costs     : V2 taker fee on every taker fill, both legs. Makers 0 fee; the 20% rebate is
              reported separately and NOT counted in the pass test.
  Size      : 5 shares (Polymarket minimum order), prices walked through 5 visible levels.
  Latency   : taker fills at the book 0.25 s after the decision; maker orders and cancels
              take 0.10 s (no taker delay for passive orders).
  Spot data : Binance 1-second klines, used only after the second has closed (+50 ms).
              Fair value of "Up": FV = Phi( ln(S/K) / (sigma * sqrt(tau)) ),
              S = spot, K = spot at the window start, sigma = 30-min realised 1s vol, tau = s left.
  Strategies (every threshold chosen on the 10 VALIDATION contracts by total profit):
    S1 Paper signal      : honest MBOFormer h=100, top-10% confidence, taker in, taker out after 100 steps
    S2 Cost-aware + spot : MBOFusion (paper's context branch) with spot/fair-value features, trained
                           on the TRADABLE label: up if  bid(t+0.25s+H) - ask(t+0.25s) - fees > 0,
                           down if the mirror (buy "Down") is profitable, else flat. H = 10 s, 30 s.
    S3 Cost-aware events : same label, MBOFormer (no spot)                 -> what spot adds
    S4 Spot logistic     : same label, logistic regression on classic + spot features -> what the transformer adds
    S5 Fair value, hold  : buy Up (Down) when FV beats the ask (1-bid) by > m after fee, hold to resolution
    S6 Maker             : quote 5 shares at best bid and best ask, inventory cap one lot, hold leftover
                           inventory to resolution. Gates that pull a quote:
                             none / paper MBOFormer / fair value / S2 model / fair value + S2 model
                           Fill = executions at our price beyond the visible queue ahead of us (cancels
                           assumed to come from behind us), or the market trading through our price.
  Unit      : $ per 15-minute contract at 5 shares, averaged over ALL 40 test contracts (no trade = $0).
  PASS      : mean > 0 with contract-bootstrap p < 0.05 after Holm over every strategy x family,
              in BOTH BTC and ETH.

Usage
  ../venv/bin/python mboformer_v3.py --train "BTC 15m" --threads 3     # load, train, save (one process per family)
  ../venv/bin/python mboformer_v3.py --train "Sports & esports" --threads 2   # science part only
  ../venv/bin/python mboformer_v3.py --simulate                         # all strategies from saved files
  ../venv/bin/python mboformer_v3.py --report                           # charts + verdict
"""

import argparse
import json
import os
import pickle
import time
import urllib.request
import numpy as np
import pandas as pd
import test_mboformer as base
import retest_mboformer as rt

# ── SETTINGS ────────────────────────────────────────────────────────────────
CRYPTO = {"BTC 15m": "BTCUSDT", "ETH 15m": "ETHUSDT"}
SPORTS = "Sports & esports"
N_TRAIN, N_VAL, N_TEST = 24, 10, 40
TEST_PERIOD = ("2026-10-23 00:00:00", "2026-10-25 00:00:00")
HORIZONS = [100, 10, 20, 50]           # paper's horizons; h=100 first because S1 and a maker gate use it
TRADE_H_S = [10, 30]
LAG_S = 0.25
LAGS_S = [0.0, 0.1, 0.2, 0.25, 0.35, 0.5]
MAKER_LAG_S = 0.10
MAKER_LAGS_S = [0.05, 0.10, 0.20]
SIZE = 5
LEVELS_KEEP = 5
FEE_RATE_V2 = 0.07
REBATE = 0.20
SEEDS_TRADE = [0, 1]
MAKER_STOP_S = 30                       # stop quoting in the last 30 s; leftover inventory goes to resolution
FV_GAP_S = 30                           # fair-value strategy: at most one entry per 30 s
SPOT_DELIVERY_MS = 50
VOL_WINDOW_S = 1800
OUTDIR = "mboformer_v3"
SPOT = ["spot_ret1", "spot_ret5", "spot_ret30", "spot_z", "fv_gap"]
BOOK_COLS = [f"{n}{k}" for n in ("bpx", "bsx", "apx", "asx") for k in range(LEVELS_KEEP)]
GRID_P = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
GRID_FV = [0.0, 0.02, 0.04, 0.06, 0.10]
GRID_GATE_P = [0.3, 0.4, 0.5, 0.6]
GRID_GATE_FV = [-0.02, 0.0, 0.01, 0.02, 0.04]


def fee(p):
    """Polymarket Fee Structure V2, crypto: fee per share = 0.07 * p * (1 - p)  (max 1.75c at p = 0.5)."""
    return FEE_RATE_V2 * p * (1 - p)


def fee_v1(p):
    """What v1/v2 used (January 2026 schedule): 0.25 * p * (p * (1 - p))^2  (max 0.78c)."""
    return 0.25 * p * (p * (1 - p)) ** 2


# ── DATA ────────────────────────────────────────────────────────────────────
def fills_sql(mid, start, end):
    # dir = +1 if the taker bought "Up" (or sold "Down"), -1 otherwise
    return f"""
SELECT toUnixTimestamp64Milli(timestamp) AS ms, toInt32(round(toFloat64(normalized_price) * 1000)) AS pm,
       toFloat32(size) AS sz, toInt8(if((outcome.index = 0) = (taker_side = 'BUY'), 1, -1)) AS dir
FROM fills
WHERE platform = 'POLYMARKET' AND market_id = '{mid}'
  AND timestamp >= toDateTime('{start}') - INTERVAL 30 SECOND
  AND timestamp <  toDateTime('{end}') + INTERVAL 30 SECOND
ORDER BY timestamp
"""


def query(client, sql, tries=5):
    """base.ch_query with retries: long downloads sometimes lose the connection."""
    for k in range(tries):
        try:
            return base.ch_query(client, sql)
        except Exception as err:
            if k == tries - 1:
                raise
            print(f"   query failed ({type(err).__name__}), retrying", flush=True)
            client.disconnect()
            time.sleep(5 * (k + 1))


def payouts_sql(mids):
    ids = ",".join(f"'{m}'" for m in mids)
    return f"""
SELECT toString(id) AS mid, outcomes, resolution_outcome_payouts AS payouts
FROM markets FINAL WHERE id IN ({ids})
"""


def payout_up(client, mids):
    """Payout of the outcome with index 0 ("Up"), 1 or 0."""
    df = base.ch_query(client, payouts_sql(mids))
    out = {}
    for _, r in df.iterrows():
        up = [o[0] for o in r["outcomes"] if o[3] == 0][0]
        pay = {p[0]: float(p[1]) for p in r["payouts"]}
        out[r["mid"]] = pay.get(up, np.nan)
    return out


def pick(client, fam):
    """Train + validation: evenly spaced over base.PERIOD. Test: evenly spaced over TEST_PERIOD."""
    def crypto(period, n):
        old = base.PERIOD
        base.PERIOD = (period[0][:10], period[1][:10])
        try:
            c = base.ch_query(client, base.crypto_markets_sql())
        finally:
            base.PERIOD = old
        c["closes_at"] = pd.to_datetime(c["closes_at"])
        c = c[(c["closes_at"] >= period[0]) & (c["closes_at"] < period[1])]
        c = c[c["title"].map(base.window_minutes) == base.WINDOW_MIN]
        c = c[c["title"].str.match(base.FAMILIES[fam]["pattern"])].sort_values("closes_at").reset_index(drop=True)
        c = c.iloc[np.linspace(0, len(c) - 1, n).round().astype(int)].copy()
        c["end"] = c["closes_at"]
        c["start"] = c["end"] - pd.Timedelta(minutes=base.WINDOW_MIN)
        return c[["mid", "title", "start", "end"]]
    if fam == SPORTS:
        c = base.pick_contracts(client, False)[fam]
        s = ["train"] * 12 + ["val"] * 3 + ["test"] * 5
        return c.assign(split=s[:len(c)])
    tv = crypto((base.PERIOD[0] + " 00:00:00", base.PERIOD[1] + " 00:00:00"), N_TRAIN + N_VAL)
    te = crypto(TEST_PERIOD, N_TEST)
    assert te["start"].min() >= tv["end"].max(), "test contracts must come after train/validation"
    return pd.concat([tv.assign(split=["train"] * N_TRAIN + ["val"] * N_VAL), te.assign(split="test")],
                     ignore_index=True)


def build_events_v3(b, fills):
    """Feature events (types 0 add / 1 decrease / 2 trade print) + ground-truth executions for the maker simulator.
    Columns as base.build_events: snap, ts, type, side, price, |size|, depth, at-best, mid change."""
    f = fills.assign(sec=fills["ms"] // 1000)
    ev = base.build_events(b, f)                         # type 2 here = decrease matched to a fill in [-2 s, +6 s]
    gt = ev[ev[:, 2] == 2][:, [0, 3, 4, 5]].copy()       # snap, side, price, size  (simulator only)
    ev[ev[:, 2] == 2, 2] = 1                             # a decrease is just a decrease at the time it happens
    ts = b["ts"]
    # trade prints: block timestamps are whole seconds -> knowable at the end of that second
    T = (f["ms"].values // 1000 + 1) * 1000
    snap = np.searchsorted(ts, T, side="left")           # first book at or after the print
    ok = (snap < len(ts)) & (snap > 0)
    rows = []
    for t, s, p, sz, dr in zip(T[ok], snap[ok], f["pm"].values[ok], f["sz"].values[ok], f["dir"].values[ok]):
        bb = b["bp"][s - 1, 0] if b["nb"][s - 1] > 0 else np.nan
        ba = b["ap"][s - 1, 0] if b["na"][s - 1] > 0 else np.nan
        ref = ba if dr > 0 else bb
        depth = abs(p - ref) / (base.TICK * 1000) if np.isfinite(ref) else 0.0
        rows.append((s, t, 2, dr, p, sz, depth, float(p == bb or p == ba), 0.0))
    if rows:
        ev = np.vstack([ev, np.array(rows, dtype=np.float64)])
        ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    return ev, gt


def klines(symbol, start_ms, end_ms):
    """Binance 1-second klines [open_ms, open, close], cached on disk."""
    os.makedirs("cache_binance", exist_ok=True)
    path = f"cache_binance/{symbol}_{start_ms}_{end_ms}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    rows, t = [], start_ms
    while t < end_ms:
        url = (f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1s"
               f"&startTime={t}&endTime={end_ms - 1}&limit=1000")
        for attempt in range(5):
            try:
                part = json.loads(urllib.request.urlopen(url, timeout=20).read())
                break
            except Exception:
                time.sleep(2 + attempt * 3)
        else:
            raise RuntimeError(url)
        if not part:
            break
        rows += [(r[0], float(r[1]), float(r[4])) for r in part]
        t = part[-1][0] + 1000
    k = np.array(rows, dtype=np.float64).reshape(-1, 3)
    pickle.dump(k, open(path, "wb"))
    return k


def spot_features(ts, start_ms, end_ms, k, mid):
    """Spot returns (bp), fair value of "Up" and its gap to the Polymarket mid, using closed klines only."""
    from scipy.stats import norm
    avail = k[:, 0] + 1000 + SPOT_DELIVERY_MS
    i = np.clip(np.searchsorted(avail, ts, side="right") - 1, 0, len(k) - 1)
    lc = np.log(k[:, 2])
    r = np.r_[0, np.diff(lc)]
    sig = pd.Series(r).rolling(VOL_WINDOW_S, min_periods=120).std().bfill().values
    s0 = np.searchsorted(k[:, 0], start_ms)
    K = np.log(k[min(s0, len(k) - 1), 1])
    out = {}
    for n in (1, 5, 30):
        out[f"spot_ret{n}"] = (lc[i] - lc[np.maximum(i - n, 0)]) * 1e4
    tau = np.maximum((end_ms - ts) / 1000, 1.0)
    z = np.clip((lc[i] - K) / (np.maximum(sig[i], 1e-6) * np.sqrt(tau)), -8, 8)
    out["spot_z"] = z
    out["fv"] = norm.cdf(z)
    out["fv_gap"] = np.clip((out["fv"] - mid) / base.TICK, -50, 50)
    return out, lc[-1] - K


def load(client, fam):
    """All contracts of one family -> one dataset dict (events, decisions, snapshots, ground truth)."""
    contracts = pick(client, fam)
    kind = base.FAMILIES[fam]["kind"]
    base.KEEP_BOOK = True
    pays = payout_up(client, contracts["mid"].tolist())
    parts, ev_types, ev_cont, l2s, gts, ev_off = [], [], [], [], {}, 0
    for k, c in contracts.iterrows():
        s, e = c["start"].strftime("%Y-%m-%d %H:%M:%S"), c["end"].strftime("%Y-%m-%d %H:%M:%S")
        t = time.time()
        raw = base.clean_books(query(client, base.book_sql(c["mid"], s, e)))
        fills = query(client, fills_sql(c["mid"], s, e))
        if len(raw) < base.SAMPLE_EVERY * (max(HORIZONS) + base.SMOOTH + 50):
            print(f"   skip {c['title'][:60]}: only {len(raw)} book rows", flush=True)
            continue
        b = base.to_arrays(raw)
        ev, gt = build_events_v3(b, fills)
        end_ms = int(c["end"].value // 10**6)
        d, l2 = base.decisions(b, ev, end_ms, kind)
        if len(ev) == 0 or len(d) == 0:
            continue
        d = d.drop(columns=[f"{n}{j}" for n in ("bpx", "bsx", "apx", "asx") for j in range(LEVELS_KEEP, base.LEVELS)])
        typ, cont = base.event_features(ev)
        d["ev_end"] += ev_off
        d["ev_start"] = ev_off
        d["contract"] = k
        d["split"] = c["split"]
        d["block"] = k * 10_000 + (d["ts"] - d["ts"].iloc[0]) // (base.BLOCK_S * 1000)
        d["payout"] = pays.get(c["mid"], np.nan)
        for h in HORIZONS:
            d[f"delta{h}"] = base.smoothed_change(d["mid"].values, h)
        if fam in CRYPTO:
            start_ms = int(c["start"].value // 10**6)
            kl = klines(CRYPTO[fam], start_ms - VOL_WINDOW_S * 1000, end_ms + 2000)
            sp, final = spot_features(d["ts"].values, start_ms, end_ms, kl, d["mid"].values)
            for nm, v in sp.items():
                d[nm] = v
            d["spot_final_logret"] = final
        num = d.select_dtypes("float64").columns
        d[num] = d[num].astype(np.float32)
        d["ts"] = d["ts"].astype(np.int64)
        gts[k] = gt.astype(np.float32)
        parts.append(d)
        ev_types.append(typ.astype(np.int8))
        ev_cont.append(cont)
        l2s.append(l2)
        ev_off += len(ev)
        print(f"   [{fam}] {k + 1}/{len(contracts)} {c['split']:5s} {c['title'][:50]:50s} {len(raw):>8,} books "
              f"{len(ev):>8,} events {len(d):>7,} decisions {time.time() - t:4.0f}s", flush=True)
    d = pd.concat(parts, ignore_index=True)
    for col in ("mid", "bid", "ask", "spread"):
        d[col] = d[col].astype(np.float64)
    return dict(dec=d, ev_type=np.concatenate(ev_types).astype(np.int64), ev_cont=np.concatenate(ev_cont),
                l2=np.concatenate(l2s), gt=gts, contracts=contracts)


# ── TARGETS AND EXECUTION ───────────────────────────────────────────────────
def book_prices(d, q):
    L = LEVELS_KEEP
    ask = rt.vwap(d[[f"apx{k}" for k in range(L)]].values.astype(np.float64),
                  d[[f"asx{k}" for k in range(L)]].values.astype(np.float64), q)
    bid = rt.vwap(d[[f"bpx{k}" for k in range(L)]].values.astype(np.float64),
                  d[[f"bsx{k}" for k in range(L)]].values.astype(np.float64), q)
    return bid, ask


def entry_exit(d, lag_s, hold_s=None, hold_steps=None):
    """entry j = first decision >= ts + lag (same contract); exit k = j + steps, or first decision >= ts_j + hold_s."""
    con = d["contract"].values.astype(np.int64)
    ts = d["ts"].values.astype(np.int64)
    key = con * 10**13 + ts
    n = len(d)
    j = np.searchsorted(key, key + int(round(lag_s * 1000)))
    jc = np.minimum(j, n - 1)
    k = j + hold_steps if hold_steps is not None else np.searchsorted(key, key[jc] + int(hold_s * 1000))
    kc = np.minimum(k, n - 1)
    ok = (j < n) & (k < n) & (con[jc] == con) & (con[kc] == con)
    return jc, kc, ok


def tradable_pnl(d, bid, ask, lag_s, hold_s=None, hold_steps=None, fee_fn=fee):
    """Per-share profit of buying "Up" (long) or buying "Down" (short) at the entry, selling at the exit, as taker."""
    j, k, ok = entry_exit(d, lag_s, hold_s, hold_steps)
    lng = bid[k] - ask[j] - fee_fn(ask[j]) - fee_fn(bid[k])
    sht = bid[j] - ask[k] - fee_fn(1 - bid[j]) - fee_fn(1 - ask[k])     # "Down" bought at 1-bid, sold at 1-ask
    ok &= np.isfinite(lng) & np.isfinite(sht)
    return lng, sht, j, k, ok


def tradable_label(d, H):
    bid, ask = book_prices(d, SIZE)
    lng, sht, _, _, ok = tradable_pnl(d, bid, ask, LAG_S, hold_s=H)
    y = np.ones(len(d), dtype=np.int64)
    y[ok & (lng > 0)] = 2
    y[ok & (sht > 0)] = 0
    return y, ok


# ── TRAINING ────────────────────────────────────────────────────────────────
def train_predict(name, data, batcher, y, rows, seeds, tag, outdir):
    """Seed-averaged class probabilities on val + test rows; cached per (tag, seed)."""
    import torch
    d = data["dec"]
    pr = np.concatenate([rows["val"], rows["test"]])
    ps = []
    for seed in seeds:
        path = f"{outdir}/{tag}_s{seed}.npy"
        if os.path.exists(path):
            ps.append(np.load(path))
            continue
        t = time.time()
        model, ep = base.train_one(name, data, batcher, y, rows, seed)
        p = np.full((len(d), 3), np.nan, dtype=np.float32)
        p[pr] = base.predict(model, batcher, pr)
        np.save(path, p)
        torch.save(model.state_dict(), f"{outdir}/{tag}_s{seed}.pt")
        ps.append(p)
        f1 = base.macro_f1(y[rows["test"]], p[rows["test"]].argmax(1))
        print(f"   {tag} seed {seed}: test macro-F1 {f1:.3f} ({ep} ep, {time.time() - t:.0f}s)", flush=True)
    return np.mean(ps, 0)


def run_train(fam, threads):
    from dotenv import load_dotenv
    from clickhouse_driver import Client
    import torch
    torch.set_num_threads(threads)
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    client = Client(host=os.environ["PROBALYTICS_CLICKHOUSE_HOST"], user=os.environ["PROBALYTICS_CLICKHOUSE_USERNAME"],
                    password=os.environ["PROBALYTICS_CLICKHOUSE_PASSWORD"], database="probalytics",
                    secure=True, port=9440)
    slug = base.slug(fam)
    out = f"{OUTDIR}/{slug}"
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    print(f"[{fam}] loading", flush=True)
    data = load(client, fam)
    d = data["dec"]
    data["l2w"] = base.l2_windows(data)
    data["contracts"].to_csv(f"{out}/contracts.csv")
    print(f"[{fam}] loaded in {(time.time() - t0) / 60:.0f} min:", d.groupby("split")["contract"].nunique().to_dict(),
          flush=True)
    if fam in CRYPTO:
        base.CONTEXT = base.CONTEXT + SPOT
        resolved = d.groupby("contract")[["payout", "spot_final_logret"]].first().dropna()
        agree = float(((resolved["spot_final_logret"] >= 0) == (resolved["payout"] > 0.5)).mean())
        print(f"[{fam}] Binance spot agrees with Polymarket resolution in {agree:.0%} of contracts", flush=True)
        json.dump(dict(spot_resolution_agreement=agree, n=len(resolved)), open(f"{out}/spot_check.json", "w"))
    split = d["split"].values
    full = (d["ev_end"].values - d["ev_start"].values >= base.T_EVENTS) & (data["l2w"][:, 0] >= 0)
    f1_rows = []
    keep = ["contract", "split", "ts", "block", "mid", "bid", "ask", "spread", "min_to_end", "payout", "snap"] + BOOK_COLS
    keep += [c for c in ["fv", "spot_z", "fv_gap", "spot_ret1", "spot_ret5", "spot_ret30"] if c in d]

    def save_sim():
        pickle.dump(dict(d=d[keep], full=full, gt=data["gt"]), open(f"{out}/sim_data.pkl", "wb"))

    save_sim()

    def science(h):
        delta = d[f"delta{h}"].values
        has = np.isfinite(delta) & full
        alpha = 0.5 * np.mean(np.abs(delta[has & (split == "train")]))
        y = base.label(np.nan_to_num(delta), alpha)
        rows = {s: np.where(has & (split == s))[0] for s in ("train", "val", "test")}
        norm = base.norm_stats(data, rows["train"])
        batcher = base.Batcher(data, norm)
        te = rows["test"]
        yte, bl = y[te], d["block"].values[te]
        probs = {"Always flat": np.tile([0, 1, 0], (len(te), 1)).astype(float)}
        lp = base.logistic(data, y, rows)
        probs["Logistic (12 features)"] = lp
        for name in ("MBOFormer", "L2Former"):
            probs[name] = train_predict(name, data, batcher, y, rows, base.SEEDS, f"paper_{name}_h{h}", out)[te]
        shares = np.bincount(yte, minlength=3) / len(yte)
        for name, p in probs.items():
            f1_rows.append(dict(family=fam, h=h, model=name, f1=base.macro_f1(yte, p.argmax(1)), n_test=len(te),
                                alpha_c=alpha * 100, share_down=shares[0], share_flat=shares[1], share_up=shares[2]))
        for a, bname in [("MBOFormer", "L2Former"), ("MBOFormer", "Logistic (12 features)")]:
            diff, lo, hi = base.f1_diff_boot(yte, probs[a].argmax(1), probs[bname].argmax(1), bl)
            f1_rows.append(dict(family=fam, h=h, model=f"{a} - {bname}", f1=diff, ci_lo=lo, ci_hi=hi))
        pd.DataFrame(f1_rows).to_csv(f"{out}/f1.csv", index=False)
        print(f"[{fam}] science h={h} done ({(time.time() - t0) / 60:.0f} min)", flush=True)

    science(HORIZONS[0])
    if fam in CRYPTO:
        lab = {}
        for H in TRADE_H_S:
            y, ok = tradable_label(d, H)
            lab[H] = (y, ok)
            rows = {s: np.where(full & ok & (split == s))[0] for s in ("train", "val", "test")}
            yt = y[rows["train"]]
            print(f"[{fam}] tradable label H={H}s: train down/none/up = "
                  f"{np.bincount(yt, minlength=3) / len(yt)}", flush=True)
            norm = base.norm_stats(data, rows["train"])
            batcher = base.Batcher(data, norm)
            train_predict("MBOFusion", data, batcher, y, rows, SEEDS_TRADE, f"cost_MBOFusionSpot_H{H}", out)
            train_predict("MBOFormer", data, batcher, y, rows, [0], f"cost_MBOFormer_H{H}", out)
            # logistic on classic + spot features, same label
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            X = d[base.CLASSIC + SPOT + ["min_to_end"]].values
            sc = StandardScaler().fit(X[rows["train"]])
            m = LogisticRegression(max_iter=1000).fit(sc.transform(X[rows["train"]]), y[rows["train"]])
            p = np.full((len(d), 3), np.nan, dtype=np.float32)
            pr = np.concatenate([rows["val"], rows["test"]])
            p[pr] = m.predict_proba(sc.transform(X[pr]))
            np.save(f"{out}/cost_LogisticSpot_H{H}_s0.npy", p)
            print(f"[{fam}] tradable models H={H}s done ({(time.time() - t0) / 60:.0f} min)", flush=True)
    for h in HORIZONS[1:]:
        science(h)
    print(f"[{fam}] all done in {(time.time() - t0) / 60:.0f} min", flush=True)


# ── STRATEGY SIMULATORS ─────────────────────────────────────────────────────
def per_contract(trades, contracts, q=SIZE):
    """$ per contract at q shares, zero for contracts with no trade."""
    s = trades.groupby("contract")["after_fees"].sum() / 100 * q if len(trades) else pd.Series(dtype=float)
    return s.reindex(contracts, fill_value=0.0)


def sim_taker(d, sig, bid, ask, lag_s, hold_s=None, hold_steps=None, fee_fn=fee):
    """One position at a time; taker in at ts + lag, taker out after the hold. Cents per share."""
    lng, sht, j, k, ok = tradable_pnl(d, bid, ask, lag_s, hold_s, hold_steps, fee_fn)
    mid = d["mid"].values
    nz = np.where(sig != 0)[0]
    rows, ip = [], 0
    while ip < len(nz):
        i = nz[ip]
        if not ok[i]:
            ip += 1
            continue
        up = sig[i] == 1
        entry, exit_ = (ask[j[i]], bid[k[i]]) if up else (bid[j[i]], ask[k[i]])
        gross = (exit_ - entry) if up else (entry - exit_)
        rows.append((d["contract"].values[i], sig[i], 100 * sig[i] * (mid[k[i]] - mid[j[i]]), 100 * gross,
                     100 * (lng[i] if up else sht[i])))
        ip = np.searchsorted(nz, k[i] + 1)
    return pd.DataFrame(rows, columns=["contract", "dir", "mid", "after_spread", "after_fees"])


def sim_fv_hold(d, bid, ask, m, lag_s):
    """Buy "Up" when FV - ask - fee > m, buy "Down" when (1-FV) - (1-bid) - fee > m; hold to resolution (no exit fee)."""
    fv = d["fv"].values
    j, _, _ = entry_exit(d, lag_s, hold_steps=0)
    con, ts, pay = d["contract"].values, d["ts"].values, d["payout"].values
    edge_up = fv - ask - fee(ask)
    edge_dn = bid - fv - fee(1 - bid)
    cand = np.where((edge_up > m) | (edge_dn > m))[0]
    rows, last = [], {}
    for i in cand:
        c = con[i]
        if ts[i] - last.get(c, -10**15) < FV_GAP_S * 1000 or con[j[i]] != c or not np.isfinite(pay[i]):
            continue
        last[c] = ts[i]
        if edge_up[i] > m:
            p = ask[j[i]]
            pnl = pay[i] - p - fee(p)
            gross = pay[i] - p
            dr = 1
        else:
            p = 1 - bid[j[i]]
            pnl = (1 - pay[i]) - p - fee(p)
            gross = (1 - pay[i]) - p
            dr = -1
        if np.isfinite(pnl):
            rows.append((c, dr, 100 * gross, 100 * gross, 100 * pnl))
    return pd.DataFrame(rows, columns=["contract", "dir", "mid", "after_spread", "after_fees"])


def sim_maker_contract(c, gt, gate_bid, gate_ask, lag_ms, q=SIZE):
    """Quote q shares at the best bid and best ask; inventory capped at one lot; leftover held to resolution.
    Returns $ profit, rebate $, number of fills, spread captured and markout parts."""
    n = len(c)
    ts = c["ts"].values
    bp = np.rint(c[[f"bpx{k}" for k in range(LEVELS_KEEP)]].values * 1000).astype(np.int64)
    bs = c[[f"bsx{k}" for k in range(LEVELS_KEEP)]].values
    ap = np.rint(c[[f"apx{k}" for k in range(LEVELS_KEEP)]].values * 1000).astype(np.int64)
    asz = c[[f"asx{k}" for k in range(LEVELS_KEEP)]].values
    bb = np.where(bs[:, 0] > 0, bp[:, 0], -1)
    ba = np.where(asz[:, 0] > 0, ap[:, 0], 10**6)
    end_ms = ts[0] + c["min_to_end"].values[0] * 60_000
    stop = end_ms - MAKER_STOP_S * 1000
    pay = c["payout"].values[0]
    mid = c["mid"].values
    # ground-truth executions -> per decision index: {(side, price): size}
    ex = {}
    if len(gt):
        di = np.searchsorted(c["snap"].values, gt[:, 0], side="left")
        for t, sd, p, sz in zip(di, gt[:, 1].astype(int), gt[:, 2].astype(np.int64), gt[:, 3]):
            if t < n:
                ex.setdefault(t, {})
                ex[t][(sd, p)] = ex[t].get((sd, p), 0.0) + sz
    inv, cash, rebate, nfill = 0, 0.0, 0.0, 0
    markout = []
    orders = {1: None, -1: None}
    for t in range(n):
        et = ex.get(t)
        for side in (1, -1):
            o = orders[side]
            if o is None:
                continue
            if o["cancel_at"] is not None and ts[t] >= o["cancel_at"]:
                orders[side] = None
                continue
            if ts[t] < o["live_at"]:
                continue
            P = o["price"]
            if o["qa"] is None:                                   # just arrived: post-only, join the queue
                if (side == 1 and P >= ba[t]) or (side == -1 and P <= bb[t]):
                    orders[side] = None
                    continue
                lv = (bp[t] == P) if side == 1 else (ap[t] == P)
                o["qa"] = float((bs[t] if side == 1 else asz[t])[lv].sum())
                continue
            if et:
                o["qa"] -= et.get((side, P), 0.0)
            through = (side == 1 and ba[t] <= P) or (side == -1 and bb[t] >= P)
            if o["qa"] <= -q or through:
                p = P / 1000
                inv += side * q
                cash -= side * q * p
                rebate += REBATE * fee(p) * q
                nfill += 1
                # markout: mid 10 s later vs fill price (per share, signed for us)
                t10 = min(np.searchsorted(ts, ts[t] + 10_000), n - 1)
                markout.append(side * (mid[t10] - p))
                orders[side] = None
        for side in (1, -1):
            nxt = inv + side * q
            want = abs(nxt) <= q and ts[t] < stop and not (gate_bid[t] if side == 1 else gate_ask[t])
            target = bb[t] if side == 1 else ba[t]
            valid = (0 < target < 1000)
            o = orders[side]
            if o is not None and o["cancel_at"] is None and (not want or target != o["price"]):
                o["cancel_at"] = ts[t] + lag_ms
            if want and valid and orders[side] is None:
                orders[side] = dict(price=int(target), live_at=ts[t] + lag_ms, qa=None, cancel_at=None)
    pnl = cash + inv * pay
    return dict(pnl=pnl, rebate=rebate, fills=nfill, inv_end=inv,
                markout_c=100 * np.mean(markout) if markout else np.nan)


def sim_maker(d, gt, gate_bid, gate_ask, contracts, lag_s):
    res = []
    for c in contracts:
        m = (d["contract"].values == c)
        idx = np.where(m)[0]
        r = sim_maker_contract(d.iloc[idx], gt.get(c, np.zeros((0, 4))), gate_bid[idx], gate_ask[idx],
                               int(lag_s * 1000))
        res.append(dict(contract=c, **r))
    return pd.DataFrame(res)


def boot(per_c, n=4000, seed=0):
    v = np.asarray(per_c, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), float((means <= 0).mean())


# ── STRATEGY RUNNER ─────────────────────────────────────────────────────────
def load_probs(out, tag, seeds):
    fs = [f"{out}/{tag}_s{s}.npy" for s in seeds]
    fs = [f for f in fs if os.path.exists(f)]
    return np.mean([np.load(f) for f in fs], 0) if fs else None


def run_simulate():
    results, per_contract_rows, latency_rows, maker_rows, fee_rows = [], [], [], [], []
    for fam in CRYPTO:
        slug = base.slug(fam)
        out = f"{OUTDIR}/{slug}"
        if not os.path.exists(f"{out}/sim_data.pkl"):
            continue
        z = pickle.load(open(f"{out}/sim_data.pkl", "rb"))
        d, full, gt = z["d"], z["full"], z["gt"]
        # A few contracts have no resolution in the markets table yet: use the final price when it is decisive
        # (where both exist they agree in every contract)
        last = d.groupby("contract")["mid"].transform("last").values
        guess = np.where(last >= 0.95, 1.0, np.where(last <= 0.05, 0.0, np.nan))
        d["payout"] = np.where(np.isfinite(d["payout"].values), d["payout"].values, guess)
        split = d["split"].values
        cons = {s: sorted(d.loc[split == s, "contract"].unique()) for s in ("val", "test")}
        bid, ask = book_prices(d, SIZE)
        t0 = time.time()

        def evaluate(name, make_trades, grid, kind="taker"):
            """Pick the grid value with the best total validation profit, then run it once on test."""
            best, best_v = None, -np.inf
            for g in grid:
                pv = make_trades(g, "val")
                if pv.sum() > best_v:
                    best, best_v = g, pv.sum()
            pt = make_trades(best, "test")
            per_contract_rows.extend(dict(family=fam, strategy=name, contract=c, usd=v) for c, v in pt.items())
            mu, lo, hi, p = boot(pt.values)
            results.append(dict(family=fam, strategy=name, kind=kind, param=best, val_total=best_v,
                                usd_per_contract=mu, lo=lo, hi=hi, p=p,
                                contracts_profitable=int((pt > 0).sum()), contracts=len(pt)))
            print(f"   [{fam}] {name:42s} param={best!s:8s} val ${best_v:+.2f}  test ${mu:+.3f}/contract "
                  f"({lo:+.3f}..{hi:+.3f}) p={p:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            return best

        def taker_maker(sigfn, hold_s=None, hold_steps=None, lag=LAG_S, fee_fn=fee, keep=None):
            def f(g, s):
                sig = sigfn(g)
                m = split == s
                ds = d[m].reset_index(drop=True)
                tr = sim_taker(ds, sig[m], bid[m], ask[m], lag, hold_s, hold_steps, fee_fn)
                if keep is not None and s == "test":
                    keep.append(tr)
                return per_contract(tr, cons[s])
            return f

        # S1: paper MBOFormer h=100, top-10% confidence cut on validation
        pm = load_probs(out, "paper_MBOFormer_h100", base.SEEDS)
        has = full & np.isfinite(pm[:, 0])
        conf = np.where(has, np.nan_to_num(pm).max(1), 0)
        dirn = np.where(has, np.nan_to_num(pm).argmax(1) - 1, 0)
        s1, _ = rt.cutoff_signal(dirn, conf, has & (split == "val"))
        evaluate("S1 Paper signal (MBOFormer h=100, top 10%)", taker_maker(lambda g: s1, hold_steps=100), [None])

        def prob_signal(p, th):
            p = np.nan_to_num(p)
            up = (p[:, 2] >= th) & (p[:, 2] >= p[:, 0])
            dn = (p[:, 0] >= th) & (p[:, 0] > p[:, 2])
            return np.where(full, up.astype(int) - dn.astype(int), 0)

        best_cost = {}
        for H in TRADE_H_S:
            for tag, label in [("cost_MBOFusionSpot", "S2 Cost-aware MBOFusion + spot"),
                               ("cost_MBOFormer", "S3 Cost-aware MBOFormer (events only)"),
                               ("cost_LogisticSpot", "S4 Cost-aware logistic + spot")]:
                p = load_probs(out, f"{tag}_H{H}", SEEDS_TRADE if tag == "cost_MBOFusionSpot" else [0])
                if p is None:
                    continue
                th = evaluate(f"{label}, H={H}s", taker_maker(lambda g, p=p: prob_signal(p, g), hold_s=H), GRID_P)
                best_cost[(tag, H)] = (p, th)

        # S5: fair value, hold to resolution
        def fv_trades(g, s, lag=LAG_S):
            m = split == s
            return per_contract(sim_fv_hold(d[m].reset_index(drop=True), bid[m], ask[m], g, lag), cons[s])
        m_fv = evaluate("S5 Fair value, taker in, hold to resolution", fv_trades, GRID_FV)

        # S6: maker
        def maker(gate_fn, lag=MAKER_LAG_S, store=None):
            def f(g, s):
                gb, ga = gate_fn(g)
                r = sim_maker(d, gt, gb, ga, cons[s], lag)
                if store is not None and s == "test":
                    store.append(r.assign(family=fam))
                return r.set_index("contract")["pnl"]
            return f

        fvv = d["fv"].values
        bbv, bav = d["bid"].values, d["ask"].values
        none_gate = lambda g: (np.zeros(len(d), bool), np.zeros(len(d), bool))
        paper_gate = lambda g: ((np.nan_to_num(pm[:, 0]) >= g) | ~has, (np.nan_to_num(pm[:, 2]) >= g) | ~has)
        fv_gate = lambda g: (fvv - bbv < g, bav - fvv < g)
        p2, _ = best_cost.get(("cost_MBOFusionSpot", 10), (None, None))
        has2 = full & np.isfinite(p2[:, 0]) if p2 is not None else full
        cost_gate = lambda g: ((np.nan_to_num(p2[:, 0]) >= g) | ~has2, (np.nan_to_num(p2[:, 2]) >= g) | ~has2)

        def combo_gate(g):
            a, b_ = fv_gate(g[0])
            c_, d_ = cost_gate(g[1])
            return a | c_, b_ | d_

        store = []
        evaluate("S6a Maker, no gate", maker(none_gate, store=store), [None], "maker")
        evaluate("S6b Maker, paper MBOFormer gate", maker(paper_gate, store=store), GRID_GATE_P, "maker")
        g_fv = evaluate("S6c Maker, fair-value gate", maker(fv_gate, store=store), GRID_GATE_FV, "maker")
        if p2 is not None:
            evaluate("S6d Maker, cost-aware model gate", maker(cost_gate, store=store), GRID_GATE_P, "maker")
            g_combo = evaluate("S6e Maker, fair value + cost-aware gate", maker(combo_gate, store=store),
                               [(a, b_) for a in GRID_GATE_FV for b_ in GRID_GATE_P], "maker")
        maker_rows += store

        # Sensitivities (descriptive, not part of the pass test)
        for lag in LAGS_S:
            r1 = per_contract(sim_taker(d[split == "test"].reset_index(drop=True), s1[split == "test"],
                                        bid[split == "test"], ask[split == "test"], lag, hold_steps=100),
                              cons["test"])
            latency_rows.append(dict(family=fam, strategy="S1 Paper signal", lag_s=lag, usd=r1.mean(),
                                     lo=boot(r1.values)[1], hi=boot(r1.values)[2]))
            for H in TRADE_H_S:
                if ("cost_MBOFusionSpot", H) in best_cost:
                    p, th = best_cost[("cost_MBOFusionSpot", H)]
                    m = split == "test"
                    r2 = per_contract(sim_taker(d[m].reset_index(drop=True), prob_signal(p, th)[m], bid[m], ask[m],
                                                lag, hold_s=H), cons["test"])
                    latency_rows.append(dict(family=fam, strategy=f"S2 Cost-aware + spot, H={H}s", lag_s=lag,
                                             usd=r2.mean(), lo=boot(r2.values)[1], hi=boot(r2.values)[2]))
            r5 = fv_trades(m_fv, "test", lag)
            latency_rows.append(dict(family=fam, strategy="S5 Fair value, hold", lag_s=lag, usd=r5.mean(),
                                     lo=boot(r5.values)[1], hi=boot(r5.values)[2]))
        for lag in MAKER_LAGS_S:
            r6 = maker(fv_gate, lag)(g_fv, "test")
            latency_rows.append(dict(family=fam, strategy="S6c Maker, fair-value gate", lag_s=lag, usd=r6.mean(),
                                     lo=boot(r6.values)[1], hi=boot(r6.values)[2]))
        # Cost stack for S1 at 0.25 s: mid -> spread -> fee (V1 vs V2), cents per share per trade
        m = split == "test"
        ds = d[m].reset_index(drop=True)
        for lab, ff in (("V1 fee (what v1/v2 assumed)", fee_v1), ("V2 fee (actual)", fee)):
            tr = sim_taker(ds, s1[m], bid[m], ask[m], LAG_S, hold_steps=100, fee_fn=ff)
            fee_rows.append(dict(family=fam, strategy="S1 Paper signal", fee=lab, trades=len(tr),
                                 mid_c=tr["mid"].mean(), after_spread_c=tr["after_spread"].mean(),
                                 after_fees_c=tr["after_fees"].mean()))
        for H in TRADE_H_S:
            if ("cost_MBOFusionSpot", H) in best_cost:
                p, th = best_cost[("cost_MBOFusionSpot", H)]
                tr = sim_taker(ds, prob_signal(p, th)[m], bid[m], ask[m], LAG_S, hold_s=H)
                fee_rows.append(dict(family=fam, strategy=f"S2 Cost-aware + spot, H={H}s", fee="V2 fee (actual)",
                                     trades=len(tr), mid_c=tr["mid"].mean(), after_spread_c=tr["after_spread"].mean(),
                                     after_fees_c=tr["after_fees"].mean()))
        tr = sim_fv_hold(ds, bid[m], ask[m], m_fv, LAG_S)
        fee_rows.append(dict(family=fam, strategy="S5 Fair value, hold", fee="V2 fee (actual)", trades=len(tr),
                             mid_c=tr["mid"].mean() if len(tr) else np.nan,
                             after_spread_c=tr["after_spread"].mean() if len(tr) else np.nan,
                             after_fees_c=tr["after_fees"].mean() if len(tr) else np.nan))
    res = pd.DataFrame(results)
    # Holm over every strategy x family
    order = res["p"].fillna(1).sort_values().index
    m_tests, running = len(res), 0.0
    res["p_holm"] = np.nan
    for i, ix in enumerate(order):
        running = max(running, min(1.0, res.loc[ix, "p"] * (m_tests - i)))
        res.loc[ix, "p_holm"] = running
    res["passes_family"] = (res["usd_per_contract"] > 0) & (res["p_holm"] < 0.05)
    res.to_csv(f"{OUTDIR}/strategies.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(f"{OUTDIR}/latency.csv", index=False)
    pd.concat(maker_rows).to_csv(f"{OUTDIR}/maker_detail.csv", index=False)
    pd.DataFrame(fee_rows).to_csv(f"{OUTDIR}/cost_stack.csv", index=False)
    pd.DataFrame(per_contract_rows).to_csv(f"{OUTDIR}/per_contract.csv", index=False)
    both = res.groupby("strategy")["passes_family"].all()
    verdict = dict(PASS=bool(both.any()), passing=[s for s, v in both.items() if v], spec=dict(
        lag_s=LAG_S, maker_lag_s=MAKER_LAG_S, size=SIZE, fee="0.07*p*(1-p)", test_contracts=N_TEST))
    json.dump(verdict, open(f"{OUTDIR}/verdict.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny end-to-end check of the plumbing")
    a = ap.parse_args()
    if a.quick:
        N_TRAIN, N_VAL, N_TEST, HORIZONS, SEEDS_TRADE = 4, 2, 4, [100], [0]
        base.SEEDS, base.MAX_EPOCHS, base.TRAIN_PER_EPOCH = [0], 1, 3000
        OUTDIR = "mboformer_v3_quick"
    if a.train:
        run_train(a.train, a.threads)
    if a.simulate:
        run_simulate()
    if a.report:
        import mboformer_v3_report
        mboformer_v3_report.report()
