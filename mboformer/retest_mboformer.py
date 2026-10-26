"""
Clean re-test of the one promising result from test_mboformer.py
================================================================
The first run found that trading only the 10% most CONFIDENT MBOFormer signals,
held 50-100 decision steps (3-6 s), made money after spread and fees in BTC and
ETH 15-minute markets. That result had four problems:
  (a) the "top 10%" cutoff was chosen on the TEST contracts themselves,
  (b) only 5 test contracts (ETH: only 2, because of a split bug, now fixed),
  (c) trades overlapped (a new trade every ~0.03-0.06 s, each held for seconds),
  (d) latency was 1 decision step (30-70 ms): far faster than a real order.
This file re-runs it cleanly.

PRE-REGISTERED SPEC (written before looking at any re-test number)
  Train   : first 12 contracts of the original sample (time order), same recipe as test_mboformer.py
  Tune    : next 3 contracts. The confidence cutoff is fixed HERE:
              cutoff = 90th percentile of the model's confidence on the validation contracts
              confidence = max(P(down), P(flat), P(up)),   signal = argmax - 1 if confidence >= cutoff
  Test    : 30 FRESH contracts per family that close after EVERY contract used before
            (2026-10-23 00:00 -> 2026-10-24 06:00 UTC). Never seen, never tuned on.
  PRIMARY : MBOFormer top-10%, h = 100 steps, latency 0.5 s, 1 share, ONE POSITION AT A TIME
            (no new trade until the last one is closed), Polymarket taker fee on both legs.
  PASS    : mean profit per trade after spread + fees > 0 in BOTH BTC and ETH, with
            contract-clustered bootstrap p < 0.05 after Holm adjustment over the 2 families,
            AND better than the simple rules below at the same settings.
  Everything else (h = 50, other latencies, 100 shares, overlapping trades) is sensitivity.

Simple rules it has to beat (cutoffs also fixed on validation, top 10%)
  Momentum          : direction = sign(mid change over the last 20 steps); confidence = |that change|
  Drift to 0/1      : direction = sign(mid - 0.5)  (bet the favourite keeps winning);
                      confidence = |mid - 0.5| * (time elapsed in the 15-min window / 15 min)
  Random direction  : MBOFormer's own top-10% moments, coin-flip direction. Tells us whether the
                      model's DIRECTION matters or only its TIMING.

Execution model
  Signal at decision t (time ts_t). Your order arrives LAG seconds later:
      entry j = first decision point with ts >= ts_t + LAG   (same contract)
      exit  k = j + h                                         (same contract, else no trade)
  Price for Q shares = walk the visible book (10 levels):
      VWAP_ask(Q) = sum_levels fill_l * ask_l / Q,   fill_l = min(size_l, Q - shares already filled)
      If the 10 visible levels hold fewer than Q shares, the trade is skipped (counted as unfilled).
  Profit per share:
      long  (signal up)  : VWAP_bid(k) - VWAP_ask(j) - fee(VWAP_ask(j)) - fee(VWAP_bid(k))
      short (signal down): VWAP_bid(j) - VWAP_ask(k) - fee(VWAP_bid(j)) - fee(VWAP_ask(k))
      fee(p) = 0.25 * p * (p * (1 - p))^2   (Polymarket fee table; max 0.78c at p = 0.5)

Usage
  ../venv/bin/python retest_mboformer.py --family "BTC 15m"   # train + predict + simulate one family
  ../venv/bin/python retest_mboformer.py --family "ETH 15m"   # (run both side by side)
  ../venv/bin/python retest_mboformer.py --report             # verdict + charts from saved results
"""

import argparse
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import test_mboformer as base

# ── SETTINGS ────────────────────────────────────────────────────────────────
FAMILIES = ["BTC 15m", "ETH 15m"]
FRESH_PERIOD = ("2026-10-23 00:00:00", "2026-10-24 06:00:00")   # after every contract used before
N_FRESH = 30
HS = [50, 100]
LAGS_S = [0.0, 0.25, 0.5, 1.0]
SIZES = [1, 100]
TOP = 0.10
PRIMARY = dict(signal="MBOFormer top-10%", h=100, lag_s=0.5, size=1, mode="one at a time")
ALPHA = 0.05
N_BOOT = 2000
OUTDIR = "mboformer_retest"
SIGNALS = ["MBOFormer top-10%", "Momentum top-10%", "Drift to 0/1 top-10%",
           "Random direction (MBOFormer timing)", "MBOFormer every signal"]


# ── DATA ────────────────────────────────────────────────────────────────────
def fresh_contracts(client, fam):
    """N_FRESH 15-minute contracts, evenly spaced, closing inside FRESH_PERIOD."""
    old = base.PERIOD
    base.PERIOD = (FRESH_PERIOD[0][:10], "2026-10-25")
    try:
        c = base.ch_query(client, base.crypto_markets_sql())
    finally:
        base.PERIOD = old
    c["closes_at"] = pd.to_datetime(c["closes_at"])
    c = c[(c["closes_at"] >= FRESH_PERIOD[0]) & (c["closes_at"] < FRESH_PERIOD[1])]
    c = c[c["title"].map(base.window_minutes) == base.WINDOW_MIN]
    c = c[c["title"].str.match(base.FAMILIES[fam]["pattern"])].sort_values("closes_at").reset_index(drop=True)
    c = c.iloc[np.linspace(0, len(c) - 1, N_FRESH).round().astype(int)].copy()
    c["end"] = c["closes_at"]
    c["start"] = c["end"] - pd.Timedelta(minutes=base.WINDOW_MIN)
    return c[["mid", "title", "start", "end"]]


def get_data(client, fam):
    """Original 20 contracts (train / val / old test) + fresh contracts, one event array."""
    base.KEEP_BOOK = True
    orig = base.pick_contracts(client, False)[fam]
    fresh = fresh_contracts(client, fam)
    assert fresh["start"].min() >= orig["end"].max(), "fresh contracts must come after all old ones"
    allc = pd.concat([orig.assign(origin="orig"), fresh.assign(origin="fresh")], ignore_index=True)
    data = base.load_family(client, fam, allc)
    d = data["dec"]
    present = sorted(d["contract"].unique())
    orig_present = [k for k in present if allc.loc[k, "origin"] == "orig"]
    split = {k: "train" for k in orig_present[:12]}
    split.update({k: "val" for k in orig_present[12:15]})
    split.update({k: "old_test" for k in orig_present[15:]})
    split.update({k: "fresh" for k in present if allc.loc[k, "origin"] == "fresh"})
    d["split"] = d["contract"].map(split)
    data["contracts"] = allc
    return data


# ── SIGNALS ─────────────────────────────────────────────────────────────────
def cutoff_signal(direction, conf, val_mask):
    """Keep a signal only if its confidence is in the top TOP share of the VALIDATION contracts."""
    thr = np.quantile(conf[val_mask], 1 - TOP)
    return np.where(conf >= thr, direction, 0).astype(int), float(thr)


def simple_rules(d, val_mask):
    out, thr = {}, {}
    out["Momentum top-10%"], thr["momentum"] = cutoff_signal(np.sign(d["ret20"].values), np.abs(d["ret20"].values), val_mask)
    elapsed = np.clip((base.WINDOW_MIN - d["min_to_end"].values) / base.WINDOW_MIN, 0, 1)
    gap = d["mid"].values - 0.5
    out["Drift to 0/1 top-10%"], thr["drift"] = cutoff_signal(np.sign(gap), np.abs(gap) * elapsed, val_mask)
    return out, thr


# ── EXECUTION SIMULATOR ─────────────────────────────────────────────────────
def vwap(prices, sizes, q):
    """Average price to trade q shares walking the levels; NaN if the visible book is too thin."""
    cum = np.cumsum(sizes, 1)
    prev = cum - sizes
    fill = np.clip(np.minimum(sizes, q - prev), 0, None)
    px = (fill * prices).sum(1) / q
    return np.where(cum[:, -1] >= q - 1e-9, px, np.nan)


def simulate(d, sig, h, lag_s, q, one_at_a_time):
    """Profit per share (cents) for every trade the signal makes. See the module docstring."""
    con = d["contract"].values
    ts = d["ts"].values.astype(np.int64)
    key = con.astype(np.int64) * 10**13 + ts                 # sorted: contract, then time
    L = base.LEVELS
    ask = vwap(d[[f"apx{k}" for k in range(L)]].values, d[[f"asx{k}" for k in range(L)]].values, q)
    bid = vwap(d[[f"bpx{k}" for k in range(L)]].values, d[[f"bsx{k}" for k in range(L)]].values, q)
    mid = d["mid"].values
    n = len(d)

    def trade(i):
        j = np.searchsorted(key, key[i] + int(lag_s * 1000))
        k = j + h
        ok = (k < n) & (np.minimum(j, n - 1) >= 0)
        jj, kk = np.minimum(j, n - 1), np.minimum(k, n - 1)
        ok = ok & (con[jj] == con[i]) & (con[kk] == con[i])
        return jj, kk, ok

    if one_at_a_time:
        # jump from one signal to the next; while a position is open, ignore new signals
        nz = np.where(sig != 0)[0]
        rows, i_pos = [], 0
        while i_pos < len(nz):
            i = nz[i_pos]
            jj, kk, ok = trade(np.array([i]))
            if ok[0]:
                rows.append((i, jj[0], kk[0]))
                i_pos = np.searchsorted(nz, kk[0] + 1)
            else:
                i_pos += 1
        if not rows:
            return pd.DataFrame()
        i, jj, kk = map(np.array, zip(*rows))
    else:
        i = np.where(sig != 0)[0]
        jj, kk, ok = trade(i)
        i, jj, kk = i[ok], jj[ok], kk[ok]
    dirn = sig[i]
    long_ = dirn == 1
    entry = np.where(long_, ask[jj], bid[jj])
    exit_ = np.where(long_, bid[kk], ask[kk])
    gross = np.where(long_, exit_ - entry, entry - exit_)
    fees = base.fee(entry) + base.fee(exit_)
    out = pd.DataFrame({"contract": con[i], "ts": ts[i], "block": d["block"].values[i], "dir": dirn,
                        "mid": dirn * (mid[kk] - mid[jj]) * 100, "after_spread": gross * 100,
                        "after_fees": (gross - fees) * 100, "hold_s": (ts[kk] - ts[jj]) / 1000})
    out["filled"] = np.isfinite(out["after_fees"])
    return out


def boot_mean(v, clusters, n=N_BOOT, seed=0):
    """Mean, 95% CI and one-sided p-value (H0: mean <= 0), resampling whole clusters."""
    v, clusters = np.asarray(v), np.asarray(clusters)
    if len(v) < 2:
        return np.nan, np.nan, np.nan, np.nan
    g = pd.DataFrame({"v": v, "c": clusters}).groupby("c")["v"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    means = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(means, 2.5), np.percentile(means, 97.5), float((means <= 0).mean())


# ── MAIN: train, predict, simulate one family ───────────────────────────────
def run(fam):
    from dotenv import load_dotenv
    from clickhouse_driver import Client
    import torch
    torch.set_num_threads(base.THREADS)
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    client = Client(host=os.environ["PROBALYTICS_CLICKHOUSE_HOST"], user=os.environ["PROBALYTICS_CLICKHOUSE_USERNAME"],
                    password=os.environ["PROBALYTICS_CLICKHOUSE_PASSWORD"], database="probalytics",
                    secure=True, port=9440)
    os.makedirs(OUTDIR, exist_ok=True)
    slug = base.slug(fam)
    t0 = time.time()
    print(f"[{fam}] loading data (original + {N_FRESH} fresh contracts)", flush=True)
    data = get_data(client, fam)
    d = data["dec"]
    data["l2w"] = base.l2_windows(data)
    data["contracts"].to_csv(f"{OUTDIR}/{slug}_contracts.csv")
    print(d.groupby("split")["contract"].nunique().to_dict(), flush=True)
    split = d["split"].values
    val_mask = split == "val"
    rules, thr = simple_rules(d, val_mask)
    results, f1_rows, cutoffs = [], [], dict(thr)
    rng = np.random.default_rng(7)
    for h in HS:
        delta = d[f"delta{h}"].values
        has = np.isfinite(delta) & (d["ev_end"].values - d["ev_start"].values >= base.T_EVENTS) & (data["l2w"][:, 0] >= 0)
        alpha = 0.5 * np.mean(np.abs(delta[has & (split == "train")]))        # from TRAIN only
        y = base.label(np.nan_to_num(delta), alpha)
        rows = {s: np.where(has & (split == s))[0] for s in ("train", "val", "old_test", "fresh")}
        norm = base.norm_stats(data, rows["train"])
        batcher = base.Batcher(data, norm)
        preds = {}
        for name in ["MBOFormer", "L2Former"]:
            ps = []
            for seed in base.SEEDS:
                t = time.time()
                model, ep = base.train_one(name, data, batcher, y, rows, seed)
                torch.save(model.state_dict(), f"{OUTDIR}/{slug}_{name}_h{h}_s{seed}.pt")
                p = np.full((len(d), 3), np.nan)
                pr = np.concatenate([rows["val"], rows["old_test"], rows["fresh"]])
                p[pr] = base.predict(model, batcher, pr)
                ps.append(p)
                print(f"   h={h} {name} seed {seed}: fresh macro-F1 "
                      f"{base.macro_f1(y[rows['fresh']], p[rows['fresh']].argmax(1)):.3f} ({ep} ep, {time.time() - t:.0f}s)",
                      flush=True)
            preds[name] = np.mean(ps, 0)
            np.save(f"{OUTDIR}/{slug}_{name}_h{h}_probs.npy", preds[name].astype(np.float32))
        # Does the event-vs-snapshot result hold on fresh contracts?
        yf, bf = y[rows["fresh"]], d["block"].values[rows["fresh"]]
        for name in ["MBOFormer", "L2Former"]:
            f1_rows.append(dict(family=fam, h=h, model=name,
                                f1=base.macro_f1(yf, preds[name][rows["fresh"]].argmax(1))))
        f1_rows.append(dict(family=fam, h=h, model="Always flat", f1=base.macro_f1(yf, np.ones_like(yf))))
        diff, lo, hi = base.f1_diff_boot(yf, preds["MBOFormer"][rows["fresh"]].argmax(1),
                                         preds["L2Former"][rows["fresh"]].argmax(1), bf)
        f1_rows.append(dict(family=fam, h=h, model="MBOFormer - L2Former", f1=diff, ci_lo=lo, ci_hi=hi))
        # Signals. No prediction (end of contract, warm-up) = no trade.
        pm = preds["MBOFormer"]
        conf = np.where(has, np.nanmax(np.nan_to_num(pm, nan=0), 1), 0)
        dirn = np.where(has, np.nanargmax(np.nan_to_num(pm, nan=0), 1) - 1, 0)
        sigs = dict(rules)
        sigs["MBOFormer top-10%"], cutoffs[f"mboformer_h{h}"] = cutoff_signal(dirn, conf, val_mask & has)
        sigs["MBOFormer every signal"] = dirn
        coin = rng.choice([-1, 1], len(d))
        sigs["Random direction (MBOFormer timing)"] = np.where(sigs["MBOFormer top-10%"] != 0, coin, 0)
        for test_split in ("fresh", "old_test"):
            m = split == test_split
            ds = d[m].reset_index(drop=True)
            for sname in SIGNALS:
                s = sigs[sname][m]
                for lag in LAGS_S:
                    for q in SIZES:
                        for mode in ("one at a time", "overlapping"):
                            if test_split == "old_test" and (q != 1 or mode != "one at a time"):
                                continue
                            tr = simulate(ds, s, h, lag, q, mode == "one at a time")
                            row = dict(family=fam, split=test_split, signal=sname, h=h, lag_s=lag, size=q, mode=mode,
                                       n_signals=int((s != 0).sum()), n_trades=len(tr))
                            if len(tr):
                                f = tr[tr["filled"]]
                                row["fill_rate"] = len(f) / len(tr)
                                row["median_hold_s"] = float(tr["hold_s"].median())
                                cl = f["contract"] if mode == "one at a time" else f["block"]
                                for col in ("mid", "after_spread", "after_fees"):
                                    mu, lo, hi, p = boot_mean(f[col].values, cl.values)
                                    row.update({f"{col}_c": mu, f"{col}_lo": lo, f"{col}_hi": hi, f"{col}_p": p})
                                row["win_rate"] = float((f["after_fees"] > 0).mean())
                                per_c = f.groupby("contract")["after_fees"].sum()
                                row["contracts_traded"] = len(per_c)
                                row["contracts_profitable"] = int((per_c > 0).sum())
                                row["usd_per_contract"] = float(per_c.sum() / d.loc[m, "contract"].nunique() / 100 * q)
                                if (test_split == "fresh" and sname in ("MBOFormer top-10%", "Momentum top-10%",
                                                                         "Drift to 0/1 top-10%")
                                        and h == PRIMARY["h"] and lag == PRIMARY["lag_s"] and q == PRIMARY["size"]
                                        and mode == PRIMARY["mode"]):
                                    f.assign(family=fam, signal=sname).to_csv(
                                        f"{OUTDIR}/{slug}_trades_{base.slug(sname)}.csv", index=False)
                            results.append(row)
        print(f"   h={h} simulations done", flush=True)
    pd.DataFrame(results).to_csv(f"{OUTDIR}/{slug}_results.csv", index=False)
    pd.DataFrame(f1_rows).to_csv(f"{OUTDIR}/{slug}_f1.csv", index=False)
    json.dump(cutoffs, open(f"{OUTDIR}/{slug}_cutoffs.json", "w"), indent=1)
    print(f"[{fam}] done in {(time.time() - t0) / 60:.0f} min", flush=True)


# ── VERDICT + CHARTS ────────────────────────────────────────────────────────
def load(kind):
    fs = [f"{OUTDIR}/{base.slug(f)}_{kind}.csv" for f in FAMILIES]
    return pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)], ignore_index=True)


def verdict(res):
    """Apply the pre-registered PASS rule. Holm: sort the p-values, compare the i-th smallest to alpha/(m - i)."""
    sel = res[(res["split"] == "fresh") & (res["h"] == PRIMARY["h"]) & (res["lag_s"] == PRIMARY["lag_s"])
              & (res["size"] == PRIMARY["size"]) & (res["mode"] == PRIMARY["mode"])]
    prim = sel[sel["signal"] == PRIMARY["signal"]].set_index("family")
    ps = prim["after_fees_p"].fillna(1.0).sort_values()
    holm = {}
    for i, (fam, p) in enumerate(ps.items()):
        holm[fam] = min(1.0, max(p * (len(ps) - i), max(holm.values()) if holm else 0))
    out = {"primary": PRIMARY, "families": {}}
    passed = True
    for fam in FAMILIES:
        if fam not in prim.index or not np.isfinite(prim.loc[fam].get("after_fees_c", np.nan)):
            out["families"][fam] = dict(passes=False, note="no trades")
            passed = False
            continue
        r = prim.loc[fam]
        rules = sel[(sel["family"] == fam) & sel["signal"].isin(["Momentum top-10%", "Drift to 0/1 top-10%"])]
        best_rule = rules["after_fees_c"].max()
        ok = (r["after_fees_c"] > 0) and (holm[fam] < ALPHA) and (r["after_fees_c"] > best_rule)
        passed &= bool(ok)
        out["families"][fam] = dict(profit_c=r["after_fees_c"], ci=[r["after_fees_lo"], r["after_fees_hi"]],
                                    p_holm=holm[fam], trades=int(r["n_trades"]), best_rule_c=best_rule,
                                    contracts_profitable=f"{int(r['contracts_profitable'])}/{int(r['contracts_traded'])}",
                                    passes=bool(ok))
    out["PASS"] = passed
    return out


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from test_mboformer import style, SURFACE, INK, INK2, GRID, BLUE, ORANGE, AQUA, YELLOW, MAGENTA
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    res = load("results")
    f1 = load("f1")
    v = verdict(res)
    json.dump(v, open(f"{OUTDIR}/verdict.json", "w"), indent=1, default=float)
    print(json.dumps(v, indent=1, default=float))
    print(f1.round(3).to_string(index=False))
    colors = {"MBOFormer top-10%": BLUE, "Momentum top-10%": ORANGE, "Drift to 0/1 top-10%": AQUA,
              "Random direction (MBOFormer timing)": INK2, "MBOFormer every signal": YELLOW}

    # R1: first run (cutoff chosen on test) vs clean re-test
    old = base.read_parts("trading")
    old = old[(old["signal"] == "MBOFormer, top-10% confidence") & (old["lag"] == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        x = np.arange(len(HS))
        series = [("first run: cutoff set on test, overlapping, instant fill", old[old["family"] == fam].set_index("h"),
                   "fees_c", "fees_lo", "fees_hi", MAGENTA, "D"),
                  ("re-test: old test contracts, val cutoff, 0.5 s, one at a time",
                   res[(res["family"] == fam) & (res["split"] == "old_test") & (res["signal"] == "MBOFormer top-10%")
                       & (res["lag_s"] == 0.5) & (res["size"] == 1) & (res["mode"] == "one at a time")].set_index("h"),
                   "after_fees_c", "after_fees_lo", "after_fees_hi", ORANGE, "s"),
                  ("re-test: 30 fresh contracts, val cutoff, 0.5 s, one at a time",
                   res[(res["family"] == fam) & (res["split"] == "fresh") & (res["signal"] == "MBOFormer top-10%")
                       & (res["lag_s"] == 0.5) & (res["size"] == 1) & (res["mode"] == "one at a time")].set_index("h"),
                   "after_fees_c", "after_fees_lo", "after_fees_hi", BLUE, "o"),
                  ("re-test: 30 fresh contracts, val cutoff, instant fill (0 s), one at a time",
                   res[(res["family"] == fam) & (res["split"] == "fresh") & (res["signal"] == "MBOFormer top-10%")
                       & (res["lag_s"] == 0) & (res["size"] == 1) & (res["mode"] == "one at a time")].set_index("h"),
                   "after_fees_c", "after_fees_lo", "after_fees_hi", AQUA, "^")]
        for k, (lab, s, c, lo, hi, col, mk) in enumerate(series):
            s = s.reindex(HS)
            ax.errorbar(x + (k - 1.5) * 0.16, s[c], yerr=[s[c] - s[lo], s[hi] - s[c]], fmt=mk, color=col, ms=7, lw=2,
                        markeredgecolor=SURFACE, label=lab)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in HS])
        style(ax, fam, "cents per share per trade" if ax is axes[0] else None)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, fontsize=8.5)
    fig.suptitle("The 'confident signals' result, before and after a clean re-test\nprofit after spread + fee, 95% CI; above 0 = profitable",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f"{OUTDIR}/retest_1_before_after.png", dpi=150)
    plt.close(fig)

    # R2: profit vs latency, MBOFormer against the simple rules (fresh, h=100, 1 share, one at a time)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        for sname, col in colors.items():
            s = res[(res["family"] == fam) & (res["split"] == "fresh") & (res["signal"] == sname) & (res["h"] == 100)
                    & (res["size"] == 1) & (res["mode"] == "one at a time")].sort_values("lag_s")
            ax.plot(s["lag_s"], s["after_fees_c"], color=col, lw=2, marker="o", ms=5, label=sname,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.fill_between(s["lag_s"], s["after_fees_lo"], s["after_fees_hi"], color=col, alpha=0.10, lw=0)
        ax.axhline(0, color=INK2, lw=1)
        ax.axvline(PRIMARY["lag_s"], color=GRID, lw=1, ls="--")
        ax.set_xticks(LAGS_S)
        style(ax, fam, "profit per share per trade, cents" if ax is axes[0] else None, "latency: signal -> your order fills (s)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, fontsize=8.5)
    fig.suptitle("Fresh contracts, h=100: profit after spread + fee as orders arrive later\nband = 95% CI; dashed line = pre-registered 0.5 s",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{OUTDIR}/retest_2_latency.png", dpi=150)
    plt.close(fig)

    # R3: every fresh contract, primary spec: total profit per 15-minute window
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        p = f"{OUTDIR}/{base.slug(fam)}_trades_{base.slug(PRIMARY['signal'])}.csv"
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        per = t.groupby("contract")["after_fees"].sum().sort_index() / 100
        x = np.arange(len(per))
        cols = [BLUE if v > 0 else MAGENTA for v in per.values]
        ax.bar(x, per.values, color=cols, width=0.7)
        ax.axhline(0, color=INK2, lw=1)
        ax.text(0.01, 0.97, f"{(per > 0).sum()} of {len(per)} windows profitable   total ${per.sum():+.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
        style(ax, fam, "profit per 15-min window, $ (1 share)" if ax is axes[0] else None, "fresh contracts, in time order")
    fig.suptitle("Pre-registered strategy on each fresh contract (h=100, 0.5 s latency, one position at a time)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/retest_3_per_contract.png", dpi=150)
    plt.close(fig)

    # R4: costs breakdown for the primary spec, 1 vs 100 shares
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        x = np.arange(len(SIZES))
        for k, (c, col, lab) in enumerate([("mid", BLUE, "at mid"), ("after_spread", ORANGE, "after spread"),
                                           ("after_fees", MAGENTA, "after spread + fee")]):
            s = res[(res["family"] == fam) & (res["split"] == "fresh") & (res["signal"] == PRIMARY["signal"])
                    & (res["h"] == 100) & (res["lag_s"] == 0.5) & (res["mode"] == "one at a time")].set_index("size").reindex(SIZES)
            ax.bar(x + (k - 1) * 0.26, s[f"{c}_c"], width=0.24, color=col, label=lab)
            ax.errorbar(x + (k - 1) * 0.26, s[f"{c}_c"], yerr=[s[f"{c}_c"] - s[f"{c}_lo"], s[f"{c}_hi"] - s[f"{c}_c"]],
                        fmt="none", ecolor=INK, lw=1)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{q} share{'s' if q > 1 else ''}" for q in SIZES])
        style(ax, fam, "cents per share per trade" if ax is axes[0] else None)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Where the edge goes: mid -> spread -> fee, and with bigger orders (fresh, primary spec)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/retest_4_costs_size.png", dpi=150)
    plt.close(fig)
    print("charts in", OUTDIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        for f in (a.family or FAMILIES):
            run(f)
