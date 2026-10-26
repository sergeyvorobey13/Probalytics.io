"""Stress-test the one strategy that passed in a family: maker quoting gated by the paper MBOFormer (S6b).
Parameters are the ones chosen on validation by mboformer_v3.py --simulate; nothing is re-tuned on test.
  1. Where the profit comes from: spread capture vs leftover inventory held to resolution.
  2. Fill rule: strict (only when the market trades THROUGH our price) and queue x2 (twice the queue ahead).
  3. Cancel/placement latency 0.05 / 0.10 / 0.20 / 0.30 s.
  4. Placebo: the same quoting time (same share of moments quoted), gates placed at random.
"""
import json
import pickle
import numpy as np
import pandas as pd
import mboformer_v3 as v
import test_mboformer as base

OUT = v.OUTDIR
res = pd.read_csv(f"{OUT}/strategies.csv")


def variant_contract(c, gt, gb, ga, lag_ms, mode):
    """Copy of v.sim_maker_contract with a switchable fill rule and a resolution split."""
    if mode == "queue x2":
        gt = gt.copy()
        gt[:, 3] = gt[:, 3] / 2            # executions count half -> twice the queue ahead
    if mode == "strict":
        gt = np.zeros((0, 4), dtype=np.float32)
    r = v.sim_maker_contract(c, gt, gb, ga, lag_ms)
    return r


def run(d, gt, gb, ga, cons, lag_s, mode):
    rows = []
    for k in cons:
        idx = np.where(d["contract"].values == k)[0]
        c = d.iloc[idx]
        r = variant_contract(c, gt.get(k, np.zeros((0, 4))), gb[idx], ga[idx], int(lag_s * 1000), mode)
        last_mid = float(c["mid"].values[-1])
        r["resolution_part"] = r["inv_end"] * (c["payout"].values[0] - last_mid)
        r["contract"] = k
        rows.append(r)
    return pd.DataFrame(rows)


out = []
for fam in v.CRYPTO:
    slug = base.slug(fam)
    z = pickle.load(open(f"{OUT}/{slug}/sim_data.pkl", "rb"))
    d, full, gt = z["d"], z["full"], z["gt"]
    last = d.groupby("contract")["mid"].transform("last").values
    guess = np.where(last >= 0.95, 1.0, np.where(last <= 0.05, 0.0, np.nan))
    d["payout"] = np.where(np.isfinite(d["payout"].values), d["payout"].values, guess)
    split = d["split"].values
    cons = sorted(d.loc[split == "test", "contract"].unique())
    pm = v.load_probs(f"{OUT}/{slug}", "paper_MBOFormer_h100", base.SEEDS)
    has = full & np.isfinite(pm[:, 0])
    th = float(res[(res.family == fam) & (res.strategy == "S6b Maker, paper MBOFormer gate")]["param"].iloc[0])
    gb = (np.nan_to_num(pm[:, 0]) >= th) | ~has
    ga = (np.nan_to_num(pm[:, 2]) >= th) | ~has
    # placebo: same number of gated moments per side, shuffled within each contract
    rng = np.random.default_rng(0)
    pb, pa = gb.copy(), ga.copy()
    for k in cons:
        idx = np.where(d["contract"].values == k)[0]
        pb[idx] = rng.permutation(gb[idx])
        pa[idx] = rng.permutation(ga[idx])
    tests = [("as tested", 0.10, "base", gb, ga)]
    tests += [(f"latency {l:.2f} s", l, "base", gb, ga) for l in (0.05, 0.20, 0.30)]
    tests += [("strict fills (trade-through only)", 0.10, "strict", gb, ga),
              ("queue ahead x2", 0.10, "queue x2", gb, ga),
              ("placebo: same quoting time, random moments", 0.10, "base", pb, pa),
              ("no gate", 0.10, "base", np.zeros(len(d), bool), np.zeros(len(d), bool))]
    for name, lag, mode, b_, a_ in tests:
        r = run(d, gt, b_, a_, cons, lag, mode)
        mu, lo, hi, p = v.boot(r["pnl"].values)
        spread_part = r["pnl"] - r["resolution_part"]
        out.append(dict(family=fam, test=name, threshold=th, usd=mu, lo=lo, hi=hi, p=p,
                        profitable=f"{(r['pnl'] > 0).sum()}/{len(r)}", fills=r["fills"].mean(),
                        markout_c=r["markout_c"].mean(), rebate=r["rebate"].mean(),
                        usd_before_resolution=spread_part.mean(), usd_resolution=r["resolution_part"].mean()))
        print(f"[{fam}] {name:44s} ${mu:+.3f} ({lo:+.3f}..{hi:+.3f}) {out[-1]['profitable']:>6s} "
              f"fills {out[-1]['fills']:.1f} markout {out[-1]['markout_c']:+.2f}c "
              f"pre-resolution ${out[-1]['usd_before_resolution']:+.3f} resolution ${out[-1]['usd_resolution']:+.3f}",
              flush=True)
pd.DataFrame(out).to_csv(f"{OUT}/maker_check.csv", index=False)
