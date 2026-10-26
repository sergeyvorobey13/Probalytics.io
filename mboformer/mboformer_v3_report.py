"""Collect every v3 result (plus the leaky v1/v2 numbers it replaces) into one JSON for the artifact's charts."""
import json
import os
import pickle
import numpy as np
import pandas as pd
import test_mboformer as base

OUTDIR = "mboformer_v3"
FAMS = ["BTC 15m", "ETH 15m", "Sports & esports"]
CRYPTO = ["BTC 15m", "ETH 15m"]


def num(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def records(df):
    return [{k: (num(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in df.to_dict("records")]


def calibration(fam):
    """Brier score of the Polymarket mid and of the Binance fair value against the outcome, by time left (test)."""
    z = pickle.load(open(f"{OUTDIR}/{base.slug(fam)}/sim_data.pkl", "rb"))
    d = z["d"]
    d = d[d["split"] == "test"].copy()
    last = d.groupby("contract")["mid"].transform("last")
    pay = d["payout"].where(d["payout"].notna(), np.where(last >= 0.95, 1.0, np.where(last <= 0.05, 0.0, np.nan)))
    d["pay"] = pay
    d = d[d["pay"].notna()]
    d["s_left"] = d["min_to_end"] * 60
    bins = [0, 60, 180, 420, 900]
    labels = ["last minute", "1-3 min left", "3-7 min left", "7-15 min left"]
    d["bucket"] = pd.cut(d["s_left"], bins, labels=labels, include_lowest=True)
    rows = []
    for b, g in d.groupby("bucket", observed=True):
        rows.append(dict(family=fam, bucket=str(b), n=len(g), contracts=g["contract"].nunique(),
                         brier_market=float(((g["mid"] - g["pay"]) ** 2).mean()),
                         brier_fv=float(((g["fv"] - g["pay"]) ** 2).mean()),
                         gap_c=float((g["fv"] - g["mid"]).abs().mean() * 100)))
    return rows


def report():
    out = {}
    f1 = []
    for fam in FAMS:
        p = f"{OUTDIR}/{base.slug(fam)}/f1.csv"
        if os.path.exists(p):
            f1.append(pd.read_csv(p))
    f1 = pd.concat(f1, ignore_index=True)
    out["f1_honest"] = records(f1)
    leaky = []
    for fam in FAMS:
        p = f"mboformer_results/{base.slug(fam)}_f1.csv"
        if os.path.exists(p):
            g = pd.read_csv(p)
            g = g[g["model"] == "MBOFormer - L2Former"]
            leaky += [dict(family=fam, h=int(r.h), run="v1 replication (leaky)", diff=r.f1_mean, lo=r.ci_lo, hi=r.ci_hi)
                      for r in g.itertuples()]
        p = f"mboformer_retest/{base.slug(fam)}_f1.csv"
        if os.path.exists(p):
            g = pd.read_csv(p)
            g = g[g["model"] == "MBOFormer - L2Former"]
            leaky += [dict(family=fam, h=int(r.h), run="v2 re-test (leaky)", diff=r.f1, lo=r.ci_lo, hi=r.ci_hi)
                      for r in g.itertuples()]
    out["f1_leaky_diff"] = records(pd.DataFrame(leaky))
    for name in ("strategies", "latency", "cost_stack", "per_contract", "maker_check"):
        p = f"{OUTDIR}/{name}.csv"
        if os.path.exists(p):
            out[name] = records(pd.read_csv(p))
    p = f"{OUTDIR}/maker_detail.csv"
    if os.path.exists(p):
        m = pd.read_csv(p)
        out["maker_summary"] = records(m.groupby(["family"]).agg(
            fills=("fills", "mean"), markout_c=("markout_c", "mean"), rebate=("rebate", "mean")).reset_index())
    out["verdict"] = json.load(open(f"{OUTDIR}/verdict.json")) if os.path.exists(f"{OUTDIR}/verdict.json") else None
    cal, spot = [], {}
    for fam in CRYPTO:
        if os.path.exists(f"{OUTDIR}/{base.slug(fam)}/sim_data.pkl"):
            cal += calibration(fam)
        p = f"{OUTDIR}/{base.slug(fam)}/spot_check.json"
        if os.path.exists(p):
            spot[fam] = json.load(open(p))
    out["calibration"] = cal
    out["spot_check"] = spot
    counts = {}
    for fam in FAMS:
        p = f"{OUTDIR}/{base.slug(fam)}/contracts.csv"
        if os.path.exists(p):
            counts[fam] = pd.read_csv(p)["split"].value_counts().to_dict()
    out["contracts"] = counts
    json.dump(out, open(f"{OUTDIR}/report_data.json", "w"), indent=1, default=lambda x: None)
    print("wrote", f"{OUTDIR}/report_data.json", {k: (len(v) if isinstance(v, list) else 1) for k, v in out.items()})


if __name__ == "__main__":
    report()
