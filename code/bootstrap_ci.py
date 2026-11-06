"""Bootstrap intervals on the fleet-median nMAE in Table II.

Resamples the 500 test transformers with replacement. Reports, per strategy and fold,
the median and a 95 percent percentile interval, and for chosen pairs the paired
interval on the difference of medians (same resampled transformers for both).
Writes results/bootstrap_ci.json.
"""
import json, os, numpy as np, pandas as pd

# run from code/ in the bundle (results in ../results) or from the working bench folder
import paths
RES = paths.RESULTS

B, SEED = 4000, 7
STRATS = ["seasonal_naive", "persistence", "ridge_local", "local", "cluster", "global",
          "global_cluster", "topdown_rolling", "topdown_cluster", "lstm_global",
          "global_cluster_q90", "local_q90"]
PAIRS = [("cluster", "global"), ("cluster", "local"), ("global", "local"),
         ("lstm_global", "cluster"), ("ridge_local", "cluster")]
out = {}
for fold in ("winter", "summer"):
    S = {s: pd.read_csv(f"{RES}/per_xfmr_smartds_{fold}_{s}.csv").set_index("col").nmae
         for s in STRATS}
    idx = S["cluster"].index
    S = {s: v.reindex(idx).to_numpy() for s, v in S.items()}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(idx), (B, len(idx)))
    f = out[fold] = {"n": int(len(idx)), "strategy": {}, "pair": {}}
    for s, a in S.items():
        bm = np.nanmedian(a[draws], axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        f["strategy"][s] = dict(median=float(np.nanmedian(a)), lo=float(lo), hi=float(hi),
                                half_width=float((hi - lo) / 2))
    for x, y in PAIRS:
        d = np.nanmedian(S[x][draws], 1) - np.nanmedian(S[y][draws], 1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        f["pair"][f"{x}-{y}"] = dict(diff=float(np.nanmedian(S[x]) - np.nanmedian(S[y])),
                                     lo=float(lo), hi=float(hi))
json.dump(out, open(f"{RES}/bootstrap_ci.json", "w"), indent=1)
w = out["winter"]
hw = [w["strategy"][s]["half_width"] for s in ("cluster", "global", "local", "lstm_global", "ridge_local")]
print("winter half-widths of learned strategies: %.2f to %.2f points" % (100 * min(hw), 100 * max(hw)))
for k, v in w["pair"].items():
    print(f"  winter {k}: {100*v['diff']:+.2f}  [{100*v['lo']:+.2f}, {100*v['hi']:+.2f}]")
