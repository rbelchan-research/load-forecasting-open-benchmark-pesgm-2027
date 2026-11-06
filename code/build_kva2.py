"""Rebuild the apparent-power panel with the fitted kW -> kVA map, and check the
rebuild against the transformers where the power flow was actually run."""
import json, os, shutil
import paths
import numpy as np, pandas as pd

FIT = json.load(open(paths.need(f"{paths.RESULTS}/kva_fit.json", "run fit_kva.py first")))
SRC = paths.need(paths.PANEL_KW, "run the VPS build in code/vps to regenerate it from SMART-DS")
DST = paths.PANEL
SIM = paths.SIM

kw = pd.read_parquet(f"{SRC}/loads.parquet")
meta = pd.read_csv(f"{SRC}/meta.csv").set_index("id")
meta = meta.loc[kw.columns]

def reg_key(r):
    return "rural" if "rural" in str(r) else "urban"

def apply_map(P, kva, region):
    f = FIT[region]
    a, b, c = f["a"], f["b"], f["c"]
    sc = kva[None, :] / 25.0
    return np.sqrt((P + a * sc) ** 2 + (b * P + c * sc) ** 2)

P = kw.to_numpy(np.float64)
kva = meta.kva.to_numpy(np.float64)
S = np.empty_like(P)
for region in ("rural", "urban"):
    m = np.array([reg_key(r) == region for r in meta.region])
    if m.any():
        S[:, m] = apply_map(P[:, m], kva[m], region)
S = S.astype(np.float32)
os.makedirs(DST, exist_ok=True)
out = pd.DataFrame(S, index=kw.index, columns=kw.columns)
out.to_parquet(f"{DST}/loads.parquet")
m2 = meta.copy()
for region in ("rural", "urban"):
    sel = [reg_key(r) == region for r in m2.region]
    for k in ("a", "b", "c"):
        m2.loc[sel, f"fit_{k}"] = FIT[region][k]
m2.to_csv(f"{DST}/meta.csv")
shutil.copy(f"{SRC}/weather.parquet", f"{DST}/weather.parquet")
print(f"wrote {DST}: {out.shape}")

# ------------------------------------------------------------------ validation
rows = []
for region, path in SIM.items():
    s = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = [c for c in s.columns if c in kw.columns]
    truth = s[cols].to_numpy(np.float64)
    fitted = out[cols].to_numpy(np.float64)
    flatf = FIT[region]["flat_factor"]
    flat = kw[cols].to_numpy(np.float64) * flatf
    nk = meta.loc[cols, "kva"].to_numpy(np.float64)
    def summarize(nm, X):
        ok = np.isfinite(truth) & np.isfinite(X) & (truth > 0)
        pk = np.nanmax(np.where(np.isfinite(X), X, np.nan), axis=0)
        pkt = np.nanmax(np.where(np.isfinite(truth), truth, np.nan), axis=0)
        return dict(region=region, variant=nm,
                    hourly_mape=round(float(np.mean(np.abs(X[ok] - truth[ok]) / truth[ok]) * 100), 2),
                    peak_bias_pct=round(float(np.mean(pk / pkt - 1) * 100), 2),
                    n_over_100=int((pk > nk).sum()), n_over_80=int((pk > 0.8 * nk).sum()),
                    n_over_100_true=int((pkt > nk).sum()), n_over_80_true=int((pkt > 0.8 * nk).sum()),
                    n=len(cols))
    rows.append(summarize("fitted", fitted))
    rows.append(summarize("flat factor", flat))
    rows.append(summarize("summed kW", kw[cols].to_numpy(np.float64)))
v = pd.DataFrame(rows)
v.to_csv(f"{paths.RESULTS}/kva_validation.csv", index=False)
print(v.to_string(index=False))

# fleet loading under each construction
for nm, X in (("fitted", out), ("flat", None)):
    pass
pk_fit = out.max().to_numpy()
util = 100 * pk_fit / meta.kva.to_numpy()
print(f"\nfitted panel fleet: median utilization {np.median(util):.1f}%, "
      f"over 80% {np.mean(util>80)*100:.2f}%, over 100% {np.mean(util>100)*100:.2f}%")
