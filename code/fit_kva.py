"""Fit a physically shaped map from summed customer kW to transformer apparent power,
using the transformers where a full-year power flow was actually run.

Model: S(t)^2 = (P(t) + a)^2 + (b*P(t) + c)^2
  a  constant real loss (core loss plus a small conductor term)
  b  reactive power per unit of real power (the load's power factor)
  c  constant reactive draw (magnetizing current, plus leakage at light load)
Fitted per region by least squares on S, with a, b, c shared across the region's
transformers but a scaled per transformer by nameplate, since core loss tracks size.
"""
import numpy as np, pandas as pd, json
import paths
from scipy.optimize import least_squares
import os

SIM = paths.SIM
KW = pd.read_parquet(paths.need(f"{paths.PANEL_KW}/loads.parquet", "run the VPS build in code/vps to regenerate it from SMART-DS"))
META = pd.read_csv(f"{paths.PANEL_KW}/meta.csv").set_index("id")

def fit(region, path):
    s = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = [c for c in s.columns if c in KW.columns]
    S = s[cols].to_numpy(np.float64)
    P = KW[cols].to_numpy(np.float64)
    kva = META.loc[cols, "kva"].to_numpy(np.float64)
    ok = np.isfinite(S) & np.isfinite(P) & (S > 0)
    Sv, Pv = S[ok], P[ok]
    kv = np.broadcast_to(kva[None, :], P.shape)[ok]
    def model(th, P, kv):
        a, b, c = th
        return np.sqrt((P + a * kv / 25.0) ** 2 + (b * P + c * kv / 25.0) ** 2)
    def res(th):
        return model(th, Pv, kv) - Sv
    r = least_squares(res, [0.05, 0.3, 0.3], bounds=([0, 0, 0], [5, 2, 5]))
    a, b, c = r.x
    pred = model(r.x, Pv, kv)
    flat = Pv * (Sv / Pv).mean()
    def err(x):
        return dict(mape=round(float(np.mean(np.abs(x - Sv) / Sv) * 100), 3),
                    rmse=round(float(np.sqrt(np.mean((x - Sv) ** 2))), 4))
    # peak-hour accuracy, which is what the screen depends on
    Pm = np.where(np.isfinite(P) & np.isfinite(S), P, np.nan)
    pk = np.nanargmax(np.where(np.isfinite(S), S, -1), axis=0)
    jj = np.arange(len(cols))
    Sp, Pp = S[pk, jj], Pm[pk, jj]
    fitted_pk = model(r.x, Pp, kva)
    flatf = float(np.nanmean(S / np.where(P > 0, P, np.nan)))
    out = dict(region=region, n_xfmr=len(cols), a=round(float(a), 4), b=round(float(b), 4),
               c=round(float(c), 4), flat_factor=round(flatf, 4),
               fit=err(pred), flat=err(Pv * flatf),
               peak_bias_fit=round(float(np.nanmean(fitted_pk / Sp - 1) * 100), 3),
               peak_bias_flat=round(float(np.nanmean(Pp * flatf / Sp - 1) * 100), 3))
    return out

res = {}
for reg, p in SIM.items():
    res[reg] = fit(reg, p)
    print(json.dumps(res[reg], indent=1))
os.makedirs(paths.RESULTS, exist_ok=True)
json.dump(res, open(f"{paths.RESULTS}/kva_fit.json", "w"), indent=1)
