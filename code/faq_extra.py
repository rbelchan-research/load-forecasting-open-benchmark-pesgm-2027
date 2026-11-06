"""Extra numbers for the FAQ: what daily-register data costs against hourly interval
data, seasonal peak timing, and the battery and solar scenario summaries."""
import json
import numpy as np, pandas as pd
from core import Panel
import paths

p = Panel.from_dir(paths.PANEL, name="smartds")
L = p.loads
idx = p.index
kva = p.meta.kva.to_numpy(np.float64)
out = {}

df = pd.DataFrame(L, index=idx)
dmax = df.resample("D").max().to_numpy()
dsum = df.resample("D").sum().to_numpy()          # kVAh per day
dmean = dsum / 24.0

# ---- AMI 1.0 style: only a daily register read, peak inferred from a load factor
lf_fleet = np.nanmedian(dmean / np.where(dmax > 0, dmax, np.nan))
est = dmean / lf_fleet
err = np.abs(est - dmax) / np.where(dmax > 0, dmax, np.nan)
out["daily_register"] = dict(
    fleet_load_factor=round(float(lf_fleet), 4),
    peak_ape_median=round(float(np.nanmedian(err)), 4),
    peak_ape_p90=round(float(np.nanpercentile(err[np.isfinite(err)], 90)), 4),
)
# per-transformer load factor known (a better AMI 1.0 variant: one year of monthly peaks)
lf_i = np.nanmedian(dmean / np.where(dmax > 0, dmax, np.nan), axis=0)
est_i = dmean / lf_i[None, :]
err_i = np.abs(est_i - dmax) / np.where(dmax > 0, dmax, np.nan)
out["daily_register_known_lf"] = dict(
    peak_ape_median=round(float(np.nanmedian(err_i)), 4),
    peak_ape_p90=round(float(np.nanpercentile(err_i[np.isfinite(err_i)], 90)), 4),
)
# screening quality of the daily-register route against the hourly truth
truth = dmax > kva[None, :]
flag = est_i > kva[None, :]
tp = int((truth & flag).sum()); fp = int((~truth & flag).sum()); fn = int((truth & ~flag).sum())
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
out["daily_register_screen"] = dict(precision=round(prec, 3), recall=round(rec, 3),
                                    f1=round(2 * prec * rec / max(prec + rec, 1e-9), 3),
                                    events=int(truth.sum()))

# ---- seasonal peak timing
scale = np.nanmean(L, axis=0)
norm = pd.DataFrame(L / scale, index=idx)
for nm, months in (("winter", [12, 1, 2]), ("summer", [6, 7, 8])):
    m = norm[idx.month.isin(months)]
    prof = m.groupby(m.index.hour).mean().mean(axis=1)
    out[f"{nm}_peak_hour"] = int(prof.idxmax())
    out[f"{nm}_peak_level"] = round(float(prof.max()), 3)
    out[f"{nm}_min_level"] = round(float(prof.min()), 3)
    out[f"{nm}_peak_to_trough"] = round(float(prof.max() / prof.min()), 2)

# ---- how much of the fleet peaks in summer vs winter
pk_month = idx[np.nanargmax(L, axis=0)].month.to_numpy()
out["share_peak_summer"] = round(float(np.isin(pk_month, [5, 6, 7, 8, 9]).mean()), 4)
out["share_peak_winter"] = round(float(np.isin(pk_month, [11, 12, 1, 2, 3]).mean()), 4)

# ---- coincidence: how many homes before the series behaves
nc = p.meta.n_cust.to_numpy()
cv = []
for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 99)):
    sel = (nc >= lo) & (nc <= hi)
    if sel.sum() < 5:
        continue
    d = pd.DataFrame(L[:, sel], index=idx).resample("D").max().to_numpy()
    cv.append(dict(homes=f"{lo}-{hi}" if lo != hi else str(lo), n=int(sel.sum()),
                   cv_daily_peak=round(float(np.nanmean(np.nanstd(d, 0) / np.nanmean(d, 0))), 4)))
out["coincidence"] = cv

json.dump(out, open(f"{paths.RESULTS}/faq_extra.json", "w"), indent=1)
print(json.dumps(out, indent=1))
