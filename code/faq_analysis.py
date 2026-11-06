"""Analyses that back the FAQ and the paper's boundary-conditions section.

Writes results/faq.json and a few csv side tables. Nothing here refits a model:
everything comes from the saved per-transformer error files and the panel itself.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
from core import Panel
import paths

DATA = paths.PANEL
RES = paths.RESULTS
OUT = {}


def selected(panel, mx):
    sel = np.random.default_rng(3).choice(panel.N, mx, replace=False)
    return np.sort(sel)


def load_err(fold, strategy, panel, mx=500):
    f = f"{RES}/per_xfmr_smartds_{fold}_{strategy}.csv"
    d = pd.read_csv(f)
    sel = selected(panel, mx)
    d["id"] = panel.ids[sel[d.col.to_numpy()]]
    return d.set_index("id")


def buckets(s, edges, labels):
    return pd.cut(s, bins=edges, labels=labels, include_lowest=True)


def main():
    panel = Panel.from_dir(DATA, name="smartds")
    meta = panel.meta.copy()
    L = panel.loads
    idx = panel.index

    # ---------------------------------------------------------------- shape features
    day = idx.dayofyear.to_numpy()
    dmax = pd.DataFrame(L, index=idx).resample("D").max().to_numpy()
    dmean = pd.DataFrame(L, index=idx).resample("D").mean().to_numpy()
    with np.errstate(invalid="ignore"):
        cv_peak = np.nanstd(dmax, axis=0) / np.nanmean(dmax, axis=0)
        lf = np.nanmean(dmean, axis=0) / np.nanmean(dmax, axis=0)
    meta["cv_daily_peak"] = cv_peak
    meta["load_factor"] = lf
    meta["ann_peak_kva"] = np.nanmax(L, axis=0)
    meta["mean_kva"] = np.nanmean(L, axis=0)
    meta["util_pct"] = 100 * meta.ann_peak_kva / meta.kva

    # ---------------------------------------------------------------- stratified error
    strat = {}
    for fold, base in (("winter", "cluster"), ("summer", "cluster")):
        e = load_err(fold, base, panel)
        j = meta.loc[e.index].copy()
        j["nmae"] = e.nmae.to_numpy()
        j["peak_ape"] = e.peak_ape.to_numpy()
        tabs = {}
        j["homes_b"] = buckets(j.n_cust, [0, 1, 2, 4, 100], ["1", "2", "3-4", "5+"])
        j["res_b"] = buckets(j.res_share, [-0.01, 0.01, 0.5, 0.99, 1.01],
                             ["0 (non-res)", "1-50%", "50-99%", "100% res"])
        j["cv_b"] = pd.qcut(j.cv_daily_peak, 4, labels=["calmest 25%", "2nd", "3rd", "most volatile 25%"])
        j["kva_b"] = buckets(j.kva, [0, 15, 25, 50, 1000], ["<=15", "25", "50", ">50"])
        for key, col in (("homes", "homes_b"), ("res_share", "res_b"),
                         ("volatility", "cv_b"), ("nameplate", "kva_b")):
            g = j.groupby(col, observed=True).agg(
                n=("nmae", "size"), nmae_med=("nmae", "median"),
                peak_ape_med=("peak_ape", "median"), load_factor=("load_factor", "median"),
                cv=("cv_daily_peak", "median"))
            tabs[key] = g.round(4).reset_index().rename(columns={col: "bucket"}).to_dict("records")
        # correlation of error with the drivers
        tabs["corr"] = {c: round(float(np.corrcoef(j[c], j.nmae)[0, 1]), 3)
                        for c in ("n_cust", "res_share", "cv_daily_peak", "load_factor",
                                  "mean_kva", "kva")}
        strat[fold] = tabs
    OUT["stratified"] = strat

    # ---------------------------------------------------------------- seasonal shapes
    mon = pd.DataFrame(L, index=idx)
    scale = np.nanmean(L, axis=0)
    norm = mon / scale
    season = {}
    for nm, months in (("winter", [12, 1, 2]), ("spring", [3, 4, 5]),
                       ("summer", [6, 7, 8]), ("fall", [9, 10, 11])):
        m = norm[idx.month.isin(months)]
        season[nm] = dict(
            weekday=[round(float(v), 4) for v in
                     m[~m.index.dayofweek.isin([5, 6])].groupby(m[~m.index.dayofweek.isin([5, 6])].index.hour).mean().mean(axis=1)],
            weekend=[round(float(v), 4) for v in
                     m[m.index.dayofweek.isin([5, 6])].groupby(m[m.index.dayofweek.isin([5, 6])].index.hour).mean().mean(axis=1)],
        )
    OUT["seasonal_shape"] = season

    # month of each transformer's annual peak
    pk_month = idx[np.nanargmax(L, axis=0)].month.to_numpy()
    OUT["peak_month_share"] = {int(m): round(float((pk_month == m).mean()), 4)
                               for m in range(1, 13)}
    pk_hour = idx[np.nanargmax(L, axis=0)].hour.to_numpy()
    OUT["peak_hour_share"] = {int(h): round(float((pk_hour == h).mean()), 4)
                              for h in range(24)}

    # ---------------------------------------------------------------- temperature link
    t = panel.temp[:, 0]
    tb = pd.cut(t, [-30, 0, 10, 20, 25, 30, 50],
                labels=["<0C", "0-10C", "10-20C", "20-25C", "25-30C", ">30C"])
    tt = {}
    for fold, strategy in (("winter", "cluster"), ("summer", "cluster")):
        pr = pd.read_parquet(f"{RES}/pred_smartds_{fold}_{strategy}.parquet")
        tt[fold] = dict(cols=list(pr.columns)[:6], n=len(pr))
    OUT["pred_cols"] = tt
    agg = np.nanmean(L / scale, axis=1)
    dfm = pd.DataFrame(dict(temp=t, load=agg, tb=tb))
    OUT["temp_response"] = (dfm.groupby("tb", observed=True)
                            .agg(hours=("load", "size"), mean_norm_load=("load", "mean"))
                            .round(3).reset_index().astype({"tb": str}).to_dict("records"))

    # ---------------------------------------------------------------- load model test
    sims = {
        "rural": paths.SIM["rural"],
        "urban": paths.SIM["urban"],
    }
    cm = pd.read_csv(paths.CUSTOMER_MAP)
    kw_panel = pd.read_parquet(paths.need(f"{paths.PANEL_KW}/loads.parquet", "run the VPS build in code/vps to regenerate it from SMART-DS"))
    lm = {}
    for nm, path in sims.items():
        try:
            s = pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception as ex:
            lm[nm] = f"unavailable: {ex}"
            continue
        common = [c for c in s.columns if c in kw_panel.columns]
        if not common:
            lm[nm] = "no overlap with the kW panel"
            continue
        a = s[common].to_numpy(np.float64)          # simulated kVA
        b = kw_panel[common].to_numpy(np.float64)   # summed customer kW
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0.2)
        ratio = np.where(ok, a / np.maximum(b, 1e-6), np.nan)
        util = np.where(ok, b / np.nanmean(b, axis=0), np.nan)
        qs = pd.qcut(pd.Series(util[ok].ravel()), 5,
                     labels=["lightest 20%", "2nd", "3rd", "4th", "heaviest 20%"])
        lm[nm] = dict(
            n_xfmr=len(common),
            ratio_mean=round(float(np.nanmean(ratio)), 4),
            ratio_by_loading=(pd.DataFrame(dict(q=qs, r=ratio[ok].ravel()))
                              .groupby("q", observed=True).r.median().round(4).to_dict()),
            corr_hourly=round(float(np.corrcoef(a[ok].ravel(), b[ok].ravel())[0, 1]), 5),
        )
    OUT["load_model_test"] = lm

    # ---------------------------------------------------------------- fleet facts
    OUT["fleet"] = dict(
        n=int(panel.N),
        homes_mean=round(float(meta.n_cust.mean()), 2),
        share_single_home=round(float((meta.n_cust == 1).mean()), 4),
        share_nonres=round(float((meta.res_share < 0.99).mean()), 4),
        share_pure_res=round(float((meta.res_share >= 0.99).mean()), 4),
        kva_median=float(meta.kva.median()),
        util_median=round(float(meta.util_pct.median()), 2),
        util_p95=round(float(np.percentile(meta.util_pct, 95)), 2),
        share_over_100=round(float((meta.util_pct > 100).mean()), 4),
        share_over_80=round(float((meta.util_pct > 80).mean()), 4),
        cv_median=round(float(meta.cv_daily_peak.median()), 4),
        cv_p90=round(float(np.percentile(meta.cv_daily_peak, 90)), 4),
        load_factor_median=round(float(meta.load_factor.median()), 4),
    )

    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/faq.json", "w") as fh:
        json.dump(OUT, fh, indent=1)
    meta.to_csv(f"{RES}/meta_enriched.csv")
    print(json.dumps({k: v for k, v in OUT.items() if k in
                      ("fleet", "load_model_test", "peak_month_share")}, indent=1))


if __name__ == "__main__":
    main()
