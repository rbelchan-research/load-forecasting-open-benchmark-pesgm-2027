"""Benchmark runner.

  python run_experiment.py main      --data DIR [--max 600] [--rounds 400]
  python run_experiment.py grouping  --data DIR [--max 300]
  python run_experiment.py coverage  --data DIR
  python run_experiment.py dirty     --data DIR

Every command appends tidy rows to <out>/results.csv.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd

from core import (Panel, FeatureBuilder, make_groups, predict_strategy, topdown_predict,
                  donor_predict, score, degrade, weekly_profile)
from scenarios import add_ev, add_pv, add_bess, loading_summary

FOLDS = {
    "summer": dict(tr=("01-01", "06-15"), va=("06-16", "06-30"), te=("07-01", "07-31")),
    "winter": dict(tr=("01-01", "11-15"), va=("11-16", "11-30"), te=("12-01", "12-31")),
}


def rows_for(panel, span, year=None):
    year = year or panel.index[0].year
    a = pd.Timestamp(f"{year}-{span[0]}")
    b = pd.Timestamp(f"{year}-{span[1]} 23:00")
    return np.where((panel.index >= a) & (panel.index <= b))[0]


def log(out, row):
    os.makedirs(out, exist_ok=True)
    row = {k: (float(v) if isinstance(v, (np.floating,)) else
               int(v) if isinstance(v, (np.integer,)) else v) for k, v in row.items()}
    with open(f"{out}/results.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    keys = ["dataset", "fold", "experiment", "strategy", "grouping", "k", "tier", "nmae_med",
            "nrmse_med", "peak_ape_med", "ovl100_f1", "secs"]
    print(" | ".join(f"{k}={row.get(k)}" for k in keys if k in row), flush=True)


def prep(panel, fold):
    f = FOLDS[fold]
    tr, va, te = (rows_for(panel, f[k]) for k in ("tr", "va", "te"))
    tr = tr[tr >= 360]                      # feature warm-up
    scale = np.nanmean(panel.loads[tr], axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-3), scale, 1.0).astype(np.float32)
    return tr, va, te, scale


def run_main(panel, out, folds, rounds, grouping="shape_kmeans", k=8, strategies=None):
    strategies = strategies or ["seasonal_naive", "persistence", "ridge_local", "local",
                                "cluster", "global", "global_cluster", "topdown_cluster",
                                "topdown_feeder", "lstm_global", "global_cluster_q90",
                                "local_q90"]
    for fold in folds:
        tr, va, te, scale = prep(panel, fold)
        groups, glab = make_groups(grouping, panel, tr, scale, k=k)
        feeders = make_groups("feeder", panel, tr, scale)[0]
        wp = weekly_profile(panel.loads, panel.cal["how"], scale, tr)
        gprof = np.stack([wp[groups == g].mean(0) for g in np.unique(groups)])
        fb = FeatureBuilder(panel, scale, seed=11)
        fbg = None
        for s in strategies:
            t0 = time.time()
            try:
                if s.startswith("global_cluster"):
                    if fbg is None:
                        fbg = FeatureBuilder(panel, scale, groups=groups, group_profile=gprof,
                                             use_group=True, seed=11)
                    q = dict(objective="quantile", alpha=0.9) if s.endswith("q90") else None
                    pr, tt, rr, cc = predict_strategy("global", fbg, tr, va, te, rounds=rounds,
                                                      params=q)
                elif s == "local_q90":
                    pr, tt, rr, cc = predict_strategy("local", fb, tr, va, te, rounds=rounds,
                                                      params=dict(objective="quantile", alpha=0.9))
                elif s == "topdown_cluster":
                    pr, tt, rr, cc = topdown_predict(fb, panel, groups, tr, va, te, scale, rounds)
                elif s == "topdown_feeder":
                    pr, tt, rr, cc = topdown_predict(fb, panel, feeders, tr, va, te, scale, rounds)
                elif s == "topdown_rolling":
                    pr, tt, rr, cc = topdown_predict(fb, panel, groups, tr, va, te, scale, rounds,
                                                     share_mode="rolling")
                elif s == "cluster":
                    pr, tt, rr, cc = predict_strategy("cluster", fb, tr, va, te, groups=groups, rounds=rounds)
                elif s.startswith("lstm"):
                    from neural import lstm_global_predict
                    gg = groups if s.endswith("cluster") else None
                    pr, tt, rr, cc = lstm_global_predict(
                        panel, scale, tr, va, te, groups=gg,
                        n_group=int(len(np.unique(groups))) if gg is not None else 0,
                        epochs=int(os.environ.get("LSTM_EPOCHS", 18)),
                        max_train=int(os.environ.get("LSTM_MAXTRAIN", 120000)))
                else:
                    pr, tt, rr, cc = predict_strategy(s, fb, tr, va, te, rounds=rounds)
            except Exception as e:
                print(f"FAILED {s}: {type(e).__name__}: {e}", flush=True)
                continue
            m, per = score(pr, tt, rr, cc, panel, scale, tag=s)
            pd.DataFrame(dict(row=rr, col=cc, pred=pr, truth=tt)).to_parquet(
                f"{out}/pred_{panel.name}_{fold}_{s}.parquet")
            m.update(dataset=panel.name, fold=fold, experiment="main", strategy=s,
                     grouping=glab if s in ("cluster", "global_cluster", "topdown_cluster") else "-",
                     k=int(len(np.unique(groups))), tier="hourly_full", secs=round(time.time() - t0, 1))
            log(out, m)
            per.to_csv(f"{out}/per_xfmr_{panel.name}_{fold}_{s}.csv")
        del fb, fbg


def run_grouping(panel, out, fold="winter", rounds=250, ks=(2, 4, 8, 16, 32),
                 kinds=("shape_kmeans", "daily_kmeans", "feature_kmeans", "ward_corr",
                        "dtw_kmedoids", "attribute", "feeder"),
                 strategies=("cluster", "topdown")):
    tr, va, te, scale = prep(panel, fold)
    fb = FeatureBuilder(panel, scale, seed=11)
    for kind in kinds:
        kl = ks if kind in ("shape_kmeans", "daily_kmeans", "feature_kmeans", "ward_corr", "dtw_kmedoids") else (0,)
        for k in kl:
            t0 = time.time()
            groups, glab = make_groups(kind, panel, tr, scale, k=max(k, 2))
            for s in strategies:
                t1 = time.time()
                if s == "cluster":
                    pr, tt, rr, cc = predict_strategy("cluster", fb, tr, va, te, groups=groups, rounds=rounds)
                else:
                    pr, tt, rr, cc = topdown_predict(fb, panel, groups, tr, va, te, scale, rounds)
                m, _ = score(pr, tt, rr, cc, panel, scale, tag=f"{s}|{glab}")
                m.update(dataset=panel.name, fold=fold, experiment="grouping", strategy=s,
                         grouping=kind, k=int(len(np.unique(groups))), tier="hourly_full",
                         secs=round(time.time() - t1, 1))
                log(out, m)


def run_coverage(panel, out, fold="winter", rounds=300, fracs=(0.15, 0.35, 0.7), k=8, eval_frac=0.4):
    """Partial AMI coverage. Only a fraction of transformers have metered history;
    the rest are forecast with no series of their own, knowing only static attributes
    and their monthly billed energy (which fixes their mean load)."""
    from sklearn.neighbors import KNeighborsClassifier
    tr, va, te, scale = prep(panel, fold)
    rng = np.random.default_rng(5)
    fb = FeatureBuilder(panel, scale, seed=11)
    cold = fb.keep_idx("coldstart")
    all_cols = np.arange(panel.N)
    A = np.stack([np.log(panel.meta.kva.to_numpy(np.float32)), panel.meta.n_cust.to_numpy(np.float32),
                  panel.meta.res_share.to_numpy(np.float32), np.log(np.maximum(scale, 1e-3))], 1)
    # fixed evaluation set of never-metered transformers, so tiers are comparable
    unmet = np.sort(rng.choice(all_cols, max(8, int(eval_frac * panel.N)), replace=False))
    pool = np.setdiff1d(all_cols, unmet)
    for frac in fracs:
        metered = np.sort(rng.choice(pool, max(8, int(frac * len(pool))), replace=False))
        kk = int(min(k, max(2, len(metered) // 4)))
        g_attr, _ = make_groups("attribute", panel, tr, scale)
        g_one = np.zeros(panel.N, np.int32)
        sub = panel.subset(metered)
        g_m, _ = make_groups("shape_kmeans", sub, tr, scale[metered], k=kk)
        kn = KNeighborsClassifier(min(5, len(metered))).fit(A[metered], g_m)
        g_shape = np.zeros(panel.N, np.int32)
        g_shape[metered] = g_m
        g_shape[unmet] = kn.predict(A[unmet])
        wp = weekly_profile(panel.loads, panel.cal["how"], scale, tr)
        gprof = np.stack([wp[metered][g_m == g].mean(0) if (g_m == g).any() else wp[metered].mean(0)
                          for g in range(kk)])
        fbg = FeatureBuilder(panel, scale, groups=g_shape, group_profile=gprof, use_group=True, seed=11)
        coldg = fbg.keep_idx("coldstart")
        jobs = [
            ("class_profile", lambda: donor_predict(panel, g_one, tr, te, scale, unmet, metered), "-"),
            ("donor_attr", lambda: donor_predict(panel, g_attr, tr, te, scale, unmet, metered), "attribute"),
            ("donor_shape", lambda: donor_predict(panel, g_shape, tr, te, scale, unmet, metered), "shape_kmeans"),
            ("topdown_attr", lambda: topdown_predict(fb, panel, g_attr, tr, va, te, scale, rounds,
                                                     target_cols=unmet, member_cols=metered,
                                                     share_mode="mean"), "attribute"),
            ("topdown_shape", lambda: topdown_predict(fb, panel, g_shape, tr, va, te, scale, rounds,
                                                      target_cols=unmet, member_cols=metered,
                                                      share_mode="mean"), "shape_kmeans"),
            ("cold_global", lambda: predict_strategy("global", fb, tr, va, te, rounds=rounds,
                                                     cols=metered, target_cols=unmet, keep=cold), "-"),
            ("cold_global_group", lambda: predict_strategy("global", fbg, tr, va, te, rounds=rounds,
                                                           cols=metered, target_cols=unmet, keep=coldg),
             "shape_kmeans"),
            ("local_oracle", lambda: predict_strategy("local", fb, tr, va, te, rounds=rounds,
                                                      cols=unmet, target_cols=unmet), "-"),
        ]
        for name, fn, glab in jobs:
            t0 = time.time()
            try:
                pr, tt, rr, cc = fn()
            except Exception as e:
                print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)
                continue
            m, _ = score(pr, tt, rr, cc, panel, scale, tag=name)
            m.update(dataset=panel.name, fold=fold, experiment="coverage", strategy=name,
                     grouping=glab, k=kk, tier=f"ami{int(frac*100)}", n_unmetered=len(unmet),
                     n_metered=len(metered),
                     secs=round(time.time() - t0, 1))
            log(out, m)
        del fbg


def run_dirty(panel, out, fold="winter", rounds=300, grouping="shape_kmeans", k=8, cases=None):
    tr, va, te, scale = prep(panel, fold)
    hist = np.concatenate([tr, va])
    cases = cases or ([("clean", 0.0)] + [("mcar", r) for r in (0.05, 0.1, 0.2)] +
                      [("outage", 0.3), ("frozen", 0.3), ("spike", 0.01)])
    for kind, lvl in cases:
        dp = panel if kind == "clean" else degrade(panel, hist, kind, lvl, seed=9)
        dp.name = panel.name
        cfrac = getattr(dp, "corrupt_frac", 0.0)
        groups, glab = make_groups(grouping, dp, tr, scale, k=k)
        fb = FeatureBuilder(dp, scale, seed=11)
        # truth always comes from the clean panel
        for s in ("local", "cluster", "global"):
            t0 = time.time()
            pr, _, rr, cc = predict_strategy(s, fb, tr, va, te, groups=groups, rounds=rounds)
            tt = panel.loads[rr, cc] / scale[cc]
            m, _ = score(pr, tt, rr, cc, panel, scale, tag=s)
            m.update(dataset=panel.name, fold=fold, experiment="dirty", strategy=s, grouping=grouping,
                     k=int(len(np.unique(groups))), tier=f"{kind}_{lvl}", corrupt_frac=round(cfrac, 5),
                     secs=round(time.time() - t0, 1))
            log(out, m)
        del fb


def run_quantile(panel, out, fold="summer", rounds=350, grouping="shape_kmeans", k=8,
                 alphas=(0.5, 0.7, 0.8, 0.9, 0.95), ev=0.3):
    """Screening quality as the training objective moves from the mean to an upper
    quantile, on a fleet stressed by EV growth."""
    p = panel if not ev else add_ev(panel, adoption=ev, seed=17)
    p.name = panel.name
    tr, va, te, scale = prep(p, fold)
    groups, glab = make_groups(grouping, p, tr, scale, k=k)
    wp = weekly_profile(p.loads, p.cal["how"], scale, tr)
    gprof = np.stack([wp[groups == g].mean(0) for g in np.unique(groups)])
    fb = FeatureBuilder(p, scale, seed=11)
    fbg = FeatureBuilder(p, scale, groups=groups, group_profile=gprof, use_group=True, seed=11)
    for a in alphas:
        for s in ("global_cluster", "local"):
            t0 = time.time()
            q = None if a == 0.5 else dict(objective="quantile", alpha=a)
            if a == 0.5:
                q = dict(objective="l2")
            if s == "global_cluster":
                pr, tt, rr, cc = predict_strategy("global", fbg, tr, va, te, rounds=rounds, params=q)
            else:
                pr, tt, rr, cc = predict_strategy("local", fb, tr, va, te, rounds=rounds, params=q)
            m, _ = score(pr, tt, rr, cc, p, scale, tag=s)
            m.update(dataset=panel.name, fold=fold, experiment="quantile", strategy=s,
                     grouping=glab, k=int(len(np.unique(groups))), tier=f"alpha{int(a*100)}",
                     alpha=a, ev=ev, secs=round(time.time() - t0, 1))
            log(out, m)


def run_growth(panel, out, fold="summer", rounds=350, grouping="shape_kmeans", k=8,
               levels=(0.0, 0.15, 0.3, 0.5), managed=False,
               strategies=("seasonal_naive", "cluster", "global_cluster",
                           "global_cluster_q90")):
    """Overload screening as electric vehicle adoption rises."""
    for a in levels:
        p = panel if a == 0 else add_ev(panel, adoption=a, managed=managed, seed=17)
        p.name = panel.name
        ls = loading_summary(p)
        print(f"EV {int(a*100)}%: median annual loading {ls['med_loading']:.2f}, "
              f"share over 80% {ls['over80']:.3f}, over 100% {ls['over100']:.3f}", flush=True)
        tr, va, te, scale = prep(p, fold)
        groups, glab = make_groups(grouping, p, tr, scale, k=k)
        wp = weekly_profile(p.loads, p.cal["how"], scale, tr)
        gprof = np.stack([wp[groups == g].mean(0) for g in np.unique(groups)])
        fb = FeatureBuilder(p, scale, seed=11)
        fbg = FeatureBuilder(p, scale, groups=groups, group_profile=gprof, use_group=True, seed=11)
        for s in strategies:
            t0 = time.time()
            try:
                if s.startswith("global_cluster"):
                    q = dict(objective="quantile", alpha=0.9) if s.endswith("q90") else None
                    pr, tt, rr, cc = predict_strategy("global", fbg, tr, va, te, rounds=rounds,
                                                      params=q)
                elif s == "local_q90":
                    pr, tt, rr, cc = predict_strategy("local", fb, tr, va, te, rounds=rounds,
                                                      params=dict(objective="quantile", alpha=0.9))
                elif s == "cluster":
                    pr, tt, rr, cc = predict_strategy("cluster", fb, tr, va, te, groups=groups, rounds=rounds)
                else:
                    pr, tt, rr, cc = predict_strategy(s, fb, tr, va, te, rounds=rounds)
            except Exception as e:
                print(f"FAILED {s}: {type(e).__name__}: {e}", flush=True)
                continue
            m, _ = score(pr, tt, rr, cc, p, scale, tag=s)
            m.update(dataset=panel.name, fold=fold, experiment="growth", strategy=s, grouping=glab,
                     k=int(len(np.unique(groups))), tier=f"ev{int(a*100)}{'m' if managed else ''}",
                     med_loading=round(ls["med_loading"], 3), share_over100=round(ls["over100"], 4),
                     secs=round(time.time() - t0, 1))
            log(out, m)
        del fb, fbg


def run_solar(panel, out, fold="summer", rounds=300, grouping="shape_kmeans", k=8,
              levels=(0.0, 0.2, 0.4), strategies=("cluster", "global_cluster")):
    """Rooftop PV behind the meter, with and without a clear-sky feature in the model.

    The question is whether behind-the-meter generation breaks a transformer-level
    forecast, and whether telling the model where the sun is puts the accuracy back.
    """
    for a in levels:
        for feat in ((True,) if a == 0 else (False, True)):
            p = panel if a == 0 else add_pv(panel, adoption=a, seed=23, expose_feature=feat)
            p.name = panel.name
            ls = loading_summary(p)
            print(f"PV {int(a*100)}% feature={feat}: median annual loading "
                  f"{ls['med_loading']:.2f}, over 100% {ls['over100']:.3f}", flush=True)
            tr, va, te, scale = prep(p, fold)
            groups, glab = make_groups(grouping, p, tr, scale, k=k)
            wp = weekly_profile(p.loads, p.cal["how"], scale, tr)
            gprof = np.stack([wp[groups == g].mean(0) for g in np.unique(groups)])
            fb = FeatureBuilder(p, scale, seed=11)
            fbg = FeatureBuilder(p, scale, groups=groups, group_profile=gprof,
                                 use_group=True, seed=11)
            for s in strategies:
                t0 = time.time()
                if s == "global_cluster":
                    pr, tt, rr, cc = predict_strategy("global", fbg, tr, va, te, rounds=rounds)
                else:
                    pr, tt, rr, cc = predict_strategy("cluster", fb, tr, va, te, groups=groups,
                                                      rounds=rounds)
                m, _ = score(pr, tt, rr, cc, p, scale, tag=s)
                m.update(dataset=panel.name, fold=fold, experiment="solar", strategy=s,
                         grouping=glab, k=int(len(np.unique(groups))),
                         tier=f"pv{int(a*100)}_{'feat' if feat else 'nofeat'}",
                         pv=a, solar_feature=bool(feat), secs=round(time.time() - t0, 1),
                         med_loading=ls["med_loading"], share_over100=ls["over100"])
                log(out, m)
            del fb, fbg


def run_storage(panel, out, fold="winter", rounds=300, grouping="shape_kmeans", k=8,
                levels=(0.0, 0.3, 0.6), strategies=("cluster", "global_cluster")):
    """Home batteries on a fixed daily schedule behind a share of the customers."""
    for a in levels:
        p = panel if a == 0 else add_bess(panel, adoption=a, seed=29)
        p.name = panel.name
        ls = loading_summary(p)
        print(f"BESS {int(a*100)}%: median annual loading {ls['med_loading']:.2f}, "
              f"over 100% {ls['over100']:.3f}", flush=True)
        tr, va, te, scale = prep(p, fold)
        groups, glab = make_groups(grouping, p, tr, scale, k=k)
        wp = weekly_profile(p.loads, p.cal["how"], scale, tr)
        gprof = np.stack([wp[groups == g].mean(0) for g in np.unique(groups)])
        fb = FeatureBuilder(p, scale, seed=11)
        fbg = FeatureBuilder(p, scale, groups=groups, group_profile=gprof, use_group=True, seed=11)
        for s in strategies:
            t0 = time.time()
            if s == "global_cluster":
                pr, tt, rr, cc = predict_strategy("global", fbg, tr, va, te, rounds=rounds)
            else:
                pr, tt, rr, cc = predict_strategy("cluster", fb, tr, va, te, groups=groups,
                                                  rounds=rounds)
            m, _ = score(pr, tt, rr, cc, p, scale, tag=s)
            m.update(dataset=panel.name, fold=fold, experiment="storage", strategy=s,
                     grouping=glab, k=int(len(np.unique(groups))),
                     tier=f"bess{int(a*100)}", bess=a, secs=round(time.time() - t0, 1),
                     med_loading=ls["med_loading"], share_over100=ls["over100"])
            log(out, m)
        del fb, fbg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "grouping", "coverage", "dirty", "growth",
                                    "quantile", "solar", "storage"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=__import__("paths").RESULTS)
    ap.add_argument("--region", default=None)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--folds", default="summer,winter")
    ap.add_argument("--name", default=None)
    ap.add_argument("--strategies", default="")
    ap.add_argument("--ks", default="")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--levels", default="")
    ap.add_argument("--alphas", default="")
    ap.add_argument("--cases", default="")
    ap.add_argument("--fracs", default="")
    a = ap.parse_args()
    p = Panel.from_dir(a.data, region=a.region.split(",") if a.region else None, name=a.name)
    if a.max and p.N > a.max:
        sel = np.random.default_rng(3).choice(p.N, a.max, replace=False)
        p = p.subset(np.sort(sel), name=p.name)
    print(f"{p.name}: {p.N} transformers, {p.T} hours, "
          f"mean load {np.nanmean(p.loads):.2f} kW, kVA {p.meta.kva.median()}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "main":
        run_main(p, a.out, a.folds.split(","), a.rounds,
                 strategies=a.strategies.split(",") if a.strategies else None)
    elif a.cmd == "grouping":
        kw = {}
        if a.ks: kw["ks"] = tuple(int(x) for x in a.ks.split(","))
        if a.kinds: kw["kinds"] = tuple(a.kinds.split(","))
        if a.strategies: kw["strategies"] = tuple(a.strategies.split(","))
        run_grouping(p, a.out, rounds=a.rounds, **kw)
    elif a.cmd == "coverage":
        kw = {}
        if a.fracs: kw["fracs"] = tuple(float(x) for x in a.fracs.split(","))
        run_coverage(p, a.out, rounds=a.rounds, **kw)
    elif a.cmd == "dirty":
        kw = {}
        if a.cases: kw["cases"] = [(c.split(":")[0], float(c.split(":")[1])) for c in a.cases.split(",")]
        run_dirty(p, a.out, rounds=a.rounds, **kw)
    elif a.cmd == "storage":
        kw = {}
        if a.levels: kw["levels"] = tuple(float(x) for x in a.levels.split(","))
        run_storage(p, a.out, rounds=a.rounds, **kw)
    elif a.cmd == "solar":
        kw = {}
        if a.levels: kw["levels"] = tuple(float(x) for x in a.levels.split(","))
        run_solar(p, a.out, rounds=a.rounds, **kw)
    elif a.cmd == "growth":
        kw = {}
        if a.levels: kw["levels"] = tuple(float(x) for x in a.levels.split(","))
        if a.strategies: kw["strategies"] = tuple(a.strategies.split(","))
        run_growth(p, a.out, rounds=a.rounds, **kw)
    elif a.cmd == "quantile":
        kw = {}
        if a.alphas: kw["alphas"] = tuple(float(x) for x in a.alphas.split(","))
        run_quantile(p, a.out, rounds=a.rounds, **kw)
