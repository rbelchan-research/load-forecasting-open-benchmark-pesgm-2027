"""Core benchmark machinery: panel container, day-ahead feature builder, grouping
strategies, forecasters, metrics.

Convention everywhere: day-ahead forecast issued at 00:00 local time for the next
24 hours, so every feature for target hour t of day D uses data at or before
23:00 of day D-1 (plus dated weather, treated as a forecast).
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import holidays as _holidays
    US_HOL = _holidays.US(years=range(2015, 2027))
except Exception:  # pragma: no cover
    US_HOL = {}

# ----------------------------------------------------------------------------- panel


@dataclass
class Panel:
    loads: np.ndarray            # (T, N) float32, kW at the transformer
    index: pd.DatetimeIndex      # length T, hourly
    ids: np.ndarray              # (N,) transformer ids
    meta: pd.DataFrame           # indexed by id, aligned to ids order
    temp: np.ndarray             # (T, N) float32 deg C (per-transformer weather)
    name: str = "panel"
    cal: dict = field(default_factory=dict)

    def __post_init__(self):
        self.T, self.N = self.loads.shape
        assert len(self.index) == self.T and len(self.ids) == self.N
        self.meta = self.meta.loc[self.ids]
        h = self.index.hour.values
        dow = self.index.dayofweek.values
        doy = self.index.dayofyear.values
        self.cal = dict(
            hour=h.astype(np.float32), dow=dow.astype(np.float32),
            is_weekend=(dow >= 5).astype(np.float32),
            month=self.index.month.values.astype(np.float32),
            doy_sin=np.sin(2 * np.pi * doy / 365.25).astype(np.float32),
            doy_cos=np.cos(2 * np.pi * doy / 365.25).astype(np.float32),
            holiday=np.array([1.0 if d in US_HOL else 0.0 for d in self.index.date], dtype=np.float32),
            how=(dow * 24 + h).astype(np.int32),      # hour of week 0..167
            day=((self.index - self.index[0]).days).values.astype(np.int32),
        )

    def subset(self, cols=None, name=None):
        cols = np.arange(self.N) if cols is None else np.asarray(cols)
        return Panel(self.loads[:, cols].copy(), self.index, self.ids[cols],
                     self.meta.iloc[cols].copy(), self.temp[:, cols].copy(), name or self.name)

    @staticmethod
    def from_dir(path, region=None, name=None, tz=None):
        loads = pd.read_parquet(f"{path}/loads.parquet")
        meta = pd.read_csv(f"{path}/meta.csv").set_index("id")
        weather = pd.read_parquet(f"{path}/weather.parquet")
        if region is not None:
            keep = meta.index[meta.region.isin(np.atleast_1d(region))]
            loads = loads[[c for c in loads.columns if c in set(keep)]]
        meta = meta.loc[loads.columns]
        wcol = meta.region.values if set(meta.region) <= set(weather.columns) else None
        if wcol is not None:
            temp = weather[wcol].to_numpy(np.float32)
        else:
            temp = np.repeat(weather.iloc[:, [0]].to_numpy(np.float32), loads.shape[1], axis=1)
        return Panel(loads.to_numpy(np.float32), loads.index, np.array(loads.columns),
                     meta, temp, name or path.split("/")[-1])


# ------------------------------------------------------------------- feature builder

LAGS = [24, 25, 26, 48, 168, 336]
WARMUP = 336 + 24


def _shift(mat, k):
    out = np.full_like(mat, np.nan)
    if k > 0:
        out[k:] = mat[:-k]
    return out


def day_means(mat, T):
    """Mean and max over each calendar day, broadcast back to hourly rows."""
    nd = T // 24
    d = mat[: nd * 24].reshape(nd, 24, -1)
    return np.nanmean(d, axis=1), np.nanmax(d, axis=1)      # (nd, N)


class FeatureBuilder:
    """Builds the day-ahead design matrix for a panel, once, then slices it."""

    def __init__(self, panel: Panel, scale: np.ndarray, groups: np.ndarray | None = None,
                 group_profile: np.ndarray | None = None, use_group=False, temp_noise=1.0, seed=0):
        self.p, self.scale = panel, scale.astype(np.float32)
        self.use_group = use_group and groups is not None
        T, N = panel.T, panel.N
        rng = np.random.default_rng(seed)
        L = panel.loads / self.scale[None, :]                   # scale-free load
        feats, names = [], []

        for k in LAGS:
            feats.append(_shift(L, k)); names.append(f"lag{k}")
        dmean, dmax = day_means(L, T)                            # (nd, N)
        prev_dmean = np.repeat(np.vstack([np.full((1, N), np.nan), dmean[:-1]]), 24, axis=0)[:T]
        prev_dmax = np.repeat(np.vstack([np.full((1, N), np.nan), dmax[:-1]]), 24, axis=0)[:T]
        feats += [prev_dmean, prev_dmax]; names += ["prev_day_mean", "prev_day_max"]
        sh7 = np.nanmean(np.stack([_shift(L, 24 * k) for k in range(1, 8)]), axis=0)
        sh4 = np.nanmean(np.stack([_shift(L, 168 * k) for k in range(1, 5)]), axis=0)
        wk = np.nanmean(np.stack([np.repeat(np.vstack([np.full((k, N), np.nan), dmean[:-k]]), 24, axis=0)[:T]
                                  for k in range(1, 8)]), axis=0)
        feats += [sh7, sh4, wk]; names += ["same_hour_mean7", "same_hour_dow4", "prev_week_mean"]

        # weather: actual temperature perturbed to emulate a day-ahead forecast
        tp = panel.temp + rng.normal(0, temp_noise, panel.temp.shape).astype(np.float32)
        tmean, tmax = day_means(tp, T)
        tmin = np.min(tp[: (T // 24) * 24].reshape(-1, 24, N), axis=1)
        feats += [tp, np.repeat(tmax, 24, axis=0)[:T], np.repeat(tmin, 24, axis=0)[:T],
                  np.clip(tp - 18.3, 0, None), np.clip(18.3 - tp, 0, None), tp - _shift(tp, 24)]
        names += ["temp", "temp_day_max", "temp_day_min", "cdd", "hdd", "temp_chg"]

        for c in ["hour", "dow", "is_weekend", "month", "doy_sin", "doy_cos", "holiday"]:
            feats.append(np.repeat(panel.cal[c][:, None], N, axis=1)); names.append(c)

        # clear-sky irradiance, present only when a solar scenario exposes it
        if "csky" in panel.cal:
            cs = panel.cal["csky"].astype(np.float32)
            csd = np.repeat(day_means(np.repeat(cs[:, None], N, axis=1), T)[0], 24, axis=0)[:T]
            feats += [np.repeat(cs[:, None], N, axis=1), csd]
            names += ["csky", "csky_day_mean"]

        st = panel.meta
        stat = dict(log_kva=np.log(st.kva.to_numpy(np.float32)),
                    n_cust=st.n_cust.to_numpy(np.float32),
                    res_share=st.res_share.to_numpy(np.float32),
                    pv_per_cust=(st.pv_kw.to_numpy(np.float32) / np.maximum(st.n_cust.to_numpy(np.float32), 1)),
                    log_scale=np.log(np.maximum(self.scale, 1e-3)),
                    region_id=pd.factorize(st.region)[0].astype(np.float32))
        for k, v in stat.items():
            feats.append(np.repeat(v[None, :], T, axis=0)); names.append(k)

        if self.use_group:
            g = np.asarray(groups)
            feats.append(np.repeat(g.astype(np.float32)[None, :], T, axis=0)); names.append("group_id")
            if group_profile is not None:                        # (n_groups, 168) scale-free
                feats.append(group_profile[g][:, panel.cal["how"]].T.astype(np.float32))
                names.append("group_how_profile")
        self.X = np.stack(feats, axis=-1).astype(np.float32)      # (T, N, F)
        self.names = names
        self.cat = [names.index("hour"), names.index("dow"), names.index("month")] + \
                   ([names.index("group_id")] if self.use_group else [])
        self.y = L
        self.valid = np.isfinite(self.X).all(-1) & np.isfinite(self.y)

    def keep_idx(self, mode="all"):
        """Column indices for a feature regime. 'coldstart' drops every feature that
        needs the transformer's own metered history."""
        if mode == "all":
            return np.arange(len(self.names))
        bad = ("lag", "prev_day", "prev_week", "same_hour")
        return np.array([i for i, n in enumerate(self.names) if not n.startswith(bad)])

    def cat_for(self, keep):
        pos = {c: i for i, c in enumerate(keep)}
        return [pos[c] for c in self.cat if c in pos]

    def slice(self, rows, cols=None, row_frac=1.0, seed=0):
        cols = np.arange(self.p.N) if cols is None else np.asarray(cols)
        m = self.valid[np.ix_(rows, cols)]
        X = self.X[np.ix_(rows, cols)][m]
        y = self.y[np.ix_(rows, cols)][m]
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr, cc = rr[m], cc[m]
        if row_frac < 1.0:
            keep = np.random.default_rng(seed).random(len(y)) < row_frac
            X, y, rr, cc = X[keep], y[keep], rr[keep], cc[keep]
        return X, y, rr, cc


# ------------------------------------------------------------------------- grouping

def weekly_profile(loads, how, scale, rows):
    """Scale-free mean profile by hour-of-week, computed on `rows` only. -> (N,168)"""
    L = loads[rows] / scale[None, :]
    h = how[rows]
    out = np.zeros((loads.shape[1], 168), np.float32)
    for k in range(168):
        m = h == k
        out[:, k] = np.nanmean(L[m], axis=0) if m.any() else np.nan
    return np.nan_to_num(out, nan=1.0)


def shape_features(panel, rows, scale):
    """Engineered features for attribute/feature-based clustering (train rows only)."""
    L = panel.loads[rows] / scale[None, :]
    h = panel.cal["hour"][rows].astype(int)
    dow = panel.cal["dow"][rows].astype(int)
    t = panel.temp[rows]
    prof = np.stack([L[h == k].mean(0) for k in range(24)])            # (24,N)
    peak_hour = prof.argmax(0).astype(np.float32)
    lf = L.mean(0) / np.maximum(L.max(0), 1e-6)
    we = L[dow >= 5].mean(0) / np.maximum(L[dow < 5].mean(0), 1e-6)
    cv = L.std(0) / np.maximum(L.mean(0), 1e-6)
    cdd, hdd = np.clip(t - 18.3, 0, None), np.clip(18.3 - t, 0, None)
    def beta(x):
        xc = x - x.mean(0, keepdims=True)
        yc = L - L.mean(0, keepdims=True)
        return (xc * yc).sum(0) / np.maximum((xc ** 2).sum(0), 1e-6)
    F = np.stack([peak_hour / 23.0, lf, we, cv, beta(cdd), beta(hdd),
                  np.log(panel.meta.kva.to_numpy(np.float32)) / 5.0,
                  panel.meta.n_cust.to_numpy(np.float32) / 12.0,
                  panel.meta.res_share.to_numpy(np.float32)], axis=1)
    return np.nan_to_num(F).astype(np.float32)


def make_groups(kind, panel, rows, scale, k=8, seed=0):
    """Returns (groups (N,), label). Uses only `rows` (training period)."""
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    N = panel.N
    if kind == "none":
        return np.zeros(N, np.int32), "no grouping"
    if kind == "shape_kmeans":                     # normalised hour-of-week profile
        W = weekly_profile(panel.loads, panel.cal["how"], scale, rows)
        Z = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-6)
        g = KMeans(k, n_init=8, random_state=seed).fit_predict(Z)
    elif kind == "daily_kmeans":                   # normalised mean daily profile
        W = weekly_profile(panel.loads, panel.cal["how"], scale, rows).reshape(N, 7, 24).mean(1)
        Z = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-6)
        g = KMeans(k, n_init=8, random_state=seed).fit_predict(Z)
    elif kind == "feature_kmeans":                 # engineered features
        Z = StandardScaler().fit_transform(shape_features(panel, rows, scale))
        g = KMeans(k, n_init=8, random_state=seed).fit_predict(Z)
    elif kind == "ward_corr":                      # hierarchical, correlation distance
        W = weekly_profile(panel.loads, panel.cal["how"], scale, rows)
        Z = (W - W.mean(1, keepdims=True)) / np.maximum(W.std(1, keepdims=True), 1e-6)
        g = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    elif kind == "dtw_kmedoids":
        W = weekly_profile(panel.loads, panel.cal["how"], scale, rows).reshape(N, 7, 24).mean(1)
        Z = W / np.maximum(W.max(1, keepdims=True), 1e-6)
        g = _dtw_kmedoids(Z, k, seed)
    elif kind == "attribute":                      # no AMI needed
        st = panel.meta
        key = (pd.cut(st.n_cust, [0, 2, 4, 7, 11, 999], labels=False).astype(str) + "_" +
               (st.res_share > 0.5).astype(int).astype(str) + "_" +
               pd.cut(st.kva, [0, 25, 50, 100, 1e6], labels=False).astype(str) + "_" + st.region.astype(str))
        g = pd.factorize(key)[0]
    elif kind == "feeder":                         # topological
        g = pd.factorize(panel.meta.feeder)[0]
    else:
        raise ValueError(kind)
    return g.astype(np.int32), f"{kind}(k={len(np.unique(g))})"


def _dtw_kmedoids(Z, k, seed=0, iters=12, w=3):
    """k-medoids with a Sakoe-Chiba banded DTW distance on 24-point profiles."""
    n, m = Z.shape
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        D[i, i + 1:] = [_dtw(Z[i], Z[j], w) for j in range(i + 1, n)]
    D += D.T
    rng = np.random.default_rng(seed)
    med = rng.choice(n, k, replace=False)
    for _ in range(iters):
        lab = D[:, med].argmin(1)
        new = med.copy()
        for c in range(k):
            idx = np.where(lab == c)[0]
            if len(idx):
                new[c] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
        if np.array_equal(new, med):
            break
        med = new
    return D[:, med].argmin(1)


def _dtw(a, b, w=3):
    n = len(a)
    inf = np.inf
    prev = np.full(n + 1, inf); prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(n + 1, inf)
        for j in range(max(1, i - w), min(n, i + w) + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(np.sqrt(prev[n]))


# ----------------------------------------------------------------------- forecasters

LGB_PARAMS = dict(objective="l2", learning_rate=0.09, num_leaves=63, min_data_in_leaf=40,
                  feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
                  verbose=-1, num_threads=2, force_col_wise=True)


def fit_lgb(X, y, cat, rounds=400, params=None, Xv=None, yv=None):
    import lightgbm as lgb
    p = dict(LGB_PARAMS); p.update(params or {})
    ds = lgb.Dataset(X, label=y, categorical_feature=cat, free_raw_data=True)
    cbs, vs = [], None
    if Xv is not None and len(yv) > 500:
        vs = [lgb.Dataset(Xv, label=yv, categorical_feature=cat, reference=ds)]
        cbs = [lgb.early_stopping(30, verbose=False)]
    return lgb.train(p, ds, num_boost_round=rounds, valid_sets=vs, callbacks=cbs)


def predict_strategy(strategy, fb: FeatureBuilder, tr_rows, va_rows, te_rows, groups=None,
                     rounds=400, row_frac=1.0, seed=0, cols=None, target_cols=None, keep=None,
                     params=None):
    """Returns (pred, truth, rows, cols) in scale-free units for the test block."""
    cols = np.arange(fb.p.N) if cols is None else np.asarray(cols)
    tgt = cols if target_cols is None else np.asarray(target_cols)
    if strategy == "seasonal_naive":
        pr = fb.X[..., fb.names.index("lag168")]
    elif strategy == "persistence":
        pr = fb.X[..., fb.names.index("lag24")]
    if strategy in ("seasonal_naive", "persistence"):
        m = fb.valid[np.ix_(te_rows, tgt)]
        rr, cc = np.meshgrid(te_rows, tgt, indexing="ij")
        return pr[np.ix_(te_rows, tgt)][m], fb.y[np.ix_(te_rows, tgt)][m], rr[m], cc[m]

    keep = fb.keep_idx("all") if keep is None else np.asarray(keep)
    cat = fb.cat_for(keep)
    Xte, yte, rte, cte = fb.slice(te_rows, tgt)
    Xte = Xte[:, keep]
    if strategy == "global":
        Xtr, ytr, _, _ = fb.slice(tr_rows, cols, row_frac, seed)
        Xva, yva, _, _ = fb.slice(va_rows, cols, min(1.0, row_frac), seed + 1)
        mdl = fit_lgb(Xtr[:, keep], ytr, cat, rounds, params=params, Xv=Xva[:, keep], yv=yva)
        return mdl.predict(Xte), yte, rte, cte
    if strategy == "ridge_local":
        from sklearn.linear_model import Ridge
        out = np.zeros(len(yte), np.float32)
        for c in np.unique(cte):
            Xtr, ytr, _, _ = fb.slice(tr_rows, [c])
            if len(ytr) < 200:
                out[cte == c] = np.nan; continue
            r = Ridge(alpha=1.0).fit(Xtr[:, keep], ytr)
            out[cte == c] = r.predict(Xte[cte == c])
        return out, yte, rte, cte
    if strategy == "local":
        out = np.zeros(len(yte), np.float32)
        for c in np.unique(cte):
            Xtr, ytr, _, _ = fb.slice(tr_rows, [c])
            if len(ytr) < 400:
                out[cte == c] = np.nan; continue
            p_local = dict(num_leaves=31, min_data_in_leaf=20)
            p_local.update(params or {})
            mdl = fit_lgb(Xtr[:, keep], ytr, cat, min(rounds, 140), params=p_local)
            out[cte == c] = mdl.predict(Xte[cte == c])
        return out, yte, rte, cte
    if strategy == "cluster":
        assert groups is not None
        out = np.full(len(yte), np.nan, np.float32)
        for g in np.unique(groups[cols]):
            gc = cols[groups[cols] == g]
            gt = np.intersect1d(gc, tgt)
            if len(gt) == 0:
                continue
            Xtr, ytr, _, _ = fb.slice(tr_rows, gc, row_frac, seed)
            if len(ytr) < 400:
                continue
            Xva, yva, _, _ = fb.slice(va_rows, gc)
            mdl = fit_lgb(Xtr[:, keep], ytr, cat, rounds, params=params, Xv=Xva[:, keep], yv=yva)
            sel = np.isin(cte, gt)
            out[sel] = mdl.predict(Xte[sel])
        return out, yte, rte, cte
    raise ValueError(strategy)


def topdown_predict(fb: FeatureBuilder, panel: Panel, groups, tr_rows, va_rows, te_rows,
                    scale, rounds=400, target_cols=None, member_cols=None, share_mode="ldf",
                    oracle=False):
    """Forecast each group aggregate, then allocate to members.

    member_cols  transformers whose measurements form the aggregate (metered ones)
    target_cols  transformers that receive an allocated forecast
    share_mode   'ldf'  hour-of-week share from the target's own history (needs AMI)
                 'mean' target mean load / group aggregate mean (needs billing only)
                 'rolling' share from the last four same weekday-and-hour occurrences,
                           which is how a load distribution factor is kept current in
                           practice rather than averaged over a whole training window
    oracle       allocate the TRUE group total instead of its forecast, which isolates
                 the error the allocation step contributes on its own
    """
    tgt = np.arange(panel.N) if target_cols is None else np.asarray(target_cols)
    mem_pool = np.arange(panel.N) if member_cols is None else np.asarray(member_cols)
    in_pool = np.zeros(panel.N, bool); in_pool[mem_pool] = True
    gids = np.array([g for g in np.unique(groups) if in_pool[groups == g].any()])
    agg = np.stack([panel.loads[:, (groups == g) & in_pool].sum(1) for g in gids], axis=1).astype(np.float32)
    msk = [(groups == g) & in_pool for g in gids]
    ameta = pd.DataFrame(dict(
        id=[f"g{g}" for g in gids],
        region=[panel.meta.region.values[m][0] for m in msk],
        substation="agg", feeder="agg",
        kva=[panel.meta.kva.values[m].sum() for m in msk],
        n_cust=[panel.meta.n_cust.values[m].sum() for m in msk],
        res_share=[panel.meta.res_share.values[m].mean() for m in msk],
        pv_kw=[panel.meta.pv_kw.values[m].sum() for m in msk])).set_index("id")
    atemp = np.stack([panel.temp[:, m].mean(1) for m in msk], axis=1).astype(np.float32)
    ap = Panel(agg, panel.index, ameta.index.values, ameta, atemp, "agg")
    ascale = agg[tr_rows].mean(0)
    afb = FeatureBuilder(ap, ascale, seed=3)
    Xtr, ytr, _, _ = afb.slice(tr_rows)
    Xva, yva, _, _ = afb.slice(va_rows)
    mdl = fit_lgb(Xtr, ytr, afb.cat, rounds, Xv=Xva, yv=yva)
    Xte, yte_a, rte_a, cte_a = afb.slice(te_rows)
    pa = mdl.predict(Xte)                                     # scale-free group forecast
    # group forecast in kW on a (rows x groups) grid
    G = np.full((panel.T, len(gids)), np.nan, np.float32)
    G[rte_a, cte_a] = (pa * ascale[cte_a]).astype(np.float32)
    # allocation weights
    how = panel.cal["how"]
    share = np.zeros((panel.N, 168), np.float32)
    gi_of = {g: i for i, g in enumerate(gids)}
    for gi, g in enumerate(gids):
        recv = np.where(groups == g)[0]
        if share_mode == "ldf":
            for k in range(168):
                rows = tr_rows[how[tr_rows] == k]
                share[recv, k] = panel.loads[np.ix_(rows, recv)].mean(0) / max(agg[rows, gi].mean(), 1e-6)
        else:   # billing-only: flat share = own mean / group aggregate mean
            share[recv, :] = (scale[recv] / max(agg[tr_rows, gi].mean(), 1e-6))[:, None]
    if share_mode == "rolling":
        share = None            # handled per timestamp below
    unassigned = np.array([c for c in tgt if groups[c] not in gi_of])
    tgt = np.array([c for c in tgt if groups[c] in gi_of])
    if len(unassigned):
        print(f"  topdown: {len(unassigned)} targets have no metered group members, dropped", flush=True)
    gidx = np.array([gi_of.get(g, 0) for g in groups])
    if oracle:
        G = np.full((panel.T, len(gids)), np.nan, np.float32)
        G[te_rows[:, None], np.arange(len(gids))[None, :]] = agg[te_rows]
    if share_mode == "rolling":
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = panel.loads / agg[:, gidx]          # member share of its group, per hour
        lags = [168, 336, 504, 672]
        stack = []
        for L in lags:
            prev = np.full_like(ratio, np.nan)
            prev[L:] = ratio[:-L]
            stack.append(prev[te_rows][:, tgt])
        roll = np.nanmean(np.stack(stack), axis=0)
        pred_kw = G[np.ix_(te_rows, gidx[tgt])] * roll
    else:
        pred_kw = G[np.ix_(te_rows, gidx[tgt])] * share[tgt][:, how[te_rows]].T
    truth = panel.loads[np.ix_(te_rows, tgt)]
    m = np.isfinite(pred_kw) & np.isfinite(truth)
    rr, cc = np.meshgrid(te_rows, tgt, indexing="ij")
    sc = scale[tgt]
    return (pred_kw[m] / np.tile(sc, (len(te_rows), 1))[m], truth[m] / np.tile(sc, (len(te_rows), 1))[m],
            rr[m], cc[m])


def donor_predict(panel: Panel, groups, tr_rows, te_rows, scale, target_cols, source_cols):
    """Unmetered transformers: hour-of-week mean shape of the metered members of the
    same group, scaled to the target's known mean load (from monthly billing)."""
    how = panel.cal["how"]
    L = panel.loads / scale[None, :]
    prof = np.zeros((int(groups.max()) + 1, 168), np.float32)
    cnt = np.zeros(int(groups.max()) + 1)
    for g in np.unique(groups[source_cols]):
        mem = source_cols[groups[source_cols] == g]
        for k in range(168):
            rows = tr_rows[how[tr_rows] == k]
            prof[g, k] = L[np.ix_(rows, mem)].mean()
        cnt[g] = len(mem)
    glob = np.zeros(168, np.float32)
    for k in range(168):
        rows = tr_rows[how[tr_rows] == k]
        glob[k] = L[np.ix_(rows, source_cols)].mean()
    tgt = np.asarray(target_cols)
    pr = np.stack([prof[groups[c]] if cnt[groups[c]] >= 3 else glob for c in tgt])   # (n_t,168)
    pred = pr[:, how[te_rows]].T
    truth = L[np.ix_(te_rows, tgt)]
    m = np.isfinite(pred) & np.isfinite(truth)
    rr, cc = np.meshgrid(te_rows, tgt, indexing="ij")
    return pred[m], truth[m], rr[m], cc[m]


# --------------------------------------------------------------------------- metrics

def score(pred, truth, rows, cols, panel: Panel, scale, tag=""):
    """Per-transformer metrics plus fleet aggregates, in kW units."""
    ok = np.isfinite(pred) & np.isfinite(truth)
    pred, truth, rows, cols = pred[ok], truth[ok], rows[ok], cols[ok]
    p_kw, t_kw = pred * scale[cols], truth * scale[cols]
    df = pd.DataFrame(dict(row=rows, col=cols, p=p_kw, t=t_kw,
                           day=panel.cal["day"][rows], kva=panel.meta.kva.to_numpy()[cols]))
    per = df.groupby("col").apply(lambda g: pd.Series({
        "mae": np.abs(g.p - g.t).mean(),
        "rmse": np.sqrt(((g.p - g.t) ** 2).mean()),
        "mean": g.t.mean(),
        "peak": g.t.max(),
    }), include_groups=False)
    per["nmae"] = per.mae / per["mean"]
    per["nrmse"] = per.rmse / per["mean"]
    # daily peak magnitude / timing error
    d = df.sort_values(["col", "row"])
    gp = d.groupby(["col", "day"])
    pk = gp.apply(lambda g: pd.Series({
        "pe": (g.p.max() - g.t.max()) / max(g.t.max(), 1e-6),
        "ape": abs(g.p.max() - g.t.max()) / max(g.t.max(), 1e-6),
        "hr": abs(int(np.argmax(g.p.values)) - int(np.argmax(g.t.values))),
        "ld_t": g.t.max() / g.kva.iloc[0],
        "ld_p": g.p.max() / g.kva.iloc[0]}), include_groups=False).reset_index()
    per = per.join(pk.groupby("col")[["ape", "hr"]].mean().rename(columns={"ape": "peak_ape", "hr": "peak_hour_err"}))
    per["peak_bias"] = pk.groupby("col").pe.mean()
    out = dict(tag=tag, n_xfmr=per.shape[0],
               nmae_med=per.nmae.median(), nmae_iqr=per.nmae.quantile(.75) - per.nmae.quantile(.25),
               nrmse_med=per.nrmse.median(), nrmse_mean=per.nrmse.mean(),
               peak_ape_med=per.peak_ape.median(), peak_bias_med=per.peak_bias.median(),
               peak_hour_err_med=per.peak_hour_err.median(),
               fleet_nmae=np.abs(df.p - df.t).sum() / df.t.sum())
    for thr in (0.8, 1.0):
        y = pk.ld_t > thr
        yh = pk.ld_p > thr
        tp, fp, fn = (y & yh).sum(), (~y & yh).sum(), (y & ~yh).sum()
        out[f"ovl{int(thr*100)}_n"] = int(y.sum())
        out[f"ovl{int(thr*100)}_prec"] = tp / max(tp + fp, 1)
        out[f"ovl{int(thr*100)}_rec"] = tp / max(tp + fn, 1)
        out[f"ovl{int(thr*100)}_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return out, per


# ------------------------------------------------------------------ data degradation

def degrade(panel: Panel, rows, kind, level, seed=0):
    """Corrupt the *history* used for training and feature building, then repair it the
    way a utility would, and return a new panel.

    `level` is the share of history cells the fault touches, so the four kinds are
    directly comparable. Block faults are drawn as events until that share is reached.
    The returned panel carries `corrupt_frac`, the share actually touched.
    """
    rng = np.random.default_rng(seed)
    L = panel.loads.copy()
    sub = L[rows]
    cells = sub.size
    touched = np.zeros(sub.shape, bool)
    if kind == "mcar":
        touched = rng.random(sub.shape) < level
        sub[touched] = np.nan
    elif kind == "outage":                       # block gaps of 24-72 h
        budget = int(level * cells)
        while touched.sum() < budget:
            c = rng.integers(panel.N)
            d = int(rng.integers(24, 73))
            s = int(rng.integers(0, max(1, len(rows) - d)))
            sub[s:s + d, c] = np.nan
            touched[s:s + d, c] = True
    elif kind == "frozen":                       # stuck meter: one value repeats a week
        budget = int(level * cells)
        while touched.sum() < budget:
            c = rng.integers(panel.N)
            s = int(rng.integers(0, max(1, len(rows) - 168)))
            sub[s:s + 168, c] = sub[s, c]
            touched[s:s + 168, c] = True
    elif kind == "spike":
        touched = rng.random(sub.shape) < level
        sub[touched] *= rng.uniform(3, 8, int(touched.sum()))
    L[rows] = sub
    # utility-style repair: same hour previous week, then short linear interpolation
    df = pd.DataFrame(L, index=panel.index)
    df = df.fillna(df.shift(168)).fillna(df.shift(-168))
    df = df.interpolate(limit=6, limit_direction="both")
    out = Panel(df.to_numpy(np.float32), panel.index, panel.ids, panel.meta, panel.temp,
                panel.name + f"|{kind}{level}")
    out.corrupt_frac = float(touched.sum()) / cells
    return out
