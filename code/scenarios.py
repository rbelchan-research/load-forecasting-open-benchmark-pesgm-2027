"""Electric vehicle growth overlay.

SMART-DS ships PV and battery penetration scenarios but no EV charging time series,
so EV load is added with an explicit, documented model. Every parameter is stated in
the paper so the scenario can be reproduced or replaced.

Per adopting customer:
  charger    Level 2, `power_kw` at the meter, efficiency `eff`
  need       daily energy drawn from a lognormal with median `kwh_day`, charging on
             `charge_prob` of days
  start      uncontrolled: arrival ~ N(mu_arr, sd_arr), charging begins on arrival
             managed:      charging begins at `managed_start` local time
  energy is delivered at constant power until the need is met, spilling into the
  following hours
"""
from __future__ import annotations
import numpy as np
from core import Panel


def ev_profile(n_days, hours, rng, power_kw=7.2, eff=0.92, kwh_day=10.5, charge_prob=0.7,
               mu_arr=18.0, sd_arr=1.6, managed=False, managed_start=23.0):
    out = np.zeros(n_days * 24, np.float32)
    draw = rng.lognormal(np.log(kwh_day), 0.35, n_days)
    charging = rng.random(n_days) < charge_prob
    starts = (np.full(n_days, managed_start) if managed
              else np.clip(rng.normal(mu_arr, sd_arr, n_days), 12.0, 23.5))
    for d in range(n_days):
        if not charging[d]:
            continue
        need = draw[d] / eff
        h = int(starts[d])
        frac = 1.0 - (starts[d] - h)
        while need > 1e-3 and h < n_days * 24 - d * 24:
            idx = d * 24 + h
            if idx >= len(out):
                break
            take = min(need, power_kw * frac)
            out[idx] += take
            need -= take
            h += 1
            frac = 1.0
    return out[:len(out)]


def add_ev(panel: Panel, adoption=0.3, managed=False, seed=0, **kw):
    """Adds EV charging load to a fraction of the residential customers behind each
    transformer. Returns a new panel and records the vehicle count in meta."""
    rng = np.random.default_rng(seed)
    n_days = panel.T // 24
    L = panel.loads.copy()
    meta = panel.meta.copy()
    ev_n = np.zeros(panel.N, int)
    for j in range(panel.N):
        n_res = int(round(meta.n_cust.iloc[j] * meta.res_share.iloc[j]))
        k = int(rng.binomial(max(n_res, 0), adoption))
        ev_n[j] = k
        for _ in range(k):
            L[:n_days * 24, j] += ev_profile(n_days, panel.T, rng, managed=managed, **kw)
    meta["ev_count"] = ev_n
    p = Panel(L, panel.index, panel.ids, meta, panel.temp,
              panel.name + f"|ev{int(adoption*100)}{'m' if managed else ''}")
    return p


def loading_summary(panel: Panel):
    """Share of transformers whose annual peak exceeds 80 and 100 percent of rating."""
    pk = np.nanmax(panel.loads, axis=0) / panel.meta.kva.to_numpy(np.float32)
    return dict(n=panel.N, med_loading=float(np.nanmedian(pk)),
                over80=float(np.nanmean(pk > 0.8)), over100=float(np.nanmean(pk > 1.0)))


# --------------------------------------------------------------------------- solar

def clear_sky(index, lat=36.07, lon=-79.79, tz_offset=-5):
    """Normalized clear-sky plane-of-array irradiance, 0 at night and 1 at the best
    hour of the year. Plain solar geometry, no atmosphere model beyond a cosine term,
    which is enough to give the PV overlay a realistic daily and seasonal shape."""
    doy = index.dayofyear.to_numpy(np.float64)
    hour = index.hour.to_numpy(np.float64) + 0.5
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    b = 2 * np.pi * (doy - 81) / 364.0
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_time = hour + (4 * (lon - 15 * tz_offset) + eot) / 60.0
    omega = np.deg2rad(15.0 * (solar_time - 12.0))
    phi = np.deg2rad(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)
    g = np.clip(cosz, 0, None) ** 1.15
    return (g / max(g.max(), 1e-6)).astype(np.float32)


def pv_series(index, rng, kw_dc=6.0, derate=0.85, rho=0.6):
    """One rooftop system's hourly AC output. A day-to-day clearness index with
    first-order persistence stands in for weather, so output is partly predictable
    from recent days and partly not."""
    cs = clear_sky(index)
    nd = len(index) // 24
    z = np.zeros(nd)
    z[0] = rng.normal()
    for d in range(1, nd):
        z[d] = rho * z[d - 1] + np.sqrt(1 - rho ** 2) * rng.normal()
    kt = np.clip(0.72 + 0.16 * z, 0.18, 1.0)
    kt = np.repeat(kt, 24)[:len(index)]
    return (kw_dc * derate * cs * kt).astype(np.float32)


def add_pv(panel: Panel, adoption=0.3, kw_dc=6.0, seed=0, expose_feature=True):
    """Puts rooftop PV behind a share of the residential customers on each transformer.

    The transformer series is apparent power, which is a magnitude, so a feeder
    exporting power loads the transformer exactly as much as one importing it. The
    overlay therefore subtracts generation from the net and keeps the absolute value.
    """
    rng = np.random.default_rng(seed)
    L = panel.loads.copy()
    meta = panel.meta.copy()
    n_pv = np.zeros(panel.N, int)
    kwtot = np.zeros(panel.N, np.float32)
    for j in range(panel.N):
        n_res = int(round(meta.n_cust.iloc[j] * meta.res_share.iloc[j]))
        k = int(rng.binomial(max(n_res, 0), adoption))
        n_pv[j] = k
        for _ in range(k):
            L[:, j] -= pv_series(panel.index, rng, kw_dc=kw_dc)
            kwtot[j] += kw_dc
    L = np.abs(L)
    meta["pv_count"] = n_pv
    meta["pv_kw"] = kwtot
    p = Panel(L, panel.index, panel.ids, meta, panel.temp,
              panel.name + f"|pv{int(adoption*100)}")
    if expose_feature:
        p.cal["csky"] = clear_sky(panel.index)
    return p


# -------------------------------------------------------------------------- storage

def bess_dispatch(load_kw, rng, cap_kwh=13.5, p_kw=5.0, eff=0.92, mode="tou",
                  charge_h=(1, 5), discharge_h=(17, 21), jitter=0.15):
    """A simple home battery following a fixed daily schedule, which is what most
    residential systems actually do. Charges overnight, discharges in the evening,
    and stops when the pack is empty or the house load is already low."""
    T = len(load_kw)
    hours = np.arange(T) % 24
    soc = 0.5 * cap_kwh
    out = np.zeros(T, np.float32)
    pk = p_kw * (1.0 + rng.normal(0, jitter))
    for t in range(T):
        h = hours[t]
        if charge_h[0] <= h < charge_h[1]:
            take = min(pk, (cap_kwh - soc) / eff)
            soc += take * eff
            out[t] = take                      # extra load on the transformer
        elif discharge_h[0] <= h < discharge_h[1]:
            give = min(pk, soc, max(load_kw[t], 0.0))
            soc -= give
            out[t] = -give                     # load the transformer no longer sees
    return out


def add_bess(panel: Panel, adoption=0.3, seed=0, **kw):
    """Puts a home battery behind a share of the residential customers on each
    transformer and returns the new panel."""
    rng = np.random.default_rng(seed)
    L = panel.loads.copy()
    meta = panel.meta.copy()
    n_b = np.zeros(panel.N, int)
    for j in range(panel.N):
        n_res = int(round(meta.n_cust.iloc[j] * meta.res_share.iloc[j]))
        k = int(rng.binomial(max(n_res, 0), adoption))
        n_b[j] = k
        if k == 0:
            continue
        per_home = panel.loads[:, j] / max(meta.n_cust.iloc[j], 1)
        for _ in range(k):
            L[:, j] += bess_dispatch(per_home, rng, **kw)
    L = np.abs(L)
    meta["bess_count"] = n_b
    return Panel(L, panel.index, panel.ids, meta, panel.temp,
                 panel.name + f"|bess{int(adoption*100)}")
