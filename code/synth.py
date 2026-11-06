"""Synthetic service-transformer population, used only to validate the pipeline
before the real SMART-DS / Iowa data arrives.

Writes the same three files the real loaders will write:
  loads.parquet  wide hourly kW, index=timestamp, columns=transformer ids
  meta.csv       id, region, substation, feeder, kva, n_cust, res_share, pv_kw
  weather.csv    timestamp, temp_c  (one column per region)
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
STD_KVA = np.array([15, 25, 37.5, 50, 75, 100, 167])


def weather_year(year=2018, n=8760):
    t = np.arange(n)
    doy = t / 24.0
    # Charlotte-like annual + diurnal temperature signal
    annual = 15.5 - 10.5 * np.cos(2 * np.pi * (doy - 15) / 365.0)
    diurnal = 4.5 * np.sin(2 * np.pi * (t % 24 - 9) / 24.0)
    synoptic = np.convolve(RNG.normal(0, 3.2, n + 96), np.ones(72) / 72.0, mode="same")[:n]
    return annual + diurnal + synoptic + RNG.normal(0, 0.7, n)


def customer_profile(kind, temp, tstamp, rng):
    """One customer's hourly kW for the year."""
    n = len(temp)
    hour = tstamp.hour.values
    dow = tstamp.dayofweek.values
    base = {"evening": 0.55, "morning": 0.5, "flat": 0.7, "ev": 0.8, "com": 2.4}[kind]
    shape = np.ones(n)
    if kind == "evening":
        shape = 0.55 + 0.9 * np.exp(-0.5 * ((hour - 19) / 2.6) ** 2) + 0.35 * np.exp(-0.5 * ((hour - 7) / 1.8) ** 2)
    elif kind == "morning":
        shape = 0.6 + 0.8 * np.exp(-0.5 * ((hour - 7) / 1.9) ** 2) + 0.45 * np.exp(-0.5 * ((hour - 20) / 3.0) ** 2)
    elif kind == "flat":
        shape = 0.85 + 0.25 * np.sin(2 * np.pi * (hour - 14) / 24.0)
    elif kind == "ev":
        shape = 0.5 + 0.7 * np.exp(-0.5 * ((hour - 19) / 2.4) ** 2)
        charge = ((hour >= 23) | (hour <= 4)).astype(float)
        active = np.repeat(rng.random(n // 24 + 1) < 0.62, 24)[:n]
        shape = shape + 2.6 * charge * active
    elif kind == "com":
        open_h = ((hour >= 8) & (hour <= 19)).astype(float)
        shape = 0.3 + 1.4 * open_h * (1 - 0.45 * (dow >= 5))
    # weather response: cooling above 20 C, heating below 12 C
    cdd = np.clip(temp - 20.0, 0, None)
    hdd = np.clip(12.0 - temp, 0, None)
    ac = rng.uniform(0.03, 0.16)
    ht = rng.uniform(0.02, 0.13)
    wk = 1.0 - 0.06 * (dow >= 5) if kind == "com" else 1.0 + 0.07 * (dow >= 5)
    load = base * shape * wk * (1 + ac * cdd + ht * hdd)
    load *= 1 + rng.normal(0, 0.13, n)  # customer-level volatility
    return np.clip(load, 0.02, None)


def build(n_xfmr=320, year=2018, pv_share=0.0, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    regions = {"urban-suburban": int(n_xfmr * 0.65), "rural": n_xfmr - int(n_xfmr * 0.65)}
    temps = {r: weather_year(year) for r in regions}
    cols, meta, pv_profile = {}, [], None
    # one clear-sky-ish PV shape, scaled per site
    hour = idx.hour.values
    doy = idx.dayofyear.values
    elev = np.clip(np.sin(np.pi * (hour - 6.2) / 12.0), 0, None) * (0.82 + 0.18 * np.sin(2 * np.pi * (doy - 80) / 365))
    clouds = np.clip(np.convolve(rng.normal(0.78, 0.25, 8760 + 12), np.ones(6) / 6, mode="same")[:8760], 0.12, 1.0)
    pv_profile = elev * clouds

    k = 0
    for region, count in regions.items():
        temp = temps[region]
        n_sub = max(1, count // 60)
        for i in range(count):
            k += 1
            tid = f"{region[:3]}_x{k:04d}"
            if region == "urban-suburban":
                n_cust = int(rng.integers(4, 13))
            else:
                n_cust = int(rng.integers(1, 6))
            is_com = rng.random() < (0.14 if region == "urban-suburban" else 0.08)
            kinds = ["com"] * n_cust if is_com else list(
                rng.choice(["evening", "morning", "flat", "ev"], size=n_cust, p=[0.44, 0.28, 0.16, 0.12])
            )
            series = np.zeros(8760)
            for kd in kinds:
                series += customer_profile(kd, temp, idx, rng)
            pv_kw = 0.0
            if pv_share > 0:
                n_pv = int(rng.binomial(n_cust, pv_share))
                if n_pv:
                    pv_kw = float(np.round(n_pv * rng.uniform(4.0, 8.5), 1))
                    series = series - pv_kw * pv_profile
            peak = np.max(np.abs(series))
            kva = float(STD_KVA[np.searchsorted(STD_KVA, peak / rng.uniform(0.80, 1.45)).clip(0, len(STD_KVA) - 1)])
            cols[tid] = series.astype(np.float32)
            meta.append(dict(id=tid, region=region, substation=f"{region[:3]}_sub{i % n_sub + 1}",
                             feeder=f"{region[:3]}_fdr{i % (n_sub * 3) + 1}", kva=kva, n_cust=n_cust,
                             res_share=0.0 if is_com else 1.0, pv_kw=pv_kw))
    loads = pd.DataFrame(cols, index=idx)
    meta = pd.DataFrame(meta).set_index("id")
    weather = pd.DataFrame({r: temps[r] for r in regions}, index=idx)
    return loads, meta, weather


if __name__ == "__main__":
    import sys, os
    import paths
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.DATA, "synth")
    os.makedirs(out, exist_ok=True)
    loads, meta, weather = build(n_xfmr=int(os.environ.get("NX", 320)),
                                 pv_share=float(os.environ.get("PVSHARE", 0.0)))
    loads.to_parquet(f"{out}/loads.parquet")
    meta.to_csv(f"{out}/meta.csv")
    weather.to_parquet(f"{out}/weather.parquet")
    print(f"{loads.shape[1]} transformers, {loads.shape[0]} hours -> {out}")
    print(loads.iloc[:, :4].describe().round(2).to_string())
    print(meta.head().to_string())
