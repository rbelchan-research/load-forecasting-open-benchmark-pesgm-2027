"""Transformer loading from an OpenDSS time-series power flow.

Two ways to get the loading of a service transformer are used in this benchmark:

  summation   sum the customer kW behind the transformer, apply the modeled power
              factor, and add the transformer's own no-load and load losses. Cheap
              enough to run for a whole fleet for a year.
  QSTS        a quasi-static time series power flow on the published OpenDSS model,
              with a monitor on each sampled transformer, read in kVA.

The paper reports the summation loading for the full fleet and uses a QSTS run on a
random sample to state how closely the two agree, so the cheap number is validated
rather than assumed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def open_circuit(master):
    import opendssdirect as dss
    dss.Text.Command("Clear")
    dss.Text.Command(f'Redirect "{master}"')
    return dss


def transformer_table(dss):
    """name, kVA, %loadloss, %noloadloss for every transformer in the model."""
    rows = []
    i = dss.Transformers.First()
    while i:
        rows.append(dict(name=dss.Transformers.Name(), kva=dss.Transformers.kVA(),
                         loadloss=dss.Transformers.LossesPercent()
                         if hasattr(dss.Transformers, "LossesPercent") else np.nan,
                         noloadloss=dss.Transformers.NoLoadLossPercent()
                         if hasattr(dss.Transformers, "NoLoadLossPercent") else np.nan,
                         windings=dss.Transformers.NumWindings()))
        i = dss.Transformers.Next()
    return pd.DataFrame(rows)


def attach_shapes(dss, shapes: dict, npts, interval_h=1.0):
    """shapes: load name -> 1D per-unit array. Registers one LoadShape per load."""
    for name, arr in shapes.items():
        sname = f"ls_{name}"
        mult = ",".join(f"{v:.5f}" for v in np.asarray(arr, dtype=float))
        dss.Text.Command(f"New Loadshape.{sname} npts={npts} interval={interval_h} "
                         f"mult=({mult})")
        dss.Text.Command(f"Edit Load.{name} yearly={sname}")


def qsts_monitor(dss, xfmr_names, npts, step_h=1.0, mode="yearly"):
    """Attach a power monitor to each named transformer and run the time series."""
    for t in xfmr_names:
        dss.Text.Command(f"New Monitor.m_{t} element=Transformer.{t} terminal=1 mode=1 ppolar=no")
    dss.Text.Command(f"set mode={mode} stepsize={step_h}h number={npts}")
    dss.Text.Command("set controlmode=static")
    dss.Solution.Solve()
    out = {}
    for t in xfmr_names:
        dss.Monitors.Name(f"m_{t}")
        ch = dss.Monitors.NumChannels()
        # mode=1 ppolar=no writes P,Q for each conductor of the terminal
        p = sum(np.array(dss.Monitors.Channel(c)) for c in range(1, ch + 1, 2))
        q = sum(np.array(dss.Monitors.Channel(c)) for c in range(2, ch + 1, 2))
        out[t] = np.sqrt(np.asarray(p) ** 2 + np.asarray(q) ** 2)   # kVA at the primary
    return pd.DataFrame(out)


def summation_loading(cust_kw: pd.DataFrame, mapping: dict, pf=0.95,
                      loadloss_pct=1.0, noloadloss_pct=0.3, kva=None):
    """Loading estimate without a power flow.

    cust_kw   hourly customer kW, one column per customer
    mapping   customer -> transformer
    """
    g = {}
    for cust, xf in mapping.items():
        if cust in cust_kw.columns:
            g.setdefault(xf, []).append(cust)
    out = {}
    for xf, members in g.items():
        kw = cust_kw[members].sum(axis=1)
        kva_load = kw / pf
        rating = (kva or {}).get(xf, np.nan)
        losses = (noloadloss_pct / 100.0) * (rating if np.isfinite(rating) else 0.0) + \
                 (loadloss_pct / 100.0) * kva_load
        out[xf] = kva_load + losses
    return pd.DataFrame(out)


def agreement(qsts_df, summ_df):
    """Per-transformer agreement between the two loading estimates."""
    cols = [c for c in qsts_df.columns if c in summ_df.columns]
    rows = []
    for c in cols:
        a, b = qsts_df[c].to_numpy(), summ_df[c].to_numpy()[:len(qsts_df)]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
        if ok.sum() < 10:
            continue
        rows.append(dict(xfmr=c, mape=float(np.mean(np.abs(b[ok] - a[ok]) / a[ok])),
                         bias=float(np.mean((b[ok] - a[ok]) / a[ok])),
                         peak_err=float((b[ok].max() - a[ok].max()) / a[ok].max())))
    return pd.DataFrame(rows)


def map_loads_via_dss(dss, max_hops=6000):
    """Customer -> service transformer, walked on the live circuit rather than parsed
    from files: edges are every non-transformer two-terminal element, and the walk
    stops at any bus that is a transformer secondary."""
    from collections import defaultdict, deque
    adj = defaultdict(list)
    sec = {}
    names = dss.Circuit.AllElementNames()
    for full in names:
        cls = full.split(".")[0].lower()
        dss.Circuit.SetActiveElement(full)
        buses = [b.split(".")[0].lower() for b in dss.CktElement.BusNames()]
        if cls == "transformer":
            for b in buses[1:]:
                sec.setdefault(b, full.split(".", 1)[1])
        elif len(buses) >= 2 and cls in ("line", "reactor", "fuse", "switch", "capacitor"):
            adj[buses[0]].append(buses[1])
            adj[buses[1]].append(buses[0])
    out = {}
    i = dss.Loads.First()
    while i:
        nm = dss.Loads.Name()
        dss.Circuit.SetActiveElement(f"Load.{nm}")
        start = dss.CktElement.BusNames()[0].split(".")[0].lower()
        hit = sec.get(start)
        if hit is None:
            seen, q = {start}, deque([start])
            while q and len(seen) < max_hops and hit is None:
                u = q.popleft()
                for v in adj[u]:
                    if v in seen:
                        continue
                    if v in sec:
                        hit = sec[v]
                        break
                    seen.add(v)
                    q.append(v)
        out[nm] = hit
        i = dss.Loads.Next()
    return out
