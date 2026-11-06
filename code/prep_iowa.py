"""Convert the Iowa 240-bus distribution test system into the benchmark panel format.

Source: F. Bu, Y. Yuan, Z. Wang, K. Dehghanpour, A. Kimber, "A time-series distribution
test system based on real utility data," NAPS 2019. Downloads are the four zips linked
from the authors' page ("System Description", "Smart Meter Data", "OpenDSS Model",
"Quasi-static Time-series Analysis").

Usage:
  python prep_iowa.py inspect --zips DIR        # print what is inside, no assumptions
  python prep_iowa.py build   --zips DIR --out DIR

`inspect` is deliberately first: the layout of the smart meter workbook is not
documented, so look at the printout before trusting the build step.
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import zipfile

import numpy as np
import pandas as pd


def unpack(zips, work):
    os.makedirs(work, exist_ok=True)
    for z in glob.glob(os.path.join(zips, "*.zip")):
        with zipfile.ZipFile(z) as f:
            f.extractall(os.path.join(work, os.path.splitext(os.path.basename(z))[0]))
    return work


def inspect(work):
    for root, _, files in os.walk(work):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, work)
            size = os.path.getsize(p) / 1e6
            print(f"{rel}  {size:.1f} MB")
            ext = os.path.splitext(f)[1].lower()
            try:
                if ext in (".csv", ".txt") and size < 400:
                    d = pd.read_csv(p, nrows=4)
                    print("   cols:", list(d.columns)[:12], "...", d.shape[1], "columns")
                    print("   head:", d.head(2).to_string()[:300].replace("\n", " | "))
                elif ext in (".xlsx", ".xls"):
                    xl = pd.ExcelFile(p)
                    print("   sheets:", xl.sheet_names[:12])
                    d = xl.parse(xl.sheet_names[0], nrows=4)
                    print("   cols:", list(d.columns)[:12])
                elif ext == ".dss":
                    head = open(p, errors="ignore").read(300).replace("\n", " | ")
                    print("   ", head[:280])
            except Exception as e:                       # noqa: BLE001
                print("   (unreadable:", type(e).__name__, e, ")")


def ratings(work):
    """Pull transformer nameplate ratings out of any OpenDSS file in the unpacked tree.

    OpenDSS writes `New Transformer.<name> ... kVAs=[a b]` or `kVA=a`, so both forms are
    read and the first winding rating is kept. Returns {name_lower: kva}.
    """
    out = {}
    for p in glob.glob(os.path.join(work, "**", "*.dss"), recursive=True):
        try:
            txt = open(p, errors="ignore").read()
        except Exception:                                # noqa: BLE001
            continue
        for m in re.finditer(r"new\s+transformer\.(\S+)(.*?)(?=\nnew\s|\Z)", txt,
                             re.I | re.S):
            name, body = m.group(1).strip().lower(), m.group(2)
            kv = re.search(r"kvas?\s*=\s*\[?\s*([0-9.]+)", body, re.I)
            if kv:
                out[name] = float(kv.group(1))
    print(f"parsed {len(out)} transformer ratings from the OpenDSS model")
    return out


def build(work, out, year=2017):
    """Expects a wide table of hourly kWh with one column per secondary transformer,
    or a long table with columns like (timestamp, id, kwh). Both shapes are handled."""
    cands = [p for p in glob.glob(os.path.join(work, "**", "*"), recursive=True)
             if os.path.splitext(p)[1].lower() in (".csv", ".xlsx", ".xls")]
    best = None
    for p in sorted(cands, key=os.path.getsize, reverse=True):
        try:
            d = pd.read_csv(p, nrows=200) if p.lower().endswith(".csv") else pd.read_excel(p, nrows=200)
        except Exception:                                # noqa: BLE001
            continue
        if d.shape[1] >= 20 or {"kwh", "kw", "value"} & {c.lower() for c in d.columns}:
            best = p
            break
    if best is None:
        raise SystemExit("no candidate load table found; run `inspect` first")
    print("using", best)
    d = pd.read_csv(best) if best.lower().endswith(".csv") else pd.read_excel(best)
    tcol = next((c for c in d.columns if str(c).lower() in
                 ("time", "timestamp", "datetime", "date", "hour")), None)
    if tcol is not None and d.shape[1] > 5:
        d = d.set_index(tcol)
        idx = pd.date_range(f"{year}-01-01", periods=len(d), freq="h")
        d.index = idx
        loads = d.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    else:
        raise SystemExit("unexpected layout; adapt build() after looking at `inspect`")
    loads = loads.loc[:, loads.notna().mean() > 0.9]
    rate = ratings(work)
    kva = np.array([rate.get(str(c).lower(), np.nan) for c in loads.columns], float)
    if np.isnan(kva).all():
        print("WARNING: no rating matched a series name. Falling back to a sized estimate, "
              "which makes the overload metrics meaningless until the names are reconciled.")
        kva = np.round(loads.max().to_numpy() / 0.85 / 5) * 5
    meta = pd.DataFrame(dict(
        id=loads.columns,
        region="iowa", substation="iowa_sub", feeder="iowa_fdr",
        kva=kva, n_cust=np.nan, res_share=1.0, pv_kw=0.0))
    os.makedirs(out, exist_ok=True)
    loads.to_parquet(f"{out}/loads.parquet")
    meta.to_csv(f"{out}/meta.csv", index=False)
    print(f"wrote {loads.shape[1]} transformer series x {loads.shape[0]} hours to {out}")
    print("NOTE: the benchmark also wants weather.parquet, one hourly temperature column "
          "per region. The Iowa release does not ship weather, so pull the matching year "
          "for the site and write it alongside before running the experiments.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "build"])
    ap.add_argument("--zips", required=True)
    ap.add_argument("--work", default="/tmp/iowa_raw")
    ap.add_argument("--out", default="./data_iowa")
    a = ap.parse_args()
    w = unpack(a.zips, a.work)
    if a.cmd == "inspect":
        inspect(w)
    else:
        build(w, a.out)
