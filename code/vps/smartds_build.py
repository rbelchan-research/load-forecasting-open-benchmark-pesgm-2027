#!/usr/bin/env python3
"""
SMART-DS to transformer-level dataset, with an OpenDSS time-series simulation.

Run through Run-SmartDS.ps1, or directly:
    python smartds_build.py --out C:\\sds --substations 2

What it does, in order, writing output as it goes so a later failure still leaves
usable data behind:

  1  lists the OEDI data lake and downloads the OpenDSS model files (text, small)
  2  parses which customer profiles the model references, downloads only those
  3  compiles the feeder in OpenDSS and walks every customer through its service
     drop to the service transformer that feeds it, reading nameplate ratings
  4  writes the summation dataset: customer kW summed per transformer, hourly
  5  runs a quasi-static time-series power flow with a monitor on every service
     transformer, in monthly blocks, and writes the simulated loading in kVA
  6  writes weather, metadata, the customer map and a run log

Outputs in <out>\\dataset\\:
  transformer_load_sim_hourly.csv.gz    kVA at each service transformer, from the power flow
  transformer_load_sum_hourly.csv.gz    kW summed from the customers behind it
  transformer_meta.csv                  id, substation, feeder, kVA rating, customer count
  customer_map.csv                      customer -> transformer, kW, profile file
  weather_hourly.csv                    temperature for the region
  run_log.txt                           everything printed here
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict, deque

import numpy as np
import pandas as pd

BUCKET = "https://oedi-data-lake.s3.amazonaws.com"
UA = {"User-Agent": "smartds-benchmark/1.2"}
LOG_LINES = []
SSL_CTX = None
SSL_HOW = "not set up yet"


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    LOG_LINES.append(line)


# --------------------------------------------------------------------------- http

def setup_tls(insecure=False):
    """Windows Python ships no CA bundle of its own, so certificate verification fails
    out of the box. Prefer the operating system trust store, which also covers a
    corporate TLS-inspecting proxy, then fall back to certifi."""
    global SSL_CTX, SSL_HOW
    if insecure:
        SSL_CTX = ssl._create_unverified_context()
        SSL_HOW = "verification DISABLED (--insecure)"
        return
    try:
        import truststore
        SSL_CTX = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        SSL_HOW = "operating system trust store (truststore)"
        return
    except Exception:
        pass
    try:
        import certifi
        SSL_CTX = ssl.create_default_context(cafile=certifi.where())
        SSL_HOW = f"certifi bundle at {certifi.where()}"
        return
    except Exception:
        pass
    SSL_CTX = ssl.create_default_context()
    SSL_HOW = "python default context"


def http_get(url, retries=4, timeout=240):
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout, context=SSL_CTX) as r:
                return r.read()
        except ssl.SSLError as e:
            raise RuntimeError(
                f"TLS verification failed for {url}\n  using {SSL_HOW}\n  {e}\n"
                "  Fix: rerun with -Insecure, or install the trust helpers with\n"
                "  C:\\sds\\.venv\\Scripts\\python.exe -m pip install truststore certifi"
            ) from e
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET failed: {url}\n  using {SSL_HOW}\n  {last}")


def s3_list(prefix, delimiter="/", want_size=False):
    keys, prefixes, token = [], [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        xml = http_get(f"{BUCKET}/?{urllib.parse.urlencode(q)}").decode()
        if want_size:
            keys += [(m.group(1), int(m.group(2))) for m in
                     re.finditer(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", xml, re.S)]
        else:
            keys += re.findall(r"<Key>(.*?)</Key>", xml)
        prefixes += re.findall(r"<CommonPrefixes><Prefix>(.*?)</Prefix></CommonPrefixes>", xml)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m or "<IsTruncated>false</IsTruncated>" in xml:
            break
        token = m.group(1)
    return keys, prefixes


def fetch(key, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(http_get(f"{BUCKET}/{urllib.parse.quote(key)}"))
    os.replace(tmp, dest)
    return True


# ---------------------------------------------------------------------- dss parsing

TOK = re.compile(r"""(\w[\w.]*)\s*=\s*(\[[^\]]*\]|\([^)]*\)|"[^"]*"|'[^']*'|\S+)""")


def dss_statements(path, seen=None):
    seen = seen if seen is not None else set()
    path = os.path.abspath(path)
    if path in seen or not os.path.exists(path):
        return
    seen.add(path)
    base = os.path.dirname(path)
    for raw in open(path, errors="ignore").read().splitlines():
        line = raw.split("!")[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(("redirect", "compile")):
            yield from dss_statements(os.path.join(base, line.split(None, 1)[1].strip().strip('"')),
                                      seen)
            continue
        if not low.startswith(("new ", "edit ")):
            continue
        parts = line.split(None, 2)
        otype, _, oname = parts[1].partition(".")
        props = {k.lower(): v.strip("\"'") for k, v in TOK.findall(parts[2] if len(parts) > 2 else "")}
        yield otype.lower(), oname.lower(), props


def repoint_loadshapes(model_root, profile_dir):
    """The published LoadShapes.dss points at the profiles through a relative path that
    only resolves inside the bucket layout. Rewrite each reference to the local copy so
    OpenDSS can compile. The original file is kept alongside as .orig."""
    pdir = os.path.abspath(profile_dir).replace("\\", "/")
    n_files, n_refs, missing = 0, 0, set()

    def swap(m):
        nonlocal n_refs
        base = os.path.basename(m.group(2).strip("\"'"))
        local = f"{pdir}/{base}"
        if not os.path.exists(local):
            missing.add(base)
            return m.group(0)
        n_refs += 1
        return f'{m.group(1)}"{local}"'

    for r, _, fs in os.walk(model_root):
        for f in fs:
            if not f.lower().endswith(".dss"):
                continue
            p = os.path.join(r, f)
            txt = open(p, errors="ignore").read()
            if "file=" not in txt.lower():
                continue
            new = re.sub(r"(file\s*=\s*)([^,\)\s]+)", swap, txt, flags=re.I)
            if new != txt:
                if not os.path.exists(p + ".orig"):
                    open(p + ".orig", "w").write(txt)
                open(p, "w").write(new)
                n_files += 1
    log(f"  repointed {n_refs} profile references in {n_files} model files"
        + (f", {len(missing)} still missing e.g. {sorted(missing)[:3]}" if missing else ""))
    return n_refs, missing


def find_master(root):
    best = None
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower() == "master.dss":
                p = os.path.join(r, f)
                if best is None or len(p) < len(best):
                    best = p
    return best


def all_file_refs(root):
    """Every profile file the model refers to anywhere. OpenDSS reads each LoadShape
    definition at compile time, so the reactive-power shapes must be present too,
    not only the ones a load points at with yearly=."""
    refs = set()
    for r, _, fs in os.walk(root):
        for f in fs:
            if not f.lower().endswith(".dss"):
                continue
            txt = open(os.path.join(r, f), errors="ignore").read()
            for m in re.finditer(r"file\s*=\s*([^,\)\s]+)", txt, re.I):
                refs.add(os.path.basename(m.group(1).strip("\"'")))
    return sorted(refs)


def has_model(prefix):
    keys, _ = s3_list(prefix, delimiter="/")
    if any(k.lower().endswith("master.dss") for k in keys):
        return True
    _, sub = s3_list(prefix)
    for p in sub[:3]:
        k2, _ = s3_list(p, delimiter="/")
        if any(k.lower().endswith("master.dss") for k in k2):
            return True
    return False


def referenced_profiles(root):
    shapes, loads = {}, {}
    master = find_master(root)
    if not master:
        return shapes, loads, None
    for otype, oname, p in dss_statements(master):
        if otype == "loadshape":
            src = p.get("mult") or p.get("pmult") or ""
            m = re.search(r"file\s*=\s*([^,\)\s]+)", src, re.I)
            if m:
                shapes[oname] = m.group(1).strip("\"'")
        elif otype == "load":
            y = (p.get("yearly") or p.get("daily") or "").lower()
            if y:
                loads[oname] = y
    return shapes, loads, master


# ------------------------------------------------------------------- opendss model

def compile_model(master):
    import opendssdirect as dss
    dss.Text.Command("Clear")
    dss.Text.Command(f'Redirect "{master}"')
    return dss


def map_customers(dss):
    """Customer -> service transformer, customer kW, transformer ratings and phases."""
    adj, sec = defaultdict(list), {}
    for full in dss.Circuit.AllElementNames():
        cls = full.split(".")[0].lower()
        dss.Circuit.SetActiveElement(full)
        buses = [b.split(".")[0].lower() for b in dss.CktElement.BusNames()]
        if cls == "transformer":
            for b in buses[1:]:
                sec.setdefault(b, full.split(".", 1)[1])
        elif len(buses) >= 2 and cls in ("line", "reactor", "fuse", "switch", "capacitor"):
            adj[buses[0]].append(buses[1])
            adj[buses[1]].append(buses[0])
    mapping, kw = {}, {}
    i = dss.Loads.First()
    while i:
        nm = dss.Loads.Name()
        kw[nm] = dss.Loads.kW()
        dss.Circuit.SetActiveElement(f"Load.{nm}")
        start = dss.CktElement.BusNames()[0].split(".")[0].lower()
        hit = sec.get(start)
        if hit is None:
            seen, q = {start}, deque([start])
            while q and hit is None and len(seen) < 50000:
                u = q.popleft()
                for v in adj[u]:
                    if v in seen:
                        continue
                    if v in sec:
                        hit = sec[v]
                        break
                    seen.add(v)
                    q.append(v)
        mapping[nm] = hit
        i = dss.Loads.Next()
    xf = {}
    i = dss.Transformers.First()
    while i:
        nm = dss.Transformers.Name()
        dss.Circuit.SetActiveElement(f"Transformer.{nm}")
        xf[nm] = dict(kva=dss.Transformers.kVA(), phases=dss.CktElement.NumPhases(),
                      windings=dss.Transformers.NumWindings())
        i = dss.Transformers.Next()
    return mapping, kw, xf


def run_qsts(dss, xfmr_names, year, step_min, out_csv, block_days=30):
    """Monthly blocks so monitor memory stays bounded. Returns an hourly kVA frame."""
    if step_min >= 60:
        log("    note: a 60 minute step samples the 15 minute profile at the top of each "
            "hour rather than averaging it, so peaks land slightly differently than in the "
            "summation dataset. Use --step 15 for the final run.")
    for t in xfmr_names:
        dss.Text.Command(f"New Monitor.m_{t} element=Transformer.{t} terminal=1 mode=1 ppolar=no")
    dss.Text.Command("set controlmode=static")
    dss.Text.Command(f"set mode=yearly stepsize={step_min}m")
    per_day = int(24 * 60 / step_min)
    idx_all, frames = [], []
    start = pd.Timestamp(f"{year}-01-01")
    done_steps = 0
    total_steps = 365 * per_day
    t_first = None
    while done_steps < total_steps:
        n = min(block_days * per_day, total_steps - done_steps)
        t0 = time.time()
        dss.Text.Command(f"set number={n}")
        dss.Solution.Solve()
        block = {}
        for t in xfmr_names:
            dss.Monitors.Name(f"m_{t}")
            ch = dss.Monitors.NumChannels()
            if ch < 2:
                continue
            p = sum(np.asarray(dss.Monitors.Channel(c)) for c in range(1, ch + 1, 2))
            q = sum(np.asarray(dss.Monitors.Channel(c)) for c in range(2, ch + 1, 2))
            v = np.sqrt(np.asarray(p, float) ** 2 + np.asarray(q, float) ** 2)
            block[t] = v[:n] if len(v) >= n else np.pad(v, (0, n - len(v)), constant_values=np.nan)
        dss.Monitors.ResetAll()
        idx = pd.date_range(start + pd.Timedelta(minutes=step_min * done_steps),
                            periods=n, freq=f"{step_min}min")
        frames.append(pd.DataFrame(block, index=idx).astype(np.float32))
        done_steps += n
        el = time.time() - t0
        if t_first is None:
            t_first = el
            log(f"    first block of {n} steps took {el:.0f}s, "
                f"projecting about {el * total_steps / n / 60:.0f} min for the year")
        log(f"    simulated {done_steps}/{total_steps} steps ({el:.0f}s for this block)")
    df = pd.concat(frames)
    hourly = df.resample("h").mean() if step_min < 60 else df
    write_csv_gz(hourly, out_csv)
    return hourly


# ---------------------------------------------------------------------- output io

def write_csv_gz(df, path, nd=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        df.round(nd).to_csv(f, index_label="timestamp")
    log(f"    wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB, {df.shape[1]} columns)")


def weather_for(region_prefix, out_dir, year, tag=""):
    """SMART-DS solar files carry the meteorological year, temperature included."""
    keys, _ = s3_list(f"{region_prefix}solar_data/", delimiter="", want_size=True)
    csvs = [k for k, s in keys if k.lower().endswith(".csv")]
    if not csvs:
        log("  no solar_data folder found, skipping weather")
        return None
    key = sorted(csvs, key=len)[0]
    dest = os.path.join(out_dir, "raw", os.path.basename(key))
    fetch(key, dest)
    head = open(dest, errors="ignore").readline()
    skip = 0 if "Temperature" in head else 2
    d = pd.read_csv(dest, skiprows=skip)
    tcol = next((c for c in d.columns if "temperature" in str(c).lower()), None)
    if tcol is None:
        log("  temperature column not found in", os.path.basename(key), "->", list(d.columns)[:8])
        return None
    n = len(d)
    freq = "15min" if n > 20000 else "30min" if n > 12000 else "h"
    idx = pd.date_range(f"{year}-01-01", periods=n, freq=freq)
    w = pd.Series(pd.to_numeric(d[tcol], errors="coerce").to_numpy(), index=idx,
                  name="temp_c").resample("h").mean()
    name = f"weather_hourly_{tag}.csv" if tag else "weather_hourly.csv"
    p = os.path.join(out_dir, "dataset", name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    w.to_frame().round(2).to_csv(p, index_label="timestamp")
    log(f"  wrote {p} from {os.path.basename(key)}")
    return w


# ------------------------------------------------------------------------ pipeline

def main(a):
    t_start = time.time()
    setup_tls(getattr(a, "insecure", False))
    log(f"TLS: {SSL_HOW}")
    http_get(f"{BUCKET}/?list-type=2&max-keys=1&prefix=SMART-DS/")
    log("TLS check passed, the bucket is reachable")
    base = f"SMART-DS/v1.0/{a.year}/{a.region}/"
    log(f"=== SMART-DS {a.region} {a.year}, scenario {a.scenario}")
    _, subregions = s3_list(base)
    log("sub-regions:", [s.rstrip('/').split('/')[-1] for s in subregions])

    all_sum, all_sim, all_meta, all_map = [], [], [], []
    for sub in [s.strip() for s in a.subs.split(",") if s.strip()]:
        pref = f"{base}{sub}/scenarios/{a.scenario}/opendss/"
        log(f"\n=== {sub}")
        _, sts = s3_list(pref)
        names = [s.rstrip("/").split("/")[-1] for s in sts]
        if not names:
            log("  nothing under", pref)
            _, alt = s3_list(f"{base}{sub}/scenarios/")
            log("  scenarios present:", [x.rstrip('/').split('/')[-1] for x in alt])
            continue
        skip = {"analysis", "subtransmission", "full_dataset_analysis"}
        cand = [n for n in names if n.lower() not in skip]
        real = []
        for n in cand:
            if has_model(f"{pref}{n}/"):
                real.append(n)
            if len(real) >= a.substations:
                break
        log(f"  {len(names)} folders, {len(real)} carry a model, using {real}")
        if not real:
            log("  none of these folders contain Master.dss")
            continue
        weather_for(f"{base}{sub}/", a.out, a.year, tag=sub)
        names = real

        for st in names[: a.substations]:
            log(f"\n--- substation {st}")
            root = os.path.join(a.out, "raw", sub, st)
            keys, _ = s3_list(f"{pref}{st}/", delimiter="", want_size=True)
            dss_keys = [(k, s) for k, s in keys if k.lower().endswith(".dss")]
            log(f"  {len(dss_keys)} model files ({sum(s for _, s in dss_keys)/1e6:.1f} MB)")
            for k, _ in dss_keys:
                fetch(k, os.path.join(root, k[len(f"{pref}{st}/"):]))

            shapes, loads, master = referenced_profiles(root)
            kw_shapes = sorted({os.path.basename(shapes[v]) for v in set(loads.values())
                                if v in shapes})
            need = all_file_refs(root)
            log(f"  {len(loads)} loads, {len(kw_shapes)} distinct real-power profiles, "
                f"{len(need)} profile files referenced in total")
            if loads and kw_shapes:
                log(f"  profile reuse: {len(loads)/max(len(kw_shapes),1):.1f} customers per "
                    f"distinct shape")
            if not need:
                log("  no file-backed load shapes. Sample loadshape entries:",
                    list(shapes.items())[:3])
                continue
            pkeys, _ = s3_list(f"{base}{sub}/profiles/", delimiter="", want_size=True)
            by_base = {os.path.basename(k): (k, s) for k, s in pkeys}
            hit = [by_base[n] for n in need if n in by_base]
            miss = [n for n in need if n not in by_base]
            mb = sum(s for _, s in hit) / 1e6
            log(f"  profiles to download: {len(hit)} ({mb:.0f} MB)"
                + (f", {len(miss)} not found in the profiles folder, e.g. {miss[:3]}" if miss else ""))
            if a.dry_run:
                log("  dry run, stopping before the profile download")
                continue
            prof_dir = os.path.join(a.out, "raw", "profiles", sub)
            t0 = time.time()
            for n, (k, _) in enumerate(hit, 1):
                fetch(k, os.path.join(prof_dir, os.path.basename(k)))
                if n % 200 == 0:
                    log(f"    {n}/{len(hit)} ({time.time()-t0:.0f}s)")

            repoint_loadshapes(root, prof_dir)
            log("  compiling", master)
            dss = compile_model(master)
            mapping, kw, xf = map_customers(dss)
            n_ok = sum(1 for v in mapping.values() if v)
            log(f"  mapped {n_ok}/{len(mapping)} customers to a transformer")

            groups = defaultdict(list)
            for ld, t in mapping.items():
                if t:
                    groups[t].append(ld)
            svc = {t: m for t, m in groups.items()
                   if xf.get(t, {}).get("kva", 1e9) <= a.max_kva}
            log(f"  {len(svc)} service transformers at or below {a.max_kva} kVA")

            # summation dataset
            cache, cols, meta = {}, {}, []
            for t, members in svc.items():
                tot = None
                for ld in members:
                    fn = os.path.basename(shapes.get(loads.get(ld, ""), ""))
                    p = os.path.join(prof_dir, fn)
                    if not fn or not os.path.exists(p):
                        continue
                    if fn not in cache:
                        cache[fn] = pd.read_csv(p, header=None).iloc[:, 0].to_numpy(np.float32)
                    v = cache[fn] * kw.get(ld, 0.0)
                    tot = v.copy() if tot is None else tot + v[: len(tot)]
                    all_map.append(dict(customer=ld, transformer=f"{st}_{t}", kw=kw.get(ld, 0.0),
                                        profile=fn, substation=st, region=sub))
                if tot is None:
                    continue
                cols[f"{st}_{t}"] = tot
                meta.append(dict(id=f"{st}_{t}", region=sub, substation=st, feeder=st,
                                 kva=xf[t]["kva"], phases=xf[t]["phases"],
                                 n_cust=len(members), res_share=np.nan, pv_kw=0.0))
            if cols:
                n = min(len(v) for v in cols.values())
                freq = "15min" if n >= 35000 else "h"
                df = pd.DataFrame({k: v[:n] for k, v in cols.items()},
                                  index=pd.date_range(f"{a.year}-01-01", periods=n, freq=freq))
                if freq == "15min":
                    df = df.resample("h").mean()
                all_sum.append(df.astype(np.float32))
                all_meta.append(pd.DataFrame(meta))
                log(f"  summation series: {df.shape[1]} transformers x {df.shape[0]} hours")

            if a.simulate and svc:
                names_sim = list(svc)[: a.max_sim]
                log(f"  time-series power flow on {len(names_sim)} transformers, "
                    f"{a.step} minute steps")
                try:
                    sim = run_qsts(dss, names_sim, a.year, a.step,
                                   os.path.join(a.out, "dataset_full",
                                                f"transformer_load_sim_hourly_{st}.csv.gz"))
                    sim.columns = [f"{st}_{c}" for c in sim.columns]
                    all_sim.append(sim)
                except Exception as e:                            # noqa: BLE001
                    log("  QSTS failed:", type(e).__name__, e)

    out_ds = os.path.join(a.out, "dataset")
    out_full = os.path.join(a.out, "dataset_full")
    os.makedirs(out_ds, exist_ok=True)
    os.makedirs(out_full, exist_ok=True)

    meta_all = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    sum_all = pd.concat(all_sum, axis=1) if all_sum else pd.DataFrame()
    sim_all = pd.concat(all_sim, axis=1) if all_sim else pd.DataFrame()

    # everything stays on disk in full, so a wider slice never needs another download
    if len(sum_all.columns):
        write_csv_gz(sum_all, os.path.join(out_full, "transformer_load_sum_hourly_FULL.csv.gz"))
    if len(sim_all.columns):
        write_csv_gz(sim_all, os.path.join(out_full, "transformer_load_sim_hourly_FULL.csv.gz"))
    if len(meta_all):
        meta_all.to_csv(os.path.join(out_full, "transformer_meta_FULL.csv"), index=False)

    # the zip carries a random sample, stratified by substation, to keep the upload small
    keep = list(sum_all.columns)
    if a.max_transformers and len(keep) > a.max_transformers and len(meta_all):
        rng = np.random.default_rng(0)
        by_sub = meta_all[meta_all.id.isin(keep)].groupby("substation").id.apply(list)
        share = a.max_transformers / len(keep)
        keep = []
        for sub_name, ids in by_sub.items():
            n = max(1, int(round(len(ids) * share)))
            keep += list(rng.choice(ids, min(n, len(ids)), replace=False))
        log(f"  sampling {len(keep)} of {len(sum_all.columns)} transformers for the zip "
            f"(full set kept in {out_full})")

    if len(sum_all.columns):
        write_csv_gz(sum_all[[c for c in keep if c in sum_all.columns]],
                     os.path.join(out_ds, "transformer_load_sum_hourly.csv.gz"))
    if len(sim_all.columns):
        cols = [c for c in keep if c in sim_all.columns] or list(sim_all.columns)[:a.max_transformers]
        write_csv_gz(sim_all[cols], os.path.join(out_ds, "transformer_load_sim_hourly.csv.gz"))
    if len(meta_all):
        meta_all[meta_all.id.isin(keep)].to_csv(os.path.join(out_ds, "transformer_meta.csv"),
                                                index=False)
        log(f"  wrote transformer_meta.csv ({meta_all.id.isin(keep).sum()} rows, "
            f"{len(meta_all)} in the full set)")
    if all_map:
        cm = pd.DataFrame(all_map)
        cm.to_csv(os.path.join(out_full, "customer_map_FULL.csv"), index=False)
        cm[cm.transformer.isin(keep)].to_csv(os.path.join(out_ds, "customer_map.csv"), index=False)
        log(f"  wrote customer_map.csv ({int(cm.transformer.isin(keep).sum())} rows)")
    with open(os.path.join(out_ds, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))

    zp = os.path.join(a.out, "smartds_dataset.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out_ds)):
            z.write(os.path.join(out_ds, f), f)
    log(f"\n=== finished in {(time.time()-t_start)/60:.1f} min")
    log(f"SEND THIS FILE: {os.path.abspath(zp)} ({os.path.getsize(zp)/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="GSO")
    ap.add_argument("--subs", default="urban-suburban,rural")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--scenario", default="base_timeseries")
    ap.add_argument("--substations", type=int, default=2)
    ap.add_argument("--out", default="C:\\sds" if os.name == "nt" else "./sds")
    ap.add_argument("--step", type=int, default=15,
                    help="QSTS step in minutes. 15 matches the profile resolution; "
                         "60 is about four times faster but samples rather than averages")
    ap.add_argument("--max-kva", type=float, default=167.0)
    ap.add_argument("--max-sim", type=int, default=1500,
                    help="cap on monitored transformers per substation")
    ap.add_argument("--no-simulate", dest="simulate", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-transformers", type=int, default=1500,
                    help="how many transformer series go into the zip; the full set is "
                         "always written to dataset_full")
    ap.add_argument("--insecure", action="store_true",
                    help="last resort: skip TLS certificate verification")
    args = ap.parse_args()
    if args.dry_run:
        args.simulate = False
    try:
        main(args)
    except Exception as e:                                        # noqa: BLE001
        log("FAILED:", type(e).__name__, e)
        import traceback
        log(traceback.format_exc())
        os.makedirs(os.path.join(args.out, "dataset"), exist_ok=True)
        with open(os.path.join(args.out, "dataset", "run_log.txt"), "w") as f:
            f.write("\n".join(LOG_LINES))
        sys.exit(1)
