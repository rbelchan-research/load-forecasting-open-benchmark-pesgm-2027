"""Fetch and preprocess NREL SMART-DS into the benchmark panel format.

Runs anywhere with plain HTTPS access to the OEDI data lake (no AWS credentials
needed). Steps are separate so they can be run one at a time:

  python smartds.py list      --prefix SMART-DS/v1.0/2018/GSO/
  python smartds.py tree      --region GSO --sub urban-suburban          # substations/feeders
  python smartds.py dss       --region GSO --sub urban-suburban --subs SUB1,SUB2
  python smartds.py map       --root DIR                                  # parse models
  python smartds.py profiles  --root DIR --region GSO --sub urban-suburban
  python smartds.py panel     --root DIR --out DIR/panel

Output of `panel`: loads.parquet (hourly kW per transformer), meta.csv, weather.parquet.
"""
from __future__ import annotations
import argparse, io, os, re, sys, time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

BUCKET = "https://oedi-data-lake.s3.amazonaws.com"
UA = {"User-Agent": "smartds-fetch/1.0"}


# --------------------------------------------------------------------------- S3 access

def _get(url, retries=4, timeout=120):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def s3_list(prefix, delimiter="/", max_pages=200):
    """Returns (keys, common_prefixes)."""
    keys, prefixes, token = [], [], None
    for _ in range(max_pages):
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        xml = _get(f"{BUCKET}/?{urllib.parse.urlencode(q)}").decode()
        keys += re.findall(r"<Key>(.*?)</Key>", xml)
        prefixes += re.findall(r"<CommonPrefixes><Prefix>(.*?)</Prefix></CommonPrefixes>", xml)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m or "<IsTruncated>false</IsTruncated>" in xml:
            break
        token = m.group(1)
    return keys, prefixes


def s3_download(key, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    data = _get(f"{BUCKET}/{urllib.parse.quote(key)}")
    with open(dest, "wb") as f:
        f.write(data)
    return dest


# ------------------------------------------------------------------- OpenDSS parsing

TOK = re.compile(r"""(\w[\w.]*)\s*=\s*(\[[^\]]*\]|\([^)]*\)|"[^"]*"|'[^']*'|\S+)""")


def dss_lines(path):
    """Yields (verb, obj_type, obj_name, props) for New/Edit statements, following Redirect."""
    base = os.path.dirname(path)
    try:
        txt = open(path, "r", errors="ignore").read()
    except FileNotFoundError:
        return
    for raw in txt.splitlines():
        line = raw.split("!")[0].strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(("redirect", "compile")):
            tgt = line.split(None, 1)[1].strip().strip('"')
            yield from dss_lines(os.path.join(base, tgt))
            continue
        if not low.startswith(("new ", "edit ")):
            continue
        parts = line.split(None, 2)
        verb = parts[0].lower()
        full = parts[1]
        rest = parts[2] if len(parts) > 2 else ""
        if "." in full:
            otype, oname = full.split(".", 1)
        else:
            otype, oname = full, ""
        props = {k.lower(): v.strip("\"'") for k, v in TOK.findall(rest)}
        # some files write "New Load.x bus1=..." with the name carrying an = free token
        yield verb, otype.lower(), oname.lower(), props


def arr(v):
    if v is None:
        return []
    return [x for x in re.split(r"[\s,]+", v.strip("[]() ")) if x]


def bus_name(b):
    return b.split(".")[0].lower()


def parse_model(root):
    """Parse every Master.dss under root. Returns transformers, loads, lines dataframes."""
    xf, ld, ln, ls = [], [], [], {}
    masters = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs
               if f.lower() == "master.dss"]
    # deepest masters are feeder level; parse all, dedupe by object name
    seen = set()
    for m in masters:
        feeder = os.path.basename(os.path.dirname(m))
        sub = os.path.basename(os.path.dirname(os.path.dirname(m)))
        for verb, otype, oname, p in dss_lines(m):
            key = (otype, oname)
            if verb == "new" and key in seen:
                continue
            seen.add(key)
            if otype == "transformer":
                buses = arr(p.get("buses"))
                kvas = [float(x) for x in arr(p.get("kvas") or p.get("kva") or "0")]
                kvs = [float(x) for x in arr(p.get("kvs") or "0")]
                xf.append(dict(name=oname, feeder=feeder, substation=sub,
                               buses=[bus_name(b) for b in buses],
                               kva=max(kvas) if kvas else np.nan,
                               kv_hi=max(kvs) if kvs else np.nan,
                               kv_lo=min(kvs) if kvs else np.nan,
                               phases=int(float(p.get("phases", 3)))))
            elif otype == "load":
                ld.append(dict(name=oname, feeder=feeder, substation=sub,
                               bus=bus_name(p.get("bus1", "")), kw=float(p.get("kw", "nan")),
                               kvar=float(p.get("kvar", "nan") or "nan"),
                               kv=float(p.get("kv", "nan") or "nan"),
                               yearly=(p.get("yearly") or p.get("daily") or "").lower(),
                               cls=p.get("class", ""), phases=int(float(p.get("phases", 1)))))
            elif otype == "line":
                ln.append(dict(name=oname, a=bus_name(p.get("bus1", "")), b=bus_name(p.get("bus2", "")),
                               switch=str(p.get("switch", "false")).lower().startswith("t")))
            elif otype == "loadshape":
                f = p.get("mult") or p.get("pmult") or ""
                mm = re.search(r"file\s*=\s*([^,\)\s]+)", f, re.I)
                ls[oname] = dict(file=(mm.group(1).strip("\"'") if mm else ""),
                                 npts=int(float(p.get("npts", 0) or 0)),
                                 interval=float(p.get("interval", 0) or 0),
                                 useactual=str(p.get("useactual", "no")).lower().startswith("y"))
    return (pd.DataFrame(xf), pd.DataFrame(ld), pd.DataFrame(ln),
            pd.DataFrame(ls).T.rename_axis("name").reset_index() if ls else pd.DataFrame())


def map_loads_to_transformers(xf, ld, ln):
    """Walk from each load bus through lines only, until a transformer secondary bus."""
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for a, b in zip(ln.a, ln.b):
        if a and b:
            adj[a].append(b); adj[b].append(a)
    sec = {}          # bus -> transformer name  (secondary = last winding bus)
    for r in xf.itertuples():
        if len(r.buses) >= 2:
            for b in r.buses[1:]:
                sec.setdefault(b, r.name)
    out = {}
    for r in ld.itertuples():
        start = r.bus
        if start in sec:
            out[r.name] = sec[start]; continue
        seen, q, hit = {start}, deque([start]), None
        while q and len(seen) < 4000:
            u = q.popleft()
            for v in adj[u]:
                if v in seen:
                    continue
                if v in sec:
                    hit = sec[v]; q.clear(); break
                seen.add(v); q.append(v)
            if hit:
                break
        out[r.name] = hit
    ld = ld.copy()
    ld["xfmr"] = ld.name.map(out)
    return ld


# ------------------------------------------------------------------------- profiles

def fetch_profiles(needed, region, sub, root, year=2018, scenario="base_timeseries"):
    """Download referenced per-unit profile CSVs into root/profiles/."""
    pref = f"SMART-DS/v1.0/{year}/{region}/{sub}/profiles/"
    keys, _ = s3_list(pref, delimiter="")
    by_base = {os.path.basename(k): k for k in keys}
    got = {}
    for i, nm in enumerate(sorted(needed)):
        cands = [b for b in by_base if b.lower().startswith(nm.lower())] or \
                [b for b in by_base if nm.lower() in b.lower()]
        if not cands:
            continue
        b = sorted(cands, key=len)[0]
        dest = os.path.join(root, "profiles", b)
        s3_download(by_base[b], dest)
        got[nm] = dest
        if i % 50 == 0:
            print(f"  {i}/{len(needed)} profiles", flush=True)
    return got


def read_profile(path):
    v = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(np.float32)
    return v


# ---------------------------------------------------------------------------- panel

def build_panel(root, out, year=2018, min_cust=1, hourly=True):
    xf, ld, ln, ls = parse_model(os.path.join(root, "opendss"))
    print(f"parsed {len(xf)} transformers, {len(ld)} loads, {len(ln)} lines, {len(ls)} loadshapes")
    ld = map_loads_to_transformers(xf, ld, ln)
    print(f"mapped {ld.xfmr.notna().sum()}/{len(ld)} loads to a transformer")
    shp = {r["name"]: r for _, r in ls.iterrows()} if len(ls) else {}
    need = sorted({os.path.basename(shp[y]["file"]).split(".")[0] for y in ld.yearly.dropna().unique()
                   if y in shp and shp[y]["file"]})
    prof_files = {os.path.basename(p).split(".")[0]: os.path.join(root, "profiles", p)
                  for p in os.listdir(os.path.join(root, "profiles"))}
    cache, missing = {}, 0
    groups = ld.dropna(subset=["xfmr"]).groupby("xfmr")
    cols, meta = {}, []
    for tname, g in groups:
        if len(g) < min_cust:
            continue
        tot = None
        for r in g.itertuples():
            f = shp.get(r.yearly, {}).get("file", "")
            base = os.path.basename(f).split(".")[0] if f else ""
            if base not in prof_files:
                missing += 1
                continue
            if base not in cache:
                cache[base] = read_profile(prof_files[base])
            v = cache[base] * (r.kw if not shp[r.yearly].get("useactual") else 1.0)
            tot = v if tot is None else tot + v
        if tot is None:
            continue
        row = xf.set_index("name").loc[tname]
        cols[tname] = tot
        meta.append(dict(id=tname, region=root.rstrip("/").split("/")[-1],
                         substation=row.substation, feeder=row.feeder, kva=float(row.kva),
                         n_cust=len(g), res_share=float((g.cls.astype(str).str.lower().str.startswith("res")).mean()),
                         pv_kw=0.0))
    print(f"built {len(cols)} transformer series, {missing} loads without a profile file")
    idx15 = pd.date_range(f"{year}-01-01", periods=len(next(iter(cols.values()))), freq="15min")
    df = pd.DataFrame(cols, index=idx15)
    if hourly:
        df = df.resample("h").mean()
    os.makedirs(out, exist_ok=True)
    df.astype(np.float32).to_parquet(f"{out}/loads.parquet")
    pd.DataFrame(meta).to_csv(f"{out}/meta.csv", index=False)
    print(f"-> {out}/loads.parquet  {df.shape}")
    return df, pd.DataFrame(meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "tree", "dss", "map", "profiles", "panel"])
    ap.add_argument("--prefix", default="SMART-DS/v1.0/2018/GSO/")
    ap.add_argument("--region", default="GSO")
    ap.add_argument("--sub", default="urban-suburban")
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--scenario", default="base_timeseries")
    ap.add_argument("--subs", default="")
    ap.add_argument("--root", default="./smartds_gso")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.cmd == "list":
        keys, pref = s3_list(a.prefix)
        print("PREFIXES:"); [print(" ", p) for p in pref[:60]]
        print("KEYS:"); [print(" ", k) for k in keys[:40]]
    elif a.cmd == "tree":
        base = f"SMART-DS/v1.0/{a.year}/{a.region}/{a.sub}/scenarios/{a.scenario}/opendss/"
        _, subs = s3_list(base)
        print(f"{len(subs)} substations under {base}")
        for s in subs[:80]:
            _, fd = s3_list(s)
            print(" ", s.split("/")[-2], f"({len(fd)} feeders)")
    elif a.cmd == "dss":
        base = f"SMART-DS/v1.0/{a.year}/{a.region}/{a.sub}/scenarios/{a.scenario}/opendss/"
        want = [s.strip() for s in a.subs.split(",") if s.strip()]
        keys, subs = s3_list(base, delimiter="")
        n = 0
        for k in keys:
            if not k.lower().endswith(".dss"):
                continue
            if want and not any(f"/{w}/" in k for w in want):
                continue
            s3_download(k, os.path.join(a.root, "opendss", k.split("/opendss/")[1]))
            n += 1
        print(f"downloaded {n} dss files into {a.root}/opendss")
    elif a.cmd == "map":
        xf, ld, ln, ls = parse_model(os.path.join(a.root, "opendss"))
        ld = map_loads_to_transformers(xf, ld, ln)
        print(xf.head().to_string())
        print(ld.head().to_string())
        print("loadshapes:", ls.head().to_string() if len(ls) else "none")
        print("mapped:", ld.xfmr.notna().mean().round(3))
        g = ld.dropna(subset=["xfmr"]).groupby("xfmr").size()
        print("customers per transformer:", g.describe().round(2).to_dict())
    elif a.cmd == "profiles":
        xf, ld, ln, ls = parse_model(os.path.join(a.root, "opendss"))
        shp = {r["name"]: r for _, r in ls.iterrows()}
        need = sorted({os.path.basename(shp[y]["file"]).split(".")[0]
                       for y in ld.yearly.dropna().unique() if y in shp and shp[y]["file"]})
        print(f"{len(need)} distinct profiles referenced")
        fetch_profiles(need, a.region, a.sub, a.root, a.year, a.scenario)
    elif a.cmd == "panel":
        build_panel(a.root, a.out or f"{a.root}/panel", a.year)
