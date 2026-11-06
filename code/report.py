"""Turn results.jsonl into the paper's tables and a numbers sheet for the prose.

  python report.py results/results.jsonl > report.md
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

LAB = {
    "seasonal_naive": "Seasonal naive", "persistence": "Day persistence",
    "ridge_local": "Local linear", "local": "Local trees", "cluster": "Cluster trees",
    "global": "Global trees", "global_cluster": "Global + cluster",
    "topdown_cluster": "Top-down, cluster", "topdown_feeder": "Top-down, feeder",
    "lstm_global": "Global LSTM", "class_profile": "Class profile",
    "donor_attr": "Group profile, attributes", "donor_shape": "Group profile, shape",
    "topdown_attr": "Top-down, attributes", "topdown_shape": "Top-down, shape",
    "cold_global": "Cold-start global", "cold_global_group": "Cold-start global + group",
    "local_oracle": "Local trees, metered",
}
ORDER = ["seasonal_naive", "persistence", "ridge_local", "local", "cluster", "global",
         "global_cluster", "lstm_global", "topdown_cluster", "topdown_feeder"]


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100*x:.{d}f}"


def table2(df):
    d = df[df.experiment == "main"]
    rows = [["Strategy", "nMAE", "nRMSE", "Peak APE", "Peak bias", "F1 > 100%"]]
    for s in ORDER:
        g = d[d.strategy == s]
        if not len(g):
            continue
        rows.append([LAB.get(s, s), pct(g.nmae_med.mean()), pct(g.nrmse_med.mean()),
                     pct(g.peak_ape_med.mean()), pct(g.peak_bias_med.mean()),
                     f"{g.ovl100_f1.mean():.2f}"])
    return rows


def table_coverage(df):
    d = df[df.experiment == "coverage"]
    if not len(d):
        return []
    tiers = sorted(d.tier.unique(), key=lambda t: int(t.replace("ami", "")))
    rows = [["Route for unmetered transformers"] + [t.replace("ami", "") + "% metered" for t in tiers]]
    for s in ["class_profile", "donor_attr", "donor_shape", "topdown_attr", "topdown_shape",
              "cold_global", "cold_global_group", "local_oracle"]:
        g = d[d.strategy == s]
        if not len(g):
            continue
        rows.append([LAB.get(s, s)] + [pct(g[g.tier == t].nmae_med.mean()) for t in tiers])
    return rows


def findings(df):
    out = []
    d = df[df.experiment == "main"]
    if len(d):
        m = d.groupby("strategy").nmae_med.mean().sort_values()
        best, second = m.index[0], m.index[1]
        out.append(f"- Best strategy: **{LAB.get(best, best)}** at {pct(m.iloc[0])}% nMAE, "
                   f"ahead of {LAB.get(second, second)} at {pct(m.iloc[1])}%. "
                   f"Seasonal naive is {pct(m.get('seasonal_naive', np.nan))}%, so the best "
                   f"strategy improves on it by "
                   f"{100*(1 - m.iloc[0]/m.get('seasonal_naive', np.nan)):.0f}%.")
        for a, b in [("cluster", "local"), ("global_cluster", "global"), ("cluster", "global"),
                     ("topdown_cluster", "local"), ("lstm_global", "global")]:
            if a in m and b in m:
                out.append(f"- {LAB[a]} vs {LAB[b]}: {pct(m[a])}% vs {pct(m[b])}% nMAE "
                           f"({100*(m[a]/m[b]-1):+.1f}% relative).")
        pb = d.groupby("strategy").peak_bias_med.mean()
        out.append(f"- Peak bias runs from {pct(pb.min())}% to {pct(pb.max())}%; "
                   f"{(pb < 0).sum()} of {len(pb)} strategies under-forecast the daily peak.")
        f1 = d.groupby("strategy").ovl100_f1.mean().sort_values(ascending=False)
        out.append(f"- Overload F1 at 100% of rating: best {f1.index[0]} {f1.iloc[0]:.2f}, "
                   f"worst {f1.index[-1]} {f1.iloc[-1]:.2f}.")
    g = df[df.experiment == "grouping"]
    if len(g):
        cw = g[g.strategy == "cluster"]
        sw = cw[cw.grouping.isin(["shape_kmeans", "daily_kmeans", "feature_kmeans", "ward_corr",
                                  "dtw_kmedoids"])]
        out.append(f"- Across grouping methods and k, cluster-wise nMAE spans "
                   f"{pct(sw.nmae_med.min())}% to {pct(sw.nmae_med.max())}%, a spread of "
                   f"{100*(sw.nmae_med.max()/sw.nmae_med.min()-1):.0f}% relative.")
        by_k = sw.groupby("k").nmae_med.mean()
        out.append(f"- Best k on average: {by_k.idxmin()} ({pct(by_k.min())}%), "
                   f"worst {by_k.idxmax()} ({pct(by_k.max())}%).")
        by_m = sw.groupby("grouping").nmae_med.min()
        out.append("- Best per method: " + ", ".join(f"{k} {pct(v)}%" for k, v in by_m.items()))
        for nm in ["attribute", "feeder"]:
            v = cw[cw.grouping == nm].nmae_med
            if len(v):
                out.append(f"- {nm} grouping (no interval data needed): {pct(v.mean())}% nMAE.")
    c = df[df.experiment == "coverage"]
    if len(c):
        orc = c[c.strategy == "local_oracle"].nmae_med.mean()
        cls = c[c.strategy == "class_profile"].nmae_med.mean()
        best = c[c.strategy != "local_oracle"].groupby("strategy").nmae_med.mean().sort_values()
        out.append(f"- Unmetered transformers: class average profile {pct(cls)}% nMAE, best "
                   f"transfer route {best.index[0]} at {pct(best.iloc[0])}%, own meter history "
                   f"{pct(orc)}%. Grouping recovers "
                   f"{100*(cls-best.iloc[0])/max(cls-orc,1e-9):.0f}% of the gap.")
    dd = df[df.experiment == "dirty"]
    if len(dd):
        base = dd[dd.tier == "clean_0.0"].set_index("strategy").nmae_med
        dd = dd.copy()
        dd["rel"] = 100 * (dd.nmae_med / dd.strategy.map(base) - 1)
        for t, gg in dd[dd.tier != "clean_0.0"].groupby("tier"):
            out.append(f"- {t}: " + ", ".join(f"{LAB.get(r.strategy, r.strategy)} "
                                              f"{r.rel:+.1f}%" for r in gg.itertuples()))
    gr = df[df.experiment == "growth"]
    if len(gr):
        gr = gr.copy()
        gr["ev"] = gr.tier.str.extract(r"ev(\d+)")[0].astype(int)
        for ev, gg in gr.groupby("ev"):
            sh = gg.share_over100.iloc[0] if "share_over100" in gg else np.nan
            best = gg.sort_values("ovl100_f1", ascending=False).iloc[0]
            out.append(f"- EV {ev}%: {100*sh:.0f}% of transformers exceed rating at some point; "
                       f"best overload F1 {best.ovl100_f1:.2f} ({LAB.get(best.strategy, best.strategy)}), "
                       f"peak bias {pct(gg.peak_bias_med.mean())}%.")
    return out


if __name__ == "__main__":
    df = pd.read_json(sys.argv[1], lines=True)
    print("# Results sheet\n")
    print(f"rows: {len(df)}, experiments: {sorted(df.experiment.unique())}\n")
    print("## Table II (main comparison)\n")
    for r in table2(df):
        print("| " + " | ".join(r) + " |")
    print("\n## Coverage table\n")
    for r in table_coverage(df):
        print("| " + " | ".join(r) + " |")
    print("\n## Findings\n")
    for f in findings(df):
        print(f)
    print("\n## Python rows for content.py\n")
    print("TABLE2_ROWS =", table2(df))
    print("TABLE3_ROWS =", table_coverage(df))
