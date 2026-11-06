"""Turn results/results.jsonl into the paper's tables and a numbers sheet."""
import json, collections
import paths
import pandas as pd

rows = [json.loads(l) for l in open(f"{paths.RESULTS}/results.jsonl")]
d = pd.DataFrame(rows)
d["tag"] = d.tag.fillna(d.strategy)
pct = lambda s: (100 * s).round(1)

LBL = {
 "seasonal_naive": "Seasonal naive (same hour, last week)",
 "persistence": "Persistence (same hour, yesterday)",
 "ridge_local": "Ridge regression, one model per transformer",
 "local": "Gradient boosting, one model per transformer",
 "cluster": "Gradient boosting, one model per group",
 "global": "Gradient boosting, one model for the fleet",
 "global_cluster": "Fleet model with group features",
 "topdown_rolling": "Top down, rolling four week shares",
 "topdown_cluster": "Top down from the group total",
 "topdown_feeder": "Top down from the feeder total",
 "lstm_global": "Fleet LSTM, 168 hour encoder",
 "global_cluster_q90": "Fleet model, 90th percentile objective",
 "local_q90": "Per transformer, 90th percentile objective",
}
ORDER = list(LBL)

out = []
m = d[d.experiment == "main"]
t = m.pivot_table(index="tag", columns="fold",
                  values=["nmae_med", "nmae_iqr", "peak_ape_med", "peak_bias_med",
                          "peak_hour_err_med", "ovl80_f1"])
t = t.reindex(ORDER)
tab1 = pd.DataFrame({
 "Method": [LBL[i] for i in t.index],
 "nMAE winter %": pct(t[("nmae_med", "winter")]).values,
 "nMAE summer %": pct(t[("nmae_med", "summer")]).values,
 "IQR winter %": pct(t[("nmae_iqr", "winter")]).values,
 "Peak APE winter %": pct(t[("peak_ape_med", "winter")]).values,
 "Peak APE summer %": pct(t[("peak_ape_med", "summer")]).values,
 "Peak bias winter %": pct(t[("peak_bias_med", "winter")]).values,
 "Peak hour err h": t[("peak_hour_err_med", "winter")].round(2).values,
 "F1 at 80% winter": t[("ovl80_f1", "winter")].round(3).values,
})
tab1.to_csv(f"{paths.RESULTS}/table1_main.csv", index=False)
print("TABLE 1 main strategies\n", tab1.to_string(index=False), "\n")

g = d[d.experiment == "grouping"]
g = g[g.strategy == "cluster"]
tg = (g.assign(nmae=pct(g.nmae_med), peak=pct(g.peak_ape_med))
        .loc[:, ["grouping", "k", "nmae", "peak"]]
        .sort_values(["grouping", "k"]))
tg.to_csv(f"{paths.RESULTS}/table2_grouping.csv", index=False)
print("TABLE 2 grouping sweep\n", tg.to_string(index=False))
print(f"  range {tg.nmae.min()} to {tg.nmae.max()}, spread {round(tg.nmae.max()-tg.nmae.min(),2)} points\n")

c = d[d.experiment == "coverage"]
tc = c.pivot_table(index="tag", columns="tier", values="nmae_med")
tc = pct(tc)[["ami15", "ami35", "ami70"]]
tc.to_csv(f"{paths.RESULTS}/table3_coverage.csv")
print("TABLE 3 metering coverage, nMAE on never metered transformers\n", tc.to_string(), "\n")

y = d[d.experiment == "growth"]
ty = y.pivot_table(index="tag", columns="tier", values=["nmae_med", "ovl100_f1", "ovl100_rec", "ovl100_prec"])
cols = ["ev0", "ev15", "ev30", "ev50"]
print("TABLE 4 EV growth, overload screening at 100% of nameplate")
print(pd.concat([pct(ty["nmae_med"][cols]).add_prefix("nMAE "),
                 ty["ovl100_f1"][cols].round(3).add_prefix("F1 ")], axis=1).to_string(), "\n")
pd.concat([ty["nmae_med"][cols], ty["ovl100_f1"][cols], ty["ovl100_rec"][cols],
           ty["ovl100_prec"][cols]], axis=1, keys=["nmae", "f1", "recall", "precision"]).to_csv(f"{paths.RESULTS}/table4_growth.csv")

q = d[d.experiment == "quantile"]
tq = q.pivot_table(index="alpha", columns="tag", values=["nmae_med", "ovl100_f1", "ovl100_rec", "ovl100_prec"])
tq.to_csv(f"{paths.RESULTS}/table5_quantile.csv")
print("TABLE 5 quantile sweep at EV 30%")
print(pd.concat([pct(tq["nmae_med"]).add_prefix("nMAE "),
                 tq["ovl100_f1"].round(3).add_prefix("F1 "),
                 tq["ovl100_rec"].round(3).add_prefix("rec ")], axis=1).to_string(), "\n")

x = d[d.experiment == "dirty"]
tx = x.pivot_table(index="tier", columns="tag", values="nmae_med")
base = tx.loc["clean_0.0"]
rel = (100 * (tx / base - 1)).round(1)
order = ["clean_0.0", "mcar_0.05", "mcar_0.1", "mcar_0.2", "outage_0.2", "frozen_0.2",
         "spike_0.01", "spike_0.05"]
tx = pct(tx).reindex(order); rel = rel.reindex(order)
pd.concat([tx, rel], axis=1, keys=["nmae_pct", "rel_change_pct"]).to_csv(f"{paths.RESULTS}/table6_dirty.csv")
print("TABLE 6 impaired history, nMAE % and change from clean\n",
      pd.concat([tx, rel], axis=1, keys=["nMAE %", "change %"]).to_string(), "\n")

s = d[d.experiment == "solar"]
ts = s.pivot_table(index="tier", columns="tag", values=["nmae_med", "share_over100"])
ts.to_csv(f"{paths.RESULTS}/table7_solar.csv")
print("TABLE 7 rooftop PV behind the meter\n",
      pd.concat([pct(ts["nmae_med"]), ts["share_over100"].round(4)], axis=1,
                keys=["nMAE %", "share over 100%"]).to_string())
