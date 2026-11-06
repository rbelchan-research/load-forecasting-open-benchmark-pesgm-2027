"""Join the rural and urban panel archives into data/panel_kva2, the fleet every
experiment reads.

Column order is rural first, then urban, which is the order the experiments were run on.
That matters: `--max N` draws its sample of transformers by position, so a different order
would pick different units and the results would not reproduce row for row.

    python merge_panels.py          (run from data/, after unzipping the archives here)
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
halves = [os.path.join(HERE, p) for p in ("panel_kva2_rural", "panel_kva2_urban")]
for h in halves:
    if not os.path.isdir(h):
        sys.exit(f"missing {h}; unzip pesgm2027_data_1_panel_rural.zip and "
                 f"pesgm2027_data_2_panel_urban.zip into {HERE} first")
out = os.path.join(HERE, "panel_kva2")
os.makedirs(out, exist_ok=True)
loads = pd.concat([pd.read_parquet(f"{h}/loads.parquet") for h in halves], axis=1)
meta = pd.concat([pd.read_csv(f"{h}/meta.csv") for h in halves], ignore_index=True)
w = [pd.read_parquet(f"{h}/weather.parquet") for h in halves]
assert w[0].equals(w[1]), "the two halves should carry the same weather table"
assert list(meta["id"]) == list(loads.columns), "metadata and load columns out of step"
loads.to_parquet(f"{out}/loads.parquet")
meta.to_csv(f"{out}/meta.csv", index=False)
w[0].to_parquet(f"{out}/weather.parquet")
print(f"wrote {out}: {loads.shape[1]} transformers x {loads.shape[0]} hours")
