# An open benchmark for day-ahead service transformer load forecasting

This repository holds the code, data and results behind the paper

> R. K. Belchandan, D. Montenegro, and V. Cecchi, "An Open Benchmark for Day-Ahead Service
> Transformer Load Forecasting: Grouping, Metering Coverage, and Overload Screening,"
> submitted to the 2027 IEEE Power & Energy Society General Meeting.

The paper tests, on one population and with one scoring protocol, the choices that published
studies of transformer-level forecasting disagree about: whether to group transformers before
forecasting and how, how many transformers need their own meter history, how to turn summed
customer kilowatts into transformer kVA, and whether an accurate forecast also finds overloaded
units. It covers 4,134 service transformers, 13 forecasting strategies, seven grouping methods,
three metering coverage levels and 16 stress cases, in 144 controlled runs.

## Headline results

| Question | Answer |
|---|---|
| Best accuracy, December | one boosted model per group, 18.5% median nMAE |
| Best accuracy, July | fleet LSTM, 16.9% |
| Spread across 17 grouping settings | 17.7% to 18.1% |
| Grouping that needs no meter history | by feeder 17.9%, by static attributes 18.1% |
| Best error on never-metered units at 15% / 35% / 70% coverage | 26.6% / 24.7% / 25.1% |
| Daily peak miss, per-transformer models trained for average error | 16.6% (boosting) and 24.5% (ridge) low |
| Overload F1 of the 90th percentile fleet model, 30% / 50% electric vehicle adoption | 0.425 / 0.508 |
| Units above nameplate: summed kW / flat ratio / calibrated map / power flow | 0 / 23 / 3 / 3 (rural substation) |

## What is in here

```
code/          the benchmark
  paths.py          where every script reads and writes, relative to this repository
  core.py           panel container, day-ahead features, grouping methods, forecasters,
                    top-down allocation, metrics, and the meter-fault generator
  run_experiment.py the eight experiments, appending one row per run to results/results.jsonl
  neural.py         the fleet LSTM
  scenarios.py      electric vehicle, rooftop solar and home battery overlays
  qsts.py           transformer loading from an OpenDSS time series power flow
  fit_kva.py        fits the kW-to-kVA conversion on the simulated year
  build_kva2.py     rebuilds the apparent power panel with that conversion and validates it
  faq_analysis.py, faq_extra.py   stratified error, seasonal shapes, temperature response,
                    daily register comparison, peak timing and coincidence
  bootstrap_ci.py   bootstrap intervals on the medians and on paired differences
  figures_v2.py     the four paper figures
  mktables.py       turns results.jsonl into the result tables
  smartds.py        reads SMART-DS OpenDSS models
  vps/              the Windows build that downloads SMART-DS and assembles the fleet
  synth.py, prep_iowa.py, topdown_probe.py, report.py   early synthetic test data, a loader
                    for the Iowa 240-bus system, and exploratory probes
data/          the hourly panels, the customer map and the power flow results (see data/README.md)
results/       every result row, the fitted conversion, tables, per-transformer errors
figures/       the four paper figures, PDF and PNG
dashboard/     a self-contained interactive explorer of the results (open index.html)
```

## Where the data comes from

The benchmark is built from **SMART-DS**, the synthetic distribution networks developed by the
National Renewable Energy Laboratory and published through the U.S. Department of Energy's Open
Energy Data Initiative as submission 2981 (https://data.openei.org/submissions/2981) under the
Creative Commons Attribution 4.0 license. It is also mirrored in the AWS Open Data bucket
`oedi-data-lake` (us-west-2), readable without an AWS account. We used the 2018 Greensboro region at
`SMART-DS/v1.0/2018/GSO/<sub-region>/scenarios/base_timeseries/opendss/<substation>/`.

Four substations, two urban-suburban and two rural, give 4,134 service transformers rated at
167 kVA or less and 20,353 load points. Each home appears once per 120 V leg of its
center-tapped service, so the two legs are combined first, which leaves about 10,200 homes.
Experiments draw from a stratified random sample of 1,500 transformers (800 rural and 700
urban-suburban), and that sample is what `data/` holds.

Loading is apparent power rather than summed real power. A year of 15-minute OpenDSS power flow
on 1,154 transformers showed that the kVA-to-kW ratio runs near 1.25 at light load and falls to
about 1.09 at heavy load, so the panel uses a fitted conversion,
S² = (P + a)² + (bP + c)², per region instead of a flat factor.

The planned check on measured data uses the **Iowa 240-bus system** published by Iowa State
University (https://wzy.ece.iastate.edu/testsystem.html). Its producers assert no license and ask
that their 2019 NAPS paper be cited, so this repository includes only the loader
(`code/prep_iowa.py`), not their data.

## Quick start

The code needs Python 3.10 or later.

```bash
pip install -r requirements.txt
python data/merge_panels.py          # joins the rural and urban halves into data/panel_kva2
cd code
python run_experiment.py main --data ../data/panel_kva2 --name smartds --out ../results \
       --folds winter --strategies seasonal_naive --max 500
```

That last command reruns one row of the paper's Table II (same hour last week, December) and
should reproduce its median nMAE of 27.0%. Each run appends to `results/results.jsonl`, so rerun it
into a copy of `results/` if you want to keep the published rows untouched.

## Reproducing everything

```bash
cd code
python run_experiment.py main     --data ../data/panel_kva2 --name smartds --out ../results --folds winter,summer --max 500
python run_experiment.py grouping --data ../data/panel_kva2 --name smartds --out ../results --max 300
python run_experiment.py coverage --data ../data/panel_kva2 --name smartds --out ../results --max 400
python run_experiment.py dirty    --data ../data/panel_kva2 --name smartds --out ../results --max 350
python run_experiment.py growth   --data ../data/panel_kva2 --name smartds --out ../results --max 350
python run_experiment.py quantile --data ../data/panel_kva2 --name smartds --out ../results --max 350
python run_experiment.py solar    --data ../data/panel_kva2 --name smartds --out ../results --max 300
python run_experiment.py storage  --data ../data/panel_kva2 --name smartds --out ../results --max 300
python faq_analysis.py && python faq_extra.py
python figures_v2.py && python mktables.py && python bootstrap_ci.py
```

`--max N` caps the number of transformers and `--strategies a,b,c` runs a subset. An interrupted
run resumes by rerunning the same command. The saved hourly predictions (about 95 MB) are
regenerated as `results/pred_*.parquet` and are not tracked.

Refitting the kW-to-kVA conversion (`fit_kva.py`, `build_kva2.py`) needs the summed-kW panel,
which is not shipped. The Windows build in `code/vps` regenerates it from SMART-DS:

```powershell
# rename code/vps/Run-SmartDS.single-file.ps1.txt to Run-SmartDS.ps1 first
powershell -ExecutionPolicy Bypass -File .\Run-SmartDS.ps1 -DryRun
powershell -ExecutionPolicy Bypass -File .\Run-SmartDS.ps1 -Substations 2 -NoSimulate
powershell -ExecutionPolicy Bypass -File .\Run-SmartDS.ps1 -Substations 1 -Subs rural -Step 15 -MaxSim 600
```

Every script finds its inputs through `code/paths.py`, so the repository can live anywhere. Set
`PESGM_DATA`, `PESGM_RESULTS` or `PESGM_FIGURES` to keep any of those elsewhere.

## Protocol

Each forecast is issued at midnight for the next 24 hours, using data through the previous day.
There are two seasonal folds: one trained from January to mid-June and tested on July, and one
trained from January to mid-November and tested on December. Clustering, scaling and early
stopping use training and validation data only. Temperature is the recorded 2018 value plus
zero-mean Gaussian noise of 1 °C, which stands in for a day-ahead weather forecast. In the
meter-fault runs the models see corrupted history while scoring always uses the clean series.

Error is normalized by each transformer's own mean load and summarized by the fleet median and
interquartile range. Peak behavior is the daily peak absolute error, the signed peak bias and the
error in the peak hour. Overload screening is precision, recall and F1 on daily maximum loading
above 80% and 100% of nameplate.

## License

The code is released under the MIT License (`LICENSE`). The data, results and figures are
released under Creative Commons Attribution 4.0 (`LICENSE-DATA.md`) and are derived from
SMART-DS, which NREL publishes under the same license, so any reuse should credit both this
repository and SMART-DS.

## Citation

Until the paper is published, please cite this repository using `CITATION.cff` (GitHub's "Cite
this repository" button reads it). The paper citation will be added here once it appears.

## Contact

Rakesh Kumar Belchandan, University of North Carolina at Charlotte, rbelchan@charlotte.edu
