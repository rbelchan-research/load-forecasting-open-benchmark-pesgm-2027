"""Where every script reads and writes.

The repository layout is

    code/      this folder
    data/      the hourly panels, customer map and power flow results (see data/README.md)
    results/   result rows, fitted parameters, tables
    figures/   the paper figures

Any of these can be moved by setting PESGM_DATA, PESGM_RESULTS or PESGM_FIGURES.
"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
DATA = os.environ.get("PESGM_DATA", os.path.join(REPO, "data"))
RESULTS = os.environ.get("PESGM_RESULTS", os.path.join(REPO, "results"))
FIGURES = os.environ.get("PESGM_FIGURES", os.path.join(REPO, "figures"))

# the merged fleet every experiment runs on (merge the rural and urban archives into it)
PANEL = os.path.join(DATA, "panel_kva2")
# summed customer kW per transformer before the kVA conversion; written by the VPS
# build in code/vps, and only needed to refit or revalidate the conversion
PANEL_KW = os.path.join(DATA, "panel_kw")
# hourly apparent power from the two OpenDSS runs (data/simulated)
SIM = {"rural": os.path.join(DATA, "simulated", "sim_kva_hourly_rural_rhs0_1247.csv.gz"),
       "urban": os.path.join(DATA, "simulated", "sim_kva_hourly_urban_uhs0_1247.csv.gz")}
CUSTOMER_MAP = os.path.join(DATA, "customer_map_4substations.csv")


def need(path, how):
    """Stop with a plain message when an input is missing, instead of a traceback."""
    if not os.path.exists(path):
        raise SystemExit(f"missing input: {path}\n  {how}")
    return path
