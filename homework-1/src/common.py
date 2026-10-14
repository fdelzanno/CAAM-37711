import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

HW_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = HW_DIR / "data"
RESULTS_DIR = HW_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_JSON = RESULTS_DIR / "results.json"

ALE_FEATURES = ["anchor_ratio", "trans_range", "node_density", "iterations"]


def load_ale():
    df = pd.read_csv(DATA_DIR / "mcs_ds_edited_iter_shuffled.csv")
    return df[ALE_FEATURES].values, df["ale"].values


def save_results(key, values):
    RESULTS_DIR.mkdir(exist_ok=True)
    data = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    data[key] = values
    RESULTS_JSON.write_text(json.dumps(data, indent=2))


def save_table(name, header, rows, align):
    """Write a LaTeX tabular to results/<name>.tex; rows are lists of strings."""
    RESULTS_DIR.mkdir(exist_ok=True)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n")
