# Homework 2

## Running

From the repository root (see the top-level `README.md` for setup):

```
make homework-2
```

or, with the virtual environment active, to run only the code:

```
python homework-2/src/run_all.py
```

## Scripts (`src/`)

- `common.py`: shared paths and helpers for saving results and LaTeX tables.
- `p1.py`: Problem 1(c)-(e), Gaussian-NLL MLP for trip duration, baselines, and probability scores.
- `p2.py`: Problem 2(a)-(e), target encoding, embedding network, combined model, and the seed and frequency analyses.
- `p3.py`: Problem 3(a)-(d), data visualization, RKS sweep, MLP sweep, comparison, test predictions, and fitted surfaces.
- `run_all.py`: runs `p1.py`, `p2.py`, `p3.py` in order.

## Outputs (`results/`)

- `results.json`: all numerical results.
- `p*_table.tex`: LaTeX tables included by the report.
- `figures/`: Problem 3 figures.
- `rks_test_predictions.csv`, `mlp_test_predictions.csv`: Problem 3(d) predictions on `test.csv`, one row per test point in the original order.

## Report

`report/homework-2.tex` is the LaTeX source. The built report is `Delzanno-HW2.pdf`.
