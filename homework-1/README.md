# Homework 1

## Running

From the repository root (see the top-level `README.md` for setup):

```
make homework-1
```

or, with the virtual environment active, to run only the code:

```
python homework-1/src/run_all.py
```

## Scripts (`src/`)

- `common.py`: shared paths and helpers for saving results and LaTeX tables.
- `p4.py`: Problem 4, implementing regression models (linear regression, SVR, decision tree, neural network).
- `p5_classification.py`: Problem 5 (classification), logistic and probit regression on Breast Cancer Wisconsin data.
- `p5_regression.py`: Problem 5 (regression), linear regression with squared-error and Student-$t$ losses on ALE data.
- `run_all.py`: runs `p4.py`, `p5_classification.py`, `p5_regression.py` in order.

## Outputs (`results/`)

- `results.json`: all numerical results.
- `p*_table.tex`: LaTeX tables included by the report.
- `figures/`: Problem 4 QQ plots.

## Report

`report/homework-1.tex` is the LaTeX source. The built report is `Delzanno-HW1.pdf`.
