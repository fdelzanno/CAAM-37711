# CAAM-37711

Introduction to Machine Learning and AI 1: homework solutions.

## Layout

```
homework-N/
  data/      input data
  src/       Python scripts (one per problem) + run_all.py
  report/    LaTeX source for the write-up (+ theory/ for handwritten scans)
  results/   generated figures, tables, results.json
  Delzanno-HWN.pdf  final report (built copy)
```

## Running a homework

Requires Python 3.10-3.12 and `make`. LaTeX (`latexmk`) is optional.

```
git clone https://github.com/fdelzanno/CAAM-37711
cd CAAM-37711
make homework-1
```

The first run creates `.venv/` and installs `requirements.txt` into it (a few minutes, mostly PyTorch), then runs the code and
builds `homework-1/Delzanno-HW1.pdf`. Later runs reuse `.venv/`. Without `latexmk`, the code still runs and the PDF step is skipped
(the committed PDF is unchanged).

To run only the code manually:

```
source .venv/bin/activate
python homework-1/src/run_all.py
```

`requirements-lock.txt` lists the exact package versions the results were produced with.
Each homework's `README.md` lists its scripts and outputs.