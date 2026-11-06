PYTHON ?= python3

.PHONY: all results pdf clean

all: pdf

results:
	$(PYTHON) src/run_all.py

pdf: results
	@if command -v latexmk >/dev/null 2>&1; then \
	  latexmk -pdf -cd -interaction=nonstopmode report/homework-2.tex && \
	  cp report/homework-2.pdf Delzanno-HW2.pdf; \
	else \
	  echo "latexmk not found: skipping PDF build (committed Delzanno-HW2.pdf is unchanged)"; \
	fi

clean:
	-latexmk -C -cd report/homework-2.tex
	rm -rf results src/__pycache__
