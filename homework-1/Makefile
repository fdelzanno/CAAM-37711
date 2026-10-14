PYTHON ?= python3

.PHONY: all results pdf clean

all: pdf

results:
	$(PYTHON) src/run_all.py

pdf: results
	@if command -v latexmk >/dev/null 2>&1; then \
	  latexmk -pdf -cd -interaction=nonstopmode report/homework-1.tex && \
	  cp report/homework-1.pdf Delzanno-HW1.pdf; \
	else \
	  echo "latexmk not found: skipping PDF build (committed Delzanno-HW1.pdf is unchanged)"; \
	fi

clean:
	-latexmk -C -cd report/homework-1.tex
	rm -rf results src/__pycache__
