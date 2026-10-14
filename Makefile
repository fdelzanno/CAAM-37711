BASE_PYTHON ?= python3
VENV := .venv
HWS := $(wildcard homework-[0-9]*)

.PHONY: all setup clean distclean $(HWS)

all: $(HWS)

# Create .venv and install requirements (re-runs only if requirements.txt changes)
setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	$(BASE_PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	touch $@

$(HWS): $(VENV)/.installed
	$(MAKE) -C $@ PYTHON=../$(VENV)/bin/python

clean:
	for d in $(HWS); do $(MAKE) -C $$d clean; done

distclean: clean
	rm -rf $(VENV)
