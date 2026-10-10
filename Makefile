# Reproduce the paper end to end. Everything runs on CPU: no GPU, no API keys,
# and no model inference anywhere in the pipeline.
#
# decon is a separate Rust binary and is not vendored. To include it:
#   git clone https://github.com/allenai/decon && cd decon && cargo build --release
#   make decon DECON_BIN=decon/target/release/decon

PY := python
DECON_BIN ?= decon/target/release/decon

.PHONY: help all data results figures verify decon test clean

help:  ## list the targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) \
	  | sed -e 's/:.*## /\t/' -e 's/^/  make /' \
	  | expand -t 22

all: data results figures verify  ## the whole pipeline, in order

data:  ## fetch the corpora and benchmark groups from HuggingFace
	$(PY) scripts/01_prepare_data.py --nemotron

results:  ## regenerate every number the paper reports
	$(PY) scripts/02_results.py

figures:  ## redraw Figures 1 and 2 from the regenerated JSON
	$(PY) scripts/05_figures.py

verify:  ## check the paper's text against the regenerated JSON
	$(PY) scripts/04_verify_numbers.py

decon:  ## run Ai2's decon; needs DECON_BIN
	$(PY) scripts/03_run_decon.py --decon-bin "$(DECON_BIN)"

test:  ## unit tests for the filter implementations
	$(PY) -m pytest tests/ -q

clean:  ## remove decon's working copies and __pycache__; keeps data and results
	rm -rf results/paper/decon_run
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
