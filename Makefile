PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .

install-cuda:
	$(PYTHON) -m pip install -e ".[cuda,dev]"

install-frontier:
	bash scripts/setup_linux.sh --with-torchao --with-tracking

check:
	$(PYTHON) scripts/system_check.py --model configs/model/aster_moe_frontier_893m_a484m.yaml

smoke:
	$(PYTHON) scripts/smoke_train.py

test:
	$(PYTHON) -m pytest

lint:
	ruff check src scripts tests

data-dry-run:
	$(PYTHON) scripts/download_data.py --profile all --validate-first --dry-run

data:
	$(PYTHON) scripts/download_data.py --profile all --validate-first

probe:
	$(PYTHON) scripts/run_frontier_experiments.py --mode quick --steps 3

quality-ablation:
	$(PYTHON) scripts/run_quality_ablations.py --tokens 100000000 --continue-on-error

zip:
	cd .. && zip -r AsterLM-Frontier.zip AsterLM-Frontier \
		-x 'AsterLM-Frontier/.venv/*' 'AsterLM-Frontier/runs/*' \
		'AsterLM-Frontier/data/*' 'AsterLM-Frontier/**/__pycache__/*' \
		'AsterLM-Frontier/.pytest_cache/*'
