.PHONY: install test format lint run clean spatial-cv interactive-map pub-figures sensitivity package

PY = python
PIP = pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	pytest tests/

format:
	black .
	isort .

lint:
	flake8 .
	mypy .

run:
	$(PY) scripts/run_complete_pipeline.py --n-regimes 6 --pysr_iterations 40

run-quick:
	$(PY) scripts/run_complete_pipeline.py --n-regimes 6 --pysr_iterations 5

spatial-cv:
	$(PY) scripts/validation/run_spatial_cv.py --splits 5 --iterations 20

interactive-map:
	$(PY) scripts/viz/plot_interactive_regime_map.py

pub-figures:
	$(PY) scripts/viz/generate_publication_figures.py

sensitivity:
	$(PY) scripts/analysis/equation_sensitivity.py

package:
	powershell -ExecutionPolicy Bypass -File scripts/package_for_publication.ps1

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
