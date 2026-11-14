# Usage (Git Bash / WSL / Linux / macOS):  make all
# Without make, run the same commands by hand (see README).
PY ?= ./.venv/Scripts/python.exe
export PYTHONPATH := src

.PHONY: setup test eda train baselines replica analysis final api monitor all clean-cache

setup:
	python -m venv .venv && $(PY) -m pip install -r requirements.txt

test:            ## leakage tests (must pass before trusting any number)
	$(PY) -m pytest tests -q

eda:             ## figures fig1..fig5 + results/eda_stats.json
	$(PY) -m sepsis.eda

train:           ## 5-fold patient-grouped CV (cached; add ARGS=--force to redo)
	$(PY) -m sepsis.train $(ARGS)

baselines:       ## SIRS rule + logistic regression on the same folds
	$(PY) -m sepsis.baselines

replica:         ## the original (leaky) pipeline, for comparison
	$(PY) -m sepsis.replica

analysis:        ## bootstrap CI, SHAP, fig6..fig13, results/summary.json
	$(PY) -m sepsis.analysis

final:           ## final model + monitoring reference -> models/
	$(PY) -m sepsis.final

api:             ## http://localhost:8000/docs
	$(PY) -m uvicorn api.app:app --port 8000

monitor:         ## PSI drift report from logs/predictions.csv
	$(PY) -m sepsis.monitoring

all: test eda train baselines replica analysis final

clean-cache:     ## forces features + CV to be rebuilt next time
	rm -f results/X.parquet results/meta.parquet results/cv_*.csv results/oof_*.parquet
