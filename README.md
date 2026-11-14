# SepsisWatch — leak-free sepsis risk model (research prototype)

Predict, hour by hour, whether an ICU patient will develop sepsis in the next 6 hours
(PhysioNet/CinC 2019 data). The project was rewritten after an audit found data leakage in the first version.

**Headline result (honest):** hourly AUROC 0.80 with patient-grouped CV, but the model ranks *patients* and does
not tell *when* sepsis starts (timing AUROC 0.51, no better than a SIRS rule or logistic regression on that point).
See [MODEL_CARD.md](MODEL_CARD.md). The first report's recall numbers (72% → 93%) did not reproduce.

Full report (simple English, one section per page): https://docs.google.com/document/d/1SgLCopo9o1afc9v5Uht0-Q1fvGAk1SGycOkL__ZfqWE/edit  (source text: docs/REPORT.md, build with docs/build_report_docx.py)

## Layout
```
src/sepsis/    config  features  train  baselines  final  metrics  analysis  eda  replica  monitoring  timing
api/           app.py               FastAPI service (POST /predict, GET /health)
models/        lgbm_final.txt, model_meta.json, reference_stats.json (created by sepsis.final)
tests/         leakage + API + monitoring tests (13)
notebooks/     Sepsis_Colab.ipynb   (Google Colab, results cached in Drive)
docs/          PYTHON_GUIDE.md      (Serbian: Python/ML knowledge for projects like this)
data/raw/      Dataset.csv          (not in git)
results/       CV tables, summary.json (parquet caches are git-ignored)
figures/       fig1..fig13
archive/       original v0.1 code + report
Dockerfile, requirements-api.txt    (container for the API; build NOT tested yet)
```

## Run locally
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows
export PYTHONPATH=src
python -m pytest tests -q          # 1. tests (leakage, API, monitoring)
python -m sepsis.eda               # 2. figures 1-5
python -m sepsis.train             # 3. CV (first run builds the feature cache; later runs are cached)
python -m sepsis.baselines         # 4. SIRS rule + logistic regression on the same folds
python -m sepsis.replica           # 5. original leaky pipeline for comparison
python -m sepsis.analysis          # 6. bootstrap CI, SHAP, calibration, figures 6-13, results/summary.json
python -m sepsis.timing            # 6b. why timing fails (5 experiments, ~8 min)
python -m sepsis.final             # 7. final model + monitoring reference -> models/
uvicorn api.app:app --port 8000    # 8. API, docs at http://localhost:8000/docs
python -m sepsis.monitoring --log logs/predictions.csv   # 9. drift report (PSI) from logged requests
```
`make all` runs steps 1-7. Add `--force` to `sepsis.train` to recompute.

## Run on Google Colab (no re-running every time)
1. Upload the whole project folder (without `.venv/`) to Google Drive as `MyDrive/Sepsis_ML_project`.
2. Open `notebooks/Sepsis_Colab.ipynb` in Colab and run all cells (not yet tested in Colab).
3. Features, CV results and models are cached **in Drive**; the next session only loads them (seconds).

## Method in 7 lines
- Features and targets are built **per patient**; never-measured values stay NaN; one shared function for training and serving.
- Target: sepsis onset within H hours; already-septic hours removed; onset ≤ 6 h patients dropped.
- `ICULOS`, `HospAdmTime` excluded (dataset-construction leak, quantified by ablation).
- LightGBM, 5-fold `StratifiedGroupKFold` by patient; threshold = 80% recall on patient-held-out validation.
- Baselines on the same folds: SIRS rule, logistic regression. Benchmark-style task also reported.
- Metrics: AUROC, AUPRC, hourly + patient-level, alarms per 100 patient-hours, timing AUROC, calibration, patient-bootstrap CI.
- Explainability: SHAP + gain importance.

## Roadmap
Fix the timing problem (sequence model such as LSTM; five other ideas already failed, see CHANGELOG 0.4.0) → test Docker build → Streamlit demo → scheduled drift reports.

Keep this README, `CHANGELOG.md`, `MODEL_CARD.md` and the report in sync after every code change (numbers come from `results/summary.json`).
