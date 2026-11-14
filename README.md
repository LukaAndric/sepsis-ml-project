# SepsisWatch: sepsis risk and detection model (research prototype, v1.0)

[![tests](https://github.com/LukaAndric/sepsis-ml-project/actions/workflows/tests.yml/badge.svg)](https://github.com/LukaAndric/sepsis-ml-project/actions/workflows/tests.yml) MIT license

Hour-by-hour sepsis risk for ICU patients (PhysioNet/CinC 2019 data). The project started as a 6-hour early-warning
model; the final result is a risk and detection model, because the data does not support early timing. The project was rewritten after an audit found data leakage in the first version.

**Headline result (honest):** hourly AUROC 0.80 with patient-grouped CV, but the model ranks *patients* and does
not tell *when* sepsis starts (timing AUROC 0.51, no better than a SIRS rule or logistic regression on that point).
See [MODEL_CARD.md](MODEL_CARD.md). The first report's recall numbers (72% → 93%) did not reproduce.

Full report (simple English, one section per page): https://docs.google.com/document/d/1f6NR-cWK-QR78c-DTjYeqtHF0Q783H-UsZfOS74_fAo/edit  (source text: docs/REPORT.md, build with docs/build_report_docx.py)

## Layout
```
src/sepsis/    config  features  train  baselines  final  metrics  analysis  eda  replica  monitoring  timing  onset  alarms
api/           app.py               FastAPI service (POST /predict, GET /health)
models/        lgbm_final.txt, model_meta.json, reference_stats.json (created by sepsis.final)
tests/         leakage, API, monitoring, timing and onset, alarm tests (22)
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
python -m sepsis.onset             # 6c. what changes around onset, 3 targets, SOFA rules (~25 min)
python -m sepsis.alarms            # 6d. silence repeated alarms (fig17)
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

## Final result (v1.0)
- Risk ranking AUROC 0.80; detection at onset AUROC 0.87 (timing AUROC 0.67).
- No early-timing signal 6-12 h before clinical onset (timing AUROC 0.50 for every rule and model): a limit of this data and label.
- 4 h of alarm silence cuts false alarms from 7.7 to 1.9 per patient-day (-76%) and keeps 85.8% of in-window detections (was 86.8%).

## Next project: MIMIC-IV
Richer data (antibiotics, cultures, vasopressors, GCS, urine) to build full Sepsis-3 and study timing properly.
Access (check the PhysioNet site for current rules): PhysioNet account -> required online CITI course -> sign the MIMIC-IV data use agreement.

## Roadmap
Timing: 6-12 h ahead is not predictable with this data (CHANGELOG 0.4.0, 0.5.0); detection at onset works (timing AUROC 0.67) → test Docker build → Streamlit demo → scheduled drift reports.

Keep this README, `CHANGELOG.md`, `MODEL_CARD.md` and the report in sync after every code change (numbers come from `results/summary.json`).
