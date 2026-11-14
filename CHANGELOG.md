# Changelog

## 0.3.0 (2026-11-14) - baselines, trends, serving, monitoring
- Added 18 trend features (6h mean/std/change of 6 vitals): AUROC +0.004 only; timing AUROC unchanged (~0.51).
- Added baselines on the same folds: SIRS rule (AUROC 0.626) and logistic regression (0.725); LightGBM 0.801.
- Added benchmark-style task (target = SepsisLabel of the hour): AUROC 0.821, timing AUROC 0.557.
- Added calibration analysis (Brier 0.0085 vs 0.0087 no-skill) and `timing AUROC` as a standard metric.
- Added final model training (`sepsis.final`), FastAPI service (`api/app.py`), Dockerfile (not built: Docker missing),
  PSI drift monitoring (`sepsis.monitoring`) and 7 more tests (13 total).
- One shared feature function for training and serving (test checks they match).
- Cleaned up logs and temporary files.

## 0.2.0 (2026-11-14) — leak-free rewrite
- Restructured into `src/sepsis/` package, `tests/`, `notebooks/`, `docs/`, `archive/`.
- **Fixed leakage:** ffill / diff / shift now per patient; split by patient (StratifiedGroupKFold);
  imputation statistics no longer computed on the whole dataset; threshold chosen on a
  patient-held-out validation split (not on test).
- **Removed dataset-construction leak:** `ICULOS`, `Hour`, `HospAdmTime` are not features
  (non-sepsis stays are cut at ~59h; stays > 60h are 82% septic). Ablation quantifies it (+0.02 AUROC).
- **New target:** onset within next H hours, already-septic hours removed, onset <= 6h patients dropped.
- Features: 34 forward-filled variables + hours-since-last-measurement + Age, Gender (70 total). LightGBM.
- Evaluation: hourly + patient-level metrics, patient-bootstrap CI, horizons 3/6/12h, SHAP.
- Replica of the original pipeline added (`sepsis.replica`) — original report numbers did **not** reproduce.
- Found: model does not localize onset in time (AUROC within septic patients = 0.51).
- Colab notebook with Drive caching (`notebooks/Sepsis_Colab.ipynb`).
- Fixed along the way: `np.isin` on string IDs (10 min/fold) -> integer patient codes (seconds).

## 0.1.0 (2026-02) — original Colab notebook
- RandomForest on 7 features, random row split, global ffill/shift. Kept in `archive/` for reference.

## 0.3.1 (2026-11-14) - report rewrite
- Report rewritten in simple first-person English, no tables, plain ASCII text (`docs/REPORT.md`).
- `docs/build_report_docx.py` builds `docs/SepsisWatch_Report.docx` (page break before each section, 12 figures); uploaded to Google Drive as a Google Doc.
