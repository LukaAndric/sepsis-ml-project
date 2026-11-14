# Model card — SepsisWatch v1.0: sepsis risk and detection (research prototype)

**Not for clinical use.** Educational / portfolio project on retrospective public data.

## Task
Hourly prediction of "sepsis onset within the next H hours" (main H = 6; also 3 and 12) for ICU patients.
A second, benchmark-style task (target = the dataset's `SepsisLabel` of the current hour) is also reported.

## Data
PhysioNet/CinC Challenge 2019: 40,336 ICU patients, 1,552,210 patient-hours, 40 variables, 7.3% of patients septic.
After removing patients with onset in the first 6 h (706) and hours where the patient is already septic:
39,627 patients, 1,523,513 hours, 0.87% positive hours for H = 6.
Note: the dataset's `SepsisLabel` already starts 6 h before clinical onset.

## Model
LightGBM, 88 features: 34 forward-filled measurements, 34 "hours since last measured", 18 trend features
(6 h mean / std / change for HR, O2Sat, Temp, SBP, MAP, Resp), Age, Gender.
Evaluation: 5-fold CV split by patient; threshold fixed at 80% recall on a patient-held-out validation split.
Final deployable model: `models/lgbm_final.txt` (threshold in `models/model_meta.json`).

## Performance (H = 6, hourly, mean ± std over 5 folds; patient-bootstrap 95% CI on pooled out-of-fold)
| Metric | Value |
|---|---|
| AUROC | 0.801 ± 0.003 (pooled 0.794, CI 0.785–0.801) |
| AUPRC | 0.045 ± 0.003 (prevalence 0.87%; pooled CI 0.041–0.047) |
| Recall at operating point | 80.5% |
| Precision at operating point | 1.9% |
| False-positive rate (hours) | 36.1% |
| Septic patients with an alarm in the window | 86.8% ± 1.2% |
| Non-septic patients with ≥1 alarm | 59.1% ± 2.3% |
| Alarms per 100 non-septic patient-hours | 32.1 ± 1.4 |
| **Timing AUROC** (inside septic patients) | **0.509 ± 0.009** |
| Brier score | 0.0085 (no-skill 0.0087) |

Comparison (same folds, 6 h): SIRS rule AUROC 0.626 · logistic regression 0.725 · LightGBM without trend
features 0.797 · LightGBM with trend features 0.801. Timing AUROC is 0.50–0.52 for **all** of them.
Horizons: AUROC 0.804 (3 h), 0.801 (6 h), 0.796 (12 h).
Benchmark-style task (label of the current hour): AUROC 0.821 ± 0.005, AUPRC 0.097, timing AUROC 0.557.

## Known limitations (read before using any number)
1. **No temporal localization:** timing AUROC ≈ 0.51 — the alarm rate for septic patients is ~80% at every hour from
   48 h to 1 h before onset. The model ranks *patients*, not *time to onset*; it is not yet a true early-warning signal.
   Trend features did not fix this (+0.004 AUROC only). Five further attempts (vitals only, extra dynamics, training on septic
   patients only, benchmark target; `python -m sepsis.timing`) reached 0.51-0.56 at best (`figures/fig14_timing_attempts.png`).
   Within-patient, vital signs hardly change in the 48 h before onset (univariate AUC 0.43-0.56).
   Onset study (`python -m sepsis.onset`, fig15-16): relative to the CLINICAL onset (label start + 6 h), timing AUROC is
   0.50 for 6-12 h before, 0.54 for 0-6 h before and 0.67 for detection (0-3 h after; AUROC 0.87). Partial SOFA / qSOFA
   features and rules did not help; an alarm on a rise of risk did not reduce false alarms. Records end 3 h after onset.
2. **Alarm burden is unusable clinically:** 59% of non-septic patients would trigger an alarm; precision 1.9%.
3. **Calibration is acceptable but skill is low:** deciles follow the diagonal (Fig 13), but Brier is only 1.6% better than no skill.
4. **Clinician-behaviour features:** top SHAP features include `FiO2_hrs_since`, Lactate and other `*_hrs_since` features
   (test-ordering pattern). They may reflect suspicion of sepsis rather than physiology and may not transfer to another hospital.
5. **Dataset artifact:** non-sepsis stays are cut at ~59 h; `ICULOS`/`HospAdmTime` are excluded. With them AUROC would be 0.825 (inflated).
6. Single retrospective dataset, no external validation, no fairness analysis, evaluated by cross-validation on the
   training data (not on the challenge's hidden test set), so numbers are not directly comparable to the challenge leaderboard.
7. The original project report (recall 72% → 93%) could not be reproduced with the provided data (`results/replica_original.json`).

## Intended use (v1.0)
Ranking ICU patients by sepsis risk and detecting sepsis at onset (research only). NOT an early-warning system:
there is no timing signal 6-12 h before clinical onset in this data. Recommended alarm policy: 4 h of silence after an alarm
(false alarms 7.7 -> 1.9 per non-septic patient-day; in-window detection 86.8% -> 85.8%; 59% of non-septic patients still get >=1 alarm).

## Serving and monitoring
`api/app.py` (FastAPI) returns risk, alarm and threshold for one patient's hourly history and logs every request.
`python -m sepsis.monitoring` compares logged inputs with the training distribution (PSI).
The Dockerfile is provided but has **not** been built or tested on this machine (Docker is not installed).
