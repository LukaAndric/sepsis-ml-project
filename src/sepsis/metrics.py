"""Threshold selection and hourly / patient-level metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def threshold_for_recall(y, p, recall: float) -> float:
    """Highest threshold whose recall is still >= `recall` (fix on VALIDATION data only)."""
    _, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= recall)[0]
    return float(thr[ok[-1]])


def hourly_metrics(y, p, thr) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), prevalence=float(y.mean()),
                threshold=thr, recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                fpr=fp / max(fp + tn, 1))


def patient_metrics(m: pd.DataFrame, alarm: np.ndarray, y: np.ndarray) -> dict:
    """m: meta rows of the evaluated hours. A septic patient counts as detected when an alarm
    fires inside the H-hour window before onset (y == 1)."""
    d = m[["Patient_ID", "septic_patient"]].copy()
    d["alarm"] = alarm.astype(int); d["y"] = y
    sep, non = d[d.septic_patient == 1], d[d.septic_patient == 0]
    det = sep[sep.y == 1].groupby("Patient_ID").alarm.max()
    sens = det.reindex(sep.Patient_ID.unique()).fillna(0).mean()
    return dict(pat_sensitivity=float(sens),
                nonsepsis_pat_with_alarm=float(non.groupby("Patient_ID").alarm.max().mean()),
                alarms_per_100_nonsepsis_hours=float(non.alarm.mean() * 100))


def bootstrap_ci(y, p, thr_row, patient_codes, n=100, seed=42):
    """Patient-level bootstrap (resample patients, not rows) of AUROC/AUPRC/recall/precision/FPR."""
    rng = np.random.default_rng(seed)
    order = np.argsort(patient_codes, kind="stable")
    codes = patient_codes[order]
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
    out = {k: [] for k in ("auroc", "auprc", "recall", "precision", "fpr")}
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        yy, pp, tt = y[idx], p[idx], thr_row[idx]
        pred = pp >= tt
        tp = (pred & (yy == 1)).sum(); fp = (pred & (yy == 0)).sum()
        out["auroc"].append(roc_auc_score(yy, pp)); out["auprc"].append(average_precision_score(yy, pp))
        out["recall"].append(tp / max((yy == 1).sum(), 1)); out["precision"].append(tp / max(tp + fp, 1))
        out["fpr"].append(fp / max((yy == 0).sum(), 1))
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in out.items()}


def temporal_auroc(m: pd.DataFrame, y, p, max_hours: int = 48) -> float:
    """AUROC INSIDE septic patients only: rows inside the target window vs rows up to `max_hours` before
    onset. ~0.5 = the model ranks patients but does not know WHEN onset is near."""
    sel = (m.septic_patient.values == 1) & (m.hrs_to_onset.values <= max_hours)
    return float(roc_auc_score(np.asarray(y)[sel], np.asarray(p)[sel]))
