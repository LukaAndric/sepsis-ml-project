"""Baselines on the SAME folds and target as the LightGBM model (6h horizon):
  * SIRS-like rule  (>=2 of: HR>90, Resp>20, Temp>38 or <36, WBC>12 or <4)  - no training
  * Logistic regression (median impute + standardize), threshold = 80% recall on validation

    python -m sepsis.baselines
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import cached_features
from .metrics import hourly_metrics, patient_metrics, temporal_auroc, threshold_for_recall
from .train import make_folds


def sirs_score(X: pd.DataFrame) -> np.ndarray:
    """Number of SIRS criteria met (0-4) from the forward-filled values. Missing counts as 'not met'."""
    s = ((X.HR > 90).astype(int) + (X.Resp > 20).astype(int)
         + ((X.Temp > 38) | (X.Temp < 36)).astype(int) + ((X.WBC > 12) | (X.WBC < 4)).astype(int))
    return s.to_numpy(dtype=float)


def _row(meta, y, p, thr, te, k):
    return dict(fold=k, **hourly_metrics(y[te], p, thr), temporal_auroc=temporal_auroc(meta.iloc[te], y[te], p),
                **patient_metrics(meta.iloc[te], p >= thr, y[te]))


def main():
    X, meta = cached_features()
    row_fold, pcode = make_folds(meta)
    y = meta[f"y{C.MAIN_HORIZON}"].values
    score = sirs_score(X)
    sirs_rows, lr_rows = [], []
    for k in range(C.N_FOLDS):
        te = np.where(row_fold == k)[0]; tr = np.where(row_fold != k)[0]
        sirs_rows.append(_row(meta, y, score[te], 2.0, te, k))          # fixed clinical rule, nothing fitted
        rng = np.random.default_rng(C.SEED + k)
        tr_pat = np.unique(pcode[tr]); val_pat = rng.choice(tr_pat, int(0.15 * len(tr_pat)), replace=False)
        is_val = np.zeros(pcode.max() + 1, bool); is_val[val_pat] = True
        vm = is_val[pcode[tr]]; fit, val = tr[~vm], tr[vm]
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=200, C=1.0))
        lr.fit(X.iloc[fit], y[fit])
        thr = threshold_for_recall(y[val], lr.predict_proba(X.iloc[val])[:, 1], C.TARGET_RECALL)
        pt = lr.predict_proba(X.iloc[te])[:, 1]
        lr_rows.append(_row(meta, y, pt, thr, te, k))
        print(f"fold {k}: SIRS AUROC {sirs_rows[-1]['auroc']:.3f} recall {sirs_rows[-1]['recall']:.2f} "
              f"fpr {sirs_rows[-1]['fpr']:.2f} | LogReg AUROC {lr_rows[-1]['auroc']:.3f} "
              f"timing {lr_rows[-1]['temporal_auroc']:.3f}", flush=True)
    pd.DataFrame(sirs_rows).to_csv(C.RESULTS / "cv_baseline_sirs_h6.csv", index=False)
    pd.DataFrame(lr_rows).to_csv(C.RESULTS / "cv_baseline_logreg_h6.csv", index=False)


if __name__ == "__main__":
    main()
