"""Faithful replica of the ORIGINAL (leaky) pipeline, to quantify how inflated it was:
global ffill/mean over all patients, global shift(-6), random row split, RandomForest,
7 features, Youden threshold chosen on the TEST set. Then the same model with a patient split.

    python -m sepsis.replica
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config as C


def _evaluate(tag, Xo, yo, tr, te):
    m = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=C.SEED, n_jobs=-1)
    m.fit(Xo.iloc[tr], yo.iloc[tr])
    p = m.predict_proba(Xo.iloc[te])[:, 1]
    y = yo.iloc[te].values
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr)); best = thr[i]
    out = dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), prevalence=float(y.mean()),
               recall_at_0_5=recall_score(y, p >= 0.5), precision_at_0_5=precision_score(y, p >= 0.5, zero_division=0),
               youden_threshold=float(best), recall_at_youden=recall_score(y, p >= best),
               precision_at_youden=precision_score(y, p >= best), fpr_at_youden=float(fpr[i]))
    print(tag, {k: round(v, 4) for k, v in out.items()}, flush=True)
    return out


def main():
    df = pd.read_csv(C.data_path(), index_col=0).reset_index(drop=True)
    pid = df["Patient_ID"].astype(str).values
    dc = df.ffill()
    dc = dc.fillna(dc.mean())                                   # LEAK: statistics over the whole dataset
    dc["HR_Change"] = (dc["HR"] - dc["HR"].shift(1)).fillna(0)  # LEAK: crosses patient boundaries
    yo = dc["SepsisLabel"].shift(-6).fillna(0).astype(int)      # LEAK: crosses patients, includes already-septic hours
    Xo = dc[["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "HR_Change"]]
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=C.SEED, stratify=yo)
    res = {"original_random_row_split": _evaluate("original (random rows)", Xo, yo, tr, te)}
    tr_g, te_g = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED).split(idx, groups=pid))
    res["same_pipeline_patient_split"] = _evaluate("same pipeline, patient split", Xo, yo, tr_g, te_g)
    C.RESULTS.mkdir(exist_ok=True)
    (C.RESULTS / "replica_original.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
