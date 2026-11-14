"""5-fold patient-grouped CV with LightGBM. Cached: finished runs are skipped unless --force.

    python -m sepsis.train            # main horizons 6, 3, 12 + ICULOS leakage ablation
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from .features import cached_features
from .metrics import hourly_metrics, patient_metrics, temporal_auroc, threshold_for_recall


def make_folds(meta: pd.DataFrame):
    """Patient-level, stratified folds; identical for every horizon. Returns (row_fold, patient_codes)."""
    pcode, uniq = pd.factorize(meta.Patient_ID)
    pat = meta.groupby("Patient_ID", sort=False).septic_patient.max()
    codes = pd.Series(np.arange(len(uniq)), index=uniq).loc[pat.index].values
    fold_of_pat = np.full(len(uniq), -1)
    skf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    for k, (_, te) in enumerate(skf.split(codes, pat.values, groups=codes)):
        fold_of_pat[codes[te]] = k
    row_fold = fold_of_pat[pcode]
    assert (row_fold >= 0).all()
    return row_fold, pcode


def run_cv(X, meta, h, cols, tag, row_fold, pcode, force=False):
    res_path = C.RESULTS / f"cv_{tag}_h{h}.csv"
    if res_path.exists() and not force:
        print(f"[{tag} h={h}] cached -> {res_path.name}")
        return pd.read_csv(res_path)
    y = meta[f"y{h}"].values
    oof = np.full(len(meta), np.nan); thr_row = np.full(len(meta), np.nan)
    rows = []
    for k in range(C.N_FOLDS):
        te = np.where(row_fold == k)[0]; tr = np.where(row_fold != k)[0]
        rng = np.random.default_rng(C.SEED + k)                    # 15% of training PATIENTS -> validation
        tr_pat = np.unique(pcode[tr])
        val_pat = rng.choice(tr_pat, size=int(0.15 * len(tr_pat)), replace=False)
        is_val = np.zeros(pcode.max() + 1, bool); is_val[val_pat] = True
        vm = is_val[pcode[tr]]
        fit, val = tr[~vm], tr[vm]
        dtr = lgb.Dataset(X.iloc[fit][cols], y[fit])
        dva = lgb.Dataset(X.iloc[val][cols], y[val], reference=dtr)
        t0 = time.time()
        mdl = lgb.train({**C.LGBM_PARAMS, "metric": "average_precision"}, dtr, C.MAX_ROUNDS, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(C.EARLY_STOPPING, verbose=False)])
        thr = threshold_for_recall(y[val], mdl.predict(X.iloc[val][cols], num_iteration=mdl.best_iteration),
                                   C.TARGET_RECALL)
        pt = mdl.predict(X.iloc[te][cols], num_iteration=mdl.best_iteration)
        oof[te], thr_row[te] = pt, thr
        r = dict(fold=k, best_iter=mdl.best_iteration, **hourly_metrics(y[te], pt, thr),
                 temporal_auroc=temporal_auroc(meta.iloc[te], y[te], pt),
                 **patient_metrics(meta.iloc[te], pt >= thr, y[te]))
        rows.append(r)
        print(f"[{tag} h={h}] fold {k}: AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f} recall {r['recall']:.2f} "
              f"prec {r['precision']:.3f} patSens {r['pat_sensitivity']:.2f} timingAUC {r['temporal_auroc']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if k == 0 and tag == "main" and h == C.MAIN_HORIZON:
            mdl.save_model(str(C.RESULTS / "lgbm_fold0_h6.txt"))
    res = pd.DataFrame(rows)
    res.to_csv(res_path, index=False)
    pd.DataFrame({"oof": oof, "thr": thr_row, "fold": row_fold}).to_parquet(C.RESULTS / f"oof_{tag}_h{h}.parquet")
    return res


def run_physionet_task(force=False):
    """Benchmark-comparable task: every hour of every patient, target = SepsisLabel of the current hour.
    Note: the label already starts 6 h before clinical onset, so this is ~6 h of warning in clinical terms."""
    X, meta = cached_features(variant="all")
    row_fold, pcode = make_folds(meta)
    return run_cv(X, meta, 0, list(X.columns), "physionet", row_fold, pcode, force)


def main(force=False):
    X, meta = cached_features()
    row_fold, pcode = make_folds(meta)
    base = list(X.columns)
    for h in (C.MAIN_HORIZON, *[x for x in C.HORIZONS if x != C.MAIN_HORIZON]):
        run_cv(X, meta, h, base, "main", row_fold, pcode, force)
    # ablation: same model WITHOUT the 6h trend features (did rolling features help timing?)
    roll = [c for c in base if c.endswith(("_mean6", "_std6", "_delta6"))]
    run_cv(X, meta, C.MAIN_HORIZON, [c for c in base if c not in roll], "no_rolling", row_fold, pcode, force)
    # ablation: add the leaky columns back to quantify the leak
    Xl = X.copy()
    Xl["ICULOS"] = meta.ICULOS.values.astype("float32")
    Xl["HospAdmTime"] = meta.HospAdmTime.values.astype("float32")
    run_cv(Xl, meta, C.MAIN_HORIZON, list(Xl.columns), "with_iculos", row_fold, pcode, force)
    run_physionet_task(force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
