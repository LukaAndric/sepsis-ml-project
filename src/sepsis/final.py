"""Train the final deployable model (6h horizon) and save everything serving + monitoring need.

    python -m sepsis.final

Outputs in models/: lgbm_final.txt, model_meta.json (features, threshold, version), reference_stats.json
(training distribution per feature, for PSI drift monitoring).
"""
import datetime as dt
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import __version__
from . import config as C
from .features import cached_features
from .metrics import threshold_for_recall

N_BINS = 10


def reference_stats(X: pd.DataFrame, score: np.ndarray) -> dict:
    """Decile edges + expected fraction per bin (NaN is its own bin) for every feature and for the risk score."""
    def one(v):
        v = np.asarray(v, dtype=float)
        nan_frac = float(np.isnan(v).mean())
        ok = v[~np.isnan(v)]
        edges = np.unique(np.quantile(ok, np.linspace(0, 1, N_BINS + 1)[1:-1])) if len(ok) else np.array([])
        frac = np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1) / len(v)
        return dict(edges=edges.tolist(), frac=frac.tolist(), nan_frac=nan_frac)
    out = {c: one(X[c].values) for c in X.columns}
    out["_risk"] = one(score)
    return out


def main():
    X, meta = cached_features()
    h = C.MAIN_HORIZON
    y = meta[f"y{h}"].values
    pcode, _ = pd.factorize(meta.Patient_ID)
    rng = np.random.default_rng(C.SEED)
    pats = np.unique(pcode)
    val_pat = rng.choice(pats, int(0.15 * len(pats)), replace=False)
    is_val = np.zeros(pcode.max() + 1, bool); is_val[val_pat] = True
    vm = is_val[pcode]; fit, val = np.where(~vm)[0], np.where(vm)[0]
    cols = list(X.columns)
    dtr = lgb.Dataset(X.iloc[fit][cols], y[fit]); dva = lgb.Dataset(X.iloc[val][cols], y[val], reference=dtr)
    mdl = lgb.train({**C.LGBM_PARAMS, "metric": "average_precision"}, dtr, C.MAX_ROUNDS, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(C.EARLY_STOPPING, verbose=False)])
    pv = mdl.predict(X.iloc[val][cols], num_iteration=mdl.best_iteration)
    thr = threshold_for_recall(y[val], pv, C.TARGET_RECALL)
    out = C.ROOT / "models"; out.mkdir(exist_ok=True)
    mdl.save_model(str(out / "lgbm_final.txt"), num_iteration=mdl.best_iteration)
    meta_json = dict(version=__version__, horizon_hours=h, threshold=thr, target_recall_on_validation=C.TARGET_RECALL,
                     features=cols, best_iteration=int(mdl.best_iteration), n_train_patients=int(len(pats) - len(val_pat)),
                     trained_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                     note="Research prototype, not for clinical use. Threshold fixed on a patient-held-out validation split.")
    (out / "model_meta.json").write_text(json.dumps(meta_json, indent=1))
    (out / "reference_stats.json").write_text(json.dumps(reference_stats(X.iloc[fit], mdl.predict(
        X.iloc[fit][cols], num_iteration=mdl.best_iteration))))
    print(f"saved final model: threshold {thr:.5f}, iterations {mdl.best_iteration}")


if __name__ == "__main__":
    main()
