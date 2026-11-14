"""Why does the model not know WHEN sepsis starts?  Timing AUROC = AUROC inside septic patients only
(rows <= 6 h before onset vs rows 7-48 h before onset).  Experiments, 5 folds, same folds as everywhere:

  D1  univariate: does ANY single feature change before onset?  (raw, and within-patient = minus patient's own mean)
  E1  vitals only (no labs, no 'hours since measured')                  -> is the model cheating with test ordering?
  E2  + patient-baseline / short + long window dynamics                   -> richer causal time features
  E3  E2 trained only on septic patients (case-only)                      -> learn timing directly
  E4  E2 + E3 as a two-stage score (patient risk x timing)                -> informational

    python -m sepsis.timing
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import cached_features
from .train import make_folds

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
MAX_H = 48


def dynamics(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Extra CAUSAL per-patient features from forward-filled vitals (only past and current hours)."""
    pid = meta.Patient_ID
    out = {}
    for c in VITALS:
        s = X[c]
        g = s.groupby(pid, sort=False)
        out[c + "_dev_base"] = s - g.expanding(min_periods=1).mean().to_numpy()      # vs patient's own baseline so far
        out[c + "_d1"] = s - g.shift(1)
        out[c + "_d3"] = s - g.shift(3)
        out[c + "_max12"] = g.rolling(12, min_periods=1).max().to_numpy()
        out[c + "_min12"] = g.rolling(12, min_periods=1).min().to_numpy()
        m24 = g.rolling(24, min_periods=1).mean().to_numpy()
        out[c + "_mean24"] = m24
        out[c + "_dev24"] = s - m24
        out[c + "_std24"] = g.rolling(24, min_periods=2).std().to_numpy()
    return pd.DataFrame(out).astype("float32")


def timing_auc(meta, y, p, idx):
    sel = idx[(meta.septic_patient.values[idx] == 1) & (meta.hrs_to_onset.values[idx] <= MAX_H)]
    return roc_auc_score(y[sel], p[sel]) if len(np.unique(y[sel])) == 2 else np.nan


def univariate(X, meta):
    """Per feature: AUROC (window vs earlier) inside septic patients, raw and within-patient."""
    sel = (meta.septic_patient.values == 1) & (meta.hrs_to_onset.values <= MAX_H)
    m, Xs, y = meta[sel], X[sel], meta.y6.values[sel]
    rows = []
    for c in Xs.columns:
        v = Xs[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() < 500 or len(np.unique(y[ok])) < 2:
            continue
        raw = roc_auc_score(y[ok], v[ok])
        dm = v - pd.Series(v).groupby(m.Patient_ID.values).transform("mean").to_numpy()   # remove patient level
        ok2 = ~np.isnan(dm)
        within = roc_auc_score(y[ok2], dm[ok2]) if len(np.unique(y[ok2])) == 2 else np.nan
        rows.append((c, raw, within, int(ok.sum())))
    d = pd.DataFrame(rows, columns=["feature", "auc_raw", "auc_within_patient", "n"])
    d["strength_within"] = (d.auc_within_patient - 0.5).abs()
    return d.sort_values("strength_within", ascending=False)


def cv(X, meta, cols, tag, row_fold, pcode, case_only=False):
    y = meta.y6.values
    septic48 = (meta.septic_patient.values == 1) & (meta.hrs_to_onset.values <= MAX_H)
    oof = np.full(len(meta), np.nan); rows = []
    for k in range(C.N_FOLDS):
        te = np.where(row_fold == k)[0]; tr = np.where(row_fold != k)[0]
        if case_only:
            tr = tr[septic48[tr]]
        rng = np.random.default_rng(C.SEED + k)
        tp = np.unique(pcode[tr]); vp = rng.choice(tp, int(0.15 * len(tp)), replace=False)
        isv = np.zeros(pcode.max() + 1, bool); isv[vp] = True
        vm = isv[pcode[tr]]; fit, val = tr[~vm], tr[vm]
        dtr = lgb.Dataset(X.iloc[fit][cols], y[fit]); dva = lgb.Dataset(X.iloc[val][cols], y[val], reference=dtr)
        mdl = lgb.train({**C.LGBM_PARAMS, "metric": "auc"}, dtr, 600, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(60, verbose=False)])
        p = mdl.predict(X.iloc[te][cols], num_iteration=mdl.best_iteration)
        oof[te] = p
        rows.append(dict(fold=k, timing_auroc=timing_auc(meta, y, oof, te), auroc_all=roc_auc_score(y[te], p)
                         if not case_only else np.nan, iters=mdl.best_iteration))
        print(f"[{tag}] fold {k}: timing {rows[-1]['timing_auroc']:.3f}", flush=True)
    r = pd.DataFrame(rows)
    return r, oof


def main():
    X, meta = cached_features()
    row_fold, pcode = make_folds(meta)
    base = list(X.columns)
    print("D1: univariate within-patient signal ...", flush=True)
    uni = univariate(X, meta)
    uni.to_csv(C.RESULTS / "timing_univariate.csv", index=False)
    print(uni.head(12).to_string(index=False), flush=True)

    dyn = dynamics(X, meta)
    XE = pd.concat([X, dyn], axis=1)
    vit_cols = [c for c in base if c.split("_")[0] in VITALS and "hrs_since" not in c] + ["Age", "Gender"]
    exps = [("E1_vitals_only", XE, vit_cols, False),
            ("E2_vitals_plus_dynamics", XE, vit_cols + list(dyn.columns), False),
            ("E2b_all_plus_dynamics", XE, base + list(dyn.columns), False),
            ("E3_case_only_all", XE, base + list(dyn.columns), True),
            ("E3b_case_only_vitals", XE, vit_cols + list(dyn.columns), True)]
    out = {}
    for tag, XX, cols, case in exps:
        r, _ = cv(XX, meta, cols, tag, row_fold, pcode, case_only=case)
        out[tag] = dict(timing_mean=float(r.timing_auroc.mean()), timing_std=float(r.timing_auroc.std()),
                        auroc_all_mean=float(r.auroc_all.mean()) if r.auroc_all.notna().any() else None, n_features=len(cols))
        print(tag, out[tag], flush=True)
    (C.RESULTS / "timing_experiments.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()


def make_figure():
    """fig14: timing AUROC of every attempt (reads results/*.json/csv, no retraining)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = json.loads((C.RESULTS / "timing_experiments.json").read_text())
    main = pd.read_csv(C.RESULTS / "cv_main_h6.csv").temporal_auroc
    phys = pd.read_csv(C.RESULTS / "cv_physionet_h0.csv").temporal_auroc
    items = [("Final model\n(all inputs)", main.mean(), main.std()),
             ("Vitals only", e["E1_vitals_only"]["timing_mean"], e["E1_vitals_only"]["timing_std"]),
             ("Vitals + more\ndynamics", e["E2_vitals_plus_dynamics"]["timing_mean"], e["E2_vitals_plus_dynamics"]["timing_std"]),
             ("All + more\ndynamics", e["E2b_all_plus_dynamics"]["timing_mean"], e["E2b_all_plus_dynamics"]["timing_std"]),
             ("Trained on sepsis\npatients only", e["E3_case_only_all"]["timing_mean"], e["E3_case_only_all"]["timing_std"]),
             ("Sepsis only,\nvitals", e["E3b_case_only_vitals"]["timing_mean"], e["E3b_case_only_vitals"]["timing_std"]),
             ("Benchmark\ntarget", phys.mean(), phys.std())]
    plt.figure(figsize=(11, 4.5))
    b = plt.bar([i[0] for i in items], [i[1] for i in items], yerr=[i[2] for i in items], capsize=4, color="#4C72B0")
    plt.bar_label(b, fmt="%.3f", padding=3); plt.axhline(0.5, ls="--", c="k"); plt.ylim(0.4, 0.8)
    plt.ylabel("Timing AUROC (0.5 = chance)")
    plt.title("Fig 14. Every attempt to teach the model WHEN sepsis starts stays close to chance")
    plt.tight_layout(); plt.savefig(C.FIGURES / "fig14_timing_attempts.png", dpi=150); plt.close()
