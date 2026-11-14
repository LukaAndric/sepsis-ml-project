"""Can the model learn WHEN sepsis starts?  All times are relative to the CLINICAL onset
t_sepsis = first SepsisLabel hour + 6 h (the challenge label starts 6 h before t_sepsis).

  1. Diagnostic: mean trajectories of vitals, labs, partial SOFA and lab-ordering around clinical onset,
     septic patients vs non-septic controls aligned at a random hour.            -> fig15
  2. Three targets, each with rules (qSOFA, SOFA rise) and LightGBM with / without SOFA features:
       A  onset in 6-12 h  (current early-warning target)
       B  onset in 0-6 h
       C  first 6 h after onset (detection)
     Timing AUROC = target-window hours vs hours 12-48 h before onset, septic patients only.  -> fig16
  3. Alarm on a RISE of risk (risk - its minimum over the last 6 h) vs alarm on the risk level.

    python -m sepsis.onset
"""
import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .features import cached_features
from .train import make_folds

TARGETS = {"A: 6-12 h before": (6, 12), "B: 0-6 h before": (0, 6), "C: 0-6 h after (detection)": (-6, 0)}
FAR = (12, 48)          # negatives for timing AUROC: 12-48 h before clinical onset
LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total", "BUN", "PTT", "pH"]


def sofa_features(X: pd.DataFrame, pid: pd.Series) -> pd.DataFrame:
    """Partial SOFA (no GCS, urine, vasopressors; S/F ratio instead of P/F) from forward-filled values.
    Missing component = 0 points (usual practice). All causal."""
    fio2 = X.FiO2.where(X.FiO2 <= 1, X.FiO2 / 100)
    sf = X.O2Sat / fio2
    resp = np.select([sf < 89, sf < 214, sf < 357, sf < 512], [4, 3, 2, 1], 0)
    coag = np.select([X.Platelets < 20, X.Platelets < 50, X.Platelets < 100, X.Platelets < 150], [4, 3, 2, 1], 0)
    liver = np.select([X.Bilirubin_total >= 12, X.Bilirubin_total >= 6, X.Bilirubin_total >= 2,
                       X.Bilirubin_total >= 1.2], [4, 3, 2, 1], 0)
    cardio = (X.MAP < 70).astype(int).to_numpy()
    renal = np.select([X.Creatinine >= 5, X.Creatinine >= 3.5, X.Creatinine >= 2, X.Creatinine >= 1.2], [4, 3, 2, 1], 0)
    d = pd.DataFrame({"sofa_resp": resp, "sofa_coag": coag, "sofa_liver": liver, "sofa_cardio": cardio,
                      "sofa_renal": renal})
    d["sofa"] = d.sum(axis=1)
    d["qsofa"] = ((X.Resp >= 22).astype(int) + (X.SBP <= 100).astype(int)).to_numpy()   # no mentation in data
    g = d.groupby(pid.values, sort=False)
    for c in ["sofa", "sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_renal"]:
        d[c + "_rise24"] = d[c] - g[c].rolling(24, min_periods=1).min().to_numpy()
    d["sofa_rise_base"] = d.sofa - g["sofa"].expanding(min_periods=1).min().to_numpy()
    return d.astype("float32")


def trajectories(X, S, meta, hc):
    """fig15: means by hour relative to clinical onset; controls aligned at a random hour."""
    rng = np.random.default_rng(C.SEED)
    rel = -hc                                                         # hours after clinical onset
    non = meta.septic_patient.values == 0
    # controls: per non-septic patient a random anchor hour; relative time = ICULOS - anchor
    stay = meta[non].groupby("Patient_ID").ICULOS.max()
    anchor = pd.Series(rng.integers(13, np.maximum(stay.values, 14) + 1), index=stay.index)
    rel_c = (meta.ICULOS - meta.Patient_ID.map(anchor)).where(non)
    labs_now = (X[[c + "_hrs_since" for c in LABS]] == 0).any(axis=1).astype(float)
    panels = [("HR", X.HR), ("Resp", X.Resp), ("Temp", X.Temp), ("MAP", X.MAP), ("Lactate (ffill)", X.Lactate),
              ("WBC (ffill)", X.WBC), ("Partial SOFA", S.sofa), ("SOFA rise vs 24 h min", S.sofa_rise24),
              ("Share of hours with a new lab test", labs_now)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5))
    out = {}
    for a, (name, v) in zip(axes.ravel(), panels):
        v = pd.Series(np.asarray(v, dtype=float))
        sep = v.groupby(rel.round()).mean().loc[-48:24]
        ctl = v.groupby(rel_c.round()).mean().loc[-48:24]
        a.plot(sep.index, sep.values, color="#C44E52", label="sepsis (0 = clinical onset)")
        a.plot(ctl.index, ctl.values, color="#8C8C8C", label="controls (random hour)")
        a.axvline(0, ls=":", c="k"); a.axvspan(-12, -6, color="#C44E52", alpha=.08)
        a.set_title(name, fontsize=10)
        out[name] = {"sepsis_-24h": float(sep.get(-24, np.nan)), "sepsis_-6h": float(sep.get(-6, np.nan)),
                     "sepsis_0h": float(sep.get(0, np.nan)), "sepsis_+12h": float(sep.get(12, np.nan))}
    axes[0, 0].legend(fontsize=8)
    for a in axes[2]:
        a.set_xlabel("hours relative to clinical onset")
    plt.suptitle("Fig 15. What changes around sepsis onset? (shaded: 6-12 h before = current target window)")
    plt.tight_layout(); plt.savefig(C.FIGURES / "fig15_onset_trajectories.png", dpi=150); plt.close()
    return out


def run_target(XF, cols, meta, hc, window, row_fold, pcode, tag):
    lo, hi = window
    septic = meta.septic_patient.values == 1
    pos = septic & (hc > lo) & (hc <= hi)
    drop = septic & (hc <= lo)                                        # hours at/after the window
    y = pos.astype(int)
    far = septic & (hc > FAR[0]) & (hc <= FAR[1])
    oof = np.full(len(meta), np.nan); res = []
    for k in range(C.N_FOLDS):
        te = np.where((row_fold == k) & ~drop)[0]; tr = np.where((row_fold != k) & ~drop)[0]
        rng = np.random.default_rng(C.SEED + k)
        tp = np.unique(pcode[tr]); vp = rng.choice(tp, int(0.15 * len(tp)), replace=False)
        isv = np.zeros(pcode.max() + 1, bool); isv[vp] = True
        vm = isv[pcode[tr]]; fit, val = tr[~vm], tr[vm]
        dtr = lgb.Dataset(XF.iloc[fit][cols], y[fit]); dva = lgb.Dataset(XF.iloc[val][cols], y[val], reference=dtr)
        mdl = lgb.train({**C.LGBM_PARAMS, "metric": "auc"}, dtr, 600, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[te] = mdl.predict(XF.iloc[te][cols], num_iteration=mdl.best_iteration)
        sel = te[pos[te] | far[te]]
        res.append(dict(auroc=roc_auc_score(y[te], oof[te]), timing=roc_auc_score(y[sel], oof[sel])))
        print(f"[{tag}] fold {k}: AUROC {res[-1]['auroc']:.3f} timing {res[-1]['timing']:.3f}", flush=True)
    r = pd.DataFrame(res)
    return dict(auroc=float(r.auroc.mean()), auroc_sd=float(r.auroc.std()), timing=float(r.timing.mean()),
                timing_sd=float(r.timing.std())), oof, pos, drop


def rule_timing(score, meta, hc, window):
    lo, hi = window
    septic = meta.septic_patient.values == 1
    pos = septic & (hc > lo) & (hc <= hi)
    far = septic & (hc > FAR[0]) & (hc <= FAR[1])
    sel = pos | far
    return float(roc_auc_score(pos[sel], np.asarray(score)[sel]))


def burden(score, pos, drop, meta, recall=0.8):
    """Threshold that reaches `recall` on target-window hours (chosen on pooled out-of-fold scores, the same
    way for both alarm types, so the comparison is fair but absolute numbers are slightly optimistic)."""
    ok = ~drop & ~np.isnan(score)
    thr = np.quantile(score[ok & pos], 1 - recall)
    alarm = score >= thr
    non = (meta.septic_patient.values == 0) & ok
    pat = pd.Series(alarm[non]).groupby(meta.Patient_ID.values[non]).max()
    return dict(alarms_per_100_nonsepsis_hours=float(alarm[non].mean() * 100),
                nonsepsis_patients_with_alarm=float(pat.mean()),
                precision=float((alarm & pos & ok).sum() / max((alarm & ok).sum(), 1)))


def main():
    X, meta = cached_features(variant="all")
    hc = (meta.hrs_to_onset + 6).to_numpy(dtype=float)               # hours to CLINICAL onset (NaN if never)
    print("hours to clinical onset available: min", np.nanmin(hc), "max", np.nanmax(hc), flush=True)
    S = sofa_features(X, meta.Patient_ID)
    traj = trajectories(X, S, meta, hc)
    XF = pd.concat([X, S], axis=1)
    base, full = list(X.columns), list(X.columns) + list(S.columns)
    row_fold, pcode = make_folds(meta)
    results, oofs = {}, {}
    for name, window in TARGETS.items():
        septic = meta.septic_patient.values == 1
        npos = int((septic & (hc > window[0]) & (hc <= window[1])).sum())
        if npos < 500:
            print(f"skip {name}: only {npos} positive hours", flush=True); continue
        r = {"positive_hours": npos,
             "rule_qsofa_timing": rule_timing(S.qsofa, meta, hc, window),
             "rule_sofa_rise24_timing": rule_timing(S.sofa_rise24, meta, hc, window),
             "rule_sofa_timing": rule_timing(S.sofa, meta, hc, window)}
        r["lgbm_base"], _, _, _ = run_target(XF, base, meta, hc, window, row_fold, pcode, name + " base")
        r["lgbm_sofa"], oof, pos, drop = run_target(XF, full, meta, hc, window, row_fold, pcode, name + " +SOFA")
        trend = (pd.Series(oof) - pd.Series(oof).groupby(meta.Patient_ID.values, sort=False)
                 .rolling(6, min_periods=1).min().to_numpy()).to_numpy()
        r["alarm_on_level"] = burden(oof, pos, drop, meta)
        r["alarm_on_rise"] = burden(trend, pos, drop, meta)
        r["rise_score_timing"] = rule_timing(np.nan_to_num(trend, nan=-1), meta, hc, window)
        results[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
    (C.RESULTS / "onset_experiments.json").write_text(json.dumps({"targets": results, "trajectories": traj}, indent=1))
    make_figure(results)


def make_figure(results=None):
    if results is None:
        results = json.loads((C.RESULTS / "onset_experiments.json").read_text())["targets"]
    names = list(results)
    series = [("qSOFA rule", lambda r: r["rule_qsofa_timing"]), ("SOFA rise rule", lambda r: r["rule_sofa_rise24_timing"]),
              ("LightGBM", lambda r: r["lgbm_base"]["timing"]), ("LightGBM + SOFA", lambda r: r["lgbm_sofa"]["timing"])]
    x = np.arange(len(names)); w = 0.2
    plt.figure(figsize=(11, 4.8))
    for i, (lab, f) in enumerate(series):
        v = [f(results[n]) for n in names]
        b = plt.bar(x + (i - 1.5) * w, v, w, label=lab)
        plt.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
    plt.axhline(0.5, ls="--", c="k"); plt.axhline(0.65, ls=":", c="#C44E52", label="success line 0.65")
    plt.xticks(x, names); plt.ylim(0.4, 0.9); plt.ylabel("Timing AUROC")
    plt.title("Fig 16. Timing AUROC by target window (septic patients only; 0.5 = chance)")
    plt.legend(fontsize=8, ncol=3); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig16_onset_targets.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
