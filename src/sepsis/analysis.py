"""Turn CV outputs into the report: tables (results/summary.*), bootstrap CIs and figures fig6..fig11.

    python -m sepsis.analysis
"""
import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C
from .features import cached_features
from .metrics import bootstrap_ci
from .train import make_folds

RED, BLUE, GREY = "#C44E52", "#4C72B0", "#8C8C8C"
COLS = ["auroc", "auprc", "prevalence", "recall", "precision", "fpr", "temporal_auroc", "pat_sensitivity",
        "nonsepsis_pat_with_alarm", "alarms_per_100_nonsepsis_hours"]


def _cv(tag, h):
    return pd.read_csv(C.RESULTS / f"cv_{tag}_h{h}.csv")


def tables():
    rows = {}
    for tag, h in [("main", 3), ("main", 6), ("main", 12), ("with_iculos", 6), ("no_rolling", 6),
                   ("baseline_sirs", 6), ("baseline_logreg", 6), ("physionet", 0)]:
        d = _cv(tag, h)[COLS]
        rows[f"{tag}_h{h}"] = {c: (float(d[c].mean()), float(d[c].std())) for c in COLS}
    return rows


def fig_roc_pr(y, p, thr_row):
    fpr, tpr, _ = roc_curve(y, p); prec, rec, _ = precision_recall_curve(y, p)
    pred = p >= thr_row
    op_rec = (pred & (y == 1)).sum() / (y == 1).sum(); op_fpr = (pred & (y == 0)).sum() / (y == 0).sum()
    op_prec = (pred & (y == 1)).sum() / pred.sum()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, color=BLUE); ax[0].plot([0, 1], [0, 1], "--", c=GREY)
    ax[0].scatter(op_fpr, op_rec, color=RED, zorder=5, label=f"operating point\nrecall {op_rec:.0%}, FPR {op_fpr:.0%}")
    ax[0].set(xlabel="False positive rate (hourly)", ylabel="Recall (hourly)", title="ROC"); ax[0].legend(loc="lower right")
    ax[1].plot(rec, prec, color=BLUE); ax[1].axhline(y.mean(), ls="--", c=GREY, label=f"prevalence {y.mean():.2%}")
    ax[1].scatter(op_rec, op_prec, color=RED, zorder=5, label=f"operating point\nprecision {op_prec:.1%}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall"); ax[1].legend()
    plt.suptitle("Fig 6. Out-of-fold performance, 6h horizon (patient-grouped 5-fold CV)"); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig6_roc_pr.png", dpi=150); plt.close()


def fig_detection(meta, p, thr_row):
    d = pd.DataFrame({"h": meta.hrs_to_onset.values, "alarm": p >= thr_row, "sep": meta.septic_patient.values})
    base = d[d.sep == 0].alarm.mean()
    s = d[(d.sep == 1) & (d.h >= 1) & (d.h <= 48)].groupby("h").alarm.mean()
    plt.figure(figsize=(9, 4.5))
    plt.plot(s.index, s.values * 100, "o-", color=RED, label="septic patients: hours before onset")
    plt.axhline(base * 100, ls="--", c=GREY, label=f"non-septic patients (baseline {base:.0%})")
    plt.axvspan(0.5, C.MAIN_HORIZON + .5, color=RED, alpha=.08, label="6h target window")
    plt.gca().invert_xaxis(); plt.xlabel("Hours before sepsis onset"); plt.ylabel("% of hours with an alarm")
    plt.title("Fig 7. When does the model start to 'see' sepsis? (alarm rate vs time to onset)")
    plt.legend(); plt.tight_layout(); plt.savefig(C.FIGURES / "fig7_detection_curve.png", dpi=150); plt.close()
    return s


def fig_shap(X, meta, row_fold):
    import shap
    mdl = lgb.Booster(model_file=str(C.RESULTS / "lgbm_fold0_h6.txt"))
    te = np.where(row_fold == 0)[0]
    rng = np.random.default_rng(C.SEED)
    pos = te[meta.y6.values[te] == 1]
    idx = np.r_[rng.choice(te, 15000, replace=False), rng.choice(pos, min(1500, len(pos)), replace=False)]
    Xs = X.iloc[idx]
    sv = shap.TreeExplainer(mdl).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure(); shap.summary_plot(sv, Xs, max_display=15, show=False)
    plt.title("Fig 8a. SHAP summary (fold-0 model; sample = 15k random + 1.5k positive hours)")
    plt.tight_layout(); plt.savefig(C.FIGURES / "fig8a_shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    imp = pd.Series(mdl.feature_importance("gain"), index=mdl.feature_name()).sort_values().tail(15)
    plt.figure(figsize=(8, 5)); plt.barh(imp.index, imp.values / imp.sum() * 100, color=BLUE)
    plt.xlabel("% of total gain (top 15 shown)"); plt.title("Fig 8b. LightGBM gain importance"); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig8b_importance.png", dpi=150); plt.close()
    top = pd.Series(np.abs(sv).mean(0), index=Xs.columns).sort_values(ascending=False).head(15)
    return top


def fig_patient(X, meta, p, thr_row, row_fold):
    m = meta.assign(p=p, thr=thr_row, fold=row_fold)
    cand = []
    for pid, g in m[(m.fold == 0) & (m.septic_patient == 1)].groupby("Patient_ID"):
        if 24 <= g.onset.iloc[0] <= 72:
            cand.append((g.loc[g.y6 == 1, "p"].max(), pid))
    cand.sort(); pid = cand[len(cand) // 2][1]          # MEDIAN case, not the best one (no cherry-picking)
    idx = m.index[m.Patient_ID == pid]; g = m.loc[idx]
    fig, ax = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
    for a, v in zip(ax[:4], ["HR", "MAP", "Resp", "Temp"]):
        a.plot(g.ICULOS, X.loc[idx, v], color=BLUE); a.set_ylabel(v)
    ax[4].plot(g.ICULOS, g.p, color=RED); ax[4].axhline(g.thr.iloc[0], ls="--", c="k", label="alarm threshold")
    ax[4].set_ylabel("risk (6h)"); ax[4].set_yscale("log"); ax[4].legend()
    on = g.onset.iloc[0]
    for a in ax:
        a.axvline(on, color=RED, ls=":"); a.axvspan(on - 6, on, color=RED, alpha=.08)
    ax[4].set_xlabel("ICU hour"); ax[0].set_title(f"Fig 9. Median-case septic patient {pid}: red dotted = onset, shaded = 6h window")
    plt.tight_layout(); plt.savefig(C.FIGURES / "fig9_patient_timeline.png", dpi=150); plt.close()
    return pid


def fig_horizons(tab):
    hs = [3, 6, 12]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, m, t in zip(ax, ["auroc", "auprc"], ["AUROC", "AUPRC"]):
        v = [tab[f"main_h{h}"][m] for h in hs]
        a.bar([f"{h}h" for h in hs], [x[0] for x in v], yerr=[x[1] for x in v], color=BLUE, capsize=4)
        a.set_title(t); a.set_xlabel("prediction horizon")
    ax[1].set_title("AUPRC (prevalence: 0.4% / 0.9% / 1.7%)")
    plt.suptitle("Fig 10. Horizon comparison (mean ± std over 5 folds)"); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig10_horizons.png", dpi=150); plt.close()


def fig_models(tab):
    names = [("baseline_sirs_h6", "SIRS rule (>=2)"), ("baseline_logreg_h6", "Logistic regr."),
             ("no_rolling_h6", "LightGBM, no trends"), ("main_h6", "LightGBM + trends")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, m, t in zip(ax, ["auroc", "temporal_auroc"],
                       ["AUROC: which patients are at risk?", "Timing AUROC: is onset close? (0.5 = chance)"]):
        v = [tab[k][m] for k, _ in names]
        b = a.bar([n.replace(" ", chr(10), 1) for _, n in names], [x[0] for x in v], yerr=[x[1] for x in v], capsize=4,
                  color=[GREY, GREY, BLUE, BLUE])
        a.bar_label(b, fmt="%.2f", padding=3); a.set_title(t); a.set_ylim(0.4, 0.9)
    ax[1].axhline(0.5, ls="--", c="k")
    plt.suptitle("Fig 12. Model comparison, 6h target, 5-fold patient-grouped CV"); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig12_model_comparison.png", dpi=150); plt.close()


def fig_calibration(y, p):
    q = pd.qcut(p, 10, duplicates="drop")
    d = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"))
    brier = float(np.mean((p - y) ** 2)); ref = float(np.mean((y.mean() - y) ** 2))
    plt.figure(figsize=(5.5, 5)); lim = [min(d.pred.min(), d.obs.min()) * .7, max(d.pred.max(), d.obs.max()) * 1.4]
    plt.plot(lim, lim, "--", c=GREY, label="perfect calibration"); plt.plot(d.pred, d.obs, "o-", color=BLUE, label="model")
    plt.xscale("log"); plt.yscale("log"); plt.xlabel("Predicted risk (mean per decile)"); plt.ylabel("Observed rate")
    plt.title(f"Fig 13. Calibration (Brier {brier:.4f}, no-skill {ref:.4f})"); plt.legend(); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig13_calibration.png", dpi=150); plt.close()
    return dict(brier=brier, brier_no_skill=ref, deciles=d.round(5).reset_index(drop=True).to_dict("list"))


def fig_replica(tab):
    rp = json.loads((C.RESULTS / "replica_original.json").read_text())
    o, g = rp["original_random_row_split"], rp["same_pipeline_patient_split"]
    labels = ["Original code\n(random rows, RF)", "Original code,\npatient split", "Corrected\n(LightGBM, 6h)"]
    vals = [o["auroc"], g["auroc"], tab["main_h6"]["auroc"][0]]
    plt.figure(figsize=(8, 4.5)); b = plt.bar(labels, vals, color=[GREY, GREY, BLUE])
    plt.bar_label(b, fmt="%.3f"); plt.ylim(0.5, 0.9); plt.ylabel("AUROC (hourly)")
    plt.title("Fig 11. Leakage inflates the original estimate (not the same task: see report)")
    plt.tight_layout(); plt.savefig(C.FIGURES / "fig11_replica.png", dpi=150); plt.close()


def temporal_auroc(meta, p):
    """AUROC INSIDE septic patients only: rows <=6h before onset vs rows 7-48h before onset.
    ~0.5 means the model does not know *when* sepsis is near (it only ranks patients)."""
    from sklearn.metrics import roc_auc_score
    m = (meta.septic_patient.values == 1) & (meta.hrs_to_onset.values <= 48)
    return float(roc_auc_score(meta.y6.values[m], p[m]))


def main():
    C.FIGURES.mkdir(exist_ok=True)
    X, meta = cached_features()
    row_fold, pcode = make_folds(meta)
    o = pd.read_parquet(C.RESULTS / "oof_main_h6.parquet")
    p, thr_row, y = o.oof.values, o.thr.values, meta.y6.values
    tab = tables()
    ci = bootstrap_ci(y, p, thr_row, pcode, n=C.N_BOOTSTRAP, seed=C.SEED)
    fig_roc_pr(y, p, thr_row); det = fig_detection(meta, p, thr_row)
    top = fig_shap(X, meta, row_fold); pid = fig_patient(X, meta, p, thr_row, row_fold)
    fig_horizons(tab); fig_replica(tab); fig_models(tab); calib = fig_calibration(y, p)
    out = dict(cv_mean_std=tab, bootstrap_ci_h6={k: dict(mean=v[0], lo=v[1], hi=v[2]) for k, v in ci.items()},
               temporal_auroc_within_septic_h6=temporal_auroc(meta, p), calibration=calib,
               patient_timeline_id=pid, shap_top15=top.round(4).to_dict(),
               detection_rate_by_hours_to_onset={int(k): float(v) for k, v in det.items()},
               replica=json.loads((C.RESULTS / "replica_original.json").read_text()))
    (C.RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print("temporal AUROC within septic patients (h6):", out["temporal_auroc_within_septic_h6"])


if __name__ == "__main__":
    main()
