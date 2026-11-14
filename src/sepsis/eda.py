"""Exploratory data analysis: figures fig1..fig5 + printed audit. `python -m sepsis.eda`"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C
from .features import load_raw

RED, BLUE = "#C44E52", "#4C72B0"


def main():
    C.FIGURES.mkdir(exist_ok=True); C.RESULTS.mkdir(exist_ok=True)
    df = load_raw()
    pat = df.groupby("Patient_ID").SepsisLabel.max()
    los = df.groupby("Patient_ID").size()
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    onset = df[df.SepsisLabel == 1].groupby("Patient_ID").ICULOS.min()
    g = pd.DataFrame({"n": los, "s": pat})
    stats = dict(rows=len(df), patients=int(len(pat)), positive_rows_pct=float(df.SepsisLabel.mean() * 100),
                 septic_patients=int(pat.sum()), septic_patients_pct=float(pat.mean() * 100),
                 median_icu_hours=float(los.median()), cols_over_80_missing=int((miss > 80).sum()),
                 onset_median_iculos=float(onset.median()), onset_le_6h=int((onset <= 6).sum()),
                 nonsepsis_stay_p99=float(g[g.s == 0].n.quantile(.99)),
                 share_septic_if_stay_gt_60h=float(g[g.n > 60].s.mean()), patients_stay_gt_60h=int((g.n > 60).sum()))
    (C.RESULTS / "eda_stats.json").write_text(json.dumps(stats, indent=1)); print(stats)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (v0, v1, t) in zip(ax, [((df.SepsisLabel == 0).sum(), (df.SepsisLabel == 1).sum(), "Hourly rows (log scale)"),
                                   ((pat == 0).sum(), (pat == 1).sum(), "Patients")]):
        a.bar(["Non-sepsis", "Sepsis"], [v0, v1], color=[BLUE, RED]); a.set_title(t)
        for p in a.patches:
            a.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
    ax[0].set_yscale("log")
    plt.suptitle("Fig 1. Class imbalance: rows vs patients"); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig1_class_balance.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 9)); m = miss.sort_values()
    plt.barh(m.index, m.values, color=np.where(m.values > 80, RED, BLUE)); plt.axvline(80, ls="--", c="k")
    plt.xlabel("% missing"); plt.title("Fig 2. Missingness per variable (red > 80%)"); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig2_missingness.png", dpi=150); plt.close()

    pid = np.random.default_rng(0).choice(df.Patient_ID.unique())
    s = df[df.Patient_ID == pid].head(72).set_index("ICULOS")
    plt.figure(figsize=(12, 7))
    sns.heatmap(s[C.MEASURED].isna().T, cbar=False, cmap=[BLUE, "#EEEEEE"], yticklabels=True)
    plt.xlabel("ICU hour (ICULOS)"); plt.title("Fig 3. Measurement pattern of one patient (blue = measured, grey = missing)")
    plt.tight_layout(); plt.savefig(C.FIGURES / "fig3_measurement_pattern.png", dpi=150); plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for a, v in zip(axes.ravel(), ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp"]):
        sns.boxplot(x="SepsisLabel", y=v, data=df, ax=a, showfliers=False, palette=[BLUE, RED]); a.set_title(v)
    plt.suptitle("Fig 4. Raw vitals by SepsisLabel (no imputation)"); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig4_vitals_by_label.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    for lab, name, col in [(0, "Non-sepsis", BLUE), (1, "Sepsis", RED)]:
        plt.hist(los[pat == lab].clip(upper=200).values, bins=40, density=True, alpha=.6, label=name, color=col)
    plt.legend(); plt.xlabel("ICU hours (clipped at 200)")
    plt.title("Fig 5. ICU stay length: non-sepsis stays are cut at ~59h (leakage source)"); plt.tight_layout()
    plt.savefig(C.FIGURES / "fig5_icu_length.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
