"""Fewer false alarms without new data: after an alarm, the same patient gets no new alarm for N hours.
Uses the out-of-fold predictions of the main 6 h model and the threshold of each fold (80% recall on validation).

    python -m sepsis.alarms        -> results/alarm_suppression.csv, figures/fig17_alarm_suppression.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .features import cached_features

HOURS = (0, 2, 4, 6, 12, 24)


def suppress(alarm: np.ndarray, patient: np.ndarray, hours: int) -> np.ndarray:
    """Rows must be sorted by patient and time. An alarm 'fires' only if no alarm fired for the same
    patient in the previous `hours` rows. hours=0 means every alarm hour fires."""
    fired = np.zeros(len(alarm), bool)
    last = -10**9
    prev = None
    for i in range(len(alarm)):
        if patient[i] != prev:
            prev, last = patient[i], -10**9
        if alarm[i] and i - last > hours:
            fired[i] = True
            last = i
    return fired


def evaluate(fired, meta, y):
    non = meta.septic_patient.values == 0
    sep = ~non
    pid = meta.Patient_ID.values
    win = sep & (y == 1)
    pre24 = sep & (meta.hrs_to_onset.values <= 24)
    det_win = pd.Series(fired[win]).groupby(pid[win]).max()
    det_24 = pd.Series(fired[pre24]).groupby(pid[pre24]).max()
    return dict(alarms_per_nonsepsis_patient_day=float(fired[non].sum() / (non.sum() / 24)),
                nonsepsis_patients_with_alarm=float(pd.Series(fired[non]).groupby(pid[non]).max().mean()),
                septic_alarmed_in_6h_window=float(det_win.mean()),
                septic_alarmed_in_24h_before=float(det_24.mean()),
                share_of_alarms_in_6h_window=float((fired & win).sum() / max(fired.sum(), 1)))


def main():
    _, meta = cached_features()
    o = pd.read_parquet(C.RESULTS / "oof_main_h6.parquet")
    y = meta.y6.values
    alarm = (o.oof.values >= o.thr.values)
    pid = pd.factorize(meta.Patient_ID)[0]
    rows = []
    for h in HOURS:
        r = dict(silence_hours=h, **evaluate(suppress(alarm, pid, h), meta, y))
        rows.append(r)
        print(r, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(C.RESULTS / "alarm_suppression.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(d.silence_hours, d.alarms_per_nonsepsis_patient_day, "o-", color="#4C72B0")
    ax[0].set(xlabel="hours of silence after an alarm", ylabel="false alarms per patient per day",
              title="False alarms (patients without sepsis)")
    for c, lab, col in [("septic_alarmed_in_6h_window", "alarm inside the 6 h window", "#C44E52"),
                        ("septic_alarmed_in_24h_before", "alarm in the 24 h before onset", "#DD8452")]:
        ax[1].plot(d.silence_hours, d[c] * 100, "o-", label=lab, color=col)
    ax[1].set(xlabel="hours of silence after an alarm", ylabel="% of sepsis patients", ylim=(0, 100),
              title="Sepsis patients that still get an alarm")
    ax[1].legend()
    plt.suptitle("Fig 17. Silencing repeated alarms (main 6 h model, threshold = 80% recall on validation)")
    plt.tight_layout(); plt.savefig(C.FIGURES / "fig17_alarm_suppression.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
