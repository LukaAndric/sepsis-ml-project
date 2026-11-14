"""Leakage tests. If one of these fails, the model results cannot be trusted."""
import numpy as np
import pandas as pd
import pytest

from sepsis import config as C
from sepsis.features import build_features


def make_raw(patients):
    """patients: dict id -> dict(n=hours, onset=ICULOS or None, hr=list/None). All other columns NaN."""
    frames = []
    for pid, spec in patients.items():
        n = spec["n"]
        d = pd.DataFrame(np.nan, index=range(n), columns=C.MEASURED)
        if "hr" in spec:
            d["HR"] = spec["hr"]
        d["ICULOS"] = np.arange(1, n + 1)
        d["HospAdmTime"] = -1.0
        d["Age"] = 60.0
        d["Gender"] = 1
        d["SepsisLabel"] = 0
        if spec.get("onset"):
            d.loc[d.ICULOS >= spec["onset"], "SepsisLabel"] = 1
        d["Patient_ID"] = pid
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def test_no_forward_fill_across_patients():
    raw = make_raw({"A": dict(n=5, hr=[80, 81, 82, 83, 84]),
                    "B": dict(n=5, hr=[np.nan] * 5)})
    X, meta = build_features(raw)
    b = X[meta.Patient_ID == "B"]
    assert b.HR.isna().all(), "HR of patient A leaked into patient B via ffill"


def test_forward_fill_within_patient_and_hours_since():
    raw = make_raw({"A": dict(n=5, hr=[80, np.nan, np.nan, 90, np.nan])})
    X, _ = build_features(raw)
    assert X.HR.tolist() == [80, 80, 80, 90, 90]
    assert X.HR_hrs_since.tolist() == [0, 1, 2, 0, 1]


def test_target_only_before_onset_and_within_horizon():
    raw = make_raw({"S": dict(n=30, onset=20)})          # onset at ICULOS 20
    X, meta = build_features(raw)
    assert (meta.SepsisLabel == 0).all() and meta.ICULOS.max() == 19   # septic hours removed
    y6 = meta.set_index("ICULOS").y6
    assert y6.loc[14:19].eq(1).all() and y6.loc[:13].eq(0).all()      # 6h before onset (hours 14..19)


def test_early_onset_patients_are_dropped():
    raw = make_raw({"E": dict(n=30, onset=4), "S": dict(n=30, onset=20)})
    _, meta = build_features(raw)
    assert "E" not in set(meta.Patient_ID) and "S" in set(meta.Patient_ID)


def test_leaky_columns_not_in_features():
    raw = make_raw({"A": dict(n=10)})
    X, _ = build_features(raw)
    assert not set(C.LEAKY_COLS) & set(X.columns) and "Hour" not in X.columns


def test_nonseptic_patients_have_no_positive_targets():
    raw = make_raw({"N": dict(n=40)})
    _, meta = build_features(raw)
    assert meta[["y3", "y6", "y12"]].sum().sum() == 0


def test_rolling_features_are_causal():
    """Changing FUTURE values must not change features of earlier hours."""
    hr = list(np.arange(60.0, 90.0))
    a = make_raw({"A": dict(n=30, hr=hr)})
    hr2 = hr.copy(); hr2[20:] = [200.0] * 10               # alter the future from hour 21 on
    b = make_raw({"A": dict(n=30, hr=hr2)})
    Xa, _ = build_features(a); Xb, _ = build_features(b)
    cols = [c for c in Xa.columns if c.startswith("HR")]
    pd.testing.assert_frame_equal(Xa.loc[:19, cols], Xb.loc[:19, cols])


def test_rolling_not_carried_across_patients():
    raw = make_raw({"A": dict(n=10, hr=[100.0] * 10), "B": dict(n=10, hr=[50.0] * 10)})
    X, meta = build_features(raw)
    b0 = X[meta.Patient_ID == "B"].iloc[0]
    assert b0.HR_mean6 == 50.0 and np.isnan(b0.HR_delta6)


def test_inference_features_match_training_features():
    """Serving must compute exactly the same features as training (no train/serve skew)."""
    raw = make_raw({"A": dict(n=25, hr=list(np.random.default_rng(0).normal(90, 10, 25)))})
    raw.loc[5:9, "Lactate"] = 2.0
    X, _ = build_features(raw)
    hist = raw.iloc[:15][["ICULOS", "Age", "Gender", "HR", "Lactate"]]
    row = __import__("sepsis.features", fromlist=["x"]).inference_features(hist)
    pd.testing.assert_series_equal(row.iloc[0], X.iloc[14], check_names=False)
