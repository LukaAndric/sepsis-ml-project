"""Tests for the timing study (sepsis.timing) and the onset study (sepsis.onset).
They check that the extra features never look into the future or across patients, that the SOFA points
follow the published cut-offs, and that the timing metrics pick the right hours."""
import numpy as np
import pandas as pd
import pytest

from sepsis.onset import burden, rule_timing, sofa_features
from sepsis.timing import VITALS, dynamics, timing_auc


def vitals_frame(n_a=30, n_b=10):
    rng = np.random.default_rng(0)
    rows = n_a + n_b
    X = pd.DataFrame({c: rng.normal(80, 5, rows) for c in VITALS})
    pid = pd.Series(["A"] * n_a + ["B"] * n_b)
    meta = pd.DataFrame({"Patient_ID": pid})
    return X, meta


# ---------- sepsis.timing ----------

def test_dynamics_are_causal():
    X, meta = vitals_frame()
    X2 = X.copy()
    X2.loc[20:29, "HR"] = 250.0                                   # change only the FUTURE of patient A
    a, b = dynamics(X, meta), dynamics(X2, meta)
    cols = [c for c in a.columns if c.startswith("HR")]
    pd.testing.assert_frame_equal(a.loc[:19, cols], b.loc[:19, cols])


def test_dynamics_do_not_cross_patients():
    X, meta = vitals_frame()
    d = dynamics(X, meta)
    first_b = d.iloc[30]                                          # first hour of patient B
    assert np.isnan(first_b["HR_d1"]) and np.isnan(first_b["HR_d3"])
    assert first_b["HR_dev_base"] == pytest.approx(0.0)           # baseline = its own first value
    assert first_b["HR_max12"] == pytest.approx(X.loc[30, "HR"])


def test_timing_auc_uses_only_septic_rows_within_48h():
    meta = pd.DataFrame({"septic_patient": [1, 1, 1, 1, 0, 0, 1],
                         "hrs_to_onset": [2, 3, 20, 30, np.nan, np.nan, 100]})
    y = np.array([1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.99, 0.99, 0.99])          # non-septic and >48 h rows must be ignored
    assert timing_auc(meta, y, p, np.arange(7)) == pytest.approx(1.0)


# ---------- sepsis.onset ----------

def sofa_frame(**cols):
    base = dict(FiO2=np.nan, O2Sat=np.nan, Platelets=np.nan, Bilirubin_total=np.nan, MAP=np.nan,
                Creatinine=np.nan, Resp=np.nan, SBP=np.nan)
    base.update(cols)
    n = max(len(np.atleast_1d(v)) for v in base.values())
    return pd.DataFrame({k: np.broadcast_to(np.atleast_1d(v).astype(float), n) for k, v in base.items()})


def test_sofa_points_follow_cutoffs():
    X = sofa_frame(FiO2=0.5, O2Sat=95, Platelets=40, Bilirubin_total=1.5, MAP=65, Creatinine=2.5, Resp=24, SBP=90)
    s = sofa_features(X, pd.Series(["A"])).iloc[0]
    assert s.sofa_resp == 3          # S/F = 190 -> 3 points
    assert s.sofa_coag == 3          # platelets 40 -> 3
    assert s.sofa_liver == 1         # bilirubin 1.5 -> 1
    assert s.sofa_cardio == 1        # MAP < 70 -> 1
    assert s.sofa_renal == 2         # creatinine 2.5 -> 2
    assert s.sofa == 10 and s.qsofa == 2


def test_fio2_in_percent_is_converted_and_missing_counts_zero():
    pct = sofa_features(sofa_frame(FiO2=50, O2Sat=95), pd.Series(["A"])).iloc[0]
    assert pct.sofa_resp == 3                                     # 50 % treated as 0.5
    empty = sofa_features(sofa_frame(), pd.Series(["A"])).iloc[0]
    assert empty.sofa == 0 and empty.qsofa == 0


def test_sofa_rise_is_causal_and_per_patient():
    plt_values = [200, 200, 90, 40, 300, 300]                     # patient A gets worse, then patient B starts
    X = sofa_frame(Platelets=plt_values)
    pid = pd.Series(["A", "A", "A", "A", "B", "B"])
    s = sofa_features(X, pid)
    assert s.sofa_rise24.tolist() == [0, 0, 2, 3, 0, 0]           # rise vs own minimum so far, reset for B
    X2 = sofa_frame(Platelets=[200, 200, 90, 10, 300, 300])       # change a later hour of A
    assert sofa_features(X2, pid).sofa_rise24.iloc[:3].tolist() == [0, 0, 2]


def test_rule_timing_picks_window_and_far_hours():
    hc = np.array([3, 5, 20, 30, 9, np.nan])                      # hours to clinical onset
    meta = pd.DataFrame({"septic_patient": [1, 1, 1, 1, 1, 0]})
    score = np.array([1.0, 0.9, 0.1, 0.2, 5.0, 9.0])              # hc=9 is outside both sets, NaN row is non-septic
    assert rule_timing(score, meta, hc, (0, 6)) == pytest.approx(1.0)
    assert rule_timing(-score, meta, hc, (0, 6)) == pytest.approx(0.0)


def test_burden_reaches_target_recall():
    meta = pd.DataFrame({"septic_patient": [1, 1, 1, 1, 1, 0, 0, 0, 0],
                         "Patient_ID": ["s"] * 5 + ["n1", "n1", "n2", "n2"]})
    pos = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], bool)
    drop = np.zeros(9, bool)
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.65, 0.0, 0.0, 0.0])
    b = burden(score, pos, drop, meta, recall=0.8)
    assert b["nonsepsis_patients_with_alarm"] == pytest.approx(0.5)   # only n1 crosses the threshold
    assert b["alarms_per_100_nonsepsis_hours"] == pytest.approx(25.0)


# ---------- sepsis.alarms ----------

def test_suppress_silences_repeats_and_resets_per_patient():
    from sepsis.alarms import suppress
    alarm = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1], bool)
    patient = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1])
    assert suppress(alarm, patient, 0).tolist() == alarm.tolist()            # no silence: every alarm fires
    assert suppress(alarm, patient, 2).astype(int).tolist() == [1, 0, 0, 1, 0, 0, 1, 1, 0]
    # patient 1 starts fresh at row 7 even though patient 0 had an alarm at row 6
