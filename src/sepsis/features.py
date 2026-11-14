"""Leak-free, per-patient feature and target construction.

Rules (each one is covered by a test in tests/test_features.py):
  * every temporal operation (ffill, hours-since, onset) is done PER PATIENT, never across patients
  * ICULOS / Hour / HospAdmTime are never features (see config.LEAKY_COLS)
  * trend features (6h mean/std/delta of vitals) use only past and current hours
  * never-measured values stay NaN (LightGBM handles NaN natively)
  * target(H) = "sepsis onset within the next H hours", only for hours where the patient is
    not yet septic; patients with onset <= MIN_ONSET hours are dropped (no history to warn from)
"""
import numpy as np
import pandas as pd

from . import config as C


def load_raw(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or C.data_path(), index_col=0).reset_index(drop=True)
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    return df


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient features from rows sorted by (Patient_ID, ICULOS). Causal: uses only past/current hours.
    Used by BOTH training (build_features) and serving (inference_features) to avoid train/serve skew."""
    pid = df.Patient_ID
    g = df.groupby("Patient_ID", sort=False)
    filled = g[C.MEASURED].ffill()
    feats = {}
    for c in C.MEASURED:
        feats[c] = filled[c].astype("float32").values
        t = df.ICULOS.where(df[c].notna())
        last = t.groupby(pid, sort=False).ffill()
        feats[c + "_hrs_since"] = (df.ICULOS - last).astype("float32").values   # 0 == measured now
    for c in C.ROLLING:                                                          # trend features (last 6 h)
        s = filled[c]
        r = s.groupby(pid, sort=False).rolling(C.ROLL_WINDOW, min_periods=1)
        feats[c + "_mean6"] = r.mean().to_numpy(dtype="float32")
        feats[c + "_std6"] = (s.groupby(pid, sort=False).rolling(C.ROLL_WINDOW, min_periods=2).std()
                              .to_numpy(dtype="float32"))
        feats[c + "_delta6"] = (s - s.groupby(pid, sort=False).shift(C.ROLL_WINDOW)).astype("float32").values
    feats["Age"] = df.Age.astype("float32").values
    feats["Gender"] = df.Gender.astype("float32").values
    return pd.DataFrame(feats)


def build_features(raw: pd.DataFrame, min_onset: int = C.MIN_ONSET, horizons=C.HORIZONS,
                   keep_labeled: bool = False):
    """Return (X, meta). X: model features (float32). meta: ids, onset and one y{H} per horizon.

    keep_labeled=True builds the PhysioNet-style task: keep every hour, target y0 = SepsisLabel of that hour
    (use with min_onset=0 so that no patient is dropped)."""
    df = raw.sort_values(["Patient_ID", "ICULOS"], kind="stable").reset_index(drop=True)

    onset = df[df.SepsisLabel == 1].groupby("Patient_ID").ICULOS.min()
    df["onset"] = df.Patient_ID.map(onset)               # NaN for never-septic patients
    df = df[~df.Patient_ID.isin(onset[onset <= min_onset].index)].reset_index(drop=True)

    X = _featurize(df)
    meta = pd.DataFrame({
        "Patient_ID": df.Patient_ID.values,
        "ICULOS": df.ICULOS.values,
        "HospAdmTime": df.HospAdmTime.values,
        "SepsisLabel": df.SepsisLabel.values,
        "onset": df.onset.values,
        "septic_patient": df.onset.notna().astype("int8").values,
    })
    meta["hrs_to_onset"] = meta.onset - meta.ICULOS
    for h in horizons:
        meta[f"y{h}"] = ((meta.hrs_to_onset > 0) & (meta.hrs_to_onset <= h)).astype("int8")

    meta["y0"] = meta.SepsisLabel.astype("int8")         # PhysioNet-style target (label of the current hour)
    if keep_labeled:
        return X, meta
    keep = meta.SepsisLabel.values == 0                  # only hours before the patient is septic
    return X[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)


def inference_features(history: pd.DataFrame) -> pd.DataFrame:
    """One patient's hourly history (columns: ICULOS, Age, Gender + any measured variables; missing = NaN)
    -> a single-row feature frame for the LATEST hour, computed exactly as in training."""
    h = history.copy()
    for c in C.MEASURED:
        if c not in h:
            h[c] = np.nan
    h["Patient_ID"] = "p"
    h = h.sort_values("ICULOS", kind="stable").reset_index(drop=True)
    return _featurize(h).iloc[[-1]].reset_index(drop=True)


def cached_features(force: bool = False, variant: str = "main"):
    """Build once, then load from results/ (this is what makes reruns and Colab fast).
    variant "main": early-warning task; variant "all": PhysioNet-style task (all hours, y0 = SepsisLabel)."""
    C.RESULTS.mkdir(exist_ok=True)
    suffix = "" if variant == "main" else "_all"
    px, pm = C.RESULTS / f"X{suffix}.parquet", C.RESULTS / f"meta{suffix}.parquet"
    if px.exists() and pm.exists() and not force:
        return pd.read_parquet(px), pd.read_parquet(pm)
    if variant == "main":
        X, meta = build_features(load_raw())
    else:
        X, meta = build_features(load_raw(), min_onset=0, keep_labeled=True)
    X.to_parquet(px)
    meta.to_parquet(pm)
    return X, meta
