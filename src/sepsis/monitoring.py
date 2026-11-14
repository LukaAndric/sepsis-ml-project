"""Lightweight drift monitoring: PSI of logged inputs and risk scores vs the training distribution.

    python -m sepsis.monitoring --log logs/predictions.csv

PSI < 0.10 stable | 0.10-0.25 moderate drift | > 0.25 major drift (rule of thumb).
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import config as C

MIN_ROWS = 200
EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None); a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _actual_fractions(values: np.ndarray, ref: dict) -> np.ndarray:
    """Same bins as in training; NaN rate is NOT a bin edge, so it is folded in via nan_frac comparison below."""
    v = np.asarray(values, dtype=float)
    ok = v[~np.isnan(v)]
    edges = np.asarray(ref["edges"])
    return np.bincount(np.searchsorted(edges, ok, side="right"), minlength=len(edges) + 1) / max(len(v), 1)


def feature_psi(log: pd.DataFrame, ref_stats: dict) -> pd.DataFrame:
    rows = []
    for name, ref in ref_stats.items():
        col = "risk" if name == "_risk" else name
        if col not in log:
            continue
        exp = np.append(ref["frac"], ref["nan_frac"])                   # last bin = missing
        v = log[col].to_numpy(dtype=float)
        act = np.append(_actual_fractions(v, ref), np.isnan(v).mean())
        score = psi(exp, act)
        rows.append(dict(feature=col, psi=score, status="major" if score > .25 else "moderate" if score > .10 else "stable",
                         ref_missing=ref["nan_frac"], live_missing=float(np.isnan(v).mean())))
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def main(log_path=None, out_path=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(C.ROOT / "logs" / "predictions.csv"))
    ap.add_argument("--out", default=str(C.RESULTS / "drift_report.csv"))
    args = ap.parse_args() if log_path is None else argparse.Namespace(log=log_path, out=out_path)
    ref = json.loads((C.ROOT / "models" / "reference_stats.json").read_text())
    log = pd.read_csv(args.log)
    if len(log) < MIN_ROWS:
        print(f"WARNING: only {len(log)} logged predictions (< {MIN_ROWS}); PSI is unreliable.")
    rep = feature_psi(log, ref)
    rep.to_csv(args.out, index=False)
    print(rep.head(10).to_string(index=False))
    print(f"\n{(rep.status == 'major').sum()} features with major drift, {(rep.status == 'moderate').sum()} moderate.")
    return rep


if __name__ == "__main__":
    main()
