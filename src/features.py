"""Shared feature engineering.

Both the synthetic generator and the response model import from here, so the
transform applied when *creating* the data is by construction the same one
applied when *fitting* it. If you add a driver, this is the only transform
code that needs to know.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Carryover: a[t] = x[t] + decay * a[t-1]. Media doesn't spend and vanish."""
    out = np.empty_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def apply_adstock(df: pd.DataFrame) -> pd.DataFrame:
    """Apply configured adstock within each category, in date order."""
    df = df.sort_values(["category", "date"]).copy()
    for name, spec in config.DRIVERS.items():
        decay = spec.get("adstock")
        if decay is None:
            continue
        df[name] = (
            df.groupby("category", observed=True)[name]
            .transform(lambda s: geometric_adstock(s.to_numpy(), decay))
        )
    return df


def log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Driver columns -> their log-space feature columns (prefixed `l_`)."""
    out = pd.DataFrame(index=df.index)
    for name, spec in config.DRIVERS.items():
        x = df[name].to_numpy(dtype=float)
        if spec["transform"] == "log":
            out[f"l_{name}"] = np.log(np.clip(x, 1e-6, None))
        elif spec["transform"] == "log1p":
            out[f"l_{name}"] = np.log1p(np.clip(x, 0.0, None))
        else:  # pragma: no cover - config guard
            raise ValueError(f"unknown transform for {name}: {spec['transform']}")
    return out


LOG_COLS = [f"l_{n}" for n in config.DRIVER_NAMES]


def build_design(df: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Full design matrix: log drivers + trend + category FE + month FE.

    `t0` anchors the trend so that history and future share one clock.
    """
    df = apply_adstock(df)
    X = log_features(df)

    months = (df["date"].dt.year - t0.year) * 12 + (df["date"].dt.month - t0.month)
    trend = months.to_numpy(dtype=float) / 12.0  # in years
    X["trend"] = trend

    for cat in config.CATEGORIES[1:]:  # drop first as reference level
        X[f"cat_{cat}"] = (df["category"] == cat).astype(float).to_numpy()
    for m in range(2, 13):  # January is the reference month
        X[f"mon_{m}"] = (df["date"].dt.month == m).astype(float).to_numpy()

    X.index = df.index
    return X


NUM_COLS = LOG_COLS + ["trend"]


def dummy_cols() -> list[str]:
    return [f"cat_{c}" for c in config.CATEGORIES[1:]] + [f"mon_{m}" for m in range(2, 13)]


DESIGN_COLS = NUM_COLS + dummy_cols()
