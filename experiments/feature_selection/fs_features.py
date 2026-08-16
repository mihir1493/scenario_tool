"""`src/features.py`, generalised to an arbitrary driver spec.

The shipped feature builder reads `config.DRIVERS` directly, so it can only ever
build the nine shipped drivers. This experiment needs to build a *candidate* set
that includes decoys, and then per-category subsets of it, so the spec becomes an
argument instead of a global. The transforms themselves are unchanged and the
adstock kernel is imported from `src.features` rather than reimplemented -- the
point is to widen the interface, not to fork the maths.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.features import geometric_adstock  # noqa: E402

MONTH_COLS = [f"mon_{m}" for m in range(2, 13)]
CAT_COLS = [f"cat_{c}" for c in config.CATEGORIES[1:]]


def apply_adstock(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply each driver's configured carryover within category, in date order."""
    df = df.sort_values(["category", "date"]).copy()
    for name, s in spec.items():
        decay = s.get("adstock")
        if decay is None or name not in df.columns:
            continue
        df[name] = (
            df.groupby("category", observed=True)[name]
            .transform(lambda x: geometric_adstock(x.to_numpy(), decay))
        )
    return df


def log_features(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Driver columns -> their log-space feature columns (prefixed `l_`)."""
    out = pd.DataFrame(index=df.index)
    for name, s in spec.items():
        x = df[name].to_numpy(dtype=float)
        if s["transform"] == "log":
            out[f"l_{name}"] = np.log(np.clip(x, 1e-6, None))
        elif s["transform"] == "log1p":
            out[f"l_{name}"] = np.log1p(np.clip(x, 0.0, None))
        else:  # pragma: no cover - spec guard
            raise ValueError(f"unknown transform for {name}: {s['transform']}")
    return out


def log_cols(names) -> list[str]:
    return [f"l_{n}" for n in names]


def build_design(df: pd.DataFrame, spec: dict, t0: pd.Timestamp) -> pd.DataFrame:
    """Log drivers + trend + category FE + month FE, for every driver in `spec`.

    Callers slice the columns they want: a pooled fit takes the category dummies,
    a per-category fit does not, and a selected-feature fit takes only its own
    driver columns. `t0` anchors the trend so history and future share one clock.
    """
    df = apply_adstock(df, spec)
    X = log_features(df, spec)

    months = (df["date"].dt.year - t0.year) * 12 + (df["date"].dt.month - t0.month)
    X["trend"] = months.to_numpy(dtype=float) / 12.0  # in years

    for cat in config.CATEGORIES[1:]:  # first category is the reference level
        X[f"cat_{cat}"] = (df["category"] == cat).astype(float).to_numpy()
    for m in range(2, 13):  # January is the reference month
        X[f"mon_{m}"] = (df["date"].dt.month == m).astype(float).to_numpy()

    X.index = df.index
    return X


def calendar_block(df: pd.DataFrame, t0: pd.Timestamp) -> np.ndarray:
    """Intercept + trend + month dummies, as a plain array.

    The selection engine partials this out of both the response and every driver
    before scoring anything, so a driver that only looks predictive because it
    drifts or peaks in December gets no credit for it.
    """
    df = df.sort_values(["category", "date"])
    months = (df["date"].dt.year - t0.year) * 12 + (df["date"].dt.month - t0.month)
    cols = [np.ones(len(df)), months.to_numpy(dtype=float) / 12.0]
    for m in range(2, 13):
        cols.append((df["date"].dt.month == m).astype(float).to_numpy())
    return np.column_stack(cols)
