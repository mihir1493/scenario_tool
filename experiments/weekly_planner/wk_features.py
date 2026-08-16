"""Feature engineering. Four functions, and the generator and the model both use them.

Building the data and fitting the data go through the same transforms on purpose:
if the two ever disagree, the model is chasing an artefact of its own code.

The design matrix for one category is:

    log(driver) for each SELECTED driver
  + linear trend (in years)
  + Fourier pairs for the yearly cycle, K of them
  + holiday-week dummies

There are no category dummies and no shared terms, because each category is fitted
completely on its own. That is the whole architecture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import wk_config as cfg


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Carryover: a[t] = x[t] + decay * a[t-1]. Media does not spend and vanish.

    decay=0 returns the input unchanged, which is how a category says "my media
    has no carryover" without needing a separate code path.
    """
    if decay <= 0:
        return np.asarray(x, dtype=float)
    out = np.empty(len(x), dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def apply_adstock(df: pd.DataFrame, decay: float) -> pd.DataFrame:
    """Apply one category's validated decay to every adstock-eligible driver.

    `df` must be a single category, sorted by date. One decay per category rather
    than one per driver: with 241 weeks, fitting a separate carryover rate for TV
    and for digital is more parameters than the data can support.
    """
    out = df.sort_values("date").copy()
    for name in cfg.ADSTOCK_DRIVERS:
        if name in out.columns:
            out[name] = adstock(out[name].to_numpy(dtype=float), decay)
    return out


def log_driver(values: np.ndarray, name: str) -> np.ndarray:
    """One driver column -> its log-space feature."""
    x = np.asarray(values, dtype=float)
    if cfg.DRIVERS[name]["transform"] == "log":
        return np.log(np.clip(x, 1e-6, None))
    return np.log1p(np.clip(x, 0.0, None))


def fourier_terms(dates: pd.Series, k: int) -> pd.DataFrame:
    """K sin/cos pairs for the yearly cycle, keyed on day-of-year.

    Day-of-year rather than week number so 53-week years do not shift the phase,
    which matters when the forecast runs two years past the history.
    """
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy(dtype=float)
    out = pd.DataFrame(index=range(len(doy)))
    for j in range(1, k + 1):
        out[f"sin{j}"] = np.sin(2 * np.pi * j * doy / 365.25)
        out[f"cos{j}"] = np.cos(2 * np.pi * j * doy / 365.25)
    return out


HOLIDAY_COLS = ["hol_yearend", "hol_thanksgiving", "hol_july4", "hol_easter"]


def holiday_flags(dates: pd.Series) -> pd.DataFrame:
    """Dummies for the weeks that behave differently regardless of any driver.

    Defined on ISO week number, which is what a weekly retail calendar is
    actually planned on. These are controls, not levers -- they are not drivers
    and never appear in a scenario.
    """
    wk = pd.DatetimeIndex(dates).isocalendar().week.to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "hol_yearend": np.isin(wk, [51, 52, 53, 1]).astype(float),
            "hol_thanksgiving": np.isin(wk, [47, 48]).astype(float),
            "hol_july4": np.isin(wk, [26, 27]).astype(float),
            "hol_easter": np.isin(wk, [13, 14, 15]).astype(float),
        },
        index=range(len(wk)),
    )


def build_design(df: pd.DataFrame, drivers: list[str], k: int, decay: float,
                 t0: pd.Timestamp) -> pd.DataFrame:
    """The full design matrix for ONE category.

    `drivers` is that category's selected list -- a different length and a
    different membership for every category, which is the point.
    `t0` anchors the trend so history and forecast share one clock.
    """
    df = apply_adstock(df, decay).reset_index(drop=True)

    X = pd.DataFrame(index=range(len(df)))
    for name in drivers:
        X[f"l_{name}"] = log_driver(df[name].to_numpy(), name)

    weeks = (pd.DatetimeIndex(df["date"]) - t0).days.to_numpy() / 365.25
    X["trend"] = weeks

    for col, vals in fourier_terms(df["date"], k).items():
        X[col] = vals.to_numpy()
    for col, vals in holiday_flags(df["date"]).items():
        X[col] = vals.to_numpy()
    return X


def driver_cols(drivers: list[str]) -> list[str]:
    return [f"l_{d}" for d in drivers]


def fourier_cols(k: int) -> list[str]:
    return [f"{f}{j}" for j in range(1, k + 1) for f in ("sin", "cos")]


def control_cols(k: int) -> list[str]:
    """Everything in the design that is not a driver: trend, seasonality, holidays.

    Controls are never penalised and never appear in a scenario. They explain when
    volume happens; drivers explain why.
    """
    return ["trend"] + fourier_cols(k) + HOLIDAY_COLS
