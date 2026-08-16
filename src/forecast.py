"""Forecast every driver forward, independently, with Prophet.

Category-scoped drivers get one model per (category, driver). Macro drivers are
shared, so they get one model each and are broadcast to all categories.

Prophet is the default; if it isn't installed the module falls back to a
trend + monthly-seasonal-index forecaster so the pipeline still runs.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

import config

for _name in ("cmdstanpy", "prophet"):
    _log = logging.getLogger(_name)
    _log.setLevel(logging.CRITICAL)
    _log.handlers.clear()
    _log.propagate = False
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet

    HAS_PROPHET = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PROPHET = False


def future_dates(n: int = config.FORECAST_MONTHS) -> pd.DatetimeIndex:
    start = pd.Timestamp(config.HISTORY_END) + pd.DateOffset(months=1)
    return pd.date_range(start, periods=n, freq="MS")


def _fit_prophet(dates: pd.Series, y: np.ndarray, horizon: int) -> np.ndarray:
    m = Prophet(
        yearly_seasonality=6,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        interval_width=0.8,
    )
    m.fit(pd.DataFrame({"ds": dates.to_numpy(), "y": y}))
    fut = m.make_future_dataframe(periods=horizon, freq="MS")
    fc = m.predict(fut).tail(horizon)
    return fc["yhat"].to_numpy()


def _fit_fallback(dates: pd.Series, y: np.ndarray, horizon: int) -> np.ndarray:
    """Linear trend on the last 36 months + additive month-of-year index."""
    n = len(y)
    t = np.arange(n, dtype=float)
    win = min(36, n)
    slope, intercept = np.polyfit(t[-win:], y[-win:], 1)
    resid = y - (intercept + slope * t)
    months = pd.DatetimeIndex(dates).month.to_numpy()
    idx = {m: resid[months == m].mean() for m in range(1, 13)}
    fut = future_dates(horizon)
    ft = np.arange(n, n + horizon, dtype=float)
    return intercept + slope * ft + np.array([idx.get(m, 0.0) for m in fut.month])


def _clip_to_plausible(name: str, hist: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Keep forecasts inside a plausible envelope around observed history."""
    lo, hi = hist.min(), hist.max()
    span = max(hi - lo, 1e-9)
    fc = np.clip(fc, lo - 0.35 * span, hi + 0.35 * span)
    if config.DRIVERS[name]["transform"] == "log":
        fc = np.clip(fc, max(lo * 0.4, 1e-6), None)
    else:
        fc = np.clip(fc, 0.0, None)
    if name == "distribution_acv":
        fc = np.clip(fc, 5.0, 98.0)
    return fc


def _forecast_series(dates: pd.Series, y: np.ndarray, name: str, horizon: int) -> np.ndarray:
    fit = _fit_prophet if HAS_PROPHET else _fit_fallback
    try:
        fc = fit(dates, y, horizon)
    except Exception:
        fc = _fit_fallback(dates, y, horizon)
    return _clip_to_plausible(name, y, fc)


def forecast_drivers(panel: pd.DataFrame, horizon: int = config.FORECAST_MONTHS,
                     verbose: bool = True) -> pd.DataFrame:
    """Return a future panel: date x category x every driver."""
    fut = future_dates(horizon)
    cats = sorted(config.CATEGORIES)
    out = pd.DataFrame(
        [(d, c) for c in cats for d in fut], columns=["date", "category"]
    )

    # Macro drivers: one shared path, broadcast to every category.
    macro_hist = panel.groupby("date", as_index=False)[config.MACRO_DRIVERS].first()
    for name in config.MACRO_DRIVERS:
        if verbose:
            print(f"  macro/{name}")
        fc = _forecast_series(macro_hist["date"], macro_hist[name].to_numpy(), name, horizon)
        out[name] = out["date"].map(dict(zip(fut, fc)))

    # Category drivers: one model per (category, driver).
    for name in config.COMMERCIAL_DRIVERS:
        if verbose:
            print(f"  category/{name}")
        vals = {}
        for cat in cats:
            sub = panel[panel["category"] == cat].sort_values("date")
            vals[cat] = _forecast_series(sub["date"], sub[name].to_numpy(), name, horizon)
        out[name] = np.concatenate([vals[c] for c in cats])

    return out.sort_values(["category", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    panel = pd.read_csv(config.PANEL_CSV, parse_dates=["date"])
    print(f"Forecasting {config.FORECAST_MONTHS} months (prophet={HAS_PROPHET})")
    fc = forecast_drivers(panel)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    fc.to_csv(config.DRIVER_FORECAST_CSV, index=False)
    print(fc.head())
    print(f"\n{len(fc)} rows -> {config.DRIVER_FORECAST_CSV}")
