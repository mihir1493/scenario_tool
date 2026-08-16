"""Forecast each category's SELECTED drivers forward two years.

Only the selected ones. A driver that ice cream dropped is never forecast for ice
cream, which is most of the practical payoff of doing selection first: the
forecasting work, and the ongoing maintenance of those forecasts, scales with the
drivers you actually use rather than with the drivers you happen to collect.

Each (category, driver) pair gets its own model, so the same driver can be
forecast differently for different categories -- ground coffee's price path and
baby formula's price path are not the same series and are not treated as one.

Prophet if it is installed, a transparent trend + week-of-year fallback if not.
The fallback is not a token: it is deterministic, fast, and good enough for the
smooth driver series here, and it keeps the pipeline runnable anywhere.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

import wk_config as cfg

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


def future_weeks(n: int = cfg.FORECAST_WEEKS) -> pd.DatetimeIndex:
    start = pd.Timestamp(cfg.HISTORY_END) + pd.Timedelta(weeks=1)
    return pd.date_range(start, periods=n, freq="W-SUN")


def _prophet_forecast(dates, y, horizon) -> np.ndarray:
    m = Prophet(
        yearly_seasonality=8,  # weekly data resolves a richer annual shape
        weekly_seasonality=False,  # the observations ARE weeks
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    m.fit(pd.DataFrame({"ds": pd.DatetimeIndex(dates), "y": y}))
    fut = m.make_future_dataframe(periods=horizon, freq="W-SUN")
    return m.predict(fut).tail(horizon)["yhat"].to_numpy()


def _fallback_forecast(dates, y, horizon) -> np.ndarray:
    """Linear trend over the last two years plus a week-of-year seasonal index."""
    n = len(y)
    t = np.arange(n, dtype=float)
    window = min(104, n)
    slope, intercept = np.polyfit(t[-window:], y[-window:], 1)

    resid = y - (intercept + slope * t)
    wk = pd.DatetimeIndex(dates).isocalendar().week.to_numpy(dtype=int)
    index = {w: resid[wk == w].mean() for w in np.unique(wk)}

    fut = future_weeks(horizon)
    fut_wk = fut.isocalendar().week.to_numpy(dtype=int)
    ft = np.arange(n, n + horizon, dtype=float)
    seasonal = np.array([index.get(w, 0.0) for w in fut_wk])
    return intercept + slope * ft + seasonal


def _keep_plausible(name: str, history: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Hold the forecast inside an envelope around what has actually been observed.

    Two years is a long extrapolation for a weekly series. Without this, a driver
    with a mild upward trend can drift somewhere no planner would sign off on, and
    the volume forecast inherits it.
    """
    lo, hi = float(history.min()), float(history.max())
    span = max(hi - lo, 1e-9)
    fc = np.clip(fc, lo - 0.30 * span, hi + 0.30 * span)

    if cfg.DRIVERS[name]["transform"] == "log":
        fc = np.clip(fc, max(lo * 0.5, 1e-6), None)  # strictly positive for log()
    else:
        fc = np.clip(fc, 0.0, None)
    if name == "distribution_acv":
        fc = np.clip(fc, 5.0, 99.0)
    if name in ("promo_share", "feature_display"):
        fc = np.clip(fc, 0.0, 100.0)
    return fc


def forecast_one(dates, y, name: str, horizon: int) -> np.ndarray:
    fit = _prophet_forecast if HAS_PROPHET else _fallback_forecast
    try:
        fc = fit(dates, y, horizon)
    except Exception:
        fc = _fallback_forecast(dates, y, horizon)
    return _keep_plausible(name, np.asarray(y, dtype=float), fc)


def forecast_selected(panel: pd.DataFrame, selected: dict[str, list[str]],
                      horizon: int = cfg.FORECAST_WEEKS,
                      verbose: bool = True) -> pd.DataFrame:
    """Forecast every (category, selected driver) pair.

    Returns a long frame `date x category x driver columns`. Drivers a category did
    not select come back as NaN for that category -- explicitly absent rather than
    silently zero, so nothing downstream can quietly treat "not used" as "zero".
    """
    fut = future_weeks(horizon)
    rows = pd.DataFrame(
        [(d, c) for c in cfg.CATEGORIES for d in fut], columns=["date", "category"]
    )
    for name in cfg.DRIVER_NAMES:
        rows[name] = np.nan

    for cat in cfg.CATEGORIES:
        hist = panel[panel["category"] == cat].sort_values("date")
        mask = (rows["category"] == cat).to_numpy()
        drivers = selected[cat]
        if verbose:
            print(f"  {cat:<20} {len(drivers)} drivers: {', '.join(drivers)}")
        for name in drivers:
            fc = forecast_one(hist["date"], hist[name].to_numpy(dtype=float),
                              name, horizon)
            rows.loc[mask, name] = fc

    return rows.sort_values(["category", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    import wk_model as mdl

    panel = pd.read_csv(cfg.PANEL_CSV, parse_dates=["date"])
    models = mdl.load()
    selected = {c: m.drivers for c, m in models.items()}
    print(f"Forecasting {cfg.FORECAST_WEEKS} weeks (prophet={HAS_PROPHET})")
    fc = forecast_selected(panel, selected)
    fc.to_csv(cfg.DRIVER_FORECAST_CSV, index=False)
    print(f"\n{len(fc)} rows -> {cfg.DRIVER_FORECAST_CSV}")
