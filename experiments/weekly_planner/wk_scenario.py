"""Baseline, scenario, and decomposition for one category.

The scenario loop the app implements is deliberately file-shaped rather than
slider-shaped:

    download  ->  a CSV of this category's drivers, history and forecast together
    edit      ->  in Excel, by the person who actually knows the plan
    upload    ->  the edited forecast rows become the scenario

Sliders can only express "everything moves by X%". A planner's actual plan is
"price holds until March, then up 4%, and we pull the May display event forward
two weeks" -- which is a column of numbers, not a percentage.

History is always prepended before predicting so media carryover crosses the
forecast seam correctly. Predicting the future rows on their own would restart
adstock at zero and understate the first weeks of every media plan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import wk_config as cfg
import wk_features as feat
from wk_model import CategoryModel


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------
def _predict_future(model: CategoryModel, history: pd.DataFrame,
                    future: pd.DataFrame) -> pd.DataFrame:
    """Predict the future weeks with history prepended, then return only the future.

    Also returns each driver's log level, which the waterfall needs and which must
    come from the same adstocked frame the prediction used.
    """
    cols = ["date"] + model.drivers
    combined = pd.concat([history[cols], future[cols]], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    combined["volume_pred"] = model.predict(combined)
    adstocked = feat.apply_adstock(combined, model.adstock_decay).reset_index(drop=True)
    for d in model.drivers:
        combined[f"l_{d}"] = feat.log_driver(adstocked[d].to_numpy(), d)

    cutoff = pd.Timestamp(cfg.HISTORY_END)
    return combined[combined["date"] > cutoff].reset_index(drop=True)


def baseline_forecast(model: CategoryModel, panel: pd.DataFrame,
                      driver_fc: pd.DataFrame) -> pd.DataFrame:
    """This category's two-year baseline volume, on the forecast driver paths."""
    cat = model.category
    hist = panel[panel["category"] == cat].sort_values("date")
    fut = driver_fc[driver_fc["category"] == cat].sort_values("date")
    return _predict_future(model, hist, fut)


def baseline_table(model: CategoryModel, panel: pd.DataFrame,
                   driver_fc: pd.DataFrame) -> pd.DataFrame:
    """The downloadable file: history and forecast, one row per week.

    Only this category's SELECTED drivers appear as columns. A planner cannot edit
    a driver the model does not use, because the tool would silently ignore it and
    that is worse than not offering it.
    """
    cat = model.category
    hist = panel[panel["category"] == cat].sort_values("date").reset_index(drop=True)
    fut = baseline_forecast(model, panel, driver_fc)

    hist_out = pd.DataFrame({"date": hist["date"], "category": cat, "period": "history",
                             "volume_actual": hist["volume"].to_numpy(),
                             "volume_baseline": model.predict(hist)})
    fut_out = pd.DataFrame({"date": fut["date"], "category": cat, "period": "forecast",
                            "volume_actual": np.nan,
                            "volume_baseline": fut["volume_pred"].to_numpy()})
    for d in model.drivers:
        hist_out[d] = hist[d].to_numpy()
        fut_out[d] = fut[d].to_numpy()

    return pd.concat([hist_out, fut_out], ignore_index=True)


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
def validate_upload(edited: pd.DataFrame, model: CategoryModel,
                    driver_fc: pd.DataFrame) -> list[str]:
    """Return a list of problems with an uploaded file. Empty list means usable."""
    problems = []
    for col in ["date", "period"]:
        if col not in edited.columns:
            problems.append(f"missing required column `{col}`")
    if problems:
        return problems

    missing = [d for d in model.drivers if d not in edited.columns]
    if missing:
        problems.append(f"missing driver columns: {', '.join(missing)}")

    fut = edited[edited["period"] == "forecast"]
    if fut.empty:
        problems.append("no rows with period='forecast' -- nothing to plan")
        return problems

    expected = set(
        pd.DatetimeIndex(driver_fc[driver_fc["category"] == model.category]["date"])
    )
    got = set(pd.DatetimeIndex(fut["date"]))
    if got != expected:
        if got - expected:
            problems.append(f"{len(got - expected)} forecast dates are not in the horizon")
        if expected - got:
            problems.append(f"{len(expected - got)} forecast weeks are missing")

    for d in model.drivers:
        if d in fut.columns:
            vals = pd.to_numeric(fut[d], errors="coerce")
            if vals.isna().any():
                problems.append(f"`{d}` has {int(vals.isna().sum())} blank or non-numeric values")
            elif (vals < 0).any():
                problems.append(f"`{d}` has negative values")
    return problems


def run_scenario(model: CategoryModel, panel: pd.DataFrame, driver_fc: pd.DataFrame,
                 edited: pd.DataFrame) -> dict:
    """Baseline vs an edited driver plan, with a per-driver waterfall.

    Only the `period == 'forecast'` rows of `edited` are used. History is what
    happened; editing it would move the actuals line and mean nothing.
    """
    cat = model.category
    hist = panel[panel["category"] == cat].sort_values("date")
    base_fut = driver_fc[driver_fc["category"] == cat].sort_values("date")

    scen_fut = edited[edited["period"] == "forecast"].copy()
    scen_fut["date"] = pd.to_datetime(scen_fut["date"])
    scen_fut = scen_fut.sort_values("date").reset_index(drop=True)
    for d in model.drivers:
        scen_fut[d] = pd.to_numeric(scen_fut[d], errors="coerce")

    base = _predict_future(model, hist, base_fut)
    scen = _predict_future(model, hist, scen_fut)

    paths = pd.DataFrame(
        {
            "date": base["date"],
            "baseline": base["volume_pred"].to_numpy(),
            "scenario": scen["volume_pred"].to_numpy(),
        }
    )
    paths["delta"] = paths["scenario"] - paths["baseline"]

    # Allocate the total delta across drivers in proportion to their log
    # contributions, so the parts sum exactly to the whole rather than to
    # something close to it.
    dlog = pd.DataFrame(index=base.index)
    for d in model.drivers:
        dlog[d] = float(model.elasticities[d]) * (
            scen[f"l_{d}"].to_numpy() - base[f"l_{d}"].to_numpy()
        )
    total_dlog = dlog.sum(axis=1).to_numpy()
    denom = np.where(np.abs(total_dlog) < 1e-12, np.nan, total_dlog)
    delta = paths["delta"].to_numpy()

    rows = []
    for d in model.drivers:
        share = np.where(np.isnan(denom), 0.0, dlog[d].to_numpy() / denom)
        v = float(np.nansum(delta * share))
        if abs(v) > 1e-6:
            rows.append({"driver": d, "label": cfg.DRIVERS[d]["label"],
                         "group": cfg.DRIVERS[d]["group"], "delta_volume": v})

    total_base = float(paths["baseline"].sum())
    total_scen = float(paths["scenario"].sum())
    waterfall = pd.DataFrame(rows, columns=["driver", "label", "group", "delta_volume"])
    if len(waterfall):
        waterfall["delta_pct"] = 100.0 * waterfall["delta_volume"] / total_base
        waterfall = waterfall.sort_values(
            "delta_volume", key=abs, ascending=False).reset_index(drop=True)

    # What actually changed in the plan, for a human-readable diff.
    changes = []
    for d in model.drivers:
        b, s = base_fut[d].to_numpy(dtype=float), scen_fut[d].to_numpy(dtype=float)
        if len(b) == len(s) and not np.allclose(b, s, rtol=1e-9, atol=1e-12):
            changes.append(
                {
                    "driver": d, "label": cfg.DRIVERS[d]["label"],
                    "baseline_avg": float(np.mean(b)), "scenario_avg": float(np.mean(s)),
                    "pct_change": float((np.mean(s) / np.mean(b) - 1) * 100)
                    if np.mean(b) else np.nan,
                    "weeks_changed": int(np.sum(~np.isclose(b, s, rtol=1e-9, atol=1e-12))),
                }
            )

    return {
        "paths": paths,
        "waterfall": waterfall,
        "changes": pd.DataFrame(changes),
        "scenario_drivers": scen_fut,
        "summary": {
            "baseline_volume": total_base,
            "scenario_volume": total_scen,
            "delta_volume": total_scen - total_base,
            "delta_pct": 100.0 * (total_scen / total_base - 1) if total_base else 0.0,
        },
    }


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------
def decompose_period(model: CategoryModel, panel: pd.DataFrame,
                     driver_fc: pd.DataFrame, future_drivers: pd.DataFrame | None = None
                     ) -> pd.DataFrame:
    """Weekly decomposition across history AND forecast in one continuous frame.

    Built on the combined frame so the adstock and the trend clock run unbroken
    across the seam -- decomposing the two halves separately would put a step in
    the media contribution at the first forecast week.
    """
    cat = model.category
    hist = panel[panel["category"] == cat].sort_values("date")
    fut = (driver_fc[driver_fc["category"] == cat].sort_values("date")
           if future_drivers is None else future_drivers.sort_values("date"))

    cols = ["date"] + model.drivers
    combined = pd.concat([hist[cols], fut[cols]], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    out = model.decompose(combined)
    out["period"] = np.where(out["date"] > pd.Timestamp(cfg.HISTORY_END),
                             "forecast", "history")
    return out
