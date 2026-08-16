"""Scenario engine: bend the forecast drivers, re-predict volume, explain the gap.

Everything runs off two frames -- the observed panel and the Prophet driver
forecast -- so a scenario is just "the forecast, multiplied". History is always
prepended before predicting so media adstock carries over the seam correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import features
from src.model import ResponseModel


def apply_adjustments(driver_fc: pd.DataFrame, adjustments: dict[str, float]) -> pd.DataFrame:
    """`adjustments` maps driver -> % change vs forecast (e.g. -10.0 for a 10% cut)."""
    out = driver_fc.copy()
    for name, pct in adjustments.items():
        if name not in out.columns or not pct:
            continue
        out[name] = out[name] * (1.0 + pct / 100.0)
    if "distribution_acv" in out.columns:
        out["distribution_acv"] = out["distribution_acv"].clip(1.0, 100.0)
    for name in config.DRIVER_NAMES:
        out[name] = out[name].clip(lower=0.0)
    return out


def _predict_future(model: ResponseModel, panel: pd.DataFrame,
                    driver_fc: pd.DataFrame) -> pd.DataFrame:
    """Predict the future rows with history prepended so adstock carries over."""
    hist = panel[["date", "category"] + config.DRIVER_NAMES]
    combined = pd.concat([hist, driver_fc[["date", "category"] + config.DRIVER_NAMES]],
                         ignore_index=True)
    combined = combined.sort_values(["category", "date"]).reset_index(drop=True)
    combined["volume_pred"] = model.predict(combined)

    cutoff = pd.Timestamp(config.HISTORY_END)
    fut = combined[combined["date"] > cutoff].reset_index(drop=True)

    adstocked = features.apply_adstock(combined).reset_index(drop=True)
    logs = features.log_features(adstocked)
    logs[["date", "category"]] = adstocked[["date", "category"]]
    fut_logs = logs[logs["date"] > cutoff].reset_index(drop=True)
    for name in config.DRIVER_NAMES:
        fut[f"l_{name}"] = fut_logs[f"l_{name}"]
    return fut


def run(model: ResponseModel, panel: pd.DataFrame, driver_fc: pd.DataFrame,
        adjustments: dict[str, float] | None = None,
        categories: list[str] | None = None) -> dict:
    """Baseline vs scenario over the forecast horizon, plus a driver waterfall."""
    adjustments = adjustments or {}
    cats = categories or config.CATEGORIES

    base = _predict_future(model, panel, driver_fc)
    scen = _predict_future(model, panel, apply_adjustments(driver_fc, adjustments))

    mask = base["category"].isin(cats)
    base, scen = base[mask].reset_index(drop=True), scen[mask].reset_index(drop=True)

    paths = pd.DataFrame(
        {
            "date": base["date"],
            "category": base["category"],
            "baseline": base["volume_pred"],
            "scenario": scen["volume_pred"],
        }
    )
    paths["delta"] = paths["scenario"] - paths["baseline"]

    # --- allocate the total delta across drivers, multiplicatively ---
    dlog = pd.DataFrame(index=base.index)
    for name in config.DRIVER_NAMES:
        dlog[name] = model.elasticities[name] * (
            scen[f"l_{name}"].to_numpy() - base[f"l_{name}"].to_numpy()
        )
    total_dlog = dlog.sum(axis=1).to_numpy()
    total_delta = paths["delta"].to_numpy()
    denom = np.where(np.abs(total_dlog) < 1e-12, np.nan, total_dlog)

    waterfall = {}
    for name in config.DRIVER_NAMES:
        share = np.where(np.isnan(denom), 0.0, dlog[name].to_numpy() / denom)
        waterfall[name] = float(np.nansum(total_delta * share))

    total_base = float(paths["baseline"].sum())
    total_scen = float(paths["scenario"].sum())

    wf_cols = ["driver", "label", "delta_volume", "delta_pct"]
    wf_rows = [
        {
            "driver": n,
            "label": config.DRIVERS[n]["label"],
            "delta_volume": v,
            "delta_pct": 100.0 * v / total_base if total_base else 0.0,
        }
        for n, v in waterfall.items()
        if abs(v) > 1e-6
    ]
    wf = pd.DataFrame(wf_rows, columns=wf_cols)  # keep the schema when empty
    if len(wf):
        wf = wf.sort_values("delta_volume", key=abs, ascending=False).reset_index(drop=True)

    return {
        "paths": paths,
        "waterfall": wf,
        "summary": {
            "baseline_volume": total_base,
            "scenario_volume": total_scen,
            "delta_volume": total_scen - total_base,
            "delta_pct": 100.0 * (total_scen / total_base - 1) if total_base else 0.0,
        },
    }


def fitted_history(model: ResponseModel, panel: pd.DataFrame) -> pd.DataFrame:
    """Actual vs fitted over history, for the model-fit view."""
    out = panel[["date", "category", "volume"]].copy()
    out["fitted"] = model.predict(panel)
    return out
