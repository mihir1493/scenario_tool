"""Scenario engine over a per-category elasticity matrix.

`src/scenario.py` reads `model.elasticities[driver]` -- one number per driver,
shared by every category. Once each category has its own feature set that stops
being true, so every elasticity lookup here is `elasticities.loc[category, driver]`
and the driver waterfall is allocated per row rather than per driver.

Structurally identical to the shipped engine otherwise: history is prepended
before predicting so media adstock carries over the forecast seam, and the total
volume delta is allocated across drivers in proportion to their log contributions.

The extra piece is `phantom_volume`, which exists only because of the decoys. It
moves one slider at a time and reports the volume it produces. For a decoy the
true answer is exactly zero, so whatever the number comes back as is the volume a
planner would be shown for pulling a lever that does nothing -- the cost of
skipping feature selection, denominated in cases rather than in R-squared.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

import fs_features  # noqa: E402


def apply_adjustments(driver_fc: pd.DataFrame, adjustments: dict[str, float],
                      spec: dict) -> pd.DataFrame:
    """`adjustments` maps driver -> % change vs forecast (-10.0 for a 10% cut)."""
    out = driver_fc.copy()
    for name, pct in adjustments.items():
        if name not in out.columns or not pct:
            continue
        out[name] = out[name] * (1.0 + pct / 100.0)
    if "distribution_acv" in out.columns:
        out["distribution_acv"] = out["distribution_acv"].clip(1.0, 100.0)
    for name in spec:
        if name in out.columns:
            out[name] = out[name].clip(lower=0.0)
    return out


def _predict_future(model, panel: pd.DataFrame, driver_fc: pd.DataFrame,
                    spec: dict) -> pd.DataFrame:
    """Predict the forecast rows with history prepended so adstock carries over."""
    names = list(spec)
    cols = ["date", "category"] + names
    combined = pd.concat([panel[cols], driver_fc[cols]], ignore_index=True)
    combined = combined.sort_values(["category", "date"]).reset_index(drop=True)
    combined["volume_pred"] = model.predict(combined)

    cutoff = pd.Timestamp(config.HISTORY_END)
    fut = combined[combined["date"] > cutoff].reset_index(drop=True)

    logs = fs_features.log_features(fs_features.apply_adstock(combined, spec), spec)
    logs[["date", "category"]] = combined[["date", "category"]]
    fut_logs = logs[logs["date"] > cutoff].reset_index(drop=True)
    for n in names:
        fut[f"l_{n}"] = fut_logs[f"l_{n}"]
    return fut


def run(model, panel: pd.DataFrame, driver_fc: pd.DataFrame,
        adjustments: dict[str, float] | None = None,
        categories: list[str] | None = None) -> dict:
    """Baseline vs scenario over the forecast horizon, plus a driver waterfall."""
    spec = model.spec
    names = list(spec)
    adjustments = adjustments or {}
    cats = categories or config.CATEGORIES
    elas = model.elasticity_frame()

    base = _predict_future(model, panel, driver_fc, spec)
    scen = _predict_future(model, panel,
                           apply_adjustments(driver_fc, adjustments, spec), spec)

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
    # Each row uses its own category's elasticity, which is the whole point of
    # the exercise: a 10% price cut is not one number across five categories.
    dlog = pd.DataFrame(index=base.index)
    for n in names:
        beta = base["category"].map(elas[n]).to_numpy()
        dlog[n] = beta * (scen[f"l_{n}"].to_numpy() - base[f"l_{n}"].to_numpy())
    total_dlog = dlog.sum(axis=1).to_numpy()
    total_delta = paths["delta"].to_numpy()
    denom = np.where(np.abs(total_dlog) < 1e-12, np.nan, total_dlog)

    wf_rows = []
    for n in names:
        share = np.where(np.isnan(denom), 0.0, dlog[n].to_numpy() / denom)
        v = float(np.nansum(total_delta * share))
        if abs(v) > 1e-6:
            wf_rows.append({"driver": n, "label": spec[n]["label"],
                            "group": spec[n]["group"], "delta_volume": v})

    total_base = float(paths["baseline"].sum())
    total_scen = float(paths["scenario"].sum())
    wf = pd.DataFrame(wf_rows, columns=["driver", "label", "group", "delta_volume"])
    if len(wf):
        wf["delta_pct"] = 100.0 * wf["delta_volume"] / total_base if total_base else 0.0
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


def slider_response(model, panel: pd.DataFrame, driver_fc: pd.DataFrame,
                    driver: str, pct: float = 20.0,
                    categories: list[str] | None = None) -> dict:
    """Move one slider by `pct` and report the volume the tool claims it produced."""
    r = run(model, panel, driver_fc, {driver: pct}, categories)
    return {
        "driver": driver,
        "pct_move": pct,
        "delta_volume": r["summary"]["delta_volume"],
        "delta_pct": r["summary"]["delta_pct"],
    }


def phantom_volume(model, panel: pd.DataFrame, driver_fc: pd.DataFrame,
                   drivers: list[str], pct: float = 20.0) -> pd.DataFrame:
    """Per-driver slider response. For a decoy, every non-zero row is a lie.

    Reported per category as well as in total, because an arm can look clean on
    the total while two categories cancel each other out.
    """
    rows = []
    for d in drivers:
        total = slider_response(model, panel, driver_fc, d, pct)
        row = {"driver": d, "total_delta_pct": total["delta_pct"],
               "total_delta_volume": total["delta_volume"]}
        for c in config.CATEGORIES:
            row[c] = slider_response(model, panel, driver_fc, d, pct, [c])["delta_pct"]
        rows.append(row)
    out = pd.DataFrame(rows).set_index("driver")
    out["worst_category_pct"] = out[config.CATEGORIES].abs().max(axis=1)
    return out
