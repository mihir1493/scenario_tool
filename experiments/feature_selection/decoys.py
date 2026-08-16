"""Append four known-zero-effect drivers to the panel and the driver forecast.

Volume is generated in `src/data_gen.py` from the nine real drivers only. These
four columns are bolted on afterwards and never enter that calculation, so their
true elasticity is exactly zero -- not "small", zero. That is what turns feature
selection from an aesthetic preference into something with a precision and a
recall.

Panel and forecast rows are generated together over one continuous per-category
timeline, so a decoy is the same series either side of the history/forecast seam.
Generating them separately would hand the scenario engine a decoy that jumps
discontinuously in July 2026, which would flatter it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import fs_config  # noqa: E402


def _per_category(sub: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Three category-scoped decoys, built off that category's own real drivers."""
    n = len(sub)
    t = np.arange(n, dtype=float)
    month = sub["date"].dt.month.to_numpy()

    # Collinear: promo_depth wearing a hat. ~0.8 log-correlation with the real
    # driver, which is enough to make it look excellent on its own.
    promo_echo = sub["promo_depth"].to_numpy() * np.exp(rng.normal(0, 0.22, n))

    # Calendar-shaped: trends up, peaks in spring. Correlates with volume only
    # through the trend and seasonality the model already controls for.
    search_index = (
        100.0
        * np.exp(0.035 * t / 12.0)
        * (1 + 0.10 * np.sin(2 * np.pi * (month - 3) / 12.0))
        * np.exp(rng.normal(0, 0.05, n))
    )

    # White noise. The control: if this survives, the thresholds are too loose.
    competitor_promo = 50.0 * np.exp(rng.normal(0, 0.18, n))

    return pd.DataFrame(
        {
            "date": sub["date"].to_numpy(),
            "category": sub["category"].to_numpy(),
            "promo_echo": promo_echo,
            "search_index": search_index,
            "competitor_promo": competitor_promo,
        }
    )


def build(panel: pd.DataFrame, driver_fc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(panel, driver_fc)` with the decoy columns appended to both."""
    rng = np.random.default_rng(fs_config.DECOY_SEED)

    keep = ["date", "category"] + config.DRIVER_NAMES
    combined = pd.concat(
        [panel[keep].assign(_src="hist"), driver_fc[keep].assign(_src="fc")],
        ignore_index=True,
    ).sort_values(["category", "date"], kind="stable").reset_index(drop=True)

    blocks = [
        _per_category(combined[combined["category"] == cat], rng)
        for cat in config.CATEGORIES
    ]
    decoy = pd.concat(blocks, ignore_index=True)

    # Macro decoy: one random walk shared by every category. Non-stationary, so
    # over 66 months it can drift into agreement with anything -- including the
    # unobserved AR(1) factor the response model has no column for. This is the
    # decoy most likely to survive, and the one worth watching.
    dates = np.sort(combined["date"].unique())
    walk = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.006, len(dates))))
    fx = pd.DataFrame({"date": dates, "fx_index": walk})
    decoy = decoy.merge(fx, on="date", how="left")

    out = []
    for frame in (panel, driver_fc):
        merged = frame.merge(decoy, on=["date", "category"], how="left")
        missing = merged[fs_config.DECOY_NAMES].isna().to_numpy().sum()
        if missing:  # pragma: no cover - guards a silent merge failure
            raise ValueError(f"{missing} decoy values failed to merge")
        out.append(merged)
    return out[0], out[1]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel + driver forecast, decoys included. The entry point every arm uses."""
    panel = pd.read_csv(config.PANEL_CSV, parse_dates=["date"])
    driver_fc = pd.read_csv(config.DRIVER_FORECAST_CSV, parse_dates=["date"])
    return build(panel, driver_fc)


if __name__ == "__main__":
    p, f = load()
    print(p[["date", "category"] + fs_config.DECOY_NAMES].head().to_string(index=False))
    logs = np.log(p[config.DRIVER_NAMES + fs_config.DECOY_NAMES + ["volume"]])
    print("\nLog-correlation with volume (should be ~0 for decoys except by luck):")
    print(logs.corr()["volume"].round(3).to_string())
    print("\npromo_echo vs promo_depth log-correlation:")
    print(round(float(logs["promo_echo"].corr(logs["promo_depth"])), 3))
