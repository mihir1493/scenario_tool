"""Synthetic monthly retail panel: 5 categories x 66 months of volume + drivers.

The generator is a log-log response surface with known elasticities, so the
fitted model can be scored against ground truth. Two things are deliberately
*not* recoverable: a persistent AR(1) latent factor and iid noise -- the
"measures we don't track" -- which hold in-sample R^2 down to a realistic level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import features


def _months() -> pd.DatetimeIndex:
    return pd.date_range(config.HISTORY_START, config.HISTORY_END, freq="MS")


def _macro(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Shared external context. One path, broadcast to every category."""
    n = len(dates)
    t = np.arange(n)
    month = dates.month.to_numpy()

    # Food CPI: flat 2021, inflation spike through 2022, then a slower grind up.
    infl = np.where(t < 12, 0.0015, np.where(t < 26, 0.0090, 0.0022))
    cpi = 100.0 * np.exp(np.cumsum(infl)) * np.exp(rng.normal(0, 0.0015, n))

    # Consumer confidence: mirrors the inflation shock, recovers with a lag.
    cc = 98 - 26 * np.exp(-(((t - 20) / 9.0) ** 2)) + 0.14 * t
    cc = cc * np.exp(rng.normal(0, 0.012, n))

    # Unemployment: post-COVID normalisation, then a mild loosening.
    unemp = 3.7 + 2.9 * np.exp(-t / 7.0) + 0.010 * np.maximum(t - 40, 0)
    unemp = unemp * np.exp(rng.normal(0, 0.020, n))

    # Temperature: northern-hemisphere sinusoid.
    temp = 14.0 + 11.0 * np.sin(2 * np.pi * (month - 4) / 12.0) + rng.normal(0, 0.9, n)

    return pd.DataFrame(
        {
            "date": dates,
            "cpi_food": cpi,
            "consumer_confidence": cc,
            "unemployment_rate": unemp,
            "avg_temp_c": np.clip(temp, 1.0, None),
        }
    )


def _category_drivers(
    cat: str, dates: pd.DatetimeIndex, macro: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    n = len(dates)
    t = np.arange(n)
    month = dates.month.to_numpy()

    # --- promo pressure: seasonal peaks around holidays and mid-summer ---
    promo_season = (
        0.5 * np.exp(-(((month - 11.5) / 1.3) ** 2))
        + 0.35 * np.exp(-(((month - 7.0) / 1.6) ** 2))
    )
    depth_base = {"Beverages": 17, "Snacks": 14, "Dairy": 9,
                  "Frozen Foods": 21, "Household Care": 12}[cat]
    promo_depth = depth_base * (1 + 0.55 * promo_season) * np.exp(rng.normal(0, 0.10, n))
    promo_depth = np.clip(promo_depth, 4, 38)

    share_base = {"Beverages": 34, "Snacks": 27, "Dairy": 16,
                  "Frozen Foods": 41, "Household Care": 22}[cat]
    promo_share = share_base * (1 + 0.60 * promo_season) * np.exp(rng.normal(0, 0.13, n))
    promo_share = np.clip(promo_share, 5, 70)

    # --- price: list price tracks food CPI; average price is net of promo ---
    p0 = {"Beverages": 3.10, "Snacks": 4.25, "Dairy": 2.60,
          "Frozen Foods": 5.40, "Household Care": 7.10}[cat]
    cpi_rel = macro["cpi_food"].to_numpy() / 100.0
    list_price = p0 * cpi_rel ** 0.85 * np.exp(rng.normal(0, 0.011, n))
    avg_price = list_price * (1 - (promo_share / 100.0) * (promo_depth / 100.0))

    # --- distribution: logistic build toward a category ceiling ---
    d_start, d_end = {
        "Beverages": (72, 89), "Snacks": (64, 86), "Dairy": (81, 92),
        "Frozen Foods": (55, 79), "Household Care": (68, 74),
    }[cat]
    dist = d_start + (d_end - d_start) / (1 + np.exp(-(t - 26) / 9.0))
    dist = np.clip(dist * np.exp(rng.normal(0, 0.010, n)), 30, 98)

    # --- media: a flighted plan, not a flat spend ---
    m_base = {"Beverages": 420, "Snacks": 260, "Dairy": 140,
              "Frozen Foods": 310, "Household Care": 185}[cat]
    burst_months = {"Beverages": {5, 6, 7}, "Snacks": {3, 9, 10},
                    "Dairy": {2, 9}, "Frozen Foods": {1, 10, 11},
                    "Household Care": {4, 9}}[cat]
    burst = np.isin(month, list(burst_months)).astype(float)
    media = m_base * (0.35 + 1.5 * burst) * (1 + 0.004 * t) * np.exp(rng.normal(0, 0.22, n))
    media = np.clip(media, 0, None)
    media[rng.random(n) < 0.06] = 0.0  # genuine dark months

    return pd.DataFrame(
        {
            "date": dates,
            "category": cat,
            "avg_price": avg_price,
            "distribution_acv": dist,
            "promo_depth": promo_depth,
            "promo_share": promo_share,
            "media_spend": media,
        }
    )


def _category_elasticities(rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Each category gets its own elasticities, jittered around the global truth."""
    out = {}
    for cat in config.CATEGORIES:
        j = config.CATEGORY_ELASTICITY_JITTER
        out[cat] = {
            k: v * (1 + rng.uniform(-j, j)) for k, v in config.TRUE_ELASTICITIES.items()
        }
    # A couple of hand-set overrides so the categories behave like themselves.
    out["Beverages"]["avg_temp_c"] = 0.42
    out["Frozen Foods"]["avg_temp_c"] = 0.24
    out["Household Care"]["avg_temp_c"] = -0.02
    out["Dairy"]["avg_price"] = -1.15  # staple, inelastic
    return out


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    dates = _months()
    macro = _macro(dates, rng)
    elas = _category_elasticities(rng)

    frames = [
        _category_drivers(cat, dates, macro, rng).merge(macro, on="date")
        for cat in config.CATEGORIES
    ]
    panel = pd.concat(frames, ignore_index=True)

    # Log features use the same adstock + transforms the model will use.
    panel = panel.sort_values(["category", "date"]).reset_index(drop=True)
    logs = features.log_features(features.apply_adstock(panel).reset_index(drop=True))

    volumes = np.empty(len(panel))
    for cat in config.CATEGORIES:
        m = (panel["category"] == cat).to_numpy()
        sub_logs = logs[m]
        spec = config.CATEGORY_BASE[cat]
        n = int(m.sum())
        t = np.arange(n)
        month = panel.loc[m, "date"].dt.month.to_numpy()

        # Driver contributions, centred so base_volume is volume at mean drivers.
        contrib = np.zeros(n)
        for name in config.DRIVER_NAMES:
            lx = sub_logs[f"l_{name}"].to_numpy()
            contrib += elas[cat][name] * (lx - lx.mean())

        seasonal = spec["seasonal_amp"] * np.sin(2 * np.pi * (month - 5) / 12.0)
        trend = spec["trend"] * t / 12.0

        # Unobserved: persistent AR(1) factor + iid noise.
        latent = np.zeros(n)
        for i in range(1, n):
            latent[i] = 0.72 * latent[i - 1] + rng.normal(0, config.LATENT_SHOCK_SD)
        noise = rng.normal(0, config.NOISE_SD, n)

        log_v = np.log(spec["base_volume"]) + trend + seasonal + contrib + latent + noise
        volumes[m] = np.exp(log_v)

    panel["volume"] = np.round(volumes)
    cols = ["date", "category", "volume"] + config.DRIVER_NAMES
    return panel[cols].sort_values(["category", "date"]).reset_index(drop=True)


def true_elasticity_table() -> pd.DataFrame:
    """Ground-truth elasticities, volume-weighted to match what pooling estimates."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    _ = _macro(_months(), rng)  # keep the rng stream aligned with generate()
    elas = _category_elasticities(rng)
    rows = []
    for name in config.DRIVER_NAMES:
        vals = [elas[c][name] for c in config.CATEGORIES]
        rows.append({"driver": name, "true_elasticity": float(np.mean(vals))})
    return pd.DataFrame(rows)


def true_category_elasticity_table() -> pd.DataFrame:
    """Ground-truth elasticities per category (index=category, columns=driver).

    What the per-category model is actually trying to recover, as opposed to the
    cross-category mean the pooled model can at best land on.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    _ = _macro(_months(), rng)  # keep the rng stream aligned with generate()
    elas = _category_elasticities(rng)
    return pd.DataFrame(elas).T.loc[config.CATEGORIES, config.DRIVER_NAMES]


if __name__ == "__main__":
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(config.PANEL_CSV, index=False)
    print(df.head())
    print(f"\n{len(df)} rows -> {config.PANEL_CSV}")
