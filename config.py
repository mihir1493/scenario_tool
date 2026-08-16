"""Single source of truth for the scenario tool.

Everything that defines the shape of the problem lives here: the panel dimensions,
the driver spec, the true data-generating elasticities, and the model settings.
Changing the tool means changing this file first.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

PANEL_CSV = DATA_DIR / "panel.csv"
DRIVER_FORECAST_CSV = DATA_DIR / "driver_forecast.csv"
MODEL_PKL = ARTIFACT_DIR / "response_model.pkl"
METRICS_JSON = ARTIFACT_DIR / "metrics.json"

# --------------------------------------------------------------------------
# Panel dimensions
# --------------------------------------------------------------------------
HISTORY_START = "2021-01-01"
HISTORY_END = "2026-06-01"  # last observed month
FORECAST_MONTHS = 18  # 2026-07 .. 2027-12

CATEGORIES = [
    "Beverages",
    "Snacks",
    "Dairy",
    "Frozen Foods",
    "Household Care",
]

RANDOM_SEED = 20260813

# --------------------------------------------------------------------------
# Driver spec
# --------------------------------------------------------------------------
# scope:      "category" -> varies per category | "macro" -> shared across all
# transform:  "log"      -> log(x)              | "log1p" -> log(1 + x)
# adstock:    geometric carryover applied before the transform (media only)
# slider:     (min_pct, max_pct) bounds for the scenario UI, as % change vs baseline
# sign:       expected elasticity sign, enforced during fitting. -1 / +1 / 0 (free).
#             Food CPI correlates 0.96 with the time trend, so without a constraint
#             its coefficient is not identified and can flip positive -- which would
#             make the scenario sliders give nonsense. Business priors beat noise.
DRIVERS = {
    "avg_price": {
        "label": "Average price ($/unit)",
        "scope": "category",
        "transform": "log",
        "unit": "$",
        "fmt": "{:.2f}",
        "slider": (-25, 25),
        "group": "Commercial",
        "sign": -1,
    },
    "distribution_acv": {
        "label": "Distribution (% ACV)",
        "scope": "category",
        "transform": "log",
        "unit": "%",
        "fmt": "{:.1f}",
        "slider": (-15, 15),
        "group": "Commercial",
        "sign": 1,
    },
    "promo_depth": {
        "label": "Promo depth (avg % off)",
        "scope": "category",
        "transform": "log1p",
        "unit": "%",
        "fmt": "{:.1f}",
        "slider": (-50, 50),
        "group": "Commercial",
        "sign": 1,
    },
    "promo_share": {
        "label": "Weeks on promo (% of weeks)",
        "scope": "category",
        "transform": "log1p",
        "unit": "%",
        "fmt": "{:.1f}",
        "slider": (-50, 50),
        "group": "Commercial",
        "sign": 1,
    },
    "media_spend": {
        "label": "Media spend ($k)",
        "scope": "category",
        "transform": "log1p",
        "unit": "$k",
        "fmt": "{:.0f}",
        "slider": (-60, 100),
        "group": "Commercial",
        "sign": 1,
        "adstock": 0.45,
    },
    "cpi_food": {
        "label": "Food CPI index",
        "scope": "macro",
        "transform": "log",
        "unit": "idx",
        "fmt": "{:.1f}",
        "slider": (-8, 8),
        "group": "Macro",
        "sign": -1,
    },
    "consumer_confidence": {
        "label": "Consumer confidence",
        "scope": "macro",
        "transform": "log",
        "unit": "idx",
        "fmt": "{:.1f}",
        "slider": (-20, 20),
        "group": "Macro",
        "sign": 1,
    },
    "unemployment_rate": {
        "label": "Unemployment rate (%)",
        "scope": "macro",
        "transform": "log",
        "unit": "%",
        "fmt": "{:.2f}",
        "slider": (-30, 30),
        "group": "Macro",
        "sign": -1,
    },
    "avg_temp_c": {
        "label": "Average temperature (C)",
        "scope": "macro",
        "transform": "log",
        "unit": "C",
        "fmt": "{:.1f}",
        "slider": (-15, 15),
        "group": "Macro",
        "sign": 0,  # free: genuinely differs by category
    },
}

DRIVER_NAMES = list(DRIVERS)
COMMERCIAL_DRIVERS = [k for k, v in DRIVERS.items() if v["group"] == "Commercial"]
MACRO_DRIVERS = [k for k, v in DRIVERS.items() if v["group"] == "Macro"]

# --------------------------------------------------------------------------
# True elasticities used to GENERATE the data.
# The pooled model estimates one number per driver; the generator jitters each
# category around the global value by +/- CATEGORY_ELASTICITY_JITTER so the
# pooled estimate stays a meaningful average of the truth.
# --------------------------------------------------------------------------
TRUE_ELASTICITIES = {
    "avg_price": -1.65,
    "distribution_acv": 0.85,
    "promo_depth": 0.55,
    "promo_share": 0.30,
    "media_spend": 0.070,
    "cpi_food": -0.60,
    "consumer_confidence": 0.45,
    "unemployment_rate": -0.18,
    "avg_temp_c": 0.10,
}

CATEGORY_ELASTICITY_JITTER = 0.15  # +/- 15% around the global value

# Category-level base volume (units/month) and organic trend (annualised, log pts)
CATEGORY_BASE = {
    "Beverages": {"base_volume": 2_400_000, "trend": 0.030, "seasonal_amp": 0.16},
    "Snacks": {"base_volume": 1_800_000, "trend": 0.045, "seasonal_amp": 0.08},
    "Dairy": {"base_volume": 3_100_000, "trend": 0.005, "seasonal_amp": 0.05},
    "Frozen Foods": {"base_volume": 1_250_000, "trend": 0.020, "seasonal_amp": 0.11},
    "Household Care": {"base_volume": 950_000, "trend": -0.010, "seasonal_amp": 0.04},
}

# Unobserved reality: drivers that move volume but are NOT in the feature set.
# Keeps in-sample R^2 realistic (~0.90) instead of a suspicious 0.99.
LATENT_SHOCK_SD = 0.035  # sd of a persistent AR(1) latent factor, log scale
NOISE_SD = 0.022  # iid measurement noise, log scale

# --------------------------------------------------------------------------
# Model settings
# --------------------------------------------------------------------------
RIDGE_ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]
CV_SPLITS = 4
HOLDOUT_MONTHS = 12  # final N months held out for the backtest metric

# --------------------------------------------------------------------------
# Chart palette (dataviz reference instance, light mode)
# --------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
POS = "#2a78d6"  # diverging: positive pole
NEG = "#e34948"  # diverging: negative pole
