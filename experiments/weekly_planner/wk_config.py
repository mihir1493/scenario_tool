"""Everything that defines this experiment lives here. Change the tool by changing this file.

A weekly (week-ending-Sunday) planner for four categories. The one rule that
shapes every other file: **there is no pooled model anywhere**. Each category gets
its own feature selection, its own validated hyperparameters, its own fitted
model, its own driver forecasts and its own decomposition. No category ever sees
another category's data, borrows its coefficients, or shares a fixed effect with
it. Four independent models that happen to live in the same folder.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

PANEL_CSV = DATA_DIR / "weekly_panel.csv"
DRIVER_FORECAST_CSV = DATA_DIR / "driver_forecast.csv"
MODELS_PKL = ARTIFACT_DIR / "models.pkl"
SUMMARY_JSON = ARTIFACT_DIR / "summary.json"
BASELINE_CSV = ARTIFACT_DIR / "baseline.csv"

# --------------------------------------------------------------------------
# Calendar. Weekly, week ending Sunday.
# --------------------------------------------------------------------------
HISTORY_START = "2022-01-02"  # a Sunday
HISTORY_END = "2026-08-09"  # a Sunday -- 241 weeks of history
FORECAST_WEEKS = 104  # two years: 2026-08-16 .. 2028-08-06
TEST_WEEKS = 52  # final year held out for the train/test split
CV_FOLDS = 4  # expanding-window folds used to choose each category's config

CATEGORIES = ["Ice Cream", "Ground Coffee", "Laundry Detergent", "Baby Formula"]

RANDOM_SEED = 20260815

# --------------------------------------------------------------------------
# Candidate drivers
# --------------------------------------------------------------------------
# transform    : "log" | "log1p" -- log1p wherever the driver can legitimately be 0
# sign         : expected elasticity sign. -1 / +1 / 0 (no prior, let the data speak)
# adstock      : True if eligible for carryover. The decay itself is not fixed here
#                -- it is a hyperparameter each category validates.
# controllable : True if a category team can actually set this number. It is the
#                difference between a lever and a weather report, and feature
#                selection has to know which is which -- see SELECTION_TOLERANCE
#                and MATERIALITY_PCT below.
DRIVERS = {
    "avg_price": {
        "label": "Average price ($/unit)",
        "transform": "log", "sign": -1, "adstock": False,
        "group": "Price", "controllable": True, "fmt": "{:.2f}", "unit": "$",
    },
    "competitor_price": {
        "label": "Competitor avg price ($/unit)",
        "transform": "log", "sign": 1, "adstock": False,
        "group": "Price", "controllable": False, "fmt": "{:.2f}", "unit": "$",
    },
    "promo_depth": {
        "label": "Promo depth (avg % off)",
        "transform": "log1p", "sign": 1, "adstock": False,
        "group": "Promotion", "controllable": True, "fmt": "{:.1f}", "unit": "%",
    },
    "promo_share": {
        "label": "Stores on promo (% ACV)",
        "transform": "log1p", "sign": 1, "adstock": False,
        "group": "Promotion", "controllable": True, "fmt": "{:.1f}", "unit": "%",
    },
    "feature_display": {
        "label": "Feature + display (% ACV)",
        "transform": "log1p", "sign": 1, "adstock": False,
        "group": "Promotion", "controllable": True, "fmt": "{:.1f}", "unit": "%",
    },
    "distribution_acv": {
        "label": "Distribution (% ACV)",
        "transform": "log", "sign": 1, "adstock": False,
        "group": "Distribution", "controllable": True, "fmt": "{:.1f}", "unit": "%",
    },
    "tv_grps": {
        "label": "TV GRPs",
        "transform": "log1p", "sign": 1, "adstock": True,
        "group": "Media", "controllable": True, "fmt": "{:.0f}", "unit": "grp",
    },
    "digital_spend": {
        "label": "Digital spend ($k)",
        "transform": "log1p", "sign": 1, "adstock": True,
        "group": "Media", "controllable": True, "fmt": "{:.0f}", "unit": "$k",
    },
    "avg_temp_c": {
        "label": "Average temperature (C)",
        "transform": "log", "sign": 0, "adstock": False,
        "group": "External", "controllable": False, "fmt": "{:.1f}", "unit": "C",
    },
    "cpi_food": {
        "label": "Food CPI index",
        "transform": "log", "sign": -1, "adstock": False,
        "group": "External", "controllable": False, "fmt": "{:.1f}", "unit": "idx",
    },
    "consumer_confidence": {
        "label": "Consumer confidence",
        "transform": "log", "sign": 1, "adstock": False,
        "group": "External", "controllable": False, "fmt": "{:.1f}", "unit": "idx",
    },
}

DRIVER_NAMES = list(DRIVERS)
ADSTOCK_DRIVERS = [n for n, s in DRIVERS.items() if s["adstock"]]

# --------------------------------------------------------------------------
# Ground truth used to GENERATE the data
# --------------------------------------------------------------------------
# Exact zeros are the point of this experiment. Temperature does nothing to
# laundry detergent, and competitor price does nothing to baby formula -- parents
# do not switch brands on price. A per-category feature selector should find those
# zeros, and a pooled model could not represent them even if it did.
TRUE_ELASTICITIES = {
    "Ice Cream": {
        "avg_price": -1.90, "competitor_price": 0.55, "promo_depth": 0.45,
        "promo_share": 0.30, "feature_display": 0.35, "distribution_acv": 0.90,
        "tv_grps": 0.060, "digital_spend": 0.030, "avg_temp_c": 0.55,
        "cpi_food": -0.30, "consumer_confidence": 0.35,
    },
    "Ground Coffee": {
        "avg_price": -2.20, "competitor_price": 0.70, "promo_depth": 0.55,
        "promo_share": 0.35, "feature_display": 0.25, "distribution_acv": 0.80,
        "tv_grps": 0.050, "digital_spend": 0.055, "avg_temp_c": 0.0,
        "cpi_food": -0.70, "consumer_confidence": 0.30,
    },
    "Laundry Detergent": {
        "avg_price": -1.20, "competitor_price": 0.45, "promo_depth": 0.60,
        "promo_share": 0.40, "feature_display": 0.30, "distribution_acv": 1.00,
        "tv_grps": 0.030, "digital_spend": 0.020, "avg_temp_c": 0.0,
        "cpi_food": -0.50, "consumer_confidence": 0.20,
    },
    "Baby Formula": {
        "avg_price": -0.60, "competitor_price": 0.0, "promo_depth": 0.20,
        "promo_share": 0.15, "feature_display": 0.0, "distribution_acv": 1.10,
        "tv_grps": 0.070, "digital_spend": 0.060, "avg_temp_c": 0.0,
        "cpi_food": -0.40, "consumer_confidence": 0.45,
    },
}

# Per-category shape. `fourier_k` and `adstock_decay` are the truth the validation
# grid is trying to recover -- and they genuinely differ, which is the argument
# for validating each category separately instead of picking one global config.
CATEGORY_SHAPE = {
    "Ice Cream": {
        "base_volume": 850_000, "trend": 0.035,
        # Temperature already carries the summer peak as a driver, so the Fourier
        # term only has to carry the shape temperature does not explain.
        "fourier_k": 3, "seasonal_amp": [0.07, 0.05, 0.03],
        "adstock_decay": 0.30, "peak_week": 27,
    },
    "Ground Coffee": {
        "base_volume": 1_250_000, "trend": 0.015,
        "fourier_k": 2, "seasonal_amp": [0.13, 0.04, 0.0],  # mild winter peak
        "adstock_decay": 0.50, "peak_week": 2,
    },
    "Laundry Detergent": {
        "base_volume": 2_100_000, "trend": -0.010,
        "fourier_k": 1, "seasonal_amp": [0.05, 0.0, 0.0],  # almost flat
        "adstock_decay": 0.70, "peak_week": 10,
    },
    "Baby Formula": {
        "base_volume": 430_000, "trend": 0.025,
        "fourier_k": 1, "seasonal_amp": [0.03, 0.0, 0.0],  # flat: babies are aseasonal
        "adstock_decay": 0.0, "peak_week": 1,
    },
}

# Unobserved reality, so in-sample R^2 lands somewhere believable.
LATENT_AR = 0.80  # persistence of the unobserved weekly factor
LATENT_SD = 0.030
NOISE_SD = 0.018

# --------------------------------------------------------------------------
# The validation grid. Each category picks its own row from this.
# --------------------------------------------------------------------------
GRID = {
    "fourier_k": [1, 2, 3, 4],
    "adstock_decay": [0.0, 0.3, 0.5, 0.7],
    "alpha": [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0],
}

# Backward elimination stops removing drivers once the CV RMSE penalty for the
# best available removal exceeds this fraction. 0.2% of RMSE buys a lot of
# simplicity, and a driver that cheap to lose was never carrying the forecast.
SELECTION_TOLERANCE = 0.002

# A controllable driver whose estimated impact is at least this large (% volume
# swing per one-sd move) is never eliminated, however little it does for forecast
# error.
#
# This exists because pure CV selection built the wrong tool. Our price and the
# competitor's price move together -- both track food CPI and both respond to the
# promo calendar -- so out-of-fold error genuinely cannot tell them apart, and the
# first run of this experiment dropped `avg_price` from Laundry Detergent and kept
# `competitor_price`. Forecast-wise that is a fair trade. As a planning tool it is
# useless: it deletes the one lever the category team actually sets and replaces it
# with a number they can only watch. Forecast accuracy and decision support are not
# the same objective, and where they conflict a scenario tool has to serve the
# second one.
#
# The threshold is what stops this becoming blanket protection. A controllable
# driver the model finds genuinely inert -- feature+display for baby formula, whose
# true elasticity is zero -- falls below it and is still dropped.
MATERIALITY_PCT = 1.0

# --------------------------------------------------------------------------
# Chart palette (shared with the rest of the repo)
# --------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
POS = "#2a78d6"
NEG = "#e34948"
