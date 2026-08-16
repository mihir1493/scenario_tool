"""Knobs for the feature-selection experiment. Nothing outside this folder reads it.

The shipped tool hands every category the same nine drivers. This experiment asks
a different question: *which* drivers does each category actually earn the right to
use, and does answering that per category produce better elasticities and a better
scenario tool than handing everyone the full set.

To measure that we need drivers that are known to be irrelevant. The generator in
`src/data_gen.py` gives all nine real drivers a non-zero elasticity in every
category, so on the shipped panel there is no true sparsity to recover and any
selection engine would be scored against nothing. `decoys.py` fixes that by
appending four drivers that are appended *after* volume is generated -- so their
true elasticity is exactly zero, by construction, and the engine's precision and
recall become measurable numbers rather than a matter of taste.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # repo root on the path so `config` / `src` import
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

HERE = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "artifacts"
RESULTS_JSON = ARTIFACT_DIR / "results.json"
SELECTION_CSV = ARTIFACT_DIR / "selection_table.csv"
MODEL_PKL = ARTIFACT_DIR / "model_selected.pkl"

# --------------------------------------------------------------------------
# Decoy drivers: real-looking series with zero true effect on volume
# --------------------------------------------------------------------------
# Each one is a different way for a selection engine to be fooled. A method that
# only survives the white-noise decoy has not been tested.
#
#   promo_echo        0.8 correlated with promo_depth in logs -- the collinear
#                     decoy. Marginally it predicts volume well, because the
#                     thing it echoes does. Only a method that conditions on the
#                     other drivers can tell them apart.
#   search_index      smooth, trending, mildly seasonal -- the calendar decoy.
#                     Correlates with volume through the shared time trend.
#                     Killed by residualising the calendar out first, which is
#                     exactly what the engine does before it scores anything.
#   competitor_promo  white noise. The easy case, and the control: if this
#                     survives, the thresholds are too loose.
#   fx_index          a random walk shared across categories -- the spurious
#                     regression case. Non-stationary and drifting, so in a short
#                     sample it can correlate with anything, including the
#                     unobserved AR(1) latent factor the model cannot see.
DECOYS = {
    "promo_echo": {
        "label": "Promo echo index (decoy)",
        "scope": "category",
        "transform": "log",
        "unit": "idx",
        "fmt": "{:.1f}",
        "slider": (-30, 30),
        "group": "Decoy",
        "sign": 0,
    },
    "search_index": {
        "label": "Category search index (decoy)",
        "scope": "category",
        "transform": "log",
        "unit": "idx",
        "fmt": "{:.1f}",
        "slider": (-30, 30),
        "group": "Decoy",
        "sign": 0,
    },
    "competitor_promo": {
        "label": "Competitor promo pressure (decoy)",
        "scope": "category",
        "transform": "log",
        "unit": "idx",
        "fmt": "{:.1f}",
        "slider": (-30, 30),
        "group": "Decoy",
        "sign": 0,
    },
    "fx_index": {
        "label": "Trade-weighted FX index (decoy)",
        "scope": "macro",
        "transform": "log",
        "unit": "idx",
        "fmt": "{:.1f}",
        "slider": (-15, 15),
        "group": "Decoy",
        "sign": 0,
    },
}

DECOY_NAMES = list(DECOYS)
REAL_NAMES = list(config.DRIVERS)

DECOY_SEED = 4711  # independent of config.RANDOM_SEED; vary it to re-draw decoys


def spec() -> dict:
    """The candidate driver set the engine gets to choose from: real + decoy."""
    return {**config.DRIVERS, **DECOYS}


def candidate_names() -> list[str]:
    return list(spec())


# --------------------------------------------------------------------------
# Selection engine settings
# --------------------------------------------------------------------------
# Three independent filters, all of which a driver must pass. They fail in
# different directions on purpose: stability catches drivers the data cannot
# pin down, the sign prior catches drivers the data pins down *wrongly*, and
# materiality catches drivers that are real but too small to plan against.
SELECTION = {
    # --- stability selection (the main filter) ---
    "n_bootstrap": 200,
    "block_length": 6,  # moving-block bootstrap, in months; preserves autocorrelation
    # Lasso penalties as a fraction of alpha_max, the smallest penalty that zeroes
    # everything. Selection frequency is averaged over the three, so the result
    # does not hinge on one arbitrary lambda.
    "alpha_fractions": (0.05, 0.10, 0.20),
    "tau_stability": 0.60,  # keep if selected in >=60% of (draw, lambda) pairs
    # Penalty for the light unconstrained ridge that supplies the sign and
    # magnitude used by the next two filters, as a fraction of the sample size --
    # scale-free, and light enough that the estimate stays close to OLS.
    "ridge_alpha_frac": 0.05,
    # --- sign prior ---
    # A driver whose unconstrained estimate contradicts its business prior by more
    # than this (in elasticity units) is not identified in that category. Below it,
    # the wrong sign is noise around zero and materiality will catch it anyway.
    "sign_violation_tol": 0.02,
    # --- materiality ---
    # |elasticity| x sd(log driver in this category), as % volume swing per 1sd.
    # Below this a slider is not worth showing a planner.
    "tau_impact_pct": 0.5,
    # --- the global gate ---
    # Run first, over the pooled panel, and deliberately more permissive than the
    # per-category gate. Its only job is to answer "is this a driver at all?",
    # which 330 observations can settle and 66 cannot. A driver failing here is
    # removed from the tool outright; a driver passing here but failing its
    # category keeps the pooled elasticity instead of being zeroed.
    "tau_stability_global": 0.40,
    "tau_impact_pct_global": 0.25,
    # A driver carrying a non-zero `sign` in config.py is a driver the business
    # has already asserted matters -- config.py sets those priors precisely
    # *because* the driver is weakly identified (food CPI runs 0.96 with the time
    # trend). Letting the global gate delete one would be the tool discarding a
    # stated business prior on the strength of the very collinearity the prior
    # exists to handle. Protected drivers can still lose their per-category fit
    # and fall back to the pooled elasticity; they cannot lose their slider.
    # Set False to see the engine's unmoderated verdict -- run_experiment.py
    # reports both.
    "protect_signed_priors": True,
    # A driver whose coefficient changes sign in more than 1-tau of the resamples
    # where it is selected is not plannable, whatever its selection frequency.
    "tau_sign_consistency": 0.90,
    # --- diagnostics only (recorded, not part of the keep rule) ---
    "run_loo_cv": True,  # leave-one-driver-out CV delta, for the report
    "seed": 20260815,
}

# Arms compared in run_experiment.py
ARMS = [
    "A pooled",
    "B percat all",
    "C percat selected",
    "D percat selected+pooled",
    "E oracle",
]
