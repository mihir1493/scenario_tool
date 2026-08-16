"""Partial pooling: which parts of the model should be allowed to vary by category?

Fully pooled and fully split are the two ends of a dial, not the only settings.
This module fits the middle. Every driver starts at its pooled elasticity, and a
chosen subset is allowed a per-category *deviation* from it:

    log(volume) - SUM_all_d beta_pooled[d] * log(x_d)
        ~ SUM_{d in free} delta_cat[d] * log(x_d) + trend_cat + month FE_cat

    beta_cat[d] = beta_pooled[d] + delta_cat[d]        (delta = 0 for pinned d)

The parameterisation matters more than it looks. Ridge shrinks coefficients
toward zero, so writing the free drivers as deviations makes the *pooled estimate*
the shrinkage target: with a strong penalty the arm collapses back to the pooled
model, and it only departs where a category's data insists. Fitting the free
drivers from scratch instead would shrink them toward zero elasticity, and -- with
price, food CPI and the time trend as collinear as they are here -- would let a
66-month slice reallocate the shared trend arbitrarily between them.

Two arms come out of the same machinery:

  CALENDAR-ONLY  free = []           every elasticity held at the pooled value;
                                     only trend and month effects vary. This is
                                     the control. The pooled model has no
                                     category-specific trend or seasonality
                                     either, and the generator gives categories
                                     both, so without this arm a fully-split
                                     model's win cannot be attributed to the
                                     drivers at all.

  HYBRID         free = commercial   price, distribution, promo and media may
                                     deviate per category; the macro drivers stay
                                     pooled. Macro drivers are one shared time
                                     series by construction, so a single
                                     category's 66 months hold no extra
                                     information about them.

Deviations are left sign-unconstrained -- the pooled level already carries the
business prior, and a constraint on delta would forbid a category from being
*less* price elastic than average, which is exactly the effect being looked for.
`fit` checks the resulting totals and records any sign violations in `metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src import features, model as pooled_mod
from src.model_percat import MONTH_COLS


def _offset(logs: pd.DataFrame, betas: pd.Series) -> np.ndarray:
    """Log-volume implied by every driver held at its pooled elasticity."""
    return sum(
        float(betas[d]) * logs[f"l_{d}"].to_numpy() for d in config.DRIVER_NAMES
    )


def _pipeline(num_cols: list[str]) -> Pipeline:
    """Same shape as the pooled pipeline, with two deliberate differences.

    Deviations are sign-free (the pooled level already carries the business
    prior), and only the deviation columns are penalised. Shrinking the trend and
    month effects alongside them would conflate two questions: at a strong alpha
    the arm must collapse back to *pooled elasticities with a category-specific
    calendar*, not back to the pooled model outright.
    """
    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("fe", "passthrough", MONTH_COLS)]
    )
    n = len(num_cols) + len(MONTH_COLS)
    bounds = (np.full(n, -np.inf), np.full(n, np.inf))
    weights = np.zeros(n)
    weights[: len(num_cols) - 1] = 1.0  # driver deviations only; trend is last
    return Pipeline(
        [
            ("pre", pre),
            ("ridge", pooled_mod.SignedRidge(bounds=bounds, penalty_weights=weights)),
        ]
    )


@dataclass
class PartialPoolModel:
    """Per-category deviations from a pooled set of elasticities."""

    name: str
    free: list[str]
    pinned: list[str]
    pipes: dict[str, Pipeline]
    num_cols: list[str]  # deviation columns + trend
    t0: pd.Timestamp
    alphas: dict[str, float]
    pooled_betas: pd.Series
    elasticities: pd.DataFrame  # index=category, columns=driver; pooled + deviation
    metrics: dict = field(default_factory=dict)

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        """Rows come back in (category, date) order, as with the other models."""
        adstocked = features.apply_adstock(df)
        X = features.build_design(df, self.t0)
        off = _offset(features.log_features(adstocked), self.pooled_betas)
        cats = adstocked["category"].to_numpy()
        out = np.empty(len(X), dtype=float)
        for cat, pipe in self.pipes.items():
            m = cats == cat
            if m.any():
                out[m] = pipe.predict(X.loc[m, self.num_cols + MONTH_COLS]) + off[m]
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))


def _fit_one(sub, t0, num_cols, pooled_betas, alpha=None) -> tuple[Pipeline, float]:
    """`alpha=None` selects by CV; a fixed value sets the shrinkage by hand.

    The fixed path exists because CV cannot see what goes wrong here: price, food
    CPI and the trend are collinear enough that reallocating between them barely
    moves forecast error, so CV picks the weakest penalty on the grid and the
    deviations run free. Sweeping alpha explicitly is the only way to see the
    accuracy-vs-elasticity trade-off the CV objective is blind to.
    """
    X = features.build_design(sub, t0)[num_cols + MONTH_COLS]
    logs = features.log_features(features.apply_adstock(sub))
    y = np.log(sub["volume"].to_numpy()) - _offset(logs, pooled_betas)
    if alpha is not None:
        pipe = _pipeline(num_cols).set_params(ridge__alpha=alpha).fit(X, y)
        return pipe, float(alpha)
    gs = GridSearchCV(
        _pipeline(num_cols),
        {"ridge__alpha": config.RIDGE_ALPHAS},
        cv=list(pooled_mod._time_folds(sub["date"])),
        scoring="neg_mean_squared_error",
    )
    gs.fit(X, y)
    return gs.best_estimator_, float(gs.best_params_["ridge__alpha"])


def _fit_all(panel, t0, num_cols, pooled_betas, alpha=None):
    pipes, alphas = {}, {}
    for cat in config.CATEGORIES:
        sub = panel[panel["category"] == cat]
        pipes[cat], alphas[cat] = _fit_one(sub, t0, num_cols, pooled_betas, alpha)
    return pipes, alphas


def _deviations(pipes, num_cols, free) -> pd.DataFrame:
    """Undo the scaler to read the fitted deviations back in elasticity units."""
    rows = {}
    for cat, pipe in pipes.items():
        scaler: StandardScaler = pipe.named_steps["pre"].named_transformers_["num"]
        raw = pipe.named_steps["ridge"].coef_[: len(num_cols)] / scaler.scale_
        by_col = dict(zip(num_cols, raw))
        rows[cat] = {d: float(by_col.get(f"l_{d}", 0.0)) for d in config.DRIVER_NAMES}
    return pd.DataFrame(rows).T.loc[config.CATEGORIES, config.DRIVER_NAMES]


def fit(panel: pd.DataFrame, free: list[str], pooled_betas: pd.Series,
        name: str = "partial", alpha: float | None = None) -> PartialPoolModel:
    panel = panel.sort_values(["category", "date"]).reset_index(drop=True)
    t0 = pd.Timestamp(config.HISTORY_START)
    pinned = [d for d in config.DRIVER_NAMES if d not in free]
    # `trend` is always free: category trends genuinely differ (-1% to +4.5%/yr).
    num_cols = [f"l_{d}" for d in free] + ["trend"]

    # --- backtest on the same 12-month holdout as every other arm ---
    cutoff = panel["date"].max() - pd.DateOffset(months=config.HOLDOUT_MONTHS)
    te_mask = (panel["date"] > cutoff).to_numpy()
    tr, te = panel[~te_mask], panel[te_mask]
    bt_pipes, _ = _fit_all(tr, t0, num_cols, pooled_betas, alpha)
    bt = PartialPoolModel(
        name=name, free=free, pinned=pinned, pipes=bt_pipes, num_cols=num_cols,
        t0=t0, alphas={}, pooled_betas=pooled_betas,
        elasticities=pd.DataFrame(index=config.CATEGORIES),
    )
    # Predict over the whole panel, then slice -- see the note in model.fit:
    # predicting the holdout rows alone would restart media adstock at the seam.
    bt_pred = bt.predict(panel)[te_mask]
    bt_actual = te["volume"].to_numpy()
    holdout = {
        "months": config.HOLDOUT_MONTHS,
        "mape": float(np.mean(np.abs(bt_pred / bt_actual - 1)) * 100),
        "r2_volume": float(r2_score(bt_actual, bt_pred)),
        "by_category": {
            c: float(
                np.mean(
                    np.abs(
                        bt_pred[(te["category"] == c).to_numpy()]
                        / bt_actual[(te["category"] == c).to_numpy()] - 1
                    )
                ) * 100
            )
            for c in config.CATEGORIES
        },
    }

    # --- final fit on the full history ---
    pipes, alphas = _fit_all(panel, t0, num_cols, pooled_betas, alpha)
    deltas = _deviations(pipes, num_cols, free)
    elasticities = deltas.add(pooled_betas[config.DRIVER_NAMES], axis=1)

    m = PartialPoolModel(
        name=name, free=free, pinned=pinned, pipes=pipes, num_cols=num_cols,
        t0=t0, alphas=alphas, pooled_betas=pooled_betas, elasticities=elasticities,
    )

    y = np.log(panel["volume"].to_numpy())
    fitted = m.predict_log(panel)
    violations = [
        f"{cat}/{d}"
        for d in config.DRIVER_NAMES
        for cat in config.CATEGORIES
        if config.DRIVERS[d]["sign"]
        and np.sign(elasticities.loc[cat, d]) == -config.DRIVERS[d]["sign"]
    ]
    m.metrics = {
        "alphas": alphas,
        "n_free_drivers": len(free),
        "max_abs_deviation": float(deltas.abs().to_numpy().max()),
        "sign_violations": violations,
        "r2_log": float(r2_score(y, fitted)),
        "r2_volume": float(r2_score(np.exp(y), np.exp(fitted))),
        "mape_in_sample": float(np.mean(np.abs(np.exp(fitted) / np.exp(y) - 1)) * 100),
        "holdout": holdout,
    }
    return m
