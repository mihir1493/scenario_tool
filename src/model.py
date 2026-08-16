"""Pooled log-log Ridge response model.

One model across all categories: category fixed effects absorb the level
differences, month dummies absorb seasonality, a linear trend absorbs organic
growth, and the log-driver coefficients are read directly as elasticities
(% change in volume per 1% change in the driver).

Ridge rather than OLS because average price is mechanically entangled with promo
depth and promo share -- the collinearity is real and shrinkage is the point.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src import features


class SignedRidge(BaseEstimator, RegressorMixin):
    """Ridge with per-coefficient sign constraints.

    Solved as a bounded least-squares problem on the Tikhonov-augmented system
    ``[X; sqrt(alpha) I] b ~ [y; 0]``. Both sides are centred first, so the
    intercept is fitted but never penalised or constrained.

    The constraints exist because some drivers are weakly identified -- food CPI
    is 0.96 correlated with the time trend -- and an unconstrained fit will
    happily hand back a positive price elasticity or a positive CPI elasticity.
    A scenario tool that says "raise prices to sell more" is worse than useless,
    so known-sign drivers are pinned to their business prior.
    """

    def __init__(self, alpha: float = 1.0, bounds: tuple | None = None,
                 penalty_weights: np.ndarray | None = None):
        self.alpha = alpha
        self.bounds = bounds  # (lower[], upper[]) over columns of X, or None
        # Per-column penalty multiplier; 0 exempts a column from shrinkage. Used
        # by the partial-pooling variants, which shrink driver deviations without
        # also shrinking the trend and month effects that sit in the same design.
        self.penalty_weights = penalty_weights

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self._x_mean, self._y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - self._x_mean, y - self._y_mean

        w = np.ones(p) if self.penalty_weights is None else np.asarray(
            self.penalty_weights, dtype=float
        )
        A = np.vstack([Xc, np.sqrt(self.alpha) * np.diag(w)])
        b = np.concatenate([yc, np.zeros(p)])
        lo, hi = self.bounds if self.bounds is not None else (-np.inf, np.inf)
        res = lsq_linear(A, b, bounds=(lo, hi), method="bvls", max_iter=500)

        self.coef_ = res.x
        self.intercept_ = float(self._y_mean - self._x_mean @ self.coef_)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def _coef_bounds(
    dummy_cols: list[str] | None = None, num_cols: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds over the transformed design: num_cols first, then the dummies.

    StandardScaler has a strictly positive scale, so constraining the scaled
    coefficient constrains the raw elasticity to the same sign.

    Both lists default to the pooled sets. The per-category variant passes a
    shorter `dummy_cols` (no category fixed effects); the partial-pooling variant
    also passes a shorter `num_cols`, because drivers held at their pooled value
    enter as an offset rather than as a fitted column.
    """
    dummy_cols = features.dummy_cols() if dummy_cols is None else dummy_cols
    num_cols = features.NUM_COLS if num_cols is None else num_cols
    lo, hi = [], []
    for col in num_cols:
        sign = config.DRIVERS[col[2:]]["sign"] if col.startswith("l_") else 0
        lo.append(-np.inf if sign <= 0 else 0.0)
        hi.append(np.inf if sign >= 0 else 0.0)
    for _ in dummy_cols:  # fixed effects stay free
        lo.append(-np.inf)
        hi.append(np.inf)
    return np.array(lo), np.array(hi)


def _pipeline(
    dummy_cols: list[str] | None = None, num_cols: list[str] | None = None
) -> Pipeline:
    dummy_cols = features.dummy_cols() if dummy_cols is None else dummy_cols
    num_cols = features.NUM_COLS if num_cols is None else num_cols
    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("fe", "passthrough", dummy_cols)]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("ridge", SignedRidge(bounds=_coef_bounds(dummy_cols, num_cols))),
        ]
    )


def _time_folds(dates: pd.Series, n_splits: int = config.CV_SPLITS):
    """Expanding-window CV on the panel: folds are cut on dates, not rows."""
    uniq = np.sort(dates.unique())
    edges = np.linspace(len(uniq) // 2, len(uniq), n_splits + 1).astype(int)
    d = dates.to_numpy()
    for i in range(n_splits):
        tr_end, te_end = uniq[edges[i] - 1], uniq[edges[i + 1] - 1]
        train = np.where(d <= tr_end)[0]
        test = np.where((d > tr_end) & (d <= te_end))[0]
        if len(train) and len(test):
            yield train, test


@dataclass
class ResponseModel:
    pipe: Pipeline
    t0: pd.Timestamp
    alpha: float
    elasticities: pd.Series
    ref_logs: pd.DataFrame  # per-category mean log driver level (the "baseline" anchor)
    log_sd: pd.Series  # sd of each log driver, for importance scoring
    metrics: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- predict
    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        X = features.build_design(df, self.t0)
        return self.pipe.predict(X[features.DESIGN_COLS])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))

    # ------------------------------------------------------------ attribution
    def importance(self) -> pd.DataFrame:
        """|elasticity| x sd(log driver): % volume swing per 1sd move in the driver."""
        rows = []
        for name in config.DRIVER_NAMES:
            e = float(self.elasticities[name])
            sd = float(self.log_sd[name])
            rows.append(
                {
                    "driver": name,
                    "label": config.DRIVERS[name]["label"],
                    "group": config.DRIVERS[name]["group"],
                    "elasticity": e,
                    "log_sd": sd,
                    "impact_pct": 100.0 * abs(e) * sd,
                    "direction": "increases volume" if e > 0 else "decreases volume",
                }
            )
        out = pd.DataFrame(rows).sort_values("impact_pct", ascending=False)
        out["impact_share"] = out["impact_pct"] / out["impact_pct"].sum()
        return out.reset_index(drop=True)

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split predicted volume into a base plus per-driver incremental volume.

        Contributions are measured against each category's historical mean driver
        level, then allocated multiplicatively so the parts sum to the whole.
        """
        adstocked = features.apply_adstock(df).reset_index(drop=True)
        logs = features.log_features(adstocked)
        pred = self.predict(df)

        log_contrib = pd.DataFrame(index=adstocked.index)
        for name in config.DRIVER_NAMES:
            ref = adstocked["category"].map(self.ref_logs[name]).to_numpy()
            log_contrib[name] = self.elasticities[name] * (logs[f"l_{name}"].to_numpy() - ref)

        total_log = log_contrib.sum(axis=1).to_numpy()
        base = pred / np.exp(total_log)  # volume at reference driver levels
        incremental = pred - base

        denom = np.where(np.abs(total_log) < 1e-9, np.nan, total_log)
        out = pd.DataFrame(
            {"date": adstocked["date"], "category": adstocked["category"],
             "predicted": pred, "base": base}
        )
        for name in config.DRIVER_NAMES:
            share = np.where(np.isnan(denom), 0.0, log_contrib[name].to_numpy() / denom)
            out[name] = incremental * share
        return out

    def save(self, path=config.MODEL_PKL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.MODEL_PKL) -> "ResponseModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def _fit_pipeline(
    train: pd.DataFrame, t0: pd.Timestamp, dummy_cols: list[str] | None = None
) -> tuple[Pipeline, float]:
    dummy_cols = features.dummy_cols() if dummy_cols is None else dummy_cols
    X = features.build_design(train, t0)[features.NUM_COLS + dummy_cols]
    y = np.log(train["volume"].to_numpy())
    folds = list(_time_folds(train["date"]))
    gs = GridSearchCV(
        _pipeline(dummy_cols),
        {"ridge__alpha": config.RIDGE_ALPHAS},
        cv=folds,
        scoring="neg_mean_squared_error",
    )
    gs.fit(X, y)
    return gs.best_estimator_, float(gs.best_params_["ridge__alpha"])


def fit(panel: pd.DataFrame) -> ResponseModel:
    panel = panel.sort_values(["category", "date"]).reset_index(drop=True)
    t0 = pd.Timestamp(config.HISTORY_START)

    # --- backtest: hold out the tail, fit on the rest ---
    cutoff = panel["date"].max() - pd.DateOffset(months=config.HOLDOUT_MONTHS)
    te_mask = (panel["date"] > cutoff).to_numpy()
    tr, te = panel[~te_mask], panel[te_mask]
    bt_pipe, _ = _fit_pipeline(tr, t0)
    # Design built once over the whole panel, then sliced. Rebuilding it on the
    # holdout rows alone would restart media adstock at the seam and understate
    # carryover in the first forecast months -- the same reason scenario.py
    # prepends history before predicting.
    X_bt = features.build_design(panel, t0).loc[te_mask, features.DESIGN_COLS]
    bt_pred = np.exp(bt_pipe.predict(X_bt))
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
                        / bt_actual[(te["category"] == c).to_numpy()]
                        - 1
                    )
                )
                * 100
            )
            for c in config.CATEGORIES
        },
    }

    # --- final model on the full history ---
    pipe, alpha = _fit_pipeline(panel, t0)
    X = features.build_design(panel, t0)[features.DESIGN_COLS]
    y = np.log(panel["volume"].to_numpy())

    scaler: StandardScaler = pipe.named_steps["pre"].named_transformers_["num"]
    coef = pipe.named_steps["ridge"].coef_
    raw = coef[: len(features.NUM_COLS)] / scaler.scale_
    elasticities = pd.Series(
        {n: float(raw[i]) for i, n in enumerate(config.DRIVER_NAMES)}
    )

    adstocked = features.apply_adstock(panel).reset_index(drop=True)
    logs = features.log_features(adstocked)
    logs["category"] = adstocked["category"]
    ref_logs = pd.DataFrame(
        {n: logs.groupby("category")[f"l_{n}"].mean() for n in config.DRIVER_NAMES}
    )
    log_sd = pd.Series({n: float(logs[f"l_{n}"].std()) for n in config.DRIVER_NAMES})

    fitted = pipe.predict(X)
    metrics = {
        "alpha": alpha,
        "n_obs": int(len(panel)),
        "r2_log": float(r2_score(y, fitted)),
        "r2_volume": float(r2_score(np.exp(y), np.exp(fitted))),
        "mape_in_sample": float(np.mean(np.abs(np.exp(fitted) / np.exp(y) - 1)) * 100),
        "holdout": holdout,
    }

    return ResponseModel(
        pipe=pipe, t0=t0, alpha=alpha, elasticities=elasticities,
        ref_logs=ref_logs, log_sd=log_sd, metrics=metrics,
    )


if __name__ == "__main__":
    panel = pd.read_csv(config.PANEL_CSV, parse_dates=["date"])
    m = fit(panel)
    m.save()
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_JSON.write_text(json.dumps(m.metrics, indent=2))
    print(json.dumps(m.metrics, indent=2))
    print("\nElasticities:")
    print(m.elasticities.round(3).to_string())
