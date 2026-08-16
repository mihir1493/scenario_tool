"""One model per category. Fitting, validation, and reading the elasticities back out.

The model for a single category is a log-log ridge:

    log(volume) = intercept
                + SUM_d  beta_d * log(driver_d)     <- the elasticities
                + trend + yearly seasonality + holiday weeks

`beta_d` is read directly as an elasticity: the % change in volume for a 1% change
in that driver. Ridge rather than plain least squares because price, promo depth
and promo share move together in real retail data, and shrinkage is the point.

Three things are deliberate:

  * Only the DRIVER coefficients are penalised. Trend, seasonality and holidays are
    controls -- shrinking them would distort the calendar to buy a smaller
    elasticity, which is backwards. They are fitted freely.
  * Coefficients carry sign constraints from `wk_config.DRIVERS[...]["sign"]`. A
    planner shown "raise price to sell more" stops trusting the tool, and price is
    entangled enough with promotion that an unconstrained fit will occasionally
    produce exactly that.
  * Every hyperparameter -- seasonal harmonics, media carryover, ridge strength --
    is validated PER CATEGORY on expanding-window folds. Ice cream and baby formula
    do not share a seasonal shape, and nothing here makes them.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

import wk_config as cfg
import wk_features as feat


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float,
              penalise: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """Sign-constrained ridge, solved as a bounded least-squares problem.

    The ridge penalty is applied by stacking `sqrt(alpha) * I` under the design and
    zeros under the response -- the standard Tikhonov trick -- which turns "ridge
    with bounds" into an ordinary bounded least-squares call. Both sides are
    centred first so the intercept is fitted but never penalised or constrained.

    `penalise` is a 0/1 weight per column: 1 for drivers, 0 for controls.
    """
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean

    A = np.vstack([Xc, np.sqrt(alpha) * np.diag(penalise)])
    b = np.concatenate([yc, np.zeros(X.shape[1])])
    res = lsq_linear(A, b, bounds=(lo, hi), method="bvls", max_iter=500)

    coef = res.x
    intercept = float(y_mean - x_mean @ coef)
    return coef, intercept


def _bounds(cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Sign constraints per design column. Controls are free; drivers follow config."""
    lo, hi = [], []
    for c in cols:
        sign = cfg.DRIVERS[c[2:]]["sign"] if c.startswith("l_") else 0
        lo.append(-np.inf if sign <= 0 else 0.0)
        hi.append(np.inf if sign >= 0 else 0.0)
    return np.array(lo), np.array(hi)


def _penalty_weights(cols: list[str]) -> np.ndarray:
    """Shrink elasticities, never the calendar."""
    return np.array([1.0 if c.startswith("l_") else 0.0 for c in cols])


def _scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise columns so one alpha means the same thing to every driver."""
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return X / sd, sd


def _fit_design(X: pd.DataFrame, y: np.ndarray, alpha: float):
    """Fit on a built design; returns raw (unscaled) coefficients and the intercept."""
    cols = list(X.columns)
    Xs, sd = _scale(X.to_numpy(dtype=float))
    lo, hi = _bounds(cols)
    coef_s, intercept = fit_ridge(Xs, y, alpha, _penalty_weights(cols), lo, hi)
    # Undo the scaling so coefficients read as elasticities again.
    return pd.Series(coef_s / sd, index=cols), intercept


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def time_folds(n: int, n_folds: int = cfg.CV_FOLDS):
    """Expanding-window folds: always train on the past, test on the future.

    Random k-fold would let the model train on next winter to predict last winter,
    which for a forecasting tool is cheating with extra steps.
    """
    edges = np.linspace(n // 2, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        train = np.arange(0, edges[i])
        test = np.arange(edges[i], edges[i + 1])
        if len(train) and len(test):
            yield train, test


def cv_rmse(df: pd.DataFrame, drivers: list[str], k: int, decay: float,
            alpha: float, t0: pd.Timestamp) -> float:
    """Mean out-of-fold RMSE in log space, for one category and one config.

    The design is built once over the whole history and then sliced. Rebuilding it
    per fold would restart the media adstock at each fold boundary and quietly
    understate carryover.
    """
    X = feat.build_design(df, drivers, k, decay, t0)
    y = np.log(df["volume"].to_numpy(dtype=float))

    errs = []
    for tr, te in time_folds(len(df)):
        coef, intercept = _fit_design(X.iloc[tr], y[tr], alpha)
        pred = X.iloc[te].to_numpy() @ coef.to_numpy() + intercept
        errs.append(np.sqrt(np.mean((y[te] - pred) ** 2)))
    return float(np.mean(errs))


def choose_config(df: pd.DataFrame, drivers: list[str], t0: pd.Timestamp,
                  grid: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Search the grid for this category's best (fourier_k, adstock_decay, alpha).

    Returns the winner and the full scored grid, so the choice can be inspected
    rather than taken on trust.
    """
    grid = grid or cfg.GRID
    rows = []
    for k in grid["fourier_k"]:
        for decay in grid["adstock_decay"]:
            for alpha in grid["alpha"]:
                rows.append(
                    {
                        "fourier_k": k, "adstock_decay": decay, "alpha": alpha,
                        "cv_rmse": cv_rmse(df, drivers, k, decay, alpha, t0),
                    }
                )
    table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best = table.iloc[0]
    return (
        {"fourier_k": int(best["fourier_k"]),
         "adstock_decay": float(best["adstock_decay"]),
         "alpha": float(best["alpha"])},
        table,
    )


def best_alpha(df, drivers, k, decay, t0, alphas=None) -> tuple[float, float]:
    """Tune only the ridge strength, holding the shape fixed. Used during selection."""
    alphas = alphas or cfg.GRID["alpha"]
    scored = [(cv_rmse(df, drivers, k, decay, a, t0), a) for a in alphas]
    rmse, a = min(scored)
    return a, rmse


# --------------------------------------------------------------------------
# The fitted model
# --------------------------------------------------------------------------
@dataclass
class CategoryModel:
    """One category's model. It knows nothing about any other category."""

    category: str
    drivers: list[str]
    fourier_k: int
    adstock_decay: float
    alpha: float
    coef: pd.Series  # over design columns
    intercept: float
    t0: pd.Timestamp
    ref_logs: pd.Series  # historic mean log level of each driver: the decomposition anchor
    log_sd: pd.Series  # historic sd of each log driver: the impact scale
    ref_controls: pd.Series  # historic mean of trend / seasonality / holiday columns
    metrics: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- predict
    @property
    def elasticities(self) -> pd.Series:
        return pd.Series({d: float(self.coef[f"l_{d}"]) for d in self.drivers})

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        return feat.build_design(df, self.drivers, self.fourier_k,
                                 self.adstock_decay, self.t0)

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        X = self.design(df)
        return X[self.coef.index].to_numpy() @ self.coef.to_numpy() + self.intercept

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))

    # ------------------------------------------------------------- attribution
    def impacts(self) -> pd.DataFrame:
        """How much each driver moves this category's volume, and in which direction.

        `impact_pct` is the signed % volume change from a one-standard-deviation
        move in the driver, using that driver's own historic variation. It answers
        "how much does this lever actually matter here", which elasticity alone
        does not: a big elasticity on a driver that never moves is not a lever.
        """
        rows = []
        for d in self.drivers:
            e = float(self.elasticities[d])
            sd = float(self.log_sd[d])
            rows.append(
                {
                    "driver": d,
                    "label": cfg.DRIVERS[d]["label"],
                    "group": cfg.DRIVERS[d]["group"],
                    "elasticity": e,
                    "log_sd": sd,
                    "impact_pct": 100.0 * (np.exp(e * sd) - 1),
                    "direction": "increases volume" if e > 0 else "decreases volume",
                }
            )
        out = pd.DataFrame(rows)
        out["abs_impact"] = out["impact_pct"].abs()
        return out.sort_values("abs_impact", ascending=False).drop(
            columns="abs_impact").reset_index(drop=True)

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split predicted volume into base + trend + seasonality + holiday + each driver.

        The model is additive in logs, so each block has a clean log contribution
        measured against its historic average. Those are converted to volume
        multiplicatively, which guarantees the parts sum exactly to the prediction
        rather than approximately.
        """
        X = self.design(df)
        pred = self.predict(df)

        parts = {}
        for d in self.drivers:
            col = f"l_{d}"
            parts[d] = float(self.coef[col]) * (X[col].to_numpy() - self.ref_logs[d])

        blocks = {
            "trend": ["trend"],
            "seasonality": feat.fourier_cols(self.fourier_k),
            "holiday": feat.HOLIDAY_COLS,
        }
        for name, cols in blocks.items():
            parts[name] = sum(
                float(self.coef[c]) * (X[c].to_numpy() - self.ref_controls[c])
                for c in cols
            )

        total_log = sum(parts.values())
        base = pred / np.exp(total_log)
        incremental = pred - base

        out = pd.DataFrame({"date": df["date"].to_numpy(), "predicted": pred, "base": base})
        denom = np.where(np.abs(total_log) < 1e-12, np.nan, total_log)
        for name, lp in parts.items():
            share = np.where(np.isnan(denom), 0.0, lp / denom)
            out[name] = incremental * share
        return out

    def contribution_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Total volume each block contributed over `df`, as units and as % of predicted."""
        dec = self.decompose(df)
        cols = [c for c in dec.columns if c not in ("date", "predicted")]
        total = dec["predicted"].sum()
        s = dec[cols].sum()
        return pd.DataFrame(
            {"volume": s, "pct_of_total": 100.0 * s / total}
        ).sort_values("volume", ascending=False)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_category(df: pd.DataFrame, category: str, drivers: list[str],
                 config: dict, t0: pd.Timestamp) -> CategoryModel:
    """Fit one category on the rows given. `df` must be that category only."""
    df = df.sort_values("date").reset_index(drop=True)
    k, decay, alpha = config["fourier_k"], config["adstock_decay"], config["alpha"]

    X = feat.build_design(df, drivers, k, decay, t0)
    y = np.log(df["volume"].to_numpy(dtype=float))
    coef, intercept = _fit_design(X, y, alpha)

    ref_logs = pd.Series({d: float(X[f"l_{d}"].mean()) for d in drivers})
    log_sd = pd.Series({d: float(X[f"l_{d}"].std(ddof=1)) for d in drivers})
    ref_controls = X[feat.control_cols(k)].mean()

    return CategoryModel(
        category=category, drivers=list(drivers), fourier_k=k, adstock_decay=decay,
        alpha=alpha, coef=coef, intercept=intercept, t0=t0, ref_logs=ref_logs,
        log_sd=log_sd, ref_controls=ref_controls,
    )


def score(model: CategoryModel, df: pd.DataFrame) -> dict:
    """Accuracy of a fitted model on any slice of that category's data."""
    actual = df["volume"].to_numpy(dtype=float)
    pred = model.predict(df)
    resid_log = np.log(actual) - np.log(pred)
    ss_res = float(np.sum(resid_log ** 2))
    ss_tot = float(np.sum((np.log(actual) - np.log(actual).mean()) ** 2))
    return {
        "n_weeks": int(len(df)),
        "mape": float(np.mean(np.abs(pred / actual - 1)) * 100),
        "wape": float(np.sum(np.abs(pred - actual)) / np.sum(actual) * 100),
        "r2_log": float(1 - ss_res / ss_tot) if ss_tot else float("nan"),
        "bias_pct": float((pred.sum() / actual.sum() - 1) * 100),
    }


def save(models: dict[str, CategoryModel], path=None) -> None:
    path = path or cfg.MODELS_PKL
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load(path=None) -> dict[str, CategoryModel]:
    path = path or cfg.MODELS_PKL
    with open(path, "rb") as f:
        return pickle.load(f)
