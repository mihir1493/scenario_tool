"""Per-category log-log Ridge response model (experiment against the pooled fit).

The pooled model in `src/model.py` assumes every category responds to the drivers
the same way and lets a fixed effect absorb the level difference. That is a strong
assumption: Dairy is a staple and should be far less price elastic than Beverages,
and temperature should move Beverages and Frozen Foods in ways it never moves
Household Care.

This module relaxes it in the bluntest possible way -- one independent model per
category, no sharing at all. Each fit sees ~66 months instead of 330, so the
variance cost is real; whether the bias reduction pays for it is exactly what
`scripts/experiment_per_category.py` measures.

Everything else is held identical to the pooled model on purpose: same design
matrix, same sign constraints, same alpha grid, same expanding-window CV, same
holdout. The only differences are (a) the fit is split by category and (b) the
category fixed effects are dropped, since within one category they are constant
and the intercept already carries the level.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src import features, model as pooled

# A single-category design has no category dummies -- only trend + month FE.
MONTH_COLS = [f"mon_{m}" for m in range(2, 13)]
DESIGN_COLS = features.NUM_COLS + MONTH_COLS

PERCAT_MODEL_PKL = config.ARTIFACT_DIR / "response_model_percat.pkl"


def _elasticities_from(pipe: Pipeline) -> pd.Series:
    """Undo the StandardScaler to read coefficients back as raw elasticities."""
    scaler: StandardScaler = pipe.named_steps["pre"].named_transformers_["num"]
    coef = pipe.named_steps["ridge"].coef_
    raw = coef[: len(features.NUM_COLS)] / scaler.scale_
    return pd.Series({n: float(raw[i]) for i, n in enumerate(config.DRIVER_NAMES)})


@dataclass
class PerCategoryResponseModel:
    """Mirrors the `ResponseModel` surface, but keyed by category throughout.

    `elasticities` is a DataFrame (category x driver) rather than a Series, so
    anything that consumed the pooled `model.elasticities[name]` needs to become
    `model.elasticities.loc[category, name]`.
    """

    pipes: dict[str, Pipeline]
    t0: pd.Timestamp
    alphas: dict[str, float]
    elasticities: pd.DataFrame  # index=category, columns=driver
    ref_logs: pd.DataFrame  # index=category, columns=driver
    log_sd: pd.DataFrame  # index=category, columns=driver (within-category sd)
    metrics: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- predict
    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        """Route each row to its category's model.

        Rows come back in (category, date) order, matching the pooled model --
        `build_design` sorts internally, so callers sort before comparing.
        """
        X = features.build_design(df, self.t0)  # sorted by (category, date)
        cats = features.apply_adstock(df)["category"].to_numpy()
        out = np.empty(len(X), dtype=float)
        for cat, pipe in self.pipes.items():
            m = cats == cat
            if m.any():
                out[m] = pipe.predict(X.loc[m, DESIGN_COLS])
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))

    # ------------------------------------------------------------ attribution
    def importance(self, category: str | None = None) -> pd.DataFrame:
        """|elasticity| x sd(log driver) within the category.

        With `category=None` the per-category impacts are averaged, which is the
        closest analogue to the pooled model's single importance table.
        """
        cats = list(self.pipes) if category is None else [category]
        rows = []
        for name in config.DRIVER_NAMES:
            e = float(np.mean([self.elasticities.loc[c, name] for c in cats]))
            impact = float(
                np.mean(
                    [
                        abs(self.elasticities.loc[c, name]) * self.log_sd.loc[c, name]
                        for c in cats
                    ]
                )
            )
            rows.append(
                {
                    "driver": name,
                    "label": config.DRIVERS[name]["label"],
                    "group": config.DRIVERS[name]["group"],
                    "elasticity": e,
                    "log_sd": float(np.mean([self.log_sd.loc[c, name] for c in cats])),
                    "impact_pct": 100.0 * impact,
                    "direction": "increases volume" if e > 0 else "decreases volume",
                }
            )
        out = pd.DataFrame(rows).sort_values("impact_pct", ascending=False)
        out["impact_share"] = out["impact_pct"] / out["impact_pct"].sum()
        return out.reset_index(drop=True)

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Base + per-driver incremental volume, using each category's own betas."""
        adstocked = features.apply_adstock(df).reset_index(drop=True)
        logs = features.log_features(adstocked)
        pred = self.predict(df)
        cat_col = adstocked["category"]

        log_contrib = pd.DataFrame(index=adstocked.index)
        for name in config.DRIVER_NAMES:
            beta = cat_col.map(self.elasticities[name]).to_numpy()
            ref = cat_col.map(self.ref_logs[name]).to_numpy()
            log_contrib[name] = beta * (logs[f"l_{name}"].to_numpy() - ref)

        total_log = log_contrib.sum(axis=1).to_numpy()
        base = pred / np.exp(total_log)
        incremental = pred - base

        denom = np.where(np.abs(total_log) < 1e-9, np.nan, total_log)
        out = pd.DataFrame(
            {"date": adstocked["date"], "category": cat_col,
             "predicted": pred, "base": base}
        )
        for name in config.DRIVER_NAMES:
            share = np.where(np.isnan(denom), 0.0, log_contrib[name].to_numpy() / denom)
            out[name] = incremental * share
        return out

    def save(self, path=PERCAT_MODEL_PKL) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=PERCAT_MODEL_PKL) -> "PerCategoryResponseModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def _fit_all(panel: pd.DataFrame, t0: pd.Timestamp) -> tuple[dict, dict]:
    pipes, alphas = {}, {}
    for cat in config.CATEGORIES:
        sub = panel[panel["category"] == cat]
        pipe, alpha = pooled._fit_pipeline(sub, t0, dummy_cols=MONTH_COLS)
        pipes[cat], alphas[cat] = pipe, alpha
    return pipes, alphas


def fit(panel: pd.DataFrame) -> PerCategoryResponseModel:
    panel = panel.sort_values(["category", "date"]).reset_index(drop=True)
    t0 = pd.Timestamp(config.HISTORY_START)

    # --- backtest: identical holdout protocol to the pooled model ---
    cutoff = panel["date"].max() - pd.DateOffset(months=config.HOLDOUT_MONTHS)
    te_mask = (panel["date"] > cutoff).to_numpy()
    tr, te = panel[~te_mask], panel[te_mask]
    bt_pipes, _ = _fit_all(tr, t0)

    # Design built once over the whole panel, then sliced -- see the note in
    # model.fit: rebuilding on the holdout rows restarts media adstock at the seam.
    X_te = features.build_design(panel, t0).loc[te_mask]
    bt_pred = np.empty(len(te), dtype=float)
    for cat, pipe in bt_pipes.items():
        m = (te["category"] == cat).to_numpy()
        bt_pred[m] = np.exp(pipe.predict(X_te.loc[m, DESIGN_COLS]))
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

    # --- final models on the full history ---
    pipes, alphas = _fit_all(panel, t0)

    elasticities = pd.DataFrame(
        {cat: _elasticities_from(pipes[cat]) for cat in config.CATEGORIES}
    ).T.loc[config.CATEGORIES, config.DRIVER_NAMES]

    adstocked = features.apply_adstock(panel).reset_index(drop=True)
    logs = features.log_features(adstocked)
    logs["category"] = adstocked["category"]
    ref_logs = pd.DataFrame(
        {n: logs.groupby("category")[f"l_{n}"].mean() for n in config.DRIVER_NAMES}
    )
    # Within-category sd: the pooled model uses one pooled sd, but a per-category
    # importance score has to be scored against that category's own variation.
    log_sd = pd.DataFrame(
        {n: logs.groupby("category")[f"l_{n}"].std() for n in config.DRIVER_NAMES}
    )

    y = np.log(panel["volume"].to_numpy())
    X = features.build_design(panel, t0)
    fitted = np.empty(len(panel), dtype=float)
    for cat, pipe in pipes.items():
        m = (panel["category"] == cat).to_numpy()
        fitted[m] = pipe.predict(X.loc[m, DESIGN_COLS])

    # Params burned: 9 drivers + trend + 11 month dummies + intercept, per model.
    n_params = (len(DESIGN_COLS) + 1) * len(config.CATEGORIES)
    metrics = {
        "alphas": alphas,
        "n_obs": int(len(panel)),
        "n_models": len(config.CATEGORIES),
        "n_params_nominal": int(n_params),
        "r2_log": float(r2_score(y, fitted)),
        "r2_volume": float(r2_score(np.exp(y), np.exp(fitted))),
        "mape_in_sample": float(np.mean(np.abs(np.exp(fitted) / np.exp(y) - 1)) * 100),
        "r2_log_by_category": {
            c: float(
                r2_score(y[(panel["category"] == c).to_numpy()],
                         fitted[(panel["category"] == c).to_numpy()])
            )
            for c in config.CATEGORIES
        },
        "holdout": holdout,
    }

    return PerCategoryResponseModel(
        pipes=pipes, t0=t0, alphas=alphas, elasticities=elasticities,
        ref_logs=ref_logs, log_sd=log_sd, metrics=metrics,
    )


if __name__ == "__main__":
    panel = pd.read_csv(config.PANEL_CSV, parse_dates=["date"])
    m = fit(panel)
    m.save()
    print(json.dumps(m.metrics, indent=2))
    print("\nElasticities by category:")
    print(m.elasticities.round(3).to_string())
