"""Response models over an arbitrary, possibly per-category, feature set.

Two shapes, sharing one signed-ridge core:

  `fit_pooled`   one model over every category and every candidate driver. The
                 shipped model's shape, widened to include the decoys. The
                 baseline everything else is measured against.

  `fit_percat`   one model per category, each fitted on *only the drivers that
                 category selected*. Drivers a category dropped do not vanish
                 from the world -- they enter the fit as a fixed offset at a
                 fallback elasticity, and come back out in `elasticities` at that
                 same value, so the scenario engine still has a number for every
                 driver in every category.

The fallback is the whole design question. Two answers are implemented:

  fallback="zero"    a dropped driver has elasticity 0. Honest about what the
                     category's data supports, but it hands a planner a slider
                     that does nothing, and it throws away the cross-category
                     evidence that the driver matters in general.

  fallback="pooled"  a dropped driver keeps the pooled elasticity. The category
                     could not estimate it from 66 months, so it borrows the
                     estimate from all 330. A driver dropped in *every* category
                     falls back to zero instead -- if no category can see it, the
                     pooled estimate is measuring noise, not borrowing strength.

That last clause is what makes selection worth doing for a scenario tool rather
than just for accuracy: it is the mechanism that removes a decoy's slider
entirely, instead of leaving it wired to a pooled coefficient fitted on nothing.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402
from src.model import SignedRidge, _time_folds  # noqa: E402

import fs_features  # noqa: E402

MONTH_COLS = fs_features.MONTH_COLS
CAT_COLS = fs_features.CAT_COLS


# --------------------------------------------------------------------------
# Signed-ridge plumbing, driven by a spec rather than by config.DRIVERS
# --------------------------------------------------------------------------
def coef_bounds(spec: dict, num_cols: list[str], dummy_cols: list[str]):
    """Sign constraints over the transformed design: numerics first, then dummies.

    StandardScaler's scale is strictly positive, so constraining the scaled
    coefficient constrains the raw elasticity to the same sign. Decoys carry
    sign 0 and are left free -- constraining a decoy would be cheating, since the
    engine is supposed to discover that it has no reliable sign at all.
    """
    lo, hi = [], []
    for col in num_cols:
        sign = spec[col[2:]]["sign"] if col.startswith("l_") else 0  # trend is free
        lo.append(-np.inf if sign <= 0 else 0.0)
        hi.append(np.inf if sign >= 0 else 0.0)
    for _ in dummy_cols:  # fixed effects stay free
        lo.append(-np.inf)
        hi.append(np.inf)
    return np.array(lo), np.array(hi)


def make_pipeline(spec: dict, num_cols: list[str], dummy_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("fe", "passthrough", dummy_cols)]
    )
    return Pipeline(
        [("pre", pre), ("ridge", SignedRidge(bounds=coef_bounds(spec, num_cols, dummy_cols)))]
    )


def _fit_pipeline(train, spec, t0, num_cols, dummy_cols, offset=None):
    """CV the ridge penalty on expanding time folds; `offset` is subtracted from y."""
    X = fs_features.build_design(train, spec, t0)[num_cols + dummy_cols]
    y = np.log(train["volume"].to_numpy(dtype=float))
    if offset is not None:
        y = y - offset
    gs = GridSearchCV(
        make_pipeline(spec, num_cols, dummy_cols),
        {"ridge__alpha": config.RIDGE_ALPHAS},
        cv=list(_time_folds(train["date"])),
        scoring="neg_mean_squared_error",
    )
    gs.fit(X, y)
    return gs.best_estimator_, float(gs.best_params_["ridge__alpha"])


def _raw_coefs(pipe: Pipeline, num_cols: list[str]) -> dict[str, float]:
    """Undo the StandardScaler to read coefficients back as raw elasticities."""
    scaler: StandardScaler = pipe.named_steps["pre"].named_transformers_["num"]
    raw = pipe.named_steps["ridge"].coef_[: len(num_cols)] / scaler.scale_
    return dict(zip(num_cols, (float(v) for v in raw)))


def _holdout_block(pred, actual, te) -> dict:
    return {
        "months": config.HOLDOUT_MONTHS,
        "mape": float(np.mean(np.abs(pred / actual - 1)) * 100),
        "r2_volume": float(r2_score(actual, pred)),
        "by_category": {
            c: float(
                np.mean(
                    np.abs(
                        pred[(te["category"] == c).to_numpy()]
                        / actual[(te["category"] == c).to_numpy()] - 1
                    )
                ) * 100
            )
            for c in config.CATEGORIES
        },
    }


def _reference_tables(panel: pd.DataFrame, spec: dict):
    """Per-category mean and sd of each log driver: the baseline anchor + importance scale."""
    adstocked = fs_features.apply_adstock(panel, spec).reset_index(drop=True)
    logs = fs_features.log_features(adstocked, spec)
    logs["category"] = adstocked["category"]
    names = list(spec)
    ref = pd.DataFrame({n: logs.groupby("category")[f"l_{n}"].mean() for n in names})
    sd = pd.DataFrame({n: logs.groupby("category")[f"l_{n}"].std() for n in names})
    return ref.loc[config.CATEGORIES], sd.loc[config.CATEGORIES]


# --------------------------------------------------------------------------
# Arm A: pooled over every category and every candidate driver
# --------------------------------------------------------------------------
@dataclass
class PooledModel:
    name: str
    spec: dict
    pipe: Pipeline
    t0: pd.Timestamp
    alpha: float
    num_cols: list[str]
    elasticities: pd.Series  # one number per driver, shared by all categories
    ref_logs: pd.DataFrame
    log_sd: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        X = fs_features.build_design(df, self.spec, self.t0)
        return self.pipe.predict(X[self.num_cols + CAT_COLS + MONTH_COLS])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))

    def elasticity_frame(self) -> pd.DataFrame:
        """Broadcast to category x driver so every arm scores through one interface."""
        names = list(self.spec)
        return pd.DataFrame(
            np.tile(self.elasticities[names].to_numpy(), (len(config.CATEGORIES), 1)),
            index=config.CATEGORIES,
            columns=names,
        )


def fit_pooled(panel: pd.DataFrame, spec: dict, name: str = "pooled") -> PooledModel:
    panel = panel.sort_values(["category", "date"]).reset_index(drop=True)
    t0 = pd.Timestamp(config.HISTORY_START)
    names = list(spec)
    num_cols = fs_features.log_cols(names) + ["trend"]
    dummies = CAT_COLS + MONTH_COLS

    cutoff = panel["date"].max() - pd.DateOffset(months=config.HOLDOUT_MONTHS)
    te_mask = (panel["date"] > cutoff).to_numpy()
    bt_pipe, _ = _fit_pipeline(panel[~te_mask], spec, t0, num_cols, dummies)
    # Design built over the whole panel then sliced: rebuilding it on the holdout
    # rows alone would restart media adstock at the seam.
    X_all = fs_features.build_design(panel, spec, t0)
    bt_pred = np.exp(bt_pipe.predict(X_all.loc[te_mask, num_cols + dummies]))
    holdout = _holdout_block(bt_pred, panel.loc[te_mask, "volume"].to_numpy(), panel[te_mask])

    pipe, alpha = _fit_pipeline(panel, spec, t0, num_cols, dummies)
    raw = _raw_coefs(pipe, num_cols)
    elasticities = pd.Series({n: raw[f"l_{n}"] for n in names})
    ref, sd = _reference_tables(panel, spec)

    y = np.log(panel["volume"].to_numpy())
    fitted = pipe.predict(X_all[num_cols + dummies])
    metrics = {
        "alpha": alpha,
        "n_obs": int(len(panel)),
        "n_drivers": len(names),
        "n_coefficients": int(len(num_cols) + len(dummies) + 1),
        "r2_log": float(r2_score(y, fitted)),
        "mape_in_sample": float(np.mean(np.abs(np.exp(fitted) / np.exp(y) - 1)) * 100),
        "holdout": holdout,
    }
    return PooledModel(name, spec, pipe, t0, alpha, num_cols, elasticities, ref, sd, metrics)


# --------------------------------------------------------------------------
# Arms B-E: one model per category, over that category's own feature set
# --------------------------------------------------------------------------
@dataclass
class FixedSelection:
    """A selection decided up front -- the control arms, and anything hand-specified.

    Matches the surface `fit_percat` needs from a `selection.SelectionReport`, so
    the arms with no engine behind them go through exactly the same code path.
    """

    selected: dict[str, list[str]]
    global_selected: list[str]

    def __call__(self, _frame) -> "FixedSelection":
        return self


def resolve_fallback_betas(
    selected: dict[str, list[str]], spec: dict, fallback: str,
    pooled_betas: pd.Series | None = None,
    global_selected: list[str] | None = None,
) -> pd.DataFrame:
    """Elasticity applied to each category's *dropped* drivers.

    Selected drivers get 0 here -- their coefficient comes from the fit. Under
    "pooled", a driver that failed the engine's global gate is zeroed rather than
    given the pooled estimate: the gate has already ruled it is not a driver
    anywhere, so its pooled coefficient is fitted noise, and carrying it would
    leave a decoy holding a live slider. Drivers that passed the gate but lost
    their category keep the pooled value -- that is the borrowing this is for.
    """
    names = list(spec)
    out = pd.DataFrame(0.0, index=config.CATEGORIES, columns=names)
    if fallback == "zero":
        return out
    if fallback != "pooled":  # pragma: no cover - caller guard
        raise ValueError(f"unknown fallback: {fallback}")
    if pooled_betas is None:
        raise ValueError("fallback='pooled' needs pooled_betas")

    allowed = set(names if global_selected is None else global_selected)
    for cat in config.CATEGORIES:
        for n in names:
            if n not in selected[cat] and n in allowed:
                out.loc[cat, n] = float(pooled_betas[n])
    return out


def _offset(logs: pd.DataFrame, betas: pd.Series, drivers) -> np.ndarray:
    """Log-volume implied by the dropped drivers held at their fallback elasticity."""
    if not len(drivers):
        return np.zeros(len(logs))
    return sum(float(betas[d]) * logs[f"l_{d}"].to_numpy() for d in drivers)


@dataclass
class CategoryModel:
    name: str
    spec: dict
    selected: dict[str, list[str]]
    fallback: str
    fallback_betas: pd.DataFrame
    pipes: dict[str, Pipeline]
    num_cols: dict[str, list[str]]
    t0: pd.Timestamp
    alphas: dict[str, float]
    elasticities: pd.DataFrame  # index=category, columns=driver; fitted or fallback
    ref_logs: pd.DataFrame
    log_sd: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def dropped(self, cat: str) -> list[str]:
        return [n for n in self.spec if n not in self.selected[cat]]

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        """Route each row to its category's model, then add back its offset."""
        adstocked = fs_features.apply_adstock(df, self.spec)
        X = fs_features.build_design(df, self.spec, self.t0)
        logs = fs_features.log_features(adstocked, self.spec)
        cats = adstocked["category"].to_numpy()
        out = np.empty(len(X), dtype=float)
        for cat, pipe in self.pipes.items():
            m = cats == cat
            if not m.any():
                continue
            off = _offset(logs.loc[m], self.fallback_betas.loc[cat], self.dropped(cat))
            out[m] = pipe.predict(X.loc[m, self.num_cols[cat] + MONTH_COLS]) + off
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(df))

    def elasticity_frame(self) -> pd.DataFrame:
        return self.elasticities

    def importance(self, category: str | None = None) -> pd.DataFrame:
        """|elasticity| x sd(log driver) within the category: % swing per 1sd."""
        cats = list(self.pipes) if category is None else [category]
        rows = []
        for n in self.spec:
            e = float(np.mean([self.elasticities.loc[c, n] for c in cats]))
            impact = float(
                np.mean([abs(self.elasticities.loc[c, n]) * self.log_sd.loc[c, n]
                         for c in cats])
            )
            rows.append(
                {
                    "driver": n,
                    "label": self.spec[n]["label"],
                    "group": self.spec[n]["group"],
                    "elasticity": e,
                    "impact_pct": 100.0 * impact,
                    "kept_in": sum(n in self.selected[c] for c in cats),
                }
            )
        out = pd.DataFrame(rows).sort_values("impact_pct", ascending=False)
        return out.reset_index(drop=True)

    def save(self, path=None) -> None:
        path = Path(path) if path else Path(__file__).parent / "artifacts" / f"{self.name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "CategoryModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def _fit_categories(panel, spec, t0, selected, fallback_betas):
    """One signed-ridge fit per category, over that category's own driver subset."""
    pipes, alphas, num_cols = {}, {}, {}
    for cat in config.CATEGORIES:
        sub = panel[panel["category"] == cat].sort_values("date")
        cols = fs_features.log_cols(selected[cat]) + ["trend"]
        dropped = [n for n in spec if n not in selected[cat]]
        logs = fs_features.log_features(fs_features.apply_adstock(sub, spec), spec)
        off = _offset(logs, fallback_betas.loc[cat], dropped)
        pipes[cat], alphas[cat] = _fit_pipeline(sub, spec, t0, cols, MONTH_COLS, offset=off)
        num_cols[cat] = cols
    return pipes, alphas, num_cols


def fit_percat(
    panel: pd.DataFrame,
    spec: dict,
    *,
    selector,
    fallback: str = "zero",
    pooled_betas: pd.Series | None = None,
    name: str = "percat",
) -> CategoryModel:
    """`selector(frame)` returns anything with `.selected` and `.global_selected`.

    It is called *again* on the backtest's training rows rather than reusing the
    full-sample selection. Selecting on all 66 months and then scoring on the last
    12 of them leaks the holdout into the feature set and flatters every selected
    arm; re-running it on the training slice is the only version of this number
    that means anything.
    """
    panel = panel.sort_values(["category", "date"]).reset_index(drop=True)
    t0 = pd.Timestamp(config.HISTORY_START)
    names = list(spec)

    # --- backtest, with selection re-run on the training rows only ---
    cutoff = panel["date"].max() - pd.DateOffset(months=config.HOLDOUT_MONTHS)
    te_mask = (panel["date"] > cutoff).to_numpy()
    tr = panel[~te_mask].reset_index(drop=True)
    rep_tr = selector(tr)
    sel_tr = rep_tr.selected
    fb_tr = resolve_fallback_betas(
        sel_tr, spec, fallback, pooled_betas, rep_tr.global_selected
    )
    bt_pipes, _, bt_cols = _fit_categories(tr, spec, t0, sel_tr, fb_tr)
    bt = CategoryModel(
        name=f"{name}-backtest", spec=spec, selected=sel_tr, fallback=fallback,
        fallback_betas=fb_tr, pipes=bt_pipes, num_cols=bt_cols, t0=t0, alphas={},
        elasticities=pd.DataFrame(0.0, index=config.CATEGORIES, columns=names),
        ref_logs=pd.DataFrame(), log_sd=pd.DataFrame(),
    )
    bt_pred = bt.predict(panel)[te_mask]  # predict over the full panel, then slice
    holdout = _holdout_block(bt_pred, panel.loc[te_mask, "volume"].to_numpy(), panel[te_mask])

    # --- final fit on the full history ---
    rep = selector(panel)
    selected, global_selected = rep.selected, rep.global_selected
    fallback_betas = resolve_fallback_betas(
        selected, spec, fallback, pooled_betas, global_selected
    )
    pipes, alphas, num_cols = _fit_categories(panel, spec, t0, selected, fallback_betas)

    elasticities = fallback_betas.copy()
    for cat in config.CATEGORIES:
        raw = _raw_coefs(pipes[cat], num_cols[cat])
        for n in selected[cat]:
            elasticities.loc[cat, n] = raw[f"l_{n}"]

    ref, sd = _reference_tables(panel, spec)
    m = CategoryModel(
        name=name, spec=spec, selected=selected, fallback=fallback,
        fallback_betas=fallback_betas, pipes=pipes, num_cols=num_cols, t0=t0,
        alphas=alphas, elasticities=elasticities, ref_logs=ref, log_sd=sd,
    )

    y = np.log(panel["volume"].to_numpy())
    fitted = m.predict_log(panel)
    n_coef = sum(len(c) + len(MONTH_COLS) + 1 for c in num_cols.values())
    # Borrowing strength assumes there is strength to borrow. A driver with a
    # business sign prior whose pooled elasticity came back at zero is sitting on
    # its constraint boundary -- the pooled fit could not identify it either, and
    # pinning a category to that value is not pooling, it is deletion wearing a
    # disguise. Worse, the effect does not disappear: whatever is collinear with
    # the driver absorbs it. Flagged here so the arm cannot look clean while it
    # happens.
    # Only meaningful under fallback="pooled"; a zero under fallback="zero" is the
    # stated design, not a degeneracy.
    degenerate = sorted(
        {
            n for cat in config.CATEGORIES for n in names
            if fallback == "pooled" and n not in selected[cat] and n in global_selected
            and spec[n]["sign"] != 0
            and abs(float(fallback_betas.loc[cat, n])) < 1e-8
        }
    )
    m.metrics = {
        "alphas": alphas,
        "fallback": fallback,
        "n_obs": int(len(panel)),
        "n_coefficients": int(n_coef),
        "n_selected_total": int(sum(len(v) for v in selected.values())),
        "selected": {c: list(v) for c, v in selected.items()},
        "global_selected": list(global_selected),
        "degenerate_fallbacks": degenerate,
        "selected_backtest": {c: list(v) for c, v in sel_tr.items()},
        "global_selected_backtest": list(rep_tr.global_selected),
        "r2_log": float(r2_score(y, fitted)),
        "mape_in_sample": float(np.mean(np.abs(np.exp(fitted) / np.exp(y) - 1)) * 100),
        "holdout": holdout,
    }
    return m


def all_selected(spec: dict) -> FixedSelection:
    """Every category keeps every candidate -- the no-selection control."""
    names = list(spec)
    return FixedSelection({c: list(names) for c in config.CATEGORIES}, names)


def oracle_selected(spec: dict, true_drivers: list[str]) -> FixedSelection:
    """Every category keeps exactly the drivers that truly move volume.

    Not achievable without knowing the answer -- it is the ceiling the engine is
    measured against, not a competitor.
    """
    keep = [n for n in spec if n in true_drivers]
    return FixedSelection({c: list(keep) for c in config.CATEGORIES}, keep)
