"""The feature-selection engine: which drivers has each category earned the right to use?

Two stages, because "is this a driver at all?" and "can *this category* estimate
it?" are different questions with different answers, and collapsing them into one
is the mistake that makes per-category selection dangerous.

  GLOBAL GATE (330 observations, pooled across categories, category fixed effects
  absorbed). Does this driver move volume anywhere? Food CPI does, and every
  category knows it, but no single category can prove it: within one category the
  macro series is one shared path, nearly collinear with that category's own time
  trend. A per-category filter drops it, correctly, and a naive engine would then
  conclude it is noise and delete the slider. The global gate is what tells food
  CPI apart from a random walk that happens to drift the right way. It is
  deliberately permissive -- excluding non-drivers is its whole job.

  PER-CATEGORY GATE (66 observations). Given that a driver is real, can this
  category estimate its own elasticity for it, or should it borrow the pooled one?

The three filters below run at both levels.

Sixty-six months per category and thirteen candidate drivers, several of them
badly collinear. Any single selection criterion has a failure mode here, so the
engine runs three that fail in different directions and requires a driver to pass
all of them.

  1. STABILITY  -- does the driver survive resampling?
     Everything is residualised on the calendar first (intercept, linear trend,
     eleven month dummies), so a driver gets credit only for variation the model
     does not already explain with a clock. Then: 200 moving-block bootstrap
     draws, a lasso on each, at three penalties spanning a range. A driver's
     score is the fraction of (draw, penalty) pairs in which it is non-zero.
     Blocks rather than iid resampling because the residuals are autocorrelated
     and iid draws would break the dependence the sample size actually reflects.
     This is the filter that catches drivers the data cannot pin down -- the ones
     that look decisive in the full sample and vanish when you jiggle it.

  2. SIGN PRIOR -- does the driver agree with what we know?
     A light unconstrained ridge, on the same residualised design. If the
     estimate lands on the opposite side of zero from the driver's business prior
     by more than a tolerance, the category cannot identify it: something else in
     the design is absorbing the effect. Note the fit here is deliberately
     *unconstrained*, unlike the shipped model -- constraining it would hide the
     disagreement, which is the signal being looked for. Decoys carry no prior
     and skip this filter entirely.

  3. MATERIALITY -- is the effect big enough to plan against?
     |elasticity| x sd(log driver) within the category, as a percentage volume
     swing per one-sd move. A driver that is real, correctly signed and stable
     but worth 0.2% of volume is a slider that wastes a planner's attention.

A driver is kept only if it passes all three. Leave-one-driver-out CV is computed
too, but as a diagnostic rather than a gate: with price, food CPI and the trend as
collinear as they are here, forecast error barely moves when you reallocate
between them, so CV cannot see the difference -- the same blindness documented in
`scripts/experiment_per_category.py`. It is reported so you can watch it fail to
discriminate rather than take its word for anything.

    python selection.py        # prints the full table, no fitting downstream
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import lasso_path  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

import config  # noqa: E402
from src.model import _time_folds  # noqa: E402

import fs_config  # noqa: E402
import fs_features  # noqa: E402
import fs_model  # noqa: E402


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def _residualise(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Sweep the calendar block `C` out of `A` (Frisch-Waugh). Works for 1-D or 2-D."""
    coef, *_ = np.linalg.lstsq(C, A, rcond=None)
    return A - C @ coef


def _standardise(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sd = Z.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return Z / sd, sd


def _block_resample(blocks: list[np.ndarray], block: int, rng) -> np.ndarray:
    """Moving-block resample within each contiguous group, concatenated.

    Groups are categories. Drawing blocks across a category boundary would splice
    Dairy's June onto Beverages' July and manufacture dependence that isn't there.
    """
    out = []
    for pos in blocks:
        n = len(pos)
        b = min(block, n)
        starts = rng.integers(0, n - b + 1, int(np.ceil(n / b)))
        out.append(pos[np.concatenate([np.arange(s, s + b) for s in starts])[:n]])
    return np.concatenate(out)


def stability(
    Zr: np.ndarray, yr: np.ndarray, rng: np.random.Generator,
    n_boot: int, block: int, fracs, groups: np.ndarray | None = None) -> np.ndarray:
    """Selection frequency per driver over moving-block bootstrap x lasso penalties.

    Penalties are set as fractions of each draw's own alpha_max -- the smallest
    penalty that zeroes every coefficient -- so the grid means the same thing in
    every draw and in every category, with no tuning constant to carry around.
    """
    n, p = Zr.shape
    Zs, _ = _standardise(Zr)
    if groups is None:
        blocks = [np.arange(n)]
    else:
        blocks = [np.where(groups == g)[0] for g in pd.unique(groups)]
    hits, pos, neg, trials = np.zeros(p), np.zeros(p), np.zeros(p), 0

    for _ in range(n_boot):
        idx = _block_resample(blocks, block, rng)
        Zb = Zs[idx] - Zs[idx].mean(axis=0)
        yb = yr[idx] - yr[idx].mean()

        amax = float(np.max(np.abs(Zb.T @ yb))) / len(yb)
        if amax <= 0:  # degenerate draw
            continue
        alphas = np.sort(np.asarray(fracs, dtype=float) * amax)[::-1]
        _, coefs, _ = lasso_path(Zb, yb, alphas=alphas)
        nz = np.abs(coefs) > 1e-10
        hits += nz.sum(axis=1)
        pos += (nz & (coefs > 0)).sum(axis=1)
        neg += (nz & (coefs < 0)).sum(axis=1)
        trials += len(alphas)

    freq = hits / max(trials, 1)
    # Sign consistency: of the draws where the driver was selected at all, how
    # often did it point the same way? A driver whose direction flips between
    # resamples cannot be planned against, whatever its selection frequency --
    # a planner asking "what happens if I raise this" would get a coin flip.
    with np.errstate(invalid="ignore", divide="ignore"):
        consistency = np.where(hits > 0, np.maximum(pos, neg) / np.maximum(hits, 1), 0.0)
    return freq, consistency


def ridge_elasticities(Zr: np.ndarray, yr: np.ndarray, alpha_frac: float) -> np.ndarray:
    """Light unconstrained ridge on the residualised design, in raw elasticity units.

    Used only for its sign and rough magnitude. The elasticities the tool actually
    reports come from the constrained, CV-tuned fits in `fs_model.py`.
    """
    Zs, sd = _standardise(Zr)
    p = Zs.shape[1]
    alpha = alpha_frac * len(yr)  # scale-free: alpha as a fraction of the sample size
    coef = np.linalg.solve(Zs.T @ Zs + alpha * np.eye(p), Zs.T @ yr)
    return coef / sd


def _cv_mse(sub: pd.DataFrame, spec: dict, t0: pd.Timestamp, drivers: list[str]) -> float:
    cols = fs_features.log_cols(drivers) + ["trend"]
    X = fs_features.build_design(sub, spec, t0)[cols + fs_model.MONTH_COLS]
    y = np.log(sub["volume"].to_numpy(dtype=float))
    gs = GridSearchCV(
        fs_model.make_pipeline(spec, cols, fs_model.MONTH_COLS),
        {"ridge__alpha": config.RIDGE_ALPHAS},
        cv=list(_time_folds(sub["date"])),
        scoring="neg_mean_squared_error",
    )
    gs.fit(X, y)
    return -float(gs.best_score_)


def loo_cv_delta(sub, spec, t0, candidates) -> dict[str, float]:
    """MSE(without driver) - MSE(with all). Positive means the driver earns its place."""
    base = _cv_mse(sub, spec, t0, candidates)
    return {
        d: _cv_mse(sub, spec, t0, [c for c in candidates if c != d]) - base
        for d in candidates
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _apply_filters(
    Zr: np.ndarray, yr: np.ndarray, raw_sd: np.ndarray, spec: dict, names: list[str],
    rng, p: dict, tau_stab: float, tau_impact: float, groups=None,
) -> pd.DataFrame:
    """The four filters, run over one residualised design. Shared by both stages."""
    freq, consistency = stability(Zr, yr, rng, p["n_bootstrap"], p["block_length"],
                                  p["alpha_fractions"], groups=groups)
    elas = ridge_elasticities(Zr, yr, p["ridge_alpha_frac"])
    res_sd = Zr.std(axis=0, ddof=1)

    rows = []
    for i, n in enumerate(names):
        prior = spec[n]["sign"]
        e = float(elas[i])
        violates = (
            prior != 0 and np.sign(e) == -prior and abs(e) > p["sign_violation_tol"]
        )
        impact = 100.0 * abs(e) * float(raw_sd[i])

        if freq[i] < tau_stab:
            keep, reason = False, "unstable"
        elif consistency[i] < p["tau_sign_consistency"]:
            keep, reason = False, "sign flips across resamples"
        elif violates:
            keep, reason = False, "sign contradicts prior"
        elif impact < tau_impact:
            keep, reason = False, "immaterial"
        else:
            keep, reason = True, "kept"

        rows.append(
            {
                "driver": n,
                "group": spec[n]["group"],
                "stability": float(freq[i]),
                "sign_consistency": float(consistency[i]),
                "elasticity_uc": e,
                "impact_pct": impact,
                "log_sd": float(raw_sd[i]),
                "resid_sd": float(res_sd[i]),
                "sign_ok": not violates,
                "keep": keep,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _global_stage(panel, spec, names, t0, rng, p) -> pd.DataFrame:
    """Stage 1: is each candidate a driver at all, anywhere in the panel?

    The calendar block absorbed here is richer than the per-category one: category
    fixed effects, month effects, and a *category-specific* trend. Without the
    interaction a driver could earn its keep purely by drifting at the same rate
    as one category, which is the trap `search_index` is built to spring.
    """
    X = fs_features.build_design(panel, spec, t0)
    trend = X["trend"].to_numpy()
    cat_d = X[fs_features.CAT_COLS].to_numpy()
    C = np.column_stack(
        [np.ones(len(X)), trend, cat_d, cat_d * trend[:, None],
         X[fs_features.MONTH_COLS].to_numpy()]
    )
    Z = X[fs_features.log_cols(names)].to_numpy(dtype=float)
    y = np.log(fs_features.apply_adstock(panel, spec)["volume"].to_numpy(dtype=float))

    # Materiality is scored against *within-category* variation. The pooled sd of
    # log price mostly measures that Household Care costs $7.10 and Dairy $2.60,
    # which is not variation any scenario can move.
    cats = fs_features.apply_adstock(panel, spec)["category"].to_numpy()
    raw_sd = np.array(
        [
            np.mean([Z[cats == c, i].std(ddof=1) for c in config.CATEGORIES])
            for i in range(len(names))
        ]
    )

    tbl = _apply_filters(
        _residualise(Z, C), _residualise(y, C), raw_sd, spec, names, rng, p,
        p["tau_stability_global"], p["tau_impact_pct_global"], groups=cats,
    )
    tbl["keep_data"] = tbl["keep"]  # the engine's unmoderated verdict, kept on record
    tbl["protected"] = [spec[n]["sign"] != 0 for n in tbl["driver"]]
    if p["protect_signed_priors"]:
        rescued = ~tbl["keep"] & tbl["protected"]
        tbl.loc[rescued, "keep"] = True
        tbl.loc[rescued, "reason"] = "protected (business prior)"
    return tbl


@dataclass
class SelectionReport:
    selected: dict[str, list[str]]  # category -> drivers it fits itself
    global_selected: list[str]  # drivers that survive the global gate at all
    global_selected_data: list[str]  # ... before the business-prior override
    table: pd.DataFrame  # one row per (category, driver): every score and the reason
    global_table: pd.DataFrame  # one row per driver: the stage-1 verdict
    params: dict

    def matrix(self) -> pd.DataFrame:
        """category x driver grid: 'fit' / 'pool' / '-'. The human-readable answer.

        fit   the category estimates its own elasticity
        pool  a real driver this category cannot identify; borrows the pooled value
        -     failed the global gate; not a driver, no slider
        """
        def mark(row):
            if row["driver"] not in self.global_selected:
                return "-"
            return "fit" if row["keep"] else "pool"

        m = self.table.assign(mark=self.table.apply(mark, axis=1))
        return m.pivot(index="category", columns="driver", values="mark").loc[
            config.CATEGORIES, list(self.table["driver"].unique())
        ]

    def kept_count(self) -> pd.Series:
        return (
            self.table[self.table["keep"]].groupby("driver").size()
            .reindex(self.table["driver"].unique()).fillna(0).astype(int)
        )

    def drop_reasons(self) -> pd.Series:
        return self.table.loc[~self.table["keep"], "reason"].value_counts()


def select(
    panel: pd.DataFrame, spec: dict, params: dict | None = None,
    verbose: bool = False,
) -> SelectionReport:
    """Run both stages. `panel` may be any time slice -- the backtest passes a shorter one."""
    p = {**fs_config.SELECTION, **(params or {})}
    panel = panel.sort_values(["category", "date"]).reset_index(drop=True)
    t0 = pd.Timestamp(config.HISTORY_START)
    names = list(spec)
    rng = np.random.default_rng(p["seed"])

    # --- stage 1: the global gate ---
    global_table = _global_stage(panel, spec, names, t0, rng, p)
    global_selected = global_table.loc[global_table["keep"], "driver"].tolist()
    global_selected_data = global_table.loc[global_table["keep_data"], "driver"].tolist()
    if verbose:
        cut = [n for n in names if n not in global_selected]
        saved = [n for n in global_selected if n not in global_selected_data]
        print(f"    global gate keeps {len(global_selected)}/{len(names)}; "
              f"cut: {', '.join(cut) if cut else 'none'}")
        if saved:
            print(f"    {' ' * 16}rescued by business prior: {', '.join(saved)}")

    # --- stage 2: per category ---
    frames = []
    for cat in config.CATEGORIES:
        sub = panel[panel["category"] == cat].sort_values("date").reset_index(drop=True)
        logs = fs_features.log_features(fs_features.apply_adstock(sub, spec), spec)
        Z = logs[fs_features.log_cols(names)].to_numpy(dtype=float)
        y = np.log(sub["volume"].to_numpy(dtype=float))
        C = fs_features.calendar_block(sub, t0)

        tbl = _apply_filters(
            _residualise(Z, C), _residualise(y, C), Z.std(axis=0, ddof=1),
            spec, names, rng, p, p["tau_stability"], p["tau_impact_pct"],
        )
        tbl.insert(0, "category", cat)
        # A driver the global gate rejected cannot be rescued by one category.
        tbl.loc[~tbl["driver"].isin(global_selected), ["keep", "reason"]] = (
            False, "failed global gate"
        )
        deltas = (
            loo_cv_delta(sub, spec, t0, names) if p["run_loo_cv"]
            else dict.fromkeys(names, np.nan)
        )
        tbl["cv_delta"] = tbl["driver"].map(deltas)
        frames.append(tbl)

        if verbose:
            kept = tbl.loc[tbl["keep"], "driver"].tolist()
            print(f"    {cat:<16} fits {len(kept):>2}/{len(global_selected)}: "
                  f"{', '.join(kept)}")

    table = pd.concat(frames, ignore_index=True)
    selected = {
        cat: table[(table["category"] == cat) & table["keep"]]["driver"].tolist()
        for cat in config.CATEGORIES
    }
    return SelectionReport(
        selected=selected, global_selected=global_selected,
        global_selected_data=global_selected_data, table=table,
        global_table=global_table, params=p,
    )


if __name__ == "__main__":
    import decoys

    panel, _ = decoys.load()
    spec = fs_config.spec()
    print(f"Selecting from {len(spec)} candidates "
          f"({len(fs_config.REAL_NAMES)} real + {len(fs_config.DECOY_NAMES)} decoy) "
          f"across {len(config.CATEGORIES)} categories ...\n")
    rep = select(panel, spec, verbose=True)

    pd.set_option("display.width", 250)
    print("\nStage 1 -- global gate (is it a driver at all?):")
    print(rep.global_table.round(3).to_string(index=False))
    print("\nStage 2 -- fit / pool / drop, by category:")
    print(rep.matrix().to_string())
    print("\n  fit = category estimates its own elasticity")
    print("  pool = real driver, this category cannot identify it, borrows pooled")
    print("  -    = failed the global gate; removed from the tool")
    print("\nDrop reasons:")
    print(rep.drop_reasons().to_string())
    print("\nFull per-category table:")
    print(rep.table.round(3).to_string(index=False))
