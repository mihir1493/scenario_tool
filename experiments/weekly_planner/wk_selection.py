"""Feature selection, run separately for every category.

The method is backward elimination scored on expanding-window cross-validation,
which is the version of feature selection that is easiest to defend to a planner:

    start with every candidate driver
    repeat:
        for each driver still in, ask "what happens to out-of-fold error if I drop it?"
        drop the one whose removal hurts least, provided it hurts less than the tolerance
    stop when every remaining driver is carrying its weight

Two things make it honest rather than just tidy:

  * The score is out-of-fold error on FUTURE weeks, not in-sample fit. Adding a
    driver can only ever improve in-sample fit, so in-sample selection keeps
    everything and answers nothing.
  * A driver whose fitted elasticity contradicts its business prior is removed
    before the search starts. If the data says raising price raises volume, the
    right response is not to keep the coefficient and constrain it to zero -- it is
    to conclude that this category cannot identify that driver and drop it.

Selection runs on the TRAINING rows only. Selecting on all 241 weeks and then
reporting accuracy on the last 52 of them leaks the test set into the feature list
and produces a number that means nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import wk_config as cfg
import wk_features as feat
import wk_model as mdl


def sign_check(df: pd.DataFrame, drivers: list[str], k: int, decay: float,
               t0: pd.Timestamp) -> list[str]:
    """Drivers whose UNCONSTRAINED elasticity contradicts the business prior.

    Fitted without the sign bounds on purpose: the whole point is to hear the
    disagreement. A constrained fit would silently pin the coefficient at zero and
    the driver would look fine.
    """
    X = feat.build_design(df, drivers, k, decay, t0)
    y = np.log(df["volume"].to_numpy(dtype=float))

    cols = list(X.columns)
    Xs, sd = mdl._scale(X.to_numpy(dtype=float))
    free = (np.full(len(cols), -np.inf), np.full(len(cols), np.inf))
    coef_s, _ = mdl.fit_ridge(Xs, y, alpha=1.0, penalise=mdl._penalty_weights(cols),
                              lo=free[0], hi=free[1])
    coef = pd.Series(coef_s / sd, index=cols)

    bad = []
    for d in drivers:
        prior = cfg.DRIVERS[d]["sign"]
        e = float(coef[f"l_{d}"])
        if prior != 0 and np.sign(e) == -prior and abs(e) > 0.01:
            bad.append(d)
    return bad


def protected_drivers(df: pd.DataFrame, kept: list[str], k: int, decay: float,
                      alpha: float, t0: pd.Timestamp) -> list[str]:
    """Controllable levers the model finds material. These are never eliminated.

    See the note on `MATERIALITY_PCT` in wk_config: our price and the competitor's
    price are close enough substitutes for *forecasting* that cross-validation will
    happily keep whichever it likes and drop the other. A tool that drops the price
    a category team sets, and keeps the price they can only watch, forecasts fine
    and plans nothing.
    """
    model = mdl.fit_category(df, "tmp", kept, {"fourier_k": k, "adstock_decay": decay,
                                               "alpha": alpha}, t0)
    impacts = model.impacts().set_index("driver")["impact_pct"].abs()
    return [
        d for d in kept
        if cfg.DRIVERS[d]["controllable"] and impacts.get(d, 0.0) >= cfg.MATERIALITY_PCT
    ]


def backward_eliminate(df: pd.DataFrame, drivers: list[str], k: int, decay: float,
                       t0: pd.Timestamp, tolerance: float = cfg.SELECTION_TOLERANCE,
                       verbose: bool = False) -> tuple[list[str], pd.DataFrame]:
    """Drop drivers one at a time while out-of-fold error does not meaningfully worsen.

    `tolerance` is a fraction of the current CV RMSE. A driver whose removal costs
    less than that was not carrying the forecast, and every driver kept is one more
    series someone has to forecast, maintain and explain.

    Material controllable levers are held back from the candidate list -- forecast
    error is not the only thing this model is for.
    """
    kept = list(drivers)
    alpha, rmse = mdl.best_alpha(df, kept, k, decay, t0)
    log = [{"step": 0, "removed": None, "n_drivers": len(kept),
            "cv_rmse": rmse, "alpha": alpha}]

    step = 0
    while len(kept) > 1:
        protected = protected_drivers(df, kept, k, decay, alpha, t0)
        removable = [d for d in kept if d not in protected]
        if not removable:
            break

        candidates = []
        for d in removable:
            trial = [x for x in kept if x != d]
            a, r = mdl.best_alpha(df, trial, k, decay, t0)
            candidates.append((r, d, a))
        candidates.sort()
        best_rmse, drop, best_a = candidates[0]

        if best_rmse > rmse * (1 + tolerance):
            break  # everything left is earning its place

        step += 1
        kept = [x for x in kept if x != drop]
        rmse, alpha = best_rmse, best_a
        log.append({"step": step, "removed": drop, "n_drivers": len(kept),
                    "cv_rmse": rmse, "alpha": alpha})
        if verbose:
            print(f"      drop {drop:<22} -> {len(kept):2d} drivers, "
                  f"cv_rmse {rmse:.5f}")

    return kept, pd.DataFrame(log)


def select_for_category(df: pd.DataFrame, category: str, t0: pd.Timestamp,
                        verbose: bool = True) -> dict:
    """The full selection routine for one category. `df` is that category's train rows.

    Order matters. The shape hyperparameters are settled first with every driver
    present, so the search is not comparing driver sets across different seasonal
    models. Then drivers are eliminated. Then the shape is re-validated on the
    surviving set, because the right number of harmonics can change once the
    drivers carrying part of the seasonality are gone.
    """
    candidates = list(cfg.DRIVER_NAMES)
    if verbose:
        print(f"    [1/4] tuning shape with all {len(candidates)} drivers ...")
    shape, _ = mdl.choose_config(df, candidates, t0)

    if verbose:
        print(f"          k={shape['fourier_k']}, decay={shape['adstock_decay']}, "
              f"alpha={shape['alpha']}")
        print("    [2/4] checking elasticity signs against the business prior ...")
    wrong_sign = sign_check(df, candidates, shape["fourier_k"],
                            shape["adstock_decay"], t0)
    survivors = [d for d in candidates if d not in wrong_sign]
    if verbose:
        print(f"          dropped for wrong sign: "
              f"{', '.join(wrong_sign) if wrong_sign else 'none'}")
        print("    [3/4] backward elimination on out-of-fold error ...")

    kept, log = backward_eliminate(df, survivors, shape["fourier_k"],
                                   shape["adstock_decay"], t0, verbose=verbose)

    if verbose:
        print(f"    [4/4] re-tuning shape on the {len(kept)} selected drivers ...")
    final, grid = mdl.choose_config(df, kept, t0)
    if verbose:
        print(f"          k={final['fourier_k']}, decay={final['adstock_decay']}, "
              f"alpha={final['alpha']}")

    return {
        "category": category,
        "selected": kept,
        "dropped": [d for d in candidates if d not in kept],
        "dropped_wrong_sign": wrong_sign,
        "config": final,
        "config_with_all_drivers": shape,
        "elimination_log": log,
        "grid": grid,
    }
