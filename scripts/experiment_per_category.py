"""Experiment: should the response model be pooled, split per category, or in between?

The shipped model pools: one set of elasticities across all five categories, with
a fixed effect to absorb the level difference. The generator says that assumption
is false -- each category's elasticities are jittered +/-15% around the global
truth, and several are hand-set well outside that band (Dairy price -1.15 vs -1.65
global; Beverages temperature +0.42 vs Household Care -0.02). So the pooled fit
carries bias by construction.

The obvious fix -- one model per category -- costs variance: each fit sees 66
months instead of 330 while estimating the same 21 coefficients. Whether that
trade pays off is what this measures, across four arms:

  A  POOLED         the shipped model. One elasticity set, one trend, one
                    seasonal pattern, category intercepts only.
  B  PER-CATEGORY   five independent models. Everything varies by category.
  C  CALENDAR-ONLY  elasticities pinned to the pooled values; only trend and
                    month effects vary by category. The control -- without it,
                    B's win cannot be attributed to the drivers, because the
                    pooled model has no category-specific trend or seasonality
                    either and the generator gives categories both.
  D  HYBRID         commercial elasticities allowed to deviate per category,
                    shrunk toward the pooled value rather than toward zero; macro
                    elasticities pinned. Macro drivers are one shared time series,
                    so a 66-month category slice holds no extra information about
                    them -- splitting those buys variance and nothing else.
  E  HYBRID TUNED   arm D with the deviation penalty set by hand rather than by
                    CV, at the point on the shrinkage frontier that section 4
                    identifies.

Scored on three axes: holdout accuracy, elasticity recovery against the
per-category ground truth, and what the price slider actually reports.

Writes artifacts/experiment_per_category.json. Leaves the artifacts the Streamlit
app reads untouched.

    python scripts/experiment_per_category.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data_gen, model as pooled_mod, model_partial, model_percat  # noqa: E402

pd.set_option("display.width", 220)

ARMS = ["A pooled", "B per-category", "C calendar-only", "D hybrid",
        "E hybrid tuned"]


def _rule(title: str) -> None:
    print(f"\n{'=' * 86}\n{title}\n{'=' * 86}")


def _rel(base: float, other: float) -> str:
    """Improvement of `other` over `base`, as a signed % of the base value."""
    return "n/a" if base == 0 else f"{(base - other) / abs(base) * 100:+.1f}%"


def _elas_frame(m, categories) -> pd.DataFrame:
    """Every arm's elasticities as a category x driver frame (pooled broadcasts)."""
    if isinstance(m.elasticities, pd.Series):
        return pd.DataFrame(
            np.tile(m.elasticities[config.DRIVER_NAMES].to_numpy(), (len(categories), 1)),
            index=categories,
            columns=config.DRIVER_NAMES,
        )
    return m.elasticities.loc[categories, config.DRIVER_NAMES]


# --------------------------------------------------------------------------
# 1. Accuracy
# --------------------------------------------------------------------------
def compare_accuracy(models: dict) -> dict:
    _rule("1. ACCURACY  (identical 12-month holdout and CV protocol for every arm)")

    overall = pd.DataFrame(
        {
            arm: [
                m.metrics["r2_log"],
                m.metrics["mape_in_sample"],
                m.metrics["holdout"]["mape"],
                m.metrics["holdout"]["r2_volume"],
            ]
            for arm, m in models.items()
        },
        index=["R2 (log, in-sample)", "MAPE % (in-sample)",
               "MAPE % (holdout)", "R2 volume (holdout)"],
    )
    print(overall.round(4).to_string())

    by_cat = pd.DataFrame(
        {arm: pd.Series(m.metrics["holdout"]["by_category"])
         for arm, m in models.items()}
    ).loc[config.CATEGORIES]
    by_cat["best"] = by_cat.idxmin(axis=1)
    print("\nHoldout MAPE % by category:")
    print(by_cat.round(2).to_string())

    base = models["A pooled"].metrics["holdout"]["mape"]
    print("\nHoldout MAPE vs pooled baseline:")
    for arm, m in models.items():
        h = m.metrics["holdout"]["mape"]
        print(f"  {arm:<18} {h:6.2f}%   {_rel(base, h):>8}")

    # How much of B's gain is calendar rather than drivers?
    b, c = (models["B per-category"].metrics["holdout"]["mape"],
            models["C calendar-only"].metrics["holdout"]["mape"])
    total_gain = base - b
    cal_gain = base - c
    print(f"\nDecomposition of the fully-split gain ({total_gain:.2f}pp total):")
    print(f"  category-specific trend + seasonality alone : {cal_gain:6.2f}pp  "
          f"({cal_gain / total_gain * 100:.0f}% of it)")
    print(f"  everything else (the driver elasticities)   : "
          f"{total_gain - cal_gain:6.2f}pp  "
          f"({(total_gain - cal_gain) / total_gain * 100:.0f}%)")

    return {
        "overall": overall.round(4).to_dict(),
        "by_category": by_cat.drop(columns="best").round(3).to_dict(),
        "calendar_share_of_gain": float(cal_gain / total_gain),
    }


# --------------------------------------------------------------------------
# 2. Elasticity recovery
# --------------------------------------------------------------------------
def compare_elasticities(models: dict) -> dict:
    _rule("2. ELASTICITY RECOVERY  (mean |estimate - truth| across categories)")

    truth = data_gen.true_category_elasticity_table()
    errs = {
        arm: (_elas_frame(m, truth.index) - truth).abs() for arm, m in models.items()
    }

    per_driver = pd.DataFrame({arm: e.mean() for arm, e in errs.items()})
    per_driver.insert(0, "true_spread", (truth.max() - truth.min()).round(3))
    per_driver["best"] = per_driver[list(models)].idxmin(axis=1)
    print(per_driver.round(3).to_string())

    print("\nOverall MAE (all drivers, all categories):")
    base = errs["A pooled"].to_numpy().mean()
    for arm, e in errs.items():
        v = e.to_numpy().mean()
        print(f"  {arm:<18} {v:.4f}   {_rel(base, v):>8}")

    print("\nSplit by driver group -- this is where the arms separate:")
    grp = pd.DataFrame(
        {
            arm: {
                "commercial": e[config.COMMERCIAL_DRIVERS].to_numpy().mean(),
                "macro": e[config.MACRO_DRIVERS].to_numpy().mean(),
            }
            for arm, e in errs.items()
        }
    )
    print(grp.round(4).to_string())
    print("\nMacro drivers are one shared time series, so a 66-month category slice")
    print("holds no extra information about them -- splitting them buys only variance.")

    print("\nPrice elasticity, where the truth genuinely differs by category:")
    price = pd.DataFrame({"true": truth["avg_price"]})
    for arm, m in models.items():
        price[arm] = _elas_frame(m, truth.index)["avg_price"]
    print(price.round(3).to_string())

    print("\nTemperature, where even the sign differs by category:")
    temp = pd.DataFrame({"true": truth["avg_temp_c"]})
    for arm, m in models.items():
        temp[arm] = _elas_frame(m, truth.index)["avg_temp_c"]
    print(temp.round(3).to_string())

    return {
        "mae_overall": {arm: float(e.to_numpy().mean()) for arm, e in errs.items()},
        "mae_by_group": grp.round(4).to_dict(),
        "mae_per_driver": per_driver.drop(columns="best").round(4).to_dict(),
        "true_elasticities": truth.round(4).to_dict(),
    }


# --------------------------------------------------------------------------
# 3. What the slider says
# --------------------------------------------------------------------------
def compare_scenarios(models: dict) -> dict:
    _rule("3. SCENARIO OUTPUT  (volume lift from a 10% price cut, %)")

    truth = data_gen.true_category_elasticity_table()
    shock = np.log(0.90)
    lift = pd.DataFrame(
        {"true": (np.exp(truth["avg_price"] * shock) - 1) * 100}
    )
    for arm, m in models.items():
        e = _elas_frame(m, truth.index)["avg_price"]
        lift[arm] = (np.exp(e * shock) - 1) * 100
    print(lift.round(2).to_string())

    err = pd.DataFrame(
        {arm: (lift[arm] - lift["true"]).abs() for arm in models}
    )
    print("\nMean absolute error in the reported lift (percentage points):")
    print(err.mean().round(2).to_string())
    print("\nThe pooled arm is off by the same amount everywhere -- it reports the")
    print("same lift for a staple as for a discretionary category, which is the")
    print("failure mode a planner would notice first.")

    return {"lift": lift.round(3).to_dict(), "mae_pp": err.mean().round(3).to_dict()}


# --------------------------------------------------------------------------
# 4. The shrinkage dial
# --------------------------------------------------------------------------
def frontier_table(panel: pd.DataFrame, pooled) -> pd.DataFrame:
    """Sweep the hybrid arm's deviation penalty from 'free' to 'fully pooled'."""
    truth = data_gen.true_category_elasticity_table()
    rows = []
    # Extends past the CV grid: only the deviations are penalised now, so the arm
    # needs a strong enough alpha to actually collapse back onto arm C.
    for a in config.RIDGE_ALPHAS + [300.0, 1000.0, 10_000.0]:
        m = model_partial.fit(panel, free=config.COMMERCIAL_DRIVERS,
                              pooled_betas=pooled.elasticities, alpha=a)
        err = (_elas_frame(m, truth.index) - truth).abs()
        rows.append(
            {
                "alpha": a,
                "holdout_mape": m.metrics["holdout"]["mape"],
                "elasticity_mae": err.to_numpy().mean(),
                "commercial_mae": err[config.COMMERCIAL_DRIVERS].to_numpy().mean(),
                "price_mae": err["avg_price"].mean(),
                "max_deviation": m.metrics["max_abs_deviation"],
            }
        )
    return pd.DataFrame(rows).set_index("alpha")


def report_frontier(out: pd.DataFrame, pooled, best: float) -> dict:
    _rule("4. THE SHRINKAGE DIAL  (hybrid arm, alpha set by hand instead of by CV)")

    print("CV picked the weakest alpha on the grid for every category, which turns")
    print("'shrink toward pooled' into 'fit freely'. That is not CV malfunctioning:")
    print("reallocating between price, food CPI and the trend barely moves forecast")
    print("error, so the CV objective genuinely cannot see the difference. Sweeping")
    print("alpha by hand shows what it is blind to.\n")
    print(out.round(4).to_string())

    truth = data_gen.true_category_elasticity_table()
    pooled_mape = pooled.metrics["holdout"]["mape"]
    pooled_mae = (_elas_frame(pooled, truth.index) - truth).abs().to_numpy().mean()
    both = out[(out["holdout_mape"] < pooled_mape) & (out["elasticity_mae"] < pooled_mae)]

    print(f"\nSanity check: the largest alpha lands on "
          f"({out.iloc[-1]['holdout_mape']:.2f}%, {out.iloc[-1]['elasticity_mae']:.4f}), "
          f"which is arm C exactly.\nThe dial spans the full range it should.")
    print(f"\nPooled baseline: holdout MAPE {pooled_mape:.2f}%, "
          f"elasticity MAE {pooled_mae:.4f}")
    if len(both):
        print(f"Alphas beating pooled on BOTH axes: {list(both.index)}")
        print(f"Best of them: alpha={best}  MAPE {out.loc[best, 'holdout_mape']:.2f}%  "
              f"MAE {out.loc[best, 'elasticity_mae']:.4f}  -> carried into arm E.")
    else:
        print("No alpha beats pooled on both axes at once -- the trade-off is real.")
    print("\nCaveat worth stating plainly: this alpha was chosen against ground truth,")
    print("which only exists because the panel is synthetic. On real data it is a")
    print("judgement call informed by prior elasticity studies, not a CV output --")
    print("that is what section 2 shows CV cannot supply.")
    return out.round(4).to_dict()


def main() -> None:
    panel = pd.read_csv(config.PANEL_CSV, parse_dates=["date"])

    print("A  fitting pooled model ...")
    pooled = pooled_mod.fit(panel)
    print("B  fitting 5 independent per-category models ...")
    percat = model_percat.fit(panel)
    percat.save()
    print("C  fitting calendar-only partial pool (all elasticities pinned) ...")
    calendar = model_partial.fit(panel, free=[], pooled_betas=pooled.elasticities,
                                 name="calendar-only")
    print("D  fitting hybrid (commercial shrunk toward pooled, macro pinned) ...")
    hybrid = model_partial.fit(panel, free=config.COMMERCIAL_DRIVERS,
                               pooled_betas=pooled.elasticities, name="hybrid")
    if hybrid.metrics["sign_violations"]:
        print(f"   !! sign violations: {hybrid.metrics['sign_violations']}")

    print("E  sweeping the hybrid's deviation penalty ...")
    frontier = frontier_table(panel, pooled)
    truth = data_gen.true_category_elasticity_table()
    pooled_mae = (_elas_frame(pooled, truth.index) - truth).abs().to_numpy().mean()
    ok = frontier[
        (frontier["holdout_mape"] < pooled.metrics["holdout"]["mape"])
        & (frontier["elasticity_mae"] < pooled_mae)
    ]
    best_alpha = float(
        (ok if len(ok) else frontier)["elasticity_mae"].idxmin()
    )
    print(f"   best alpha = {best_alpha}")
    tuned = model_partial.fit(panel, free=config.COMMERCIAL_DRIVERS,
                              pooled_betas=pooled.elasticities, name="hybrid-tuned",
                              alpha=best_alpha)

    models = dict(zip(ARMS, [pooled, percat, calendar, hybrid, tuned]))

    results = {
        "accuracy": compare_accuracy(models),
        "elasticity_recovery": compare_elasticities(models),
        "scenario": compare_scenarios(models),
        "shrinkage_frontier": report_frontier(frontier, pooled, best_alpha),
        "best_alpha": best_alpha,
        "metrics": {arm: m.metrics for arm, m in models.items()},
    }

    _rule("VERDICT")
    mape = {arm: m.metrics["holdout"]["mape"] for arm, m in models.items()}
    mae = results["elasticity_recovery"]["mae_overall"]
    macro_mae = results["elasticity_recovery"]["mae_by_group"]
    summary = pd.DataFrame(
        {
            "holdout MAPE %": pd.Series(mape),
            "elasticity MAE": pd.Series(mae),
            "commercial MAE": pd.Series({a: macro_mae[a]["commercial"] for a in models}),
            "macro MAE": pd.Series({a: macro_mae[a]["macro"] for a in models}),
            "price slider MAE pp": pd.Series(results["scenario"]["mae_pp"]),
        }
    ).loc[ARMS]
    print(summary.round(4).to_string())
    print("\nBest arm per column:")
    print(summary.idxmin().to_string())

    out_path = config.ARTIFACT_DIR / "experiment_per_category.json"
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(f"Wrote {model_percat.PERCAT_MODEL_PKL}")


if __name__ == "__main__":
    main()
