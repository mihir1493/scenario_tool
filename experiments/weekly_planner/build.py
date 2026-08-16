"""Run the whole pipeline and write the artifacts the Streamlit app reads.

    python build.py

Six steps, per category, with nothing shared between categories:

  1. split          last 52 weeks held out as a test set
  2. select         feature selection on the TRAIN rows only
  3. validate       seasonality / carryover / ridge strength chosen on CV folds
  4. test           fit on train, score on the untouched final year
  5. refit          re-fit on the full history with the same recipe -- this is the
                    model that ships, because throwing away the most recent year
                    to preserve a number you have already recorded is not a
                    trade a forecaster should make
  6. forecast       project the selected drivers 104 weeks, then the baseline volume

Step 2 running on train rows only is what makes step 4 mean anything. Select on
all 241 weeks and the "held out" year has already influenced which features exist.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import wk_config as cfg
import wk_data
import wk_drivers
import wk_model as mdl
import wk_scenario as scen
import wk_selection as sel


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `TEST_WEEKS` weeks are the test set. Time series split, never random."""
    cutoff = df["date"].max() - pd.Timedelta(weeks=cfg.TEST_WEEKS)
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def build_category(panel: pd.DataFrame, cat: str, t0: pd.Timestamp) -> dict:
    print(f"\n{'-' * 78}\n{cat}\n{'-' * 78}")
    df = panel[panel["category"] == cat].sort_values("date").reset_index(drop=True)
    train, test = split_train_test(df)
    print(f"  train {len(train)} weeks ({train['date'].min().date()} -> "
          f"{train['date'].max().date()}), test {len(test)} weeks")

    # --- 2 + 3: selection and validation, on training rows only ---
    result = sel.select_for_category(train, cat, t0)
    print(f"  selected {len(result['selected'])}/{len(cfg.DRIVER_NAMES)}: "
          f"{', '.join(result['selected'])}")
    print(f"  dropped: {', '.join(result['dropped']) or 'none'}")

    # --- 4: honest test score, from a model that never saw the test year ---
    train_model = mdl.fit_category(train, cat, result["selected"], result["config"], t0)
    test_metrics = mdl.score(train_model, test)
    train_metrics = mdl.score(train_model, train)
    print(f"  test  MAPE {test_metrics['mape']:5.2f}%  WAPE {test_metrics['wape']:5.2f}%"
          f"  R2(log) {test_metrics['r2_log']:.3f}  bias {test_metrics['bias_pct']:+.2f}%")

    # --- 5: the model that ships, refitted on everything ---
    final = mdl.fit_category(df, cat, result["selected"], result["config"], t0)
    final.metrics = {
        "train": train_metrics,
        "test": test_metrics,
        "full_history": mdl.score(final, df),
        "cv_rmse_log": float(result["grid"].iloc[0]["cv_rmse"]),
        "config": result["config"],
        "config_with_all_drivers": result["config_with_all_drivers"],
    }
    final.selection = {
        "selected": result["selected"],
        "dropped": result["dropped"],
        "dropped_wrong_sign": result["dropped_wrong_sign"],
        "elimination_log": result["elimination_log"].to_dict("records"),
        "grid_top10": result["grid"].head(10).to_dict("records"),
    }
    print(f"  config: fourier_k={final.fourier_k}, adstock_decay={final.adstock_decay}, "
          f"alpha={final.alpha}")
    return {"model": final, "result": result}


def main() -> None:
    t_start = time.time()
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not cfg.PANEL_CSV.exists():
        print("Generating the weekly panel ...")
        wk_data.generate().to_csv(cfg.PANEL_CSV, index=False)

    panel = pd.read_csv(cfg.PANEL_CSV, parse_dates=["date"])
    t0 = pd.Timestamp(cfg.HISTORY_START)
    print(f"Panel: {len(panel)} rows, {len(cfg.CATEGORIES)} categories, "
          f"{panel['date'].nunique()} weeks, {len(cfg.DRIVER_NAMES)} candidate drivers")

    models = {}
    for cat in cfg.CATEGORIES:
        models[cat] = build_category(panel, cat, t0)["model"]
    mdl.save(models)

    # --- 6: forecast each category's selected drivers, then the baseline ---
    print(f"\n{'-' * 78}\nForecasting selected drivers, {cfg.FORECAST_WEEKS} weeks "
          f"(prophet={wk_drivers.HAS_PROPHET})\n{'-' * 78}")
    driver_fc = wk_drivers.forecast_selected(
        panel, {c: m.drivers for c, m in models.items()})
    driver_fc.to_csv(cfg.DRIVER_FORECAST_CSV, index=False)

    baselines = [scen.baseline_table(models[c], panel, driver_fc) for c in cfg.CATEGORIES]
    baseline = pd.concat(baselines, ignore_index=True)
    baseline.to_csv(cfg.BASELINE_CSV, index=False)

    # --- summary ---
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    summary = pd.DataFrame(
        {
            c: {
                "drivers": len(m.drivers),
                "fourier_k": m.fourier_k,
                "adstock": m.adstock_decay,
                "alpha": m.alpha,
                "train MAPE %": m.metrics["train"]["mape"],
                "test MAPE %": m.metrics["test"]["mape"],
                "test R2 log": m.metrics["test"]["r2_log"],
                "test bias %": m.metrics["test"]["bias_pct"],
            }
            for c, m in models.items()
        }
    ).T.loc[cfg.CATEGORIES]
    print(summary.round(3).to_string())

    print("\nSelected drivers per category:")
    for c, m in models.items():
        print(f"  {c:<20} {', '.join(m.drivers)}")

    print("\nElasticity recovery against the truth "
          "(the panel is synthetic, so this is checkable):")
    truth = wk_data.true_elasticity_table()
    est = pd.DataFrame(
        {c: {d: float(models[c].elasticities.get(d, 0.0)) for d in cfg.DRIVER_NAMES}
         for c in cfg.CATEGORIES}
    ).T.loc[cfg.CATEGORIES, cfg.DRIVER_NAMES]
    err = (est - truth).abs()
    print(pd.DataFrame(
        {"MAE all drivers": err.mean(axis=1),
         "MAE true non-zero": pd.Series(
             {c: err.loc[c, truth.loc[c] != 0].mean() for c in cfg.CATEGORIES}),
         "true zeros dropped": pd.Series(
             {c: f"{int(sum(d not in models[c].drivers for d in truth.columns[truth.loc[c] == 0]))}"
                 f"/{int((truth.loc[c] == 0).sum())}" for c in cfg.CATEGORIES})}
    ).round(4).to_string())

    two_year = baseline[baseline["period"] == "forecast"].groupby("category")[
        "volume_baseline"].sum()
    print("\nTwo-year baseline volume:")
    for c in cfg.CATEGORIES:
        print(f"  {c:<20} {two_year[c]:>14,.0f}")

    cfg.SUMMARY_JSON.write_text(json.dumps(
        {
            "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "history": {"start": cfg.HISTORY_START, "end": cfg.HISTORY_END,
                        "weeks": int(panel["date"].nunique())},
            "forecast_weeks": cfg.FORECAST_WEEKS,
            "categories": {
                c: {
                    "drivers": m.drivers,
                    "dropped": m.selection["dropped"],
                    "config": {"fourier_k": m.fourier_k,
                               "adstock_decay": m.adstock_decay, "alpha": m.alpha},
                    "metrics": m.metrics,
                    "elasticities": {d: float(v) for d, v in m.elasticities.items()},
                    "impacts": m.impacts().to_dict("records"),
                }
                for c, m in models.items()
            },
        }, indent=2, default=str))

    print(f"\nWrote {cfg.PANEL_CSV}")
    print(f"Wrote {cfg.DRIVER_FORECAST_CSV}")
    print(f"Wrote {cfg.MODELS_PKL}")
    print(f"Wrote {cfg.BASELINE_CSV}")
    print(f"Wrote {cfg.SUMMARY_JSON}")
    print(f"\nDone in {time.time() - t_start:.0f}s.  Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
