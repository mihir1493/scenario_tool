"""Build every artifact the Streamlit app reads: panel -> forecasts -> model.

    python scripts/build.py            # full rebuild
    python scripts/build.py --no-forecast   # skip Prophet, reuse existing forecast
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data_gen, forecast, model  # noqa: E402


def main(skip_forecast: bool = False) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("1/3  generating synthetic panel")
    panel = data_gen.generate()
    panel.to_csv(config.PANEL_CSV, index=False)
    print(f"     {len(panel)} rows, {panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")

    if skip_forecast and config.DRIVER_FORECAST_CSV.exists():
        print("2/3  reusing existing driver forecast")
    else:
        print(f"2/3  forecasting drivers {config.FORECAST_MONTHS}mo "
              f"(prophet={forecast.HAS_PROPHET})")
        fc = forecast.forecast_drivers(panel)
        fc.to_csv(config.DRIVER_FORECAST_CSV, index=False)
        print(f"     {len(fc)} rows -> {config.DRIVER_FORECAST_CSV.name}")

    print("3/3  fitting pooled log-log Ridge")
    m = model.fit(panel)
    m.save()
    config.METRICS_JSON.write_text(json.dumps(m.metrics, indent=2))

    truth = data_gen.true_elasticity_table().set_index("driver")["true_elasticity"]
    comp = pd.DataFrame(
        {"estimated": m.elasticities.round(3), "true": truth.round(3)}
    )
    comp["error"] = (comp["estimated"] - comp["true"]).round(3)

    print(f"\n     alpha={m.metrics['alpha']}  R2(log)={m.metrics['r2_log']:.3f}  "
          f"holdout MAPE={m.metrics['holdout']['mape']:.1f}%")
    print("\nElasticity recovery (pooled estimate vs mean true value):")
    print(comp.to_string())


if __name__ == "__main__":
    main(skip_forecast="--no-forecast" in sys.argv)
