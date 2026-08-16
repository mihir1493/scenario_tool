"""End-to-end checks. Run after `python build.py`.

    python smoke.py

Exercises the same code paths the Streamlit app does, including a full
download -> edit -> upload round trip, so a broken scenario flow fails here rather
than in front of someone using it.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import wk_config as cfg
import wk_model as mdl
import wk_scenario as scen

panel = pd.read_csv(cfg.PANEL_CSV, parse_dates=["date"])
driver_fc = pd.read_csv(cfg.DRIVER_FORECAST_CSV, parse_dates=["date"])
baseline = pd.read_csv(cfg.BASELINE_CSV, parse_dates=["date"])
models = mdl.load()
ok = True


def check(label, cond, extra=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {extra}")


print("=== every category is genuinely its own model ===")
configs = {c: (m.fourier_k, m.adstock_decay, m.alpha) for c, m in models.items()}
check("categories chose different configs", len(set(configs.values())) > 1,
      f"({configs})")
featuresets = {c: tuple(m.drivers) for c, m in models.items()}
check("categories selected different features", len(set(featuresets.values())) > 1)
check("no category kept all 11 candidates",
      all(len(m.drivers) < len(cfg.DRIVER_NAMES) for m in models.values()),
      f"({ {c: len(m.drivers) for c, m in models.items()} })")

print("\n=== calendar is weekly, ending Sunday ===")
days = set(pd.DatetimeIndex(panel["date"]).day_name())
check("all history dates are Sundays", days == {"Sunday"}, f"({days})")
check("all forecast dates are Sundays",
      set(pd.DatetimeIndex(driver_fc["date"]).day_name()) == {"Sunday"})
gaps = panel[panel["category"] == cfg.CATEGORIES[0]]["date"].diff().dropna().unique()
check("history has no week gaps", list(gaps) == [pd.Timedelta(weeks=1)])
check(f"forecast is {cfg.FORECAST_WEEKS} weeks",
      driver_fc.groupby("category").size().eq(cfg.FORECAST_WEEKS).all())

print("\n=== only selected drivers were forecast ===")
for c, m in models.items():
    sub = driver_fc[driver_fc["category"] == c]
    filled = [d for d in cfg.DRIVER_NAMES if sub[d].notna().all()]
    check(f"{c:<20}", sorted(filled) == sorted(m.drivers),
          f"({len(filled)} forecast, {len(m.drivers)} selected)")

print("\n=== sign constraints hold ===")
for c, m in models.items():
    bad = [d for d in m.drivers
           if cfg.DRIVERS[d]["sign"]
           and np.sign(m.elasticities[d]) == -cfg.DRIVERS[d]["sign"]]
    check(f"{c:<20}", not bad, f"({', '.join(bad) or 'no violations'})")

print("\n=== price is a lever in every category ===")
check("avg_price selected everywhere",
      all("avg_price" in m.drivers for m in models.values()),
      "(a planner with no price lever has no plan)")

print("\n=== decomposition reconciles ===")
for c, m in models.items():
    dec = scen.decompose_period(m, panel, driver_fc)
    blocks = [x for x in dec.columns if x not in ("date", "predicted", "period")]
    recon = dec[blocks].sum(axis=1)
    check(f"{c:<20} parts sum to predicted",
          np.allclose(recon, dec["predicted"], rtol=1e-8),
          f"(max err {np.abs(recon / dec['predicted'] - 1).max():.2e})")

print("\n=== baseline file is well formed ===")
for c, m in models.items():
    b = baseline[baseline["category"] == c]
    check(f"{c:<20}",
          len(b) == len(panel[panel["category"] == c]) + cfg.FORECAST_WEEKS
          and set(b["period"]) == {"history", "forecast"}
          and all(d in b.columns for d in m.drivers),
          f"({len(b)} rows, {len(m.drivers)} driver columns)")

print("\n=== download -> edit -> upload round trip ===")
cat = "Ground Coffee"
m = models[cat]
edited = baseline[baseline["category"] == cat].copy()

check("unedited file reproduces the baseline exactly",
      abs(scen.run_scenario(m, panel, driver_fc, edited)["summary"]["delta_pct"]) < 1e-9)

fut = edited["period"] == "forecast"
edited.loc[fut, "avg_price"] = edited.loc[fut, "avg_price"] * 1.05
r = scen.run_scenario(m, panel, driver_fc, edited)
check("5% price rise lowers volume", r["summary"]["delta_pct"] < 0,
      f"({r['summary']['delta_pct']:+.2f}%)")
implied = (np.exp(float(m.elasticities["avg_price"]) * np.log(1.05)) - 1) * 100
check("magnitude matches the elasticity",
      abs(r["summary"]["delta_pct"] - implied) < 1.0,
      f"(got {r['summary']['delta_pct']:+.2f}%, elasticity implies {implied:+.2f}%)")
check("waterfall sums to the total delta",
      abs(r["waterfall"]["delta_volume"].sum() / r["summary"]["delta_volume"] - 1) < 1e-6)
check("only the edited driver appears in the waterfall",
      set(r["waterfall"]["driver"]) == {"avg_price"},
      f"({list(r['waterfall']['driver'])})")
check("change log spots one edited driver", len(r["changes"]) == 1)

print("\n=== a partial-year edit only moves those weeks ===")
edited2 = baseline[baseline["category"] == cat].copy()
f2 = (edited2["period"] == "forecast") & (edited2["date"].dt.year == 2027)
edited2.loc[f2, "tv_grps"] = edited2.loc[f2, "tv_grps"] * 2
r2 = scen.run_scenario(m, panel, driver_fc, edited2)
moved = r2["paths"][np.abs(r2["paths"]["delta"]) > 1e-6]
check("only 2027 weeks move (plus adstock spill)",
      moved["date"].min().year == 2027, f"(first moved week {moved['date'].min().date()})")
check("doubling TV raises volume", r2["summary"]["delta_pct"] > 0,
      f"({r2['summary']['delta_pct']:+.2f}%)")

print("\n=== upload validation catches bad files ===")
check("missing columns rejected",
      len(scen.validate_upload(pd.DataFrame({"date": [], "period": []}), m, driver_fc)) > 0)
short = baseline[baseline["category"] == cat].head(10).copy()
check("truncated horizon rejected", len(scen.validate_upload(short, m, driver_fc)) > 0)
blanks = baseline[baseline["category"] == cat].copy()
blanks.loc[blanks["period"] == "forecast", "avg_price"] = np.nan
check("blank driver values rejected", len(scen.validate_upload(blanks, m, driver_fc)) > 0)

print("\n=== test-set metrics were computed without leakage ===")
for c, m in models.items():
    check(f"{c:<20} test weeks held out",
          m.metrics["test"]["n_weeks"] == cfg.TEST_WEEKS
          and m.metrics["test"]["mape"] > 0,
          f"(MAPE {m.metrics['test']['mape']:.2f}%)")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
