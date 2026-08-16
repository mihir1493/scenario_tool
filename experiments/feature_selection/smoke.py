"""End-to-end checks on the selected-feature model: run after `python run_experiment.py`.

Same shape as `tests/smoke.py`, with the checks that only make sense once features
are selected per category: dropped drivers must be inert, kept drivers must still
move volume the right way, and the two must not have been swapped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

import decoys  # noqa: E402
import fs_config  # noqa: E402
import scenario_fs  # noqa: E402
from fs_model import CategoryModel  # noqa: E402

panel, fc = decoys.load()
m = CategoryModel.load(fs_config.MODEL_PKL)
ok = True


def check(label, cond, extra=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {extra}")


print("=== baseline continuity at the history/forecast seam ===")
base = scenario_fs.run(m, panel, fc, {})
last_act = panel[panel["date"] == panel["date"].max()]["volume"].sum()
first_fc = base["paths"][base["paths"]["date"] == base["paths"]["date"].min()]["baseline"].sum()
jump = 100 * (first_fc / last_act - 1)
check("no discontinuity", abs(jump) < 15, f"({jump:+.1f}% jump, Jun26 -> Jul26)")

print("\n=== drivers cut by the global gate are inert ===")
cut = [d for d in fs_config.DECOY_NAMES if (m.elasticities[d] == 0).all()]
check("at least one candidate was cut", len(cut) > 0, f"({', '.join(cut)})")
for d in cut:
    r = scenario_fs.run(m, panel, fc, {d: 50.0})["summary"]
    check(f"{d} +50% moves nothing", abs(r["delta_pct"]) < 1e-9,
          f"-> {r['delta_pct']:+.2e}%")

print("\n=== kept drivers still move volume the right way ===")
for d, pct, want in [("avg_price", +10, "down"), ("avg_price", -10, "up"),
                     ("media_spend", +50, "up"), ("distribution_acv", +10, "up"),
                     ("promo_depth", +25, "up"), ("promo_share", +25, "up"),
                     ("consumer_confidence", +10, "up")]:
    r = scenario_fs.run(m, panel, fc, {d: pct})["summary"]
    got = "up" if r["delta_pct"] > 0 else "down"
    check(f"{d} {pct:+d}%", got == want, f"-> volume {r['delta_pct']:+.2f}% ({got})")

print("\n=== per-category elasticities actually differ ===")
spread = m.elasticities["avg_price"].max() - m.elasticities["avg_price"].min()
check("price elasticity varies by category", spread > 0.1, f"(spread {spread:.2f})")
lift = {c: scenario_fs.run(m, panel, fc, {"avg_price": -10}, [c])["summary"]["delta_pct"]
        for c in config.CATEGORIES}
check("10% price cut gives 5 different answers",
      len({round(v, 2) for v in lift.values()}) == len(config.CATEGORIES),
      f"({', '.join(f'{c[:4]} {v:+.1f}%' for c, v in lift.items())})")

print("\n=== waterfall reconciles to the total delta ===")
r = scenario_fs.run(m, panel, fc,
                    {"avg_price": -8, "media_spend": 40, "promo_depth": 20})
tot, parts = r["summary"]["delta_volume"], r["waterfall"]["delta_volume"].sum()
check("parts sum to whole", abs(parts / tot - 1) < 1e-6,
      f"(total {tot:,.0f} vs parts {parts:,.0f})")
check("no cut driver appears in the waterfall",
      not set(r["waterfall"]["driver"]) & set(cut))

print("\n=== elasticity implied vs realised, per category (log-log consistency) ===")
for c in config.CATEGORIES:
    got = scenario_fs.run(m, panel, fc, {"avg_price": -10}, [c])["summary"]["delta_pct"]
    implied = (np.exp(float(m.elasticities.loc[c, "avg_price"]) * np.log(0.9)) - 1) * 100
    check(f"{c:<15}", abs(got - implied) < 1.5,
          f"(realised {got:+.2f}% vs implied {implied:+.2f}%)")

print("\n=== edge cases ===")
z = scenario_fs.run(m, panel, fc, {})
check("empty scenario -> zero delta", abs(z["summary"]["delta_pct"]) < 1e-9)
e = scenario_fs.run(m, panel, fc, {"media_spend": -100})["summary"]
check("media zeroed out survives", np.isfinite(e["delta_pct"]), f"({e['delta_pct']:+.2f}%)")
short = fc[fc["date"].isin(sorted(fc["date"].unique())[:3])]
s3 = scenario_fs.run(m, panel, short, {"avg_price": 5})["summary"]
check("3-month horizon runs", np.isfinite(s3["delta_pct"]), f"({s3['delta_pct']:+.2f}%)")

print("\n=== selection bookkeeping ===")
check("every category fitted at least 4 drivers",
      all(len(v) >= 4 for v in m.selected.values()),
      f"({ {c: len(v) for c, v in m.selected.items()} })")
check("fitted drivers all have non-zero elasticities",
      all(m.elasticities.loc[c, d] != 0 for c, v in m.selected.items() for d in v))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
