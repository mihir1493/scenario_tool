"""End-to-end checks on the built artifacts: run `python tests/smoke.py` after a build."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config
from src import scenario as scen
from src.model import ResponseModel

panel = pd.read_csv(config.PANEL_CSV, parse_dates=["date"])
fc = pd.read_csv(config.DRIVER_FORECAST_CSV, parse_dates=["date"])
m = ResponseModel.load()
ok = True

def check(label, cond, extra=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {extra}")

print("=== baseline continuity at the history/forecast seam ===")
base = scen.run(m, panel, fc, {}, config.CATEGORIES)
last_act = panel[panel["date"] == panel["date"].max()]["volume"].sum()
first_fc = base["paths"][base["paths"]["date"] == base["paths"]["date"].min()]["baseline"].sum()
jump = 100 * (first_fc / last_act - 1)
check("no discontinuity", abs(jump) < 15, f"({jump:+.1f}% jump, Jun26 -> Jul26)")

print("\n=== sliders move volume the right way ===")
for d, pct, want in [("avg_price", +10, "down"), ("avg_price", -10, "up"),
                     ("media_spend", +50, "up"), ("distribution_acv", +10, "up"),
                     ("promo_depth", +25, "up"), ("cpi_food", +5, "down"),
                     ("unemployment_rate", +20, "down"),
                     ("consumer_confidence", +10, "up")]:
    r = scen.run(m, panel, fc, {d: pct}, config.CATEGORIES)["summary"]
    got = "up" if r["delta_pct"] > 0 else "down"
    check(f"{d} {pct:+d}%", got == want, f"-> volume {r['delta_pct']:+.2f}% ({got})")

print("\n=== waterfall reconciles to the total delta ===")
r = scen.run(m, panel, fc, {"avg_price": -8, "media_spend": 40, "promo_depth": 20},
             config.CATEGORIES)
tot, parts = r["summary"]["delta_volume"], r["waterfall"]["delta_volume"].sum()
check("parts sum to whole", abs(parts / tot - 1) < 1e-6,
      f"(total {tot:,.0f} vs parts {parts:,.0f})")

print("\n=== decomposition reconciles ===")
dec = m.decompose(panel)
recon = dec["base"] + dec[config.DRIVER_NAMES].sum(axis=1)
check("base + contributions == predicted",
      np.allclose(recon, dec["predicted"], rtol=1e-6),
      f"(max err {np.abs(recon/dec['predicted']-1).max():.2e})")

print("\n=== elasticity implied vs realised (log-log consistency) ===")
r = scen.run(m, panel, fc, {"avg_price": -10}, config.CATEGORIES)["summary"]
implied = (np.exp(float(m.elasticities["avg_price"]) * np.log(0.9)) - 1) * 100
check("10% price cut matches elasticity", abs(r["delta_pct"] - implied) < 1.0,
      f"(realised {r['delta_pct']:+.2f}% vs implied {implied:+.2f}%)")

print("\n=== category subsetting ===")
one = scen.run(m, panel, fc, {"media_spend": 30}, ["Beverages"])["summary"]
check("single category runs", one["baseline_volume"] > 0,
      f"(Beverages baseline {one['baseline_volume']/1e6:.2f}M, {one['delta_pct']:+.2f}%)")

print("\n=== edge cases ===")
z = scen.run(m, panel, fc, {}, config.CATEGORIES)
check("empty scenario -> zero delta", abs(z["summary"]["delta_pct"]) < 1e-9)
e = scen.run(m, panel, fc, {"media_spend": -100}, config.CATEGORIES)["summary"]
check("media zeroed out survives", np.isfinite(e["delta_pct"]), f"({e['delta_pct']:+.2f}%)")
short = fc[fc["date"].isin(sorted(fc["date"].unique())[:3])]
s3 = scen.run(m, panel, short, {"avg_price": 5}, config.CATEGORIES)["summary"]
check("3-month horizon runs", np.isfinite(s3["delta_pct"]), f"({s3['delta_pct']:+.2f}%)")

print("\n=== importance table ===")
imp = m.importance()
check("shares sum to 1", abs(imp["impact_share"].sum() - 1) < 1e-9)
print(imp[["label", "elasticity", "impact_pct", "impact_share"]].round(3).to_string(index=False))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
