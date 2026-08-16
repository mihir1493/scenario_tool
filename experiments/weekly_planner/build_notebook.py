"""Generate walkthrough.ipynb, then execute it so the outputs are real.

    python build_notebook.py

The notebook is built from source here rather than edited by hand so it can be
regenerated whenever the pipeline changes, and so its outputs are never stale
relative to the code. Same idea as `scripts/build_notebook.py` at the repo root.
"""

from __future__ import annotations

import nbformat as nbf
from nbclient import NotebookClient

import wk_config as cfg

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip()))


# --------------------------------------------------------------------------
md(f"""
# Weekly category planner — walkthrough

Four categories, weekly data ending Sunday, {len(cfg.DRIVER_NAMES)} candidate drivers.
This notebook runs the whole pipeline in the order the tool does it:

1. the data
2. **feature selection, separately for every category**
3. **hyperparameter validation, separately for every category**
4. train/test split, then refit on the full history
5. driver impact — which levers matter here, and in which direction
6. forecast the selected drivers, then the two-year baseline
7. a scenario from an edited driver file
8. decomposition

The one architectural rule: **there is no pooled model**. Four categories, four
completely independent models. No shared coefficients, no category fixed effects,
no borrowed hyperparameters. Where you see the categories disagree below — on
which drivers matter, on how much seasonality to fit, on how long media carries
over — that disagreement is the reason.

The panel is synthetic, which means the true elasticities are known and every
estimate below can be checked rather than admired.
""")

code("""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import wk_config as cfg, wk_data, wk_features as feat
import wk_model as mdl, wk_selection as sel, wk_scenario as scen

plt.rcParams.update({
    "figure.figsize": (12, 3.6), "axes.grid": True, "grid.color": cfg.GRIDLINE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": cfg.INK_SECONDARY,
    "text.color": cfg.INK_PRIMARY, "xtick.color": cfg.INK_MUTED,
    "ytick.color": cfg.INK_MUTED, "font.size": 10, "figure.dpi": 110,
})
pd.set_option("display.width", 200)

panel = pd.read_csv(cfg.PANEL_CSV, parse_dates=["date"])
t0 = pd.Timestamp(cfg.HISTORY_START)
print(f"{len(panel)} rows | {panel['date'].nunique()} weeks | "
      f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
print("Every date is a Sunday:", set(panel['date'].dt.day_name()))
panel.head()
""")

# --------------------------------------------------------------------------
md("""
## 1. The data

Four categories that behave nothing like each other. Ice cream triples between
winter and summer; laundry detergent is flat all year. That difference is not a
nuisance to be absorbed by a fixed effect — it is the reason each category needs
its own model.
""")

code("""
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
for ax, cat in zip(axes, cfg.CATEGORIES):
    s = panel[panel.category == cat]
    ax.plot(s["date"], s["volume"] / 1e3, color=cfg.SERIES[0], lw=1)
    ax.set_title(cat, loc="left", fontsize=11, color=cfg.INK_PRIMARY)
    ax.set_ylabel("k units")
plt.tight_layout(); plt.show()

summer = panel[panel.date.dt.month.isin([6, 7, 8])].groupby("category")["volume"].mean()
winter = panel[panel.date.dt.month.isin([12, 1, 2])].groupby("category")["volume"].mean()
print("Summer / winter volume ratio:")
print((summer / winter).round(2).to_string())
""")

md("""
The candidate drivers. Some genuinely do nothing in some categories — temperature
does not move laundry detergent, and competitor price does not move baby formula,
because parents do not switch formula on a shelf tag. Those true zeros are what
feature selection has to find.
""")

code("""
truth = wk_data.true_elasticity_table()
print("True elasticities (0 = this driver genuinely does nothing in this category):")
truth
""")

# --------------------------------------------------------------------------
md("""
## 2. Feature selection, per category

Backward elimination scored on **expanding-window cross-validation**: always train
on the past and test on the future, because a forecasting model that gets to see
next winter while predicting last winter is not being tested.

Start with all 11 drivers. Repeatedly ask "what happens to out-of-fold error if I
drop this one?", and drop the cheapest — while it stays cheap.

Two guards stop pure error-chasing from building the wrong tool:

- a driver whose **unconstrained** elasticity contradicts its business prior is
  removed first (if the data says raising price raises volume, this category
  cannot identify price, and the fix is not to constrain the coefficient to zero
  and keep it);
- a **controllable and material** driver is never eliminated. See section 2b.

Selection runs on the **training rows only** — selecting on all 241 weeks and then
reporting accuracy on the last 52 would leak the test set into the feature list.
""")

code("""
cat = "Ground Coffee"
df = panel[panel.category == cat].sort_values("date").reset_index(drop=True)
train, test = df[df.date <= df.date.max() - pd.Timedelta(weeks=cfg.TEST_WEEKS)], \\
              df[df.date > df.date.max() - pd.Timedelta(weeks=cfg.TEST_WEEKS)]
print(f"{cat}: train {len(train)} weeks, test {len(test)} weeks\\n")

result = sel.select_for_category(train, cat, t0)
print(f"\\nKept    ({len(result['selected'])}): {', '.join(result['selected'])}")
print(f"Dropped ({len(result['dropped'])}): {', '.join(result['dropped']) or 'none'}")
""")

code("""
print("Backward elimination trace — each row drops the driver that costs least:")
result["elimination_log"]
""")

# --------------------------------------------------------------------------
md("""
### 2b. Why forecast error alone builds the wrong tool

The first version of this experiment selected purely on cross-validated error, and
it dropped `avg_price` from Laundry Detergent while keeping `competitor_price`.

That is not a bug in the search. Our price and the competitor's price move together
— both track food CPI, both respond to the same promo calendar — so out-of-fold
error genuinely cannot separate them. Swapping one for the other costs nothing you
can measure with RMSE.

As a *forecast*, fine. As a *planning tool*, useless: it deletes the one number the
category team actually sets and replaces it with a number they can only watch.

So a driver marked `controllable` in the config is protected from elimination when
its estimated impact clears `MATERIALITY_PCT`. The threshold is what keeps this
from becoming blanket protection — a controllable driver the model finds genuinely
inert is still dropped, which is exactly what happens to feature+display in baby
formula, whose true elasticity is zero.
""")

code("""
# Same category, same data, with lever protection turned off.
import copy
saved = cfg.MATERIALITY_PCT
cfg.MATERIALITY_PCT = 1e9  # nothing is protected

shape, _ = mdl.choose_config(
    panel[panel.category == "Laundry Detergent"].pipe(
        lambda d: d[d.date <= d.date.max() - pd.Timedelta(weeks=cfg.TEST_WEEKS)]),
    list(cfg.DRIVER_NAMES), t0)
ld_train = panel[panel.category == "Laundry Detergent"].pipe(
    lambda d: d[d.date <= d.date.max() - pd.Timedelta(weeks=cfg.TEST_WEEKS)])
unprotected, _ = sel.backward_eliminate(
    ld_train, list(cfg.DRIVER_NAMES), shape["fourier_k"], shape["adstock_decay"], t0)

cfg.MATERIALITY_PCT = saved
protected, _ = sel.backward_eliminate(
    ld_train, list(cfg.DRIVER_NAMES), shape["fourier_k"], shape["adstock_decay"], t0)

print("Laundry Detergent, selection WITHOUT lever protection:")
print("  ", ", ".join(unprotected))
print("   avg_price kept?", "avg_price" in unprotected)
print("\\nLaundry Detergent, selection WITH lever protection:")
print("  ", ", ".join(protected))
print("   avg_price kept?", "avg_price" in protected)
""")

# --------------------------------------------------------------------------
md("""
## 3. Hyperparameter validation, per category

Three things are tuned, and all three genuinely differ between categories:

| | what it controls |
|---|---|
| `fourier_k` | how many harmonics the yearly seasonality gets |
| `adstock_decay` | how long media carries over |
| `alpha` | ridge strength on the elasticities |

Every combination is scored on the same expanding-window folds. Ice cream needs a
richer seasonal shape than baby formula; laundry's media carries over for weeks
while baby formula's does not carry at all. One global config would be wrong for
at least three of the four.
""")

code("""
print(f"{cat}: top 8 of {len(result['grid'])} configurations")
result["grid"].head(8)
""")

code("""
g = result["grid"]
best_alpha = g.iloc[0]["alpha"]
sub = g[g.alpha == best_alpha].pivot(index="fourier_k", columns="adstock_decay",
                                     values="cv_rmse")
fig, ax = plt.subplots(figsize=(6, 3.2))
im = ax.imshow(sub.values, cmap="viridis_r", aspect="auto")
ax.set_xticks(range(len(sub.columns)), sub.columns)
ax.set_yticks(range(len(sub.index)), sub.index)
ax.set_xlabel("adstock decay"); ax.set_ylabel("fourier k")
ax.set_title(f"{cat}: CV RMSE at alpha={best_alpha} (darker is better)",
             loc="left", fontsize=10)
ax.grid(False); plt.colorbar(im); plt.tight_layout(); plt.show()
""")

# --------------------------------------------------------------------------
md("""
## 4. Train / test, then refit on everything

The test score comes from a model that saw neither the test weeks nor — crucially
— let them influence which features exist.

The model that actually ships is refitted on the full history with the same
recipe. Holding out the most recent year forever, just to preserve a number
already recorded, would throw away the most relevant data in the panel.
""")

code("""
train_model = mdl.fit_category(train, cat, result["selected"], result["config"], t0)
print("Trained on the first", len(train), "weeks only:")
print("  train:", {k: round(v, 3) for k, v in mdl.score(train_model, train).items()})
print("  test :", {k: round(v, 3) for k, v in mdl.score(train_model, test).items()})

final = mdl.fit_category(df, cat, result["selected"], result["config"], t0)
print("\\nRefitted on all", len(df), "weeks -- this is the model that ships.")

fig, ax = plt.subplots()
ax.plot(df["date"], df["volume"] / 1e3, color=cfg.INK_MUTED, lw=1, label="Actual")
ax.plot(df["date"], final.predict(df) / 1e3, color=cfg.SERIES[0], lw=1.4, label="Fitted")
ax.axvline(test["date"].min(), color=cfg.NEG, ls=":", lw=1.2)
ax.text(test["date"].min(), ax.get_ylim()[1], " test split", va="top",
        fontsize=9, color=cfg.NEG)
ax.set_ylabel("k units"); ax.legend(frameon=False, ncol=2)
ax.set_title(f"{cat}: actual vs fitted", loc="left", fontsize=11)
plt.tight_layout(); plt.show()
""")

md("Now every category, end to end. This is exactly what `build.py` does.")

code("""
models, rows = {}, []
for c in cfg.CATEGORIES:
    d = panel[panel.category == c].sort_values("date").reset_index(drop=True)
    cut = d.date.max() - pd.Timedelta(weeks=cfg.TEST_WEEKS)
    tr, te = d[d.date <= cut], d[d.date > cut]
    r = sel.select_for_category(tr, c, t0, verbose=False)
    tm = mdl.fit_category(tr, c, r["selected"], r["config"], t0)
    fm = mdl.fit_category(d, c, r["selected"], r["config"], t0)
    fm.metrics = {"train": mdl.score(tm, tr), "test": mdl.score(tm, te)}
    fm.selection = r
    models[c] = fm
    rows.append({"category": c, "drivers": len(r["selected"]),
                 **r["config"],
                 "test MAPE %": fm.metrics["test"]["mape"],
                 "test R2 log": fm.metrics["test"]["r2_log"]})
pd.DataFrame(rows).set_index("category").round(3)
""")

md("""
Read the `fourier_k` and `adstock_decay` columns above against the truth in
`wk_config.CATEGORY_SHAPE`. The categories were generated with different seasonal
shapes and different media carryover, and validating each one separately recovers
that. A single shared config could not have.
""")

code("""
print("Selected drivers, per category:\\n")
for c, m in models.items():
    dropped = [d for d in cfg.DRIVER_NAMES if d not in m.drivers]
    print(f"{c}\\n   kept    ({len(m.drivers)}): {', '.join(m.drivers)}")
    print(f"   dropped ({len(dropped)}): {', '.join(dropped) or 'none'}\\n")

print("Did selection find the true zeros?")
for c, m in models.items():
    zeros = [d for d in cfg.DRIVER_NAMES if truth.loc[c, d] == 0]
    found = [d for d in zeros if d not in m.drivers]
    kept_real = [d for d in cfg.DRIVER_NAMES if truth.loc[c, d] != 0 and d in m.drivers]
    print(f"  {c:<20} true zeros dropped {len(found)}/{len(zeros)}   "
          f"real drivers kept {len(kept_real)}/{int((truth.loc[c] != 0).sum())}")
""")

# --------------------------------------------------------------------------
md("""
## 5. Driver impact — which levers matter, and which way

Two numbers, and the difference between them matters:

- **elasticity** — % change in volume for a 1% change in the driver.
- **impact** — % change in volume for a *one-standard-deviation* move in the
  driver, using that category's own history.

Elasticity alone overstates a driver that never moves. A −2.2 price elasticity on
a price that varies by 3% is a smaller lever than a +0.06 media elasticity on a
budget that swings by half. Impact is the one to plan against; elasticity is the
one to sanity-check against the truth.
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, (c, m) in zip(axes, models.items()):
    imp = m.impacts().sort_values("impact_pct")
    colors = [cfg.POS if v > 0 else cfg.NEG for v in imp["impact_pct"]]
    ax.barh(range(len(imp)), imp["impact_pct"], color=colors)
    ax.set_yticks(range(len(imp)), imp["driver"], fontsize=8)
    ax.axvline(0, color=cfg.INK_MUTED, lw=1)
    ax.set_title(c, loc="left", fontsize=10)
    ax.set_xlabel("% per 1sd")
plt.tight_layout(); plt.show()
""")

code("""
print(f"{cat} -- impact table as the app shows it:")
models[cat].impacts().round(3)
""")

code("""
est = pd.DataFrame({c: {d: float(models[c].elasticities.get(d, 0.0))
                        for d in cfg.DRIVER_NAMES} for c in cfg.CATEGORIES}).T
err = (est - truth).abs()
print("Estimated elasticity (dropped drivers show as 0):")
print(est.round(3).to_string())
print("\\nMean absolute error vs the truth, per category:")
print(err.mean(axis=1).round(4).to_string())
""")

# --------------------------------------------------------------------------
md(f"""
## 6. Forecast the selected drivers, then the baseline

Each `(category, driver)` pair is forecast on its own — the same driver can look
completely different for two categories, and nothing forces them to agree.

Only **selected** drivers are forecast. That is most of the practical payoff of
doing selection first: the forecasting work, and the maintenance of those
forecasts, scales with the drivers you actually use rather than with everything
you happen to collect.

Then the baseline: {cfg.FORECAST_WEEKS} weeks of volume on those driver paths.
""")

code("""
import wk_drivers
driver_fc = pd.read_csv(cfg.DRIVER_FORECAST_CSV, parse_dates=["date"])
models_built = mdl.load()   # the artifacts build.py wrote

fig, axes = plt.subplots(2, 2, figsize=(13, 6))
for ax, c in zip(axes.ravel(), cfg.CATEGORIES):
    h = panel[panel.category == c]
    f = driver_fc[driver_fc.category == c]
    ax.plot(h["date"], h["avg_price"], color=cfg.INK_MUTED, lw=1, label="history")
    ax.plot(f["date"], f["avg_price"], color=cfg.SERIES[1], lw=1.4, label="forecast")
    ax.set_title(f"{c} -- avg_price", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
""")

code("""
base = pd.read_csv(cfg.BASELINE_CSV, parse_dates=["date"])
fig, axes = plt.subplots(2, 2, figsize=(13, 6))
for ax, c in zip(axes.ravel(), cfg.CATEGORIES):
    b = base[base.category == c]
    h, f = b[b.period == "history"], b[b.period == "forecast"]
    ax.plot(h["date"], h["volume_actual"] / 1e3, color=cfg.INK_MUTED, lw=0.9,
            label="actual")
    ax.plot(f["date"], f["volume_baseline"] / 1e3, color=cfg.SERIES[0], lw=1.4,
            label="baseline")
    ax.set_title(c, loc="left", fontsize=10); ax.set_ylabel("k units")
    ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print("Two-year baseline volume:")
print(base[base.period == "forecast"].groupby("category")["volume_baseline"]
      .sum().round(0).to_string())
""")

# --------------------------------------------------------------------------
md("""
## 7. A scenario

This is the loop the Streamlit app implements:

    download  ->  a CSV of this category's drivers, history and forecast together
    edit      ->  in Excel, by whoever knows the plan
    upload    ->  the edited forecast rows become the scenario

A file rather than sliders, because sliders can only say "everything moves by X%".
A real plan is "price holds until March then rises 4%, and we pull the May display
event forward two weeks" — a column of numbers.

Below, the edit is done in pandas instead of Excel, but it is the identical code
path the upload takes.
""")

code("""
cat2 = "Ice Cream"
m2 = models_built[cat2]
edited = base[base.category == cat2].copy()
fut = edited["period"] == "forecast"

# The plan: +4% price from 2027, and a heavier summer media push.
edited.loc[fut & (edited.date.dt.year >= 2027), "avg_price"] *= 1.04
summer = fut & edited.date.dt.month.isin([5, 6, 7])
edited.loc[summer, "tv_grps"] *= 1.5

problems = scen.validate_upload(edited, m2, driver_fc)
print("Validation:", problems or "file is usable")

r = scen.run_scenario(m2, panel, driver_fc, edited)
s = r["summary"]
print(f"\\nBaseline  {s['baseline_volume']:>14,.0f}")
print(f"Scenario  {s['scenario_volume']:>14,.0f}")
print(f"Change    {s['delta_volume']:>+14,.0f}   ({s['delta_pct']:+.2f}%)")
""")

code("""
print("What changed in the plan:")
display(r["changes"].round(2))
print("\\nWhere the volume difference comes from:")
display(r["waterfall"].round(1))

fig, ax = plt.subplots()
p = r["paths"]
h = panel[panel.category == cat2].tail(52)
ax.plot(h["date"], h["volume"] / 1e3, color=cfg.INK_MUTED, lw=0.9, label="actual")
ax.plot(p["date"], p["baseline"] / 1e3, color=cfg.SERIES[0], lw=1.4, label="baseline")
ax.plot(p["date"], p["scenario"] / 1e3, color=cfg.SERIES[1], lw=1.4, label="scenario")
ax.set_ylabel("k units"); ax.legend(frameon=False, ncol=3)
ax.set_title(f"{cat2}: baseline vs scenario", loc="left", fontsize=11)
plt.tight_layout(); plt.show()
""")

md("""
Note the shape of the difference: the media uplift only appears in summer weeks,
and it decays for a few weeks after each burst rather than stopping dead — that is
the validated adstock carrying over. The price effect only starts in 2027. The
model responds to *when* the plan changes, not just to how much.
""")

# --------------------------------------------------------------------------
md("""
## 8. Decomposition

The model is additive in logs, so predicted volume splits cleanly into a base plus
each block's deviation from its own historic average. Contributions are converted
back to units multiplicatively, so the parts sum **exactly** to the prediction
rather than approximately.

`base` is volume at this category's average driver levels with trend, seasonality
and holidays at their historic means. Everything else is deviation from that.
""")

code("""
dec = scen.decompose_period(models_built[cat2], panel, driver_fc)
dec["quarter"] = pd.PeriodIndex(dec["date"], freq="Q").to_timestamp()
blocks = [c for c in dec.columns
          if c not in ("date", "predicted", "period", "quarter", "base")]
q = dec.groupby("quarter")[["base"] + blocks].sum()

fig, ax = plt.subplots(figsize=(13, 5))
bottom_pos = np.zeros(len(q)); bottom_neg = np.zeros(len(q))
ax.bar(q.index, q["base"] / 1e3, width=70, color="#d9d8d0", label="base")
bottom_pos += q["base"].values / 1e3
for i, b in enumerate(blocks):
    v = q[b].values / 1e3
    pos, neg = np.clip(v, 0, None), np.clip(v, None, 0)
    ax.bar(q.index, pos, width=70, bottom=bottom_pos,
           color=cfg.SERIES[i % len(cfg.SERIES)], label=b)
    ax.bar(q.index, neg, width=70, bottom=bottom_neg,
           color=cfg.SERIES[i % len(cfg.SERIES)])
    bottom_pos += pos; bottom_neg += neg
ax.axvline(pd.Timestamp(cfg.HISTORY_END), color=cfg.INK_PRIMARY, ls=":", lw=1.4)
ax.set_ylabel("k units per quarter")
ax.set_title(f"{cat2}: what builds volume (dotted line = forecast starts)",
             loc="left", fontsize=11)
ax.legend(frameon=False, ncol=6, fontsize=8)
plt.tight_layout(); plt.show()

recon = dec[["base"] + blocks].sum(axis=1)
print("Parts sum to prediction, max relative error:",
      f"{np.abs(recon / dec['predicted'] - 1).max():.2e}")
""")

code("""
f = dec[dec.period == "forecast"]
out = pd.DataFrame({"units": [f["base"].sum()] + [f[b].sum() for b in blocks]},
                   index=["base"] + blocks)
out["% of forecast"] = (100 * out["units"] / f["predicted"].sum()).round(2)
print(f"{cat2}: contribution over the two-year forecast")
out.sort_values("units", ascending=False).round(0)
""")

# --------------------------------------------------------------------------
md("""
## What to take away

- **Per-category selection changed the answer, not just the presentation.** The
  four categories kept different drivers and validated to different seasonal and
  carryover settings. A single shared model would have imposed one of those on all
  four.
- **Cross-validated error is not the same objective as decision support.** Left to
  itself it dropped the price lever from a category because a correlated
  non-lever forecast just as well. Forecast quality was indifferent; the tool was
  not.
- **The known limits of this run.** Baby formula keeps `avg_temp_c`, whose true
  elasticity is zero — a false positive that survived because it is harmless to
  forecast error. Ice cream's test year is biased −8%, which is the unobserved
  AR(1) factor sitting high across that particular year, not a fixable modelling
  error. Both are visible in the artifacts rather than smoothed over.

Then: `streamlit run app.py`.
""")

nb["cells"] = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out = cfg.ROOT / "walkthrough.ipynb"
print(f"Executing {len(cells)} cells ...")
NotebookClient(nb, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": str(cfg.ROOT)}}).execute()
nbf.write(nb, str(out))

errors = [
    o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"
]
print(f"Wrote {out}  ({len(cells)} cells, {len(errors)} errors)")
for e in errors:
    print("  !!", e.get("ename"), e.get("evalue"))
