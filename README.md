# Category Volume Scenario Planner

Monthly category-level volume forecasting with a driver-based response model and
a Streamlit scenario front end. Synthetic data, built to be replaced by real data.

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build.py      # generate data -> forecast drivers -> fit model
.venv/bin/python tests/smoke.py        # end-to-end checks on the built artifacts
.venv/bin/streamlit run app.py
```

To understand what happens *after* Prophet — how the response model is fitted and
how the scenario engine turns a slider into a volume delta — read
[`notebooks/model_and_scenario_walkthrough.ipynb`](notebooks/model_and_scenario_walkthrough.ipynb).
It runs against the built artifacts and ships with outputs. Regenerate it with
`python scripts/build_notebook.py` (writes the notebook) followed by a re-execute.

`scripts/build.py --no-forecast` skips the Prophet step (~40s) and reuses the
existing driver forecast — use it while iterating on the response model.

## How it works

```
data_gen.py    5 categories x 66 months (2021-01 .. 2026-06), 9 drivers
    |          log-log DGP with known elasticities + an unobservable AR(1) factor
    v
forecast.py    Prophet per driver, 18 months forward.
    |          Macro drivers are shared -> one model each, broadcast to all categories.
    |          Commercial drivers -> one model per (category, driver). 29 fits total.
    v
model.py       Pooled sign-constrained log-log ridge:
    |            log(volume) ~ log(drivers) + category FE + month FE + trend
    |          Coefficients on the log drivers ARE the elasticities.
    v
scenario.py    Bend the forecast drivers by a %, re-predict, attribute the delta.
    v
app.py         Overview | Drivers | Driver impact | Scenario
```

### Why these choices

**Log-log.** Coefficients read directly as elasticities — "a 1% price cut lifts
volume 1.7%" — which is the unit a category manager actually reasons in, and it
makes the response multiplicative rather than additive.

**Ridge, not OLS.** Average price is mechanically entangled with promo depth and
promo share (average price *is* list price net of promotion), and promo depth
correlates 0.89 with promo share. The collinearity is real and shrinkage is the
point. Alpha is picked by expanding-window CV cut on dates, not on rows — a
random split would leak future months into training on a panel.

**Sign constraints.** Food CPI correlates **0.963** with the time trend (VIF 23.3),
so its coefficient is only weakly identified — what it reports depends on how much
shrinkage is applied. At near-OLS (`alpha=0.0001`) the unconstrained fit returns
*+0.41*: food inflation raising volume. Drivers with a known direction are
therefore pinned to it (`sign` in `config.py`).

Be clear about what this is doing, though: **at the CV-selected `alpha=0.05` the
constraint binds on zero of nine drivers.** Shrinkage alone already holds CPI at
−0.52, and every constrained estimate matches the unconstrained one to four
decimals. The constraint is insurance against the low-alpha regime and against
future data shifts, not a live correction. `avg_temp_c` is deliberately left free,
since its sign genuinely differs by category.

**Adstock on media.** Geometric carryover (decay 0.45) applied before the log, so
spend doesn't vanish the month it lands. History is always prepended before
scenario prediction so carryover crosses the forecast seam correctly.

**Contribution attribution.** Volume is split into a base — all drivers at their
category's historical mean — plus per-driver incremental volume, allocated by
each driver's share of the total log deviation. Parts sum to the whole exactly
(verified in `tests/smoke.py`).

## Model performance

| Metric | Value |
|---|---|
| R² (log space, in-sample) | 0.921 |
| MAPE, 12-month holdout | 13.6% |
| Ridge alpha (CV-selected) | 0.05 |
| Observations | 330 |

Elasticity recovery against the known generating values:

| Driver | Estimated | True | Error |
|---|---|---|---|
| avg_price | −1.715 | −1.598 | −0.117 |
| distribution_acv | 1.237 | 0.920 | +0.317 |
| promo_depth | 0.612 | 0.582 | +0.030 |
| promo_share | 0.294 | 0.294 | 0.000 |
| media_spend | 0.123 | 0.068 | +0.055 |
| cpi_food | −0.520 | −0.615 | +0.095 |
| consumer_confidence | 0.485 | 0.446 | +0.039 |
| unemployment_rate | −0.172 | −0.180 | +0.008 |
| avg_temp_c | 0.199 | 0.169 | +0.030 |

All signs correct, all magnitudes in the right neighbourhood.

## Known limitations

**The holdout MAPE is 13.6% against a ~4.4% noise floor, and pooling is why.**
The error concentrates almost entirely in two categories:

| Category | Holdout MAPE | Residual sd | True temp elasticity |
|---|---|---|---|
| Household Care | 23.2% | 0.209 | −0.02 |
| Beverages | 19.6% | 0.229 | +0.42 |
| Frozen Foods | 9.9% | 0.116 | +0.24 |
| Dairy | 8.4% | 0.090 | — |
| Snacks | 7.1% | 0.072 | — |

Those are exactly the two categories whose temperature response diverges most
from the pooled estimate of **+0.199**. One elasticity per driver cannot describe
both a category that loves heat and one that ignores it.

`scripts/experiment_per_category.py` tests that diagnosis, and it turns out to be
mostly wrong — see below.

## Experiment: pooled vs per-category

    .venv/bin/python scripts/experiment_per_category.py

Five arms, identical 12-month holdout and CV protocol throughout:

| Arm | What varies by category | Holdout MAPE | Elasticity MAE | Price slider MAE |
|---|---|---|---|---|
| A pooled | intercept only | 13.64% | 0.0975 | 1.96pp |
| B per-category | everything (5 independent models) | 4.49% | 0.1334 | 3.63pp |
| C calendar-only | trend + seasonality; elasticities pinned | 4.87% | 0.0975 | 1.96pp |
| D hybrid | commercial elasticities, α by CV | **4.41%** | 0.1343 | 6.45pp |
| E hybrid tuned | commercial elasticities, α = 100 | 4.77% | **0.0962** | **1.76pp** |

Elasticity MAE is mean `|estimate − truth|` against the generator's
*per-category* elasticities. The price slider column is the error in the reported
volume lift from a 10% price cut — what a planner actually reads off the tool.

**Splitting the model cuts holdout error by 67%, but almost none of that is about
drivers.** Arm C pins every elasticity to its pooled value and changes only the
trend and month effects, and it recovers **96%** of the fully-split gain. The
pooled model has one trend and one seasonal pattern for all five categories,
while the generator gives them trends from −1.0% to +4.5%/yr and seasonal
amplitudes from 0.04 to 0.16. That mis-specification, not the elasticities, is
what the holdout MAPE was measuring.

**Fully splitting makes the elasticities worse, not better** (0.0975 → 0.1334).
It helps where categories genuinely differ — temperature MAE drops 0.134 → 0.030,
and it is the only arm to get Household Care's negative sign right — but it
wrecks the macro drivers (0.075 → 0.156). Macro drivers are one shared time
series broadcast to every category, so a 66-month slice holds no information
about them that the pooled fit did not already have. Food CPI is the worst case:
correlated 0.96 with the trend, it is not identified within a single category.

**Holdout error cannot arbitrate elasticity quality.** Arm B wins the forecast and
loses the elasticities; arm A does the reverse. In arm D, CV picked the weakest
penalty on the grid for all five categories, because reallocating between price,
food CPI and the trend barely moves forecast error — so the deviations ran free
and price elasticity landed at −2.69 for Beverages and −0.01 for Dairy. Section 4
sweeps that penalty by hand and traces the whole frontier, from free-fit
(4.15% MAPE / 0.172 MAE) to fully pooled (4.87% / 0.0975).

**Arm E is the recommendation.** Commercial elasticities are written as
deviations from the pooled estimate and shrunk toward it rather than toward zero,
macro elasticities stay pooled, and trend and seasonality are per-category and
unpenalised. It beats pooling on every axis at once: 65% lower holdout error, and
elasticities and slider output that are slightly better rather than worse. It
nudges Dairy toward being the inelastic category it is (−1.63, vs pooled −1.72
against a truth of −1.15) without the blow-ups of the free fit. That is a modest
move, and it is the honest ceiling here: at the shrinkage level that keeps the
other categories sane, 66 months cannot support a bigger departure.

One caveat stated plainly: α = 100 was chosen against ground truth, which exists
only because the panel is synthetic. On real data that is a judgement call
informed by prior elasticity studies — which is precisely what the CV objective
is shown to be unable to supply.

Full numbers land in `artifacts/experiment_per_category.json`. The experiment
writes its own artifacts and does not touch the ones the app reads.

**`notebooks/per_category_vs_pooled_walkthrough.ipynb`** walks through all of the
above at a slower pace, with each step explained. It deliberately imports nothing
from `src/` or `config.py` — every transform, the solver, and all five arms are
written out in plain numpy and pandas so it reads top to bottom without opening
another file. Its numbers reproduce the table above exactly.

While building it, the backtests in all three model modules turned out to rebuild
the design matrix on the holdout rows alone, which restarted media adstock at the
seam and understated carryover in the first forecast months. That is now fixed
(build once over the panel, slice after — the same reason `scenario.py` prepends
history). It moved pooled holdout MAPE from 13.56% to 13.64% and affected all arms
about equally, so the conclusions are unchanged.

**Other things the MVP does not do:**

- Drivers are forecast independently, so their forecast paths can drift
  inconsistently (e.g. price and CPI decoupling). A VAR or a shared latent factor
  would tie them together.
- Prophet intervals are computed but not carried into the volume forecast — the
  scenario chart shows point estimates only.
- Sliders apply one uniform % across all forecast months and all selected
  categories. Per-month or per-category shaping would need a different UI.
- The response model has no diminishing returns beyond the log transform, and no
  competitive drivers (competitor price/promo were scoped out).

## Files

| File | Role |
|---|---|
| `config.py` | Every knob: categories, driver spec, signs, true elasticities, palette |
| `src/features.py` | Adstock + log transforms + design matrix. Shared by generator and model, so they cannot disagree |
| `src/data_gen.py` | Synthetic panel with known ground truth |
| `src/forecast.py` | Prophet driver forecasts, with a trend+seasonal fallback if Prophet is absent |
| `src/model.py` | `SignedRidge`, fitting, elasticities, importance, decomposition |
| `src/model_percat.py` | Experiment arm B: one independent model per category |
| `src/model_partial.py` | Experiment arms C–E: per-category deviations shrunk toward the pooled elasticities |
| `scripts/experiment_per_category.py` | Runs all five arms and scores them |
| `src/scenario.py` | Scenario application and delta attribution |
| `app.py` | Streamlit UI |
| `tests/smoke.py` | Reconciliation and directional checks |
| `notebooks/model_and_scenario_walkthrough.ipynb` | Annotated walkthrough of the model and scenario engine |
| `notebooks/per_category_vs_pooled_walkthrough.ipynb` | Self-contained walkthrough of the pooled-vs-per-category experiment; imports nothing from `src/` |
| `scripts/build_notebook.py`, `scripts/build_notebook_percat.py` | Regenerate those two notebooks |

## Swapping in real data

Replace `data_gen.generate()` with a loader returning a frame of
`date, category, volume, <driver columns>`. Then update `DRIVERS` in `config.py`
— transform, expected sign, slider range — and everything downstream follows.
