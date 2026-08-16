# Per-category feature selection for the scenario tool

Each category has its own drivers. The shipped model gives all five categories the
same nine and one shared set of elasticities. This experiment builds the pipeline
you actually asked for — **feature selection engine → per-category model on the
selected features → elasticities → response model → scenario planning** — and then
measures whether it produces a better planning tool than the alternatives.

Self-contained. Reads `data/panel.csv` and `data/driver_forecast.csv`; writes only
into `experiments/feature_selection/artifacts/`. Nothing the Streamlit app loads is
touched.

```bash
cd experiments/feature_selection
python run_experiment.py     # the whole pipeline + all five arms  (~30s)
python selection.py          # just the engine, with its full scoring table
python smoke.py              # 25 end-to-end checks on the fitted model
python decoys.py             # sanity-check the injected decoy drivers
```

---

## The measurement problem, and how it is solved

Feature selection is untestable without knowing which features are fake. The
generator in `src/data_gen.py` gives all nine real drivers a non-zero elasticity in
*every* category, so on the shipped panel there is no true sparsity to recover and
any engine would be graded against nothing.

`decoys.py` fixes that. It appends four extra drivers to the panel and the forecast
**after** volume has already been generated, so their true elasticity is exactly
zero — not small, zero. Each is a different way to fool a selector:

| decoy | what it is | the trap |
|---|---|---|
| `promo_echo` | `promo_depth` × lognormal noise | collinear with a real driver (0.84 raw, 0.53 after calendar) |
| `search_index` | smooth, trending, mildly seasonal | correlates with volume only through the trend |
| `competitor_promo` | white noise | the control — if this survives, thresholds are too loose |
| `fx_index` | a random walk shared across categories | spurious regression; non-stationary, drifts into agreement with anything |

Now precision and recall are real numbers.

---

## The engine (`selection.py`)

Two stages, because *"is this a driver at all?"* and *"can **this category**
estimate it?"* are different questions. Collapsing them is the mistake that makes
per-category selection dangerous — the first time a macro driver fails inside every
category, a one-stage engine concludes it is noise and deletes the slider.

**Stage 1 — global gate.** 330 observations, category fixed effects, month effects
and a *category-specific* trend all swept out first. Deliberately permissive; its
only job is excluding non-drivers.

**Stage 2 — per-category gate.** 66 observations, that category's own calendar swept
out. Decides whether a category fits its own elasticity or borrows the pooled one.

Four filters run at both levels; a driver must pass all of them.

1. **Stability** — 200 moving-block bootstrap draws (blocks, not iid, because the
   residuals are autocorrelated; blocks never cross a category boundary), a lasso on
   each at three penalties set as fractions of that draw's own `alpha_max`. Score is
   the fraction of (draw, penalty) pairs where the coefficient is non-zero. Catches
   drivers that look decisive in the full sample and vanish when you jiggle it.
2. **Sign consistency** — of the draws where a driver was selected, how often did it
   point the same way? A driver whose direction flips between resamples cannot be
   planned against, whatever its selection frequency.
3. **Sign prior** — a light *unconstrained* ridge; if the estimate lands opposite the
   business prior in `config.DRIVERS[...]["sign"]`, the category cannot identify it.
   Unconstrained on purpose — constraining it would hide the disagreement that is the
   whole signal.
4. **Materiality** — |elasticity| × sd(log driver) within the category, as a % volume
   swing per 1sd. A driver worth 0.2% of volume is a slider that wastes attention.

Everything is residualised on the calendar **before** any of this, so a driver earns
credit only for variation the model does not already explain with a clock.

Leave-one-driver-out CV is computed and recorded but **is not a gate**. Section 3 of
`scripts/experiment_per_category.py` already documents why: price, food CPI and the
trend are collinear enough that reallocating between them barely moves forecast
error, so CV cannot see the difference. It is reported so you can watch it fail to
discriminate rather than take its word for anything.

### What the engine decided

```
global gate keeps 10/13; cut: search_index, competitor_promo, fx_index
                rescued by business prior: cpi_food, unemployment_rate

driver         avg_price distribution_acv promo_depth promo_share media_spend cpi_food consumer_confidence unemployment_rate avg_temp_c promo_echo search_index competitor_promo fx_index
Beverages            fit             pool         fit         fit        pool     pool                 fit              pool        fit       pool            -                -        -
Snacks               fit              fit         fit         fit         fit     pool                 fit              pool        fit       pool            -                -        -
Dairy                fit             pool         fit         fit         fit     pool                 fit              pool        fit       pool            -                -        -
Frozen Foods         fit              fit         fit         fit         fit     pool                 fit              pool        fit       pool            -                -        -
Household Care       fit              fit         fit         fit         fit     pool                 fit              pool       pool        fit            -                -        -
```

`fit` = the category estimates its own elasticity · `pool` = real driver it cannot
identify, borrows the pooled value · `-` = failed the global gate, no slider at all.

**32 driver coefficients estimated instead of 65.**

|  | precision | recall | F1 | decoys kept | real drivers cut |
|---|---|---|---|---|---|
| data only | 0.875 | 0.778 | 0.824 | 1 | 2 |
| + business prior | **0.900** | **1.000** | **0.947** | 1 | 0 |

Two findings behind those numbers, both worth more than the score:

- **`cpi_food` and `unemployment_rate` fail on data alone.** Both are macro series
  that move once for the whole panel and track the time trend closely — residualising
  the calendar leaves `cpi_food` with 27% of its variation. `config.py` already pins
  their sign for exactly this reason. So the engine treats a non-zero `sign` as an
  assertion the business has already made and **demotes rather than deletes**: a
  protected driver can lose its per-category fit, never its slider. Set
  `protect_signed_priors: False` in `fs_config.py` to see the unmoderated verdict —
  both are reported either way.
- **`promo_echo` survives, and no threshold fixes it.** After category trends come
  out it still correlates 0.53 with `promo_depth`, and its sign is perfectly
  consistent across resamples. A proxy that close is not separable from what it
  proxies in 66 months. Tightening the stability threshold to cut it (0.855 vs
  `distribution_acv`'s 0.905) would be fitting the threshold to the answer. What
  bounds the damage is the fallback rule, measured below. A redundancy/VIF guard was
  tried and rejected: the strongest residual pair in the whole design is 0.71, and it
  is two *real* drivers (`avg_price`, `cpi_food`) — at any threshold that catches
  `promo_echo` it deletes food CPI.

---

## The arms

| | |
|---|---|
| **A pooled** | the shipped model's shape, widened to all 13 candidates |
| **B per-category** | five independent models, no selection — isolates what selection adds on top of splitting |
| **C selected → zero** | engine picks features; anything dropped gets elasticity 0 |
| **D selected → pooled** | same features; a driver that passed the global gate and lost only its category keeps the pooled elasticity |
| **E oracle** | per-category on exactly the nine real drivers. Not achievable — the ceiling, not a competitor |

Selection is **re-run on the training rows** inside the backtest rather than reusing
the full-sample choice. Selecting on all 66 months then scoring on the last 12 leaks
the holdout into the feature set and flatters every selected arm.

---

## Results

```
                          holdout MAPE %  elasticity MAE (real)  elasticity MAE (decoy)  price slider MAE pp  phantom volume %  coefficients
A pooled                         14.1850                 0.2434                  0.0397               10.774            2.9103            30
B percat all                      4.3285                 0.1815                  0.2256                5.778            1.3443           130
C percat selected                 9.0906                 0.2597                  0.0028                7.234            0.0697            97
D percat selected+pooled          3.9699                 0.2600                  0.0144               11.751            1.1711            97
E oracle                          4.4887                 0.1334                  0.0000                3.634            0.0000           110
```

**No arm wins everywhere, and the split is not a rounding error.**

### Selection pays for itself on accuracy and on slider honesty

Arm D forecasts better than every other arm **including the oracle** (3.97% vs
4.49% holdout MAPE), on a third fewer coefficients than arm B.

And it fixes something the accuracy column cannot show. The decoy test moves each
fake slider +20% on its own; the correct answer in every cell is 0.00%:

| arm | total phantom volume | worst single slider | live decoy sliders |
|---|---|---|---|
| A pooled | 2.91% | 1.24% | 4 |
| B percat all | 1.34% | **25.89%** | 4 |
| C selected→zero | **0.07%** | 1.01% | 1 |
| D selected→pooled | 1.17% | 1.24% | 1 |
| E oracle | 0.00% | 0.00% | 0 |

Arm B — per-category with no selection, the obvious next step from the shipped
model — will tell a planner that a 20% move in a **trade-weighted FX index shifts
Frozen Foods volume by 26%**. That number is entirely manufactured. Nothing in the
model fit looks wrong when it is produced, and splitting by category made it *worse*
than pooling, because five short samples give a random walk five chances to find a
spurious fit. This is the single strongest argument in the experiment for putting a
selection stage in front of a per-category model.

### Selection does not pay for itself on elasticity recovery

Arm D's real-driver MAE (0.260) is **worse** than arm B's (0.182), almost entirely
through price. The mechanism is worth more than the result:

```
Food CPI elasticity          true    A pooled   B percat all   D selected+pooled   E oracle
Beverages                  -0.563         0.0         -1.991                 0.0     -1.272
Dairy                      -0.612         0.0          0.000                 0.0     -0.120

Price elasticity             true    A pooled   B percat all   D selected+pooled   E oracle
Beverages                  -1.688      -2.428         -1.791              -3.313     -1.819
Snacks                     -1.793      -2.428         -0.942              -3.250     -1.914
```

The pooled `cpi_food` elasticity is **exactly 0.0** — sitting on the boundary of its
own sign constraint, because the trend absorbs the inflation signal. Selection
correctly identified that no category can estimate food CPI, and then handed each
category a fallback value that was itself unidentified. Borrowing that is not
pooling, it is deletion in disguise. And the effect does not politely vanish: price
runs 0.71 with CPI after the calendar comes out, so **price absorbs the entire
inflation signal** and blows out to −3.31 in Beverages against a truth of −1.69.

Arms B and E dodge this only by fitting CPI freely per category — not skill, their
CPI estimates are wrong too (−1.99 in Beverages), just wrong in a direction that
leaves price alone.

`fs_model.py` detects this case and `run_experiment.py` prints a **DEGENERATE
FALLBACK WARNING** rather than borrowing silently. The fix is not a better selector.
It is an external prior for food CPI — what `config.py`'s `sign` field already
gestures at, and what a real engagement would take from published elasticity work.

### Scenario planning, end to end

Price +5%, media +30%, promo depth −10%, over 2026-07 → 2027-12:

```
                arm D %  arm A %  gap pp
Beverages        -15.66   -12.85   -2.81
Snacks           -16.88   -12.77   -4.11
Dairy             -6.31   -12.67   +6.35
Frozen Foods     -13.46   -12.85   -0.60
Household Care   -11.52   -12.81   +1.28
```

Both arms land near −12.6% in total, and the total is the least useful number here.
The pooled model reports essentially the same answer for all five categories because
it structurally cannot do otherwise; arm D separates Dairy (a staple, −6.3%) from
Snacks (−16.9%) by more than 10 percentage points. That gap is the plan.

---

## How to read this

- **Use the selected model for forecasting and for deciding which sliders a planner
  is allowed to touch.** It is the most accurate arm and it removes three of four
  fake levers.
- **Do not read its price elasticity without first supplying an external prior for
  food CPI.** The warning fires automatically when this condition holds.
- **Feature selection cannot manufacture information the panel does not contain.** It
  can stop the model from pretending otherwise — but here it *relocated* the pretence
  (CPI → price) rather than removing it, and only the ground truth available in a
  synthetic panel makes that visible.

## Caveats

- One panel, one seed, one decoy draw. `fs_config.DECOY_SEED` and
  `SELECTION["seed"]` are exposed; conclusions about a *single* decoy surviving are
  one draw from a distribution, though the ordering of the arms is stable.
- The global gate assumes a driver's effects do not cancel across categories. A
  driver strongly positive in one category and equally negative in another could fail
  it wrongly. `avg_temp_c` (+0.42 Beverages, −0.02 Household Care) is the closest case
  here and passes comfortably, but the failure mode is real.
- The LOO-CV column uses `GridSearchCV`'s best inner score, which is optimistically
  biased. Consistently so across variants, and it is a diagnostic rather than a gate,
  but it is not a clean generalisation estimate.

## Files

| file | what it is |
|---|---|
| `fs_config.py` | decoy spec, selection thresholds, arm list. Every knob lives here |
| `decoys.py` | injects the four zero-effect drivers into panel + forecast |
| `fs_features.py` | `src/features.py` generalised to an arbitrary driver spec |
| `selection.py` | **the engine** — two stages, four filters |
| `fs_model.py` | pooled and per-category signed-ridge fits; the zero/pooled fallback rule |
| `scenario_fs.py` | scenario engine over a per-category elasticity matrix; `phantom_volume` |
| `run_experiment.py` | all five arms, all four scoring axes, writes `artifacts/` |
| `smoke.py` | 25 end-to-end checks on the fitted model |
