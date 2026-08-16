# Weekly category planner

Weekly (week-ending-Sunday) volume planning for four categories, built end to end:
**feature selection per category → validated model per category → elasticities and
driver impacts → two-year driver forecasts → baseline → downloadable/uploadable
scenarios → decomposition**, with a Streamlit front end.

**There is no pooled model anywhere.** Each category gets its own feature set, its
own validated hyperparameters, its own fitted coefficients, its own driver
forecasts and its own decomposition. No category sees another category's data,
borrows its coefficients, or shares a fixed effect with it. Four independent
models that happen to live in the same folder.

Self-contained — its own config, data, and artifacts. Nothing outside this folder
is read or written.

```bash
cd experiments/weekly_planner

python wk_data.py           # generate the weekly panel        (~2s)
python build.py             # select, validate, fit, forecast  (~8s)
python smoke.py             # 33 end-to-end checks
python build_notebook.py    # regenerate walkthrough.ipynb with live outputs
streamlit run app.py        # the planner
```

`build.py` generates the panel automatically if it is missing, so a cold start is
just `python build.py && streamlit run app.py`.

---

## The data

241 weeks (2022-01-02 → 2026-08-09), four categories, 11 candidate drivers, 104
forecast weeks (→ 2028-08-06).

| category | summer/winter | why it needs its own model |
|---|---|---|
| Ice Cream | **2.97×** | temperature-driven, sharp seasonal shape |
| Ground Coffee | 0.80× | winter-peaking, most price-elastic (−2.2) |
| Laundry Detergent | 1.02× | flat, promotion- and distribution-driven, long media carryover |
| Baby Formula | 0.99× | flat and inelastic (−0.6); no competitor-price response |

Drivers span price (own and competitor), promotion (depth, share, feature+display),
distribution, media (TV GRPs, digital — both with carryover), and external
(temperature, food CPI, consumer confidence).

The panel is synthetic, which is the point: the true elasticity of every driver in
every category is known, so the pipeline can be **scored** rather than admired.
Several are exactly zero — temperature does nothing to laundry detergent,
competitor price does nothing to baby formula — so feature selection has real
true-negatives to find. An AR(1) latent factor and weekly noise are deliberately
unrecoverable, keeping R² believable.

---

## The pipeline

### 1. Feature selection, per category (`wk_selection.py`)

Backward elimination scored on **expanding-window cross-validation** — always train
on the past, test on the future. Start with all 11 drivers; repeatedly drop the one
whose removal costs least in out-of-fold error, while that cost stays under
`SELECTION_TOLERANCE` (0.2% of RMSE).

In-sample fit could not do this job: adding a driver can only ever improve it.

Two guards stop pure error-chasing from building the wrong tool:

- **Sign prior.** A driver whose *unconstrained* elasticity contradicts its business
  prior is removed before the search starts. Fitted unconstrained on purpose — the
  disagreement is the signal, and a constrained fit would pin the coefficient at
  zero and let the driver look fine.
- **Lever protection.** A `controllable` driver whose estimated impact clears
  `MATERIALITY_PCT` is never eliminated. See below.

### 2. Validation, per category (`wk_model.py`)

Every combination of `fourier_k` × `adstock_decay` × `alpha` (112 configs) scored
on the same expanding-window folds. Order matters: shape is settled first with all
drivers present, then drivers are eliminated, then shape is **re-validated** on the
surviving set — the right number of harmonics can change once drivers that were
carrying part of the seasonality are gone.

### 3. Train / test, then refit (`build.py`)

Last 52 weeks held out. Selection runs on the **training rows only** — select on all
241 weeks and the "held out" year has already shaped which features exist. The
model that ships is then refitted on the full history with the same recipe;
permanently discarding the most recent year to preserve a number already recorded
is not a trade a forecaster should make.

### 4. Driver forecasts (`wk_drivers.py`)

Each `(category, selected driver)` pair forecast independently with Prophet (with a
transparent trend + week-of-year fallback). Only selected drivers are forecast —
the forecasting and maintenance work scales with the drivers you use, not with
everything you collect. Forecasts are held inside a plausible envelope around
observed history, because two years is a long extrapolation for a weekly series.

### 5. Scenarios (`wk_scenario.py`, `app.py`)

**Download → edit → upload**, not sliders. A slider can only say "everything moves
by X%". A real plan is "price holds until March then rises 4%, and we pull the May
display event forward two weeks" — a column of numbers. The download contains
history and forecast in one sheet, with only that category's selected drivers as
columns. Only `period == 'forecast'` rows are read back; history is context for
media carryover, and editing it would move the actuals line.

Uploads are validated for missing columns, wrong or missing dates, blanks and
negatives, with the problems reported rather than silently coerced.

---

## Results

```
                   drivers  fourier_k  adstock  alpha  train MAPE %  test MAPE %  test R2 log  test bias %
Ice Cream               10          3      0.5   0.01         3.116        7.885        0.937       -8.342
Ground Coffee            9          3      0.5   0.50         3.077        7.018        0.849       -6.956
Laundry Detergent        8          1      0.7   3.00         3.697        3.820        0.969       -1.026
Baby Formula             8          1      0.0   0.01         3.468        4.640        0.915       +3.398
```

**Validation recovered the per-category shape.** Media carryover was recovered
exactly for three of four categories (Laundry 0.7, Coffee 0.5, Baby Formula 0.0 —
i.e. it correctly found that baby formula media has no carryover at all), and the
seasonal harmonic count exactly for three of four. One shared config would have
been wrong for at least three categories.

**Selection found the true zeros.**

| category | true zeros dropped | real drivers kept | elasticity MAE |
|---|---|---|---|
| Ice Cream | 0 / 0 | 10 / 11 | 0.117 |
| Ground Coffee | 1 / 1 | 9 / 10 | 0.188 |
| Laundry Detergent | 1 / 1 | 8 / 10 | 0.139 |
| Baby Formula | 2 / 3 | 7 / 8 | 0.098 |

Two-year baseline: Ice Cream 113.4M, Ground Coffee 142.7M, Laundry Detergent
231.0M, Baby Formula 63.3M units.

### The finding worth reading

**Cross-validated forecast error is not the same objective as decision support,
and optimising the first alone built the wrong tool.**

The first version selected purely on CV error. It dropped `avg_price` from Laundry
Detergent and kept `competitor_price`.

That is not a bug in the search. Our price and the competitor's price move together
— both track food CPI, both respond to the same promo calendar — so out-of-fold
error genuinely cannot separate them, and swapping one for the other costs nothing
measurable. As a *forecast*, a fair trade. As a *planning tool*, useless: it deletes
the one number the category team actually sets and replaces it with a number they
can only watch.

So drivers are marked `controllable` in the config, and a controllable driver is
protected from elimination when its estimated impact clears `MATERIALITY_PCT` (1%
volume per 1sd). The threshold is what stops this becoming blanket protection — a
controllable driver the model finds genuinely inert is still dropped, which is
exactly what happens to feature+display in baby formula, whose true elasticity is
zero.

Keeping the real lever turned out to be **more accurate too**: Laundry Detergent's
elasticity MAE fell from 0.354 to 0.139, and its test MAPE improved slightly. The
collinear substitute was not a free swap, it was just a swap CV could not price.

`walkthrough.ipynb` section 2b runs the selection both ways side by side.

---

## The app

`streamlit run app.py`, four tabs per category:

- **Model & validation** — test/train metrics, actual vs fitted with the split
  marked, what selection kept and dropped, the backward-elimination trace, and the
  top of the validation grid.
- **Driver impact** — signed bar chart of % volume change per 1sd move, plus the
  elasticity table. Impact rather than raw elasticity, because a −2.2 elasticity on
  a price that varies 3% is a smaller lever than a +0.06 elasticity on a media
  budget that swings by half.
- **Baseline & scenario** — download the driver CSV, upload the edited one, see
  baseline vs scenario, what changed in the plan, and a waterfall of where the
  volume difference came from.
- **Decomposition** — quarterly stacked contributions across history and forecast
  in one continuous frame, plus a contribution table over the horizon.

---

## Known limits

- **Baby Formula keeps `avg_temp_c`**, whose true elasticity is zero. A false
  positive that survived because it is harmless to forecast error and temperature
  carries no sign prior to test it against. It is one of three true zeros; the
  other two were dropped.
- **Ice Cream's test year is biased −8%.** That is the unobserved AR(1) factor
  sitting high across that particular 52 weeks, not a fixable modelling error — the
  same model has 3.1% train MAPE. It is one draw of a test period, and a single
  test window on a series with persistent unobserved shocks will do this.
- **Ice Cream and Ground Coffee both validated to `fourier_k=3`** where the truth is
  3 and 2. Temperature and the Fourier terms are partly collinear for ice cream, so
  the split between "driver" and "seasonality" is not sharply identified.
- **One synthetic panel, one seed.** `RANDOM_SEED` in the config is exposed; the
  per-category *ordering* of results is stable, individual driver decisions less so.
- **Driver forecasts are independent per driver.** Price and promo depth are
  mechanically related in the generator, and forecasting them separately ignores
  that. Fine for a baseline; it would matter for driver forecast intervals, which
  this does not attempt.

---

## Files

| file | what it is |
|---|---|
| `wk_config.py` | categories, calendar, driver spec, ground truth, validation grid. Every knob |
| `wk_data.py` | generates the weekly panel |
| `wk_features.py` | log transforms, adstock, Fourier seasonality, holiday weeks |
| `wk_selection.py` | **per-category feature selection** — backward elimination + the two guards |
| `wk_model.py` | the signed ridge, expanding-window CV, config search, impacts, decomposition |
| `wk_drivers.py` | forecasts each category's selected drivers |
| `wk_scenario.py` | baseline, upload validation, scenario, decomposition |
| `build.py` | runs the pipeline, writes `artifacts/` |
| `app.py` | the Streamlit planner |
| `smoke.py` | 33 end-to-end checks, including a full download→edit→upload round trip |
| `build_notebook.py` | regenerates `walkthrough.ipynb` with executed outputs |
| `walkthrough.ipynb` | the whole pipeline, explained and run, using the modules above |
| `build_notebook_standalone.py` | regenerates `walkthrough_standalone.ipynb` |
| `walkthrough_standalone.ipynb` | the same walkthrough with **every function defined inline** — imports nothing from this folder, reads no CSV, loads no pickle. Copy it anywhere with numpy/pandas/matplotlib/scipy and it runs top to bottom. No decomposition section |
