"""Weekly category planner.

    streamlit run app.py

Reads only the artifacts written by `build.py` -- it never fits anything, so the
numbers on screen are always the numbers that were validated.

The scenario loop is download -> edit -> upload. Pick a category, download its
driver file (history and forecast in one sheet), change the forecast rows in
Excel, upload it back, and the app re-runs that category's model on your plan.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wk_config as cfg
import wk_model as mdl
import wk_scenario as scen

st.set_page_config(page_title="Weekly Category Planner", layout="wide")

AXIS = dict(showgrid=True, gridcolor=cfg.GRIDLINE, zeroline=False,
            linecolor="#c3c2b7", tickfont=dict(color=cfg.INK_MUTED, size=11))


def layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    opts = dict(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor=cfg.SURFACE, plot_bgcolor=cfg.SURFACE,
        font=dict(family="system-ui, -apple-system, sans-serif",
                  color=cfg.INK_SECONDARY, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=AXIS, yaxis=AXIS,
    )
    opts.update(kw)
    fig.update_layout(**opts)
    return fig


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
@st.cache_data
def load_panel():
    return pd.read_csv(cfg.PANEL_CSV, parse_dates=["date"])


@st.cache_data
def load_driver_forecast():
    return pd.read_csv(cfg.DRIVER_FORECAST_CSV, parse_dates=["date"])


@st.cache_data
def load_baseline():
    return pd.read_csv(cfg.BASELINE_CSV, parse_dates=["date"])


@st.cache_resource
def load_models():
    return mdl.load()


if not cfg.MODELS_PKL.exists():
    st.error("No artifacts found. Run `python build.py` first.")
    st.stop()

panel = load_panel()
driver_fc = load_driver_forecast()
baseline = load_baseline()
models = load_models()
summary = json.loads(cfg.SUMMARY_JSON.read_text())

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Weekly Category Planner")
category = st.sidebar.selectbox("Category", cfg.CATEGORIES)
model = models[category]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{category}**")
st.sidebar.caption(
    f"{len(model.drivers)} of {len(cfg.DRIVER_NAMES)} candidate drivers selected  \n"
    f"Fourier k = {model.fourier_k} · adstock = {model.adstock_decay} · "
    f"alpha = {model.alpha}"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Every category is fitted independently: its own features, its own validated "
    "hyperparameters, its own driver forecasts. Nothing is pooled across categories."
)

hist = panel[panel["category"] == category].sort_values("date").reset_index(drop=True)
cat_base = baseline[baseline["category"] == category].copy()

st.title(category)

tab_model, tab_impact, tab_scenario, tab_decomp = st.tabs(
    ["Model & validation", "Driver impact", "Baseline & scenario", "Decomposition"]
)

# --------------------------------------------------------------------------
# Model & validation
# --------------------------------------------------------------------------
with tab_model:
    m = model.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test MAPE", f"{m['test']['mape']:.2f}%",
              help="Last 52 weeks, held out. The model never saw them, and neither "
                   "did feature selection.")
    c2.metric("Test R² (log)", f"{m['test']['r2_log']:.3f}")
    c3.metric("Test bias", f"{m['test']['bias_pct']:+.2f}%",
              help="Total predicted vs total actual over the test year.")
    c4.metric("Train MAPE", f"{m['train']['mape']:.2f}%")

    st.markdown("#### Actual vs fitted")
    fitted = model.predict(hist)
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["volume"], name="Actual",
                    line=dict(color=cfg.INK_MUTED, width=1.5))
    fig.add_scatter(x=hist["date"], y=fitted, name="Fitted",
                    line=dict(color=cfg.SERIES[0], width=2))
    split = hist["date"].max() - pd.Timedelta(weeks=cfg.TEST_WEEKS)
    fig.add_vline(x=split, line=dict(color=cfg.INK_MUTED, dash="dot", width=1))
    fig.add_annotation(x=split, y=1.02, yref="paper", text="test split starts",
                       showarrow=False, font=dict(size=10, color=cfg.INK_MUTED))
    st.plotly_chart(layout(fig, 400, yaxis=dict(**AXIS, title="Weekly units")),
                    width='stretch')
    st.caption(
        "The fitted line is the final model, refitted on the full history. The test "
        "metrics above come from a separate model that was trained only on data "
        "before the marked split."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Feature selection")
        st.markdown(f"**Kept ({len(model.drivers)}):** "
                    + ", ".join(cfg.DRIVERS[d]["label"] for d in model.drivers))
        dropped = model.selection["dropped"]
        st.markdown(f"**Dropped ({len(dropped)}):** "
                    + (", ".join(cfg.DRIVERS[d]["label"] for d in dropped) or "none"))
        if model.selection["dropped_wrong_sign"]:
            st.warning(
                "Dropped because the fitted elasticity contradicted the business "
                "prior: "
                + ", ".join(cfg.DRIVERS[d]["label"]
                            for d in model.selection["dropped_wrong_sign"])
            )
        elim = pd.DataFrame(model.selection["elimination_log"])
        if len(elim) > 1:
            st.markdown("Backward elimination, scored on out-of-fold error:")
            st.dataframe(elim, hide_index=True, width='stretch')

    with right:
        st.markdown("#### Validation")
        st.caption(
            "Every combination of seasonal harmonics, media carryover and ridge "
            "strength, scored on expanding-window folds. Top 10 shown; this "
            "category picked row 1."
        )
        st.dataframe(pd.DataFrame(model.selection["grid_top10"]).round(5),
                     hide_index=True, width='stretch')

# --------------------------------------------------------------------------
# Driver impact
# --------------------------------------------------------------------------
with tab_impact:
    st.markdown("#### How much each driver moves volume, and in which direction")
    st.caption(
        "Impact is the % change in volume from a one-standard-deviation move in "
        "that driver, using this category's own historic variation. Elasticity "
        "alone would overstate a driver that barely moves."
    )
    imp = model.impacts()
    fig = go.Figure()
    fig.add_bar(
        x=imp["impact_pct"], y=[cfg.DRIVERS[d]["label"] for d in imp["driver"]],
        orientation="h",
        marker=dict(color=[cfg.POS if v > 0 else cfg.NEG for v in imp["impact_pct"]]),
        hovertemplate="%{y}<br>%{x:+.2f}% per 1sd<extra></extra>",
    )
    fig.add_vline(x=0, line=dict(color=cfg.INK_MUTED, width=1))
    st.plotly_chart(
        layout(fig, 60 + 34 * len(imp), hovermode="closest",
               xaxis=dict(**AXIS, title="% volume change per 1sd move"),
               yaxis=dict(**AXIS, autorange="reversed")),
        width='stretch',
    )

    show = imp.copy()
    show["elasticity"] = show["elasticity"].round(3)
    show["impact_pct"] = show["impact_pct"].round(2)
    st.dataframe(
        show[["label", "group", "elasticity", "impact_pct", "direction"]].rename(
            columns={"label": "Driver", "group": "Group", "elasticity": "Elasticity",
                     "impact_pct": "Impact % per 1sd", "direction": "Direction"}),
        hide_index=True, width='stretch',
    )
    st.caption(
        "Elasticity is the % change in volume for a 1% change in the driver. "
        "Signs are constrained to the business prior where one exists, so price is "
        "never allowed to come back positive."
    )

# --------------------------------------------------------------------------
# Baseline & scenario
# --------------------------------------------------------------------------
with tab_scenario:
    st.markdown("#### 1. Download the driver file")
    st.caption(
        f"History plus the {cfg.FORECAST_WEEKS}-week forecast, one row per week. "
        f"Only the {len(model.drivers)} drivers this category selected are included "
        "— editing a driver the model does not use would do nothing."
    )
    st.download_button(
        f"Download {category} drivers (CSV)",
        cat_base.to_csv(index=False).encode(),
        file_name=f"{category.lower().replace(' ', '_')}_drivers.csv",
        mime="text/csv",
    )

    st.markdown("#### 2. Edit the forecast rows, then upload")
    st.caption(
        "Change any driver value in the rows where `period` is `forecast`. "
        "History rows are context for media carryover and are not re-read — "
        "editing them would move the actuals line, which is not a scenario."
    )
    uploaded = st.file_uploader("Upload the edited CSV", type="csv")

    if uploaded is None:
        st.info("Upload an edited file to see a scenario. The baseline forecast is "
                "shown below in the meantime.")
        fut = cat_base[cat_base["period"] == "forecast"]
        fig = go.Figure()
        fig.add_scatter(x=hist["date"], y=hist["volume"], name="Actual",
                        line=dict(color=cfg.INK_MUTED, width=1.2))
        fig.add_scatter(x=fut["date"], y=fut["volume_baseline"], name="Baseline forecast",
                        line=dict(color=cfg.SERIES[0], width=2))
        st.plotly_chart(layout(fig, 400, yaxis=dict(**AXIS, title="Weekly units")),
                        width='stretch')
    else:
        edited = pd.read_csv(uploaded)
        problems = scen.validate_upload(edited, model, driver_fc)
        if problems:
            st.error("The uploaded file could not be used:\n\n"
                     + "\n".join(f"- {p}" for p in problems))
        else:
            result = scen.run_scenario(model, panel, driver_fc, edited)
            s = result["summary"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline (2 years)", f"{s['baseline_volume']:,.0f}")
            c2.metric("Scenario (2 years)", f"{s['scenario_volume']:,.0f}")
            c3.metric("Difference", f"{s['delta_volume']:+,.0f}",
                      f"{s['delta_pct']:+.2f}%")

            paths = result["paths"]
            fig = go.Figure()
            fig.add_scatter(x=hist["date"].tail(52), y=hist["volume"].tail(52),
                            name="Actual", line=dict(color=cfg.INK_MUTED, width=1.2))
            fig.add_scatter(x=paths["date"], y=paths["baseline"], name="Baseline",
                            line=dict(color=cfg.SERIES[0], width=2))
            fig.add_scatter(x=paths["date"], y=paths["scenario"], name="Scenario",
                            line=dict(color=cfg.SERIES[1], width=2))
            st.plotly_chart(layout(fig, 400, yaxis=dict(**AXIS, title="Weekly units")),
                            width='stretch')

            left, right = st.columns(2)
            with left:
                st.markdown("#### What changed in the plan")
                ch = result["changes"]
                if ch.empty:
                    st.info("No driver values differ from the baseline.")
                else:
                    st.dataframe(
                        ch.assign(
                            baseline_avg=ch["baseline_avg"].round(2),
                            scenario_avg=ch["scenario_avg"].round(2),
                            pct_change=ch["pct_change"].round(2),
                        )[["label", "baseline_avg", "scenario_avg", "pct_change",
                           "weeks_changed"]].rename(columns={
                            "label": "Driver", "baseline_avg": "Baseline avg",
                            "scenario_avg": "Scenario avg", "pct_change": "% change",
                            "weeks_changed": "Weeks changed"}),
                        hide_index=True, width='stretch')

            with right:
                st.markdown("#### Where the volume difference comes from")
                wf = result["waterfall"]
                if wf.empty:
                    st.info("No volume difference to attribute.")
                else:
                    fig = go.Figure()
                    fig.add_bar(
                        x=wf["delta_volume"], y=wf["label"], orientation="h",
                        marker=dict(color=[cfg.POS if v > 0 else cfg.NEG
                                           for v in wf["delta_volume"]]),
                        hovertemplate="%{y}<br>%{x:+,.0f} units<extra></extra>")
                    fig.add_vline(x=0, line=dict(color=cfg.INK_MUTED, width=1))
                    st.plotly_chart(
                        layout(fig, 60 + 34 * len(wf), hovermode="closest",
                               xaxis=dict(**AXIS, title="Units vs baseline"),
                               yaxis=dict(**AXIS, autorange="reversed")),
                        width='stretch')

            st.download_button(
                "Download the scenario forecast (CSV)",
                paths.to_csv(index=False).encode(),
                file_name=f"{category.lower().replace(' ', '_')}_scenario.csv",
                mime="text/csv",
            )

# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------
with tab_decomp:
    st.markdown("#### What builds this category's volume, week by week")
    st.caption(
        "The model is additive in logs, so every block below is measured against "
        "its own historic average and converted back to units multiplicatively — "
        "the parts sum exactly to the prediction. Aggregated to quarters to stay "
        "readable across 345 weeks."
    )

    dec = scen.decompose_period(model, panel, driver_fc)
    dec["quarter"] = pd.PeriodIndex(dec["date"], freq="Q").to_timestamp()
    blocks = [c for c in dec.columns
              if c not in ("date", "predicted", "period", "quarter", "base")]
    q = dec.groupby("quarter")[["base"] + blocks].sum().reset_index()

    fig = go.Figure()
    fig.add_bar(x=q["quarter"], y=q["base"], name="Base",
                marker=dict(color="#d9d8d0"))
    for i, b in enumerate(blocks):
        label = cfg.DRIVERS[b]["label"] if b in cfg.DRIVERS else b.title()
        fig.add_bar(x=q["quarter"], y=q[b], name=label,
                    marker=dict(color=cfg.SERIES[i % len(cfg.SERIES)]))
    fig.add_vline(x=pd.Timestamp(cfg.HISTORY_END),
                  line=dict(color=cfg.INK_PRIMARY, dash="dot", width=1.5))
    fig.add_annotation(x=pd.Timestamp(cfg.HISTORY_END), y=1.02, yref="paper",
                       text="forecast starts", showarrow=False,
                       font=dict(size=10, color=cfg.INK_MUTED))
    st.plotly_chart(
        layout(fig, 480, barmode="relative",
               yaxis=dict(**AXIS, title="Units per quarter")),
        width='stretch',
    )

    st.markdown("#### Contribution over the forecast horizon")
    fut_rows = dec[dec["period"] == "forecast"]
    total = fut_rows["predicted"].sum()
    contrib = pd.DataFrame(
        {
            "Block": ["Base"] + [cfg.DRIVERS[b]["label"] if b in cfg.DRIVERS
                                 else b.title() for b in blocks],
            "Units": [fut_rows["base"].sum()] + [fut_rows[b].sum() for b in blocks],
        }
    )
    contrib["% of forecast"] = (100 * contrib["Units"] / total).round(2)
    contrib["Units"] = contrib["Units"].round(0)
    st.dataframe(contrib.sort_values("Units", ascending=False),
                 hide_index=True, width='stretch')
    st.caption(
        "'Base' is volume at this category's average driver levels with the trend, "
        "seasonality and holiday effects at their historic means. Everything else "
        "is the deviation from that."
    )
