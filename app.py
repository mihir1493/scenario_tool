"""Category volume scenario planner.

    streamlit run app.py

Reads only the artifacts built by `scripts/build.py` -- it never fits anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import scenario as scen
from src.model import ResponseModel

st.set_page_config(page_title="Category Scenario Planner", layout="wide")

# Colour follows the entity, never its rank: a category keeps its hue no matter
# how many are selected.
CAT_COLOR = {c: config.SERIES[i] for i, c in enumerate(config.CATEGORIES)}
AXIS = dict(showgrid=True, gridcolor=config.GRIDLINE, zeroline=False,
            linecolor="#c3c2b7", tickfont=dict(color=config.INK_MUTED, size=11))


def layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    opts = dict(
        height=height,
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor=config.SURFACE,
        plot_bgcolor=config.SURFACE,
        font=dict(family="system-ui, -apple-system, sans-serif",
                  color=config.INK_SECONDARY, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=AXIS, yaxis=AXIS,
    )
    opts.update(kw)  # callers may override any default, e.g. hovermode
    fig.update_layout(**opts)
    return fig


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_csv(config.PANEL_CSV, parse_dates=["date"])


@st.cache_data
def load_forecast() -> pd.DataFrame:
    return pd.read_csv(config.DRIVER_FORECAST_CSV, parse_dates=["date"])


@st.cache_resource
def load_model() -> ResponseModel:
    return ResponseModel.load()


if not config.MODEL_PKL.exists():
    st.error("No model artifact found. Run `python scripts/build.py` first.")
    st.stop()

panel, driver_fc, model = load_panel(), load_forecast(), load_model()
metrics = json.loads(config.METRICS_JSON.read_text())

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Scenario Planner")
cats = st.sidebar.multiselect("Categories", config.CATEGORIES, default=config.CATEGORIES)
if not cats:
    st.sidebar.warning("Select at least one category.")
    st.stop()

horizon = st.sidebar.slider("Forecast horizon (months)", 3, config.FORECAST_MONTHS,
                            config.FORECAST_MONTHS, step=3)
fc_dates = sorted(driver_fc["date"].unique())[:horizon]
driver_fc_h = driver_fc[driver_fc["date"].isin(fc_dates)]

st.sidebar.divider()
st.sidebar.caption(
    f"**Model**  pooled log-log ridge (a={metrics['alpha']})  \n"
    f"R2 {metrics['r2_log']:.3f} in-sample (log)  \n"
    f"{metrics['holdout']['mape']:.1f}% MAPE on a {metrics['holdout']['months']}-month holdout  \n"
    f"{metrics['n_obs']} observations"
)

if "saved" not in st.session_state:
    st.session_state.saved = {}

tab_over, tab_drv, tab_imp, tab_scen = st.tabs(
    ["Overview", "Drivers", "Driver impact", "Scenario"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_over:
    st.subheader("Volume history and baseline forecast")

    hist = panel[panel["category"].isin(cats)]
    base = scen.run(model, panel, driver_fc_h, {}, cats)["paths"]

    h_tot = hist.groupby("date", as_index=False)["volume"].sum()
    b_tot = base.groupby("date", as_index=False)["baseline"].sum()

    last12 = h_tot.tail(12)["volume"].sum()
    prev12 = h_tot.tail(24).head(12)["volume"].sum()
    next12 = b_tot.head(min(12, len(b_tot)))["baseline"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Last 12 months", f"{last12/1e6:,.1f}M units",
              f"{100*(last12/prev12-1):+.1f}% vs prior year")
    c2.metric(f"Next {min(12,len(b_tot))} months (baseline)", f"{next12/1e6:,.1f}M units",
              f"{100*(next12/last12-1):+.1f}% vs last 12m")
    c3.metric("Holdout accuracy", f"{100-metrics['holdout']['mape']:.1f}%",
              f"{metrics['holdout']['mape']:.1f}% MAPE", delta_color="off")

    fig = go.Figure()
    for c in cats:
        h = hist[hist["category"] == c]
        b = base[base["category"] == c]
        fig.add_trace(go.Scatter(x=h["date"], y=h["volume"], name=c, legendgroup=c,
                                 mode="lines", line=dict(color=CAT_COLOR[c], width=2)))
        # Join the seam so the forecast reads as a continuation, not a new series.
        fig.add_trace(go.Scatter(
            x=pd.concat([h["date"].tail(1), b["date"]]),
            y=pd.concat([h["volume"].tail(1), b["baseline"]]),
            name=c, legendgroup=c, showlegend=False, mode="lines",
            line=dict(color=CAT_COLOR[c], width=2, dash="dot")))
    fig.add_vline(x=pd.Timestamp(config.HISTORY_END), line_width=1,
                  line_dash="dot", line_color=config.INK_MUTED)
    fig.update_yaxes(title_text="Units / month")
    st.plotly_chart(layout(fig, 420, title="Solid = actual, dotted = forecast"),
                    width="stretch")

    st.subheader("Model fit")
    fit = scen.fitted_history(model, panel)
    fit = fit[fit["category"].isin(cats)].groupby("date", as_index=False)[
        ["volume", "fitted"]].sum()
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=fit["date"], y=fit["volume"], name="Actual",
                            mode="lines", line=dict(color=config.SERIES[0], width=2)))
    f2.add_trace(go.Scatter(x=fit["date"], y=fit["fitted"], name="Fitted",
                            mode="lines", line=dict(color=config.SERIES[1], width=2)))
    f2.update_yaxes(title_text="Units / month")
    st.plotly_chart(layout(f2, 320), width="stretch")

    st.caption(
        "Holdout MAPE by category: "
        + " | ".join(f"{k} {v:.1f}%" for k, v in metrics["holdout"]["by_category"].items())
    )

# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
with tab_drv:
    st.subheader("Driver history and Prophet forecast")
    pick = st.selectbox("Driver", config.DRIVER_NAMES,
                        format_func=lambda d: config.DRIVERS[d]["label"])
    spec = config.DRIVERS[pick]

    fig = go.Figure()
    if spec["scope"] == "macro":
        h = panel.groupby("date", as_index=False)[pick].first()
        f = driver_fc_h.groupby("date", as_index=False)[pick].first()
        fig.add_trace(go.Scatter(x=h["date"], y=h[pick], name="Actual", mode="lines",
                                 line=dict(color=config.SERIES[0], width=2)))
        fig.add_trace(go.Scatter(x=pd.concat([h["date"].tail(1), f["date"]]),
                                 y=pd.concat([h[pick].tail(1), f[pick]]),
                                 name="Forecast", mode="lines",
                                 line=dict(color=config.SERIES[0], width=2, dash="dot")))
    else:
        for c in cats:
            h = panel[panel["category"] == c]
            f = driver_fc_h[driver_fc_h["category"] == c]
            fig.add_trace(go.Scatter(x=h["date"], y=h[pick], name=c, legendgroup=c,
                                     mode="lines", line=dict(color=CAT_COLOR[c], width=2)))
            fig.add_trace(go.Scatter(
                x=pd.concat([h["date"].tail(1), f["date"]]),
                y=pd.concat([h[pick].tail(1), f[pick]]),
                name=c, legendgroup=c, showlegend=False, mode="lines",
                line=dict(color=CAT_COLOR[c], width=2, dash="dot")))
    fig.add_vline(x=pd.Timestamp(config.HISTORY_END), line_width=1,
                  line_dash="dot", line_color=config.INK_MUTED)
    fig.update_yaxes(title_text=spec["label"])
    st.plotly_chart(layout(fig, 400), width="stretch")

    if spec["scope"] == "macro":
        st.caption("Macro drivers are shared across every category — one forecast, broadcast.")
    st.dataframe(
        driver_fc_h[driver_fc_h["category"].isin(cats)]
        .pivot_table(index="date", columns="category", values=pick)
        .round(2),
        width="stretch",
    )

# --------------------------------------------------------------------------
# Driver impact
# --------------------------------------------------------------------------
with tab_imp:
    st.subheader("How much each driver moves volume")
    imp = model.importance()

    fig = go.Figure(go.Bar(
        x=imp["impact_pct"][::-1], y=imp["label"][::-1], orientation="h",
        marker=dict(color=[config.POS if e > 0 else config.NEG
                           for e in imp["elasticity"][::-1]]),
        text=[f"{v:.1f}%" for v in imp["impact_pct"][::-1]],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:.1f}% volume swing per 1sd<extra></extra>",
    ))
    fig.update_xaxes(title_text="% volume swing per 1 standard deviation of the driver")
    st.plotly_chart(layout(fig, 400, showlegend=False,
                           title="Blue increases volume, red decreases it"),
                    width="stretch")

    st.markdown("**Elasticities** — % change in volume per 1% change in the driver.")
    tbl = imp[["label", "group", "elasticity", "impact_pct", "impact_share"]].copy()
    tbl.columns = ["Driver", "Group", "Elasticity", "Impact (% per 1sd)", "Share of impact"]
    st.dataframe(
        tbl.style.format({"Elasticity": "{:+.3f}", "Impact (% per 1sd)": "{:.1f}%",
                          "Share of impact": "{:.1%}"}),
        width="stretch", hide_index=True,
    )

    st.divider()
    st.subheader("Response curve")
    rc_driver = st.selectbox("Sweep driver", config.DRIVER_NAMES, key="rc",
                             format_func=lambda d: config.DRIVERS[d]["label"])
    lo, hi = config.DRIVERS[rc_driver]["slider"]
    grid = np.linspace(lo, hi, 13)
    pts = [scen.run(model, panel, driver_fc_h, {rc_driver: float(p)}, cats)
           ["summary"]["delta_pct"] for p in grid]

    f3 = go.Figure(go.Scatter(x=grid, y=pts, mode="lines+markers",
                              line=dict(color=config.SERIES[0], width=2),
                              marker=dict(size=8, color=config.SERIES[0]),
                              hovertemplate="%{x:+.0f}% driver -> %{y:+.2f}% volume<extra></extra>"))
    f3.add_hline(y=0, line_width=1, line_color="#c3c2b7")
    f3.add_vline(x=0, line_width=1, line_color="#c3c2b7")
    f3.update_xaxes(title_text=f"% change in {config.DRIVERS[rc_driver]['label']}")
    f3.update_yaxes(title_text="% change in volume")
    st.plotly_chart(layout(f3, 340, showlegend=False, hovermode="closest"),
                    width="stretch")

    st.divider()
    st.subheader("Volume decomposition")
    st.caption(
        "Predicted volume split into a base (all drivers at their historical average) "
        "plus the incremental volume each driver adds or removes."
    )
    dec = model.decompose(panel)
    dec = dec[dec["category"].isin(cats)].groupby("date", as_index=False).sum(
        numeric_only=True)
    order = imp["driver"].tolist()
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=dec["date"], y=dec["base"], name="Base", stackgroup="one",
                            mode="none", fillcolor="#c3c2b7"))
    for i, d in enumerate(order):
        f4.add_trace(go.Scatter(x=dec["date"], y=dec[d].clip(lower=0),
                                name=config.DRIVERS[d]["label"], stackgroup="one",
                                mode="none", fillcolor=config.SERIES[i % len(config.SERIES)]))
    f4.update_yaxes(title_text="Units / month")
    st.plotly_chart(layout(f4, 420, title="Positive contributions only"),
                    width="stretch")

# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
with tab_scen:
    st.subheader("Build a scenario")
    st.caption(
        f"Each slider shifts that driver by a % versus its forecast, across all "
        f"{horizon} forecast months and every selected category."
    )

    adj: dict[str, float] = {}
    for group in ["Commercial", "Macro"]:
        names = [n for n in config.DRIVER_NAMES if config.DRIVERS[n]["group"] == group]
        with st.expander(f"{group} drivers", expanded=(group == "Commercial")):
            cols = st.columns(len(names) if len(names) <= 3 else 3)
            for i, n in enumerate(names):
                lo, hi = config.DRIVERS[n]["slider"]
                adj[n] = cols[i % len(cols)].slider(
                    config.DRIVERS[n]["label"], float(lo), float(hi), 0.0, 1.0,
                    format="%+.0f%%", key=f"adj_{n}")

    active = {k: v for k, v in adj.items() if v}
    res = scen.run(model, panel, driver_fc_h, active, cats)
    s = res["summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline volume", f"{s['baseline_volume']/1e6:,.2f}M")
    c2.metric("Scenario volume", f"{s['scenario_volume']/1e6:,.2f}M",
              f"{s['delta_pct']:+.2f}%")
    c3.metric("Delta", f"{s['delta_volume']/1e6:+,.2f}M units")
    c4.metric("Horizon", f"{horizon} months", f"{len(cats)} categories",
              delta_color="off")

    paths = res["paths"].groupby("date", as_index=False)[["baseline", "scenario"]].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=paths["date"], y=paths["baseline"], name="Baseline",
                             mode="lines", line=dict(color=config.INK_MUTED, width=2,
                                                     dash="dot")))
    fig.add_trace(go.Scatter(x=paths["date"], y=paths["scenario"], name="Scenario",
                             mode="lines", line=dict(color=config.SERIES[0], width=2)))
    fig.update_yaxes(title_text="Units / month")
    st.plotly_chart(layout(fig, 380), width="stretch")

    if active:
        st.subheader("Where the change comes from")
        wf = res["waterfall"]
        f2 = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(wf) + ["total"],
            x=wf["label"].tolist() + ["Net change"],
            y=(wf["delta_volume"] / 1e6).tolist() + [s["delta_volume"] / 1e6],
            increasing=dict(marker=dict(color=config.POS)),
            decreasing=dict(marker=dict(color=config.NEG)),
            totals=dict(marker=dict(color=config.INK_SECONDARY)),
            connector=dict(line=dict(color=config.GRIDLINE, width=1)),
            text=[f"{v:+,.2f}M" for v in wf["delta_volume"] / 1e6]
                 + [f"{s['delta_volume']/1e6:+,.2f}M"],
            textposition="outside",
        ))
        f2.update_yaxes(title_text="Million units over the horizon")
        st.plotly_chart(layout(f2, 400, showlegend=False, hovermode="closest"),
                        width="stretch")

        st.dataframe(
            wf.rename(columns={"label": "Driver", "delta_volume": "Delta units",
                               "delta_pct": "% of baseline"})
              .drop(columns=["driver"])
              .style.format({"Delta units": "{:+,.0f}", "% of baseline": "{:+.2f}%"}),
            width="stretch", hide_index=True,
        )

        st.subheader("By category")
        bycat = res["paths"].groupby("category", as_index=False)[
            ["baseline", "scenario", "delta"]].sum()
        bycat["delta_pct"] = 100 * (bycat["scenario"] / bycat["baseline"] - 1)
        st.dataframe(
            bycat.rename(columns={"category": "Category", "baseline": "Baseline",
                                  "scenario": "Scenario", "delta": "Delta",
                                  "delta_pct": "% change"})
                 .style.format({"Baseline": "{:,.0f}", "Scenario": "{:,.0f}",
                                "Delta": "{:+,.0f}", "% change": "{:+.2f}%"}),
            width="stretch", hide_index=True,
        )
    else:
        st.info("Move a slider to build a scenario.")

    st.divider()
    c1, c2 = st.columns([3, 1])
    name = c1.text_input("Scenario name", placeholder="e.g. 5% price rise + promo push")
    if c2.button("Save scenario", width="stretch") and name:
        st.session_state.saved[name] = {"adjustments": dict(active), **s}
        st.success(f"Saved '{name}'")

    if st.session_state.saved:
        st.markdown("**Saved scenarios**")
        rows = []
        for k, v in st.session_state.saved.items():
            rows.append({
                "Scenario": k,
                "Volume": v["scenario_volume"],
                "vs baseline": v["delta_pct"],
                "Levers": ", ".join(
                    f"{config.DRIVERS[d]['label'].split(' (')[0]} {p:+.0f}%"
                    for d, p in v["adjustments"].items()) or "none",
            })
        st.dataframe(
            pd.DataFrame(rows).style.format({"Volume": "{:,.0f}", "vs baseline": "{:+.2f}%"}),
            width="stretch", hide_index=True,
        )
        if st.button("Clear saved"):
            st.session_state.saved = {}
            st.rerun()
