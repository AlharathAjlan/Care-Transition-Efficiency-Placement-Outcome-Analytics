import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="UAC Care Pipeline Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Data loading + cleaning (cached)
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # drop fully-blank filler rows
    df = df.dropna(how="all").reset_index(drop=True)

    # fix comma-formatted HHS Care column
    hhs_col = "Children in HHS Care"
    df[hhs_col] = df[hhs_col].astype(str).str.replace(",", "", regex=False)
    df[hhs_col] = pd.to_numeric(df[hhs_col], errors="coerce")

    # parse and sort dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # standard column names used throughout the app
    apprehended_col = [c for c in df.columns if c.startswith("Children apprehended")][0]
    df = df.rename(columns={
        apprehended_col: "CBP_Inflow",
        "Children in CBP custody": "CBP_Load",
        "Children transferred out of CBP custody": "CBP_Outflow",
        "Children in HHS Care": "HHS_Load",
        "Children discharged from HHS Care": "HHS_Outflow",
    })
    df["HHS_Inflow"] = df["CBP_Outflow"]

    return df


@st.cache_data
def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()

    # net movement
    m["CBP_Net_Movement"] = m["CBP_Inflow"] - m["CBP_Outflow"]
    m["HHS_Net_Movement"] = m["HHS_Inflow"] - m["HHS_Outflow"]

    # core ratios
    m["Transfer_Efficiency_Ratio"] = (m["CBP_Outflow"] / m["CBP_Load"]).replace([np.inf, -np.inf], np.nan)
    m["Discharge_Effectiveness"] = (m["HHS_Outflow"] / m["HHS_Load"]).replace([np.inf, -np.inf], np.nan)

    # pipeline throughput (30-report rolling)
    window = 30
    m["Total_Entries_Rolling"] = m["CBP_Inflow"].rolling(window, min_periods=10).sum()
    m["Total_Exits_Rolling"] = m["HHS_Outflow"].rolling(window, min_periods=10).sum()
    m["Pipeline_Throughput_Rate"] = (m["Total_Exits_Rolling"] / m["Total_Entries_Rolling"]).replace([np.inf, -np.inf], np.nan)

    # rolling efficiency trends
    m["Transfer_Efficiency_Rolling"] = m["Transfer_Efficiency_Ratio"].rolling(30, min_periods=10).mean()
    m["Discharge_Effectiveness_Rolling"] = m["Discharge_Effectiveness"].rolling(30, min_periods=10).mean()

    # cumulative entries/exits and backlog gap
    m["Cumulative_Entries"] = m["CBP_Inflow"].cumsum()
    m["Cumulative_Exits"] = m["HHS_Outflow"].cumsum()
    m["Cumulative_Gap"] = m["Cumulative_Entries"] - m["Cumulative_Exits"]
    m["Backlog_Accumulation_Rate"] = m["Cumulative_Gap"].diff(14) / 14

    # discharge variability / outcome stability
    m["Discharge_Rolling_Mean"] = m["HHS_Outflow"].rolling(30, min_periods=10).mean()
    m["Discharge_Rolling_Std"] = m["HHS_Outflow"].rolling(30, min_periods=10).std()
    m["Discharge_CV"] = m["Discharge_Rolling_Std"] / m["Discharge_Rolling_Mean"]
    m["Outcome_Stability_Score"] = 1 / (1 + m["Discharge_CV"])

    # week-over-week % change for sudden-drop detection
    m["Discharge_PctChange"] = m["HHS_Outflow"].pct_change(periods=7)

    # total movement (for stagnation detection)
    m["Total_Movement"] = m["CBP_Inflow"] + m["CBP_Outflow"] + m["HHS_Outflow"]

    return m


DATA_PATH = "HHS_Unaccompanied_Alien_Children_Program.csv"  # <-- point this at your actual CSV filename
df_raw = load_data(DATA_PATH)
df_full = build_metrics(df_raw)

DETECTED_CHANGEPOINT = pd.Timestamp("2025-01-30")

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("UAC Care Pipeline Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "Care Pipeline Flow Visualization",
        "Transfer & Discharge Efficiency",
        "Bottleneck Detection",
        "Outcome Trend Analysis",
        "ML: Regime & Forecasting",
    ],
)

# ---------------------------------------------------------------------
# Global filters
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

min_date, max_date = df_full["Date"].min(), df_full["Date"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df = df_full[(df_full["Date"] >= start_date) & (df_full["Date"] <= end_date)].copy()
else:
    df = df_full.copy()

if len(df) == 0:
    st.warning("No reports in the selected date range. Adjust the range in the sidebar.")
    st.stop()

st.sidebar.markdown(f"**{len(df)}** reports in range (of {len(df_full)} total)")

# ---------------------------------------------------------------------
# MODULE 1: Care Pipeline Flow Visualization
# ---------------------------------------------------------------------
if page == "Care Pipeline Flow Visualization":
    st.title("Care Pipeline Flow Visualization")
    st.caption("CBP Custody \u2192 HHS Care \u2192 Sponsor Placement (Discharge)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg CBP Custody Load", f"{df['CBP_Load'].mean():,.0f}")
    col2.metric("Avg HHS Care Load", f"{df['HHS_Load'].mean():,.0f}")
    col3.metric("Total Discharges (range)", f"{df['HHS_Outflow'].sum():,.0f}")

    st.subheader("Custody & Care Load Over Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["Date"], y=df["CBP_Load"], name="CBP Custody Load", line=dict(color="#2E86AB")))
    fig1.add_trace(go.Scatter(x=df["Date"], y=df["HHS_Load"], name="HHS Care Load", yaxis="y2", line=dict(color="#F4A261")))
    if start_date <= DETECTED_CHANGEPOINT <= end_date:
        fig1.add_vline(x=DETECTED_CHANGEPOINT, line_dash="dash", line_color="red")
    fig1.update_layout(
        yaxis=dict(title="CBP Custody Load"),
        yaxis2=dict(title="HHS Care Load", overlaying="y", side="right"),
    )
    st.plotly_chart(fig1, width="stretch")
    st.caption("Red dashed line marks the confirmed structural regime change (Jan 30, 2025), if within range.")

    st.subheader("Daily Net Movement by Stage")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["Date"], y=df["CBP_Net_Movement"], name="CBP Net Movement"))
    fig2.add_trace(go.Bar(x=df["Date"], y=df["HHS_Net_Movement"], name="HHS Net Movement"))
    fig2.add_hline(y=0, line_color="gray")
    fig2.update_layout(barmode="overlay", yaxis_title="Net Movement (Inflow \u2212 Outflow)")
    st.plotly_chart(fig2, width="stretch")
    st.caption("Positive = stage population grew that report; negative = stage population shrank.")

    st.subheader("Pipeline Stage Snapshot (most recent report in range)")
    latest = df.iloc[-1]
    sankey_fig = go.Figure(go.Sankey(
        node=dict(
            label=["Apprehensions", "CBP Custody", "HHS Care", "Discharged"],
            color=["#2E86AB", "#2E86AB", "#F4A261", "#2A9D8F"],
        ),
        link=dict(
            source=[0, 1, 2],
            target=[1, 2, 3],
            value=[
                max(latest["CBP_Inflow"], 0.1),
                max(latest["CBP_Outflow"], 0.1),
                max(latest["HHS_Outflow"], 0.1),
            ],
        ),
    ))
    sankey_fig.update_layout(title_text=f"Flow snapshot: {latest['Date'].date()}")
    st.plotly_chart(sankey_fig, width="stretch")

    st.subheader("Underlying Data")
    st.dataframe(
        df[["Date", "CBP_Inflow", "CBP_Load", "CBP_Outflow", "HHS_Load", "HHS_Outflow"]],
        width="stretch",
    )

# ---------------------------------------------------------------------
# MODULE 2: Transfer & Discharge Efficiency Panels
# ---------------------------------------------------------------------
elif page == "Transfer & Discharge Efficiency":
    st.title("Transfer & Discharge Efficiency Panels")

    metric_choice = st.radio(
        "Select ratio to display",
        ["Transfer Efficiency Ratio", "Discharge Effectiveness", "Pipeline Throughput Rate"],
        horizontal=True,
    )

    metric_map = {
        "Transfer Efficiency Ratio": ("Transfer_Efficiency_Ratio", "Transfer_Efficiency_Rolling", "CBP\u2192HHS transfer speed (Transfers \u00f7 CBP Custody)"),
        "Discharge Effectiveness": ("Discharge_Effectiveness", "Discharge_Effectiveness_Rolling", "Placement success rate (Discharges \u00f7 HHS Care)"),
        "Pipeline Throughput Rate": ("Pipeline_Throughput_Rate", "Pipeline_Throughput_Rate", "Overall system movement (Total Exits \u00f7 Total Entries, rolling)"),
    }
    raw_col, rolling_col, description = metric_map[metric_choice]
    st.caption(description)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", f"{df[raw_col].mean():.3f}")
    col2.metric("Median", f"{df[raw_col].median():.3f}")
    col3.metric("Std Dev", f"{df[raw_col].std():.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[raw_col], name="Raw (daily)", opacity=0.35, line=dict(color="lightblue")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df[rolling_col], name="30-report rolling", line=dict(color="#2E86AB", width=2)))
    if metric_choice == "Pipeline Throughput Rate":
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Balanced (1.0)")
    if start_date <= DETECTED_CHANGEPOINT <= end_date:
        fig.add_vline(x=DETECTED_CHANGEPOINT, line_dash="dash", line_color="red")
    fig.update_layout(title=metric_choice, yaxis_title=metric_choice)
    st.plotly_chart(fig, width="stretch")

    st.subheader("CBP \u2192 HHS Transition Lag")
    if st.button("Calculate lag (cross-correlation)"):
        cbp_change = df["CBP_Load"].diff().dropna()
        hhs_change = df["HHS_Load"].diff().dropna()
        min_len = min(len(cbp_change), len(hhs_change))
        if min_len > 40:
            cbp_change_v = cbp_change.iloc[-min_len:].values
            hhs_change_v = hhs_change.iloc[-min_len:].values
            correlation = signal.correlate(hhs_change_v - hhs_change_v.mean(), cbp_change_v - cbp_change_v.mean(), mode="full")
            lags = signal.correlation_lags(len(hhs_change_v), len(cbp_change_v), mode="full")
            mask = (lags >= 0) & (lags <= 30)
            best_lag = lags[mask][np.argmax(correlation[mask])]
            st.metric("Detected lag", f"{best_lag} reports (HHS responds after CBP)")
        else:
            st.info("Not enough reports in this range to calculate a reliable lag.")

# ---------------------------------------------------------------------
# MODULE 3: Bottleneck Detection Charts
# ---------------------------------------------------------------------
elif page == "Bottleneck Detection":
    st.title("Bottleneck Detection")

    st.subheader("Backlog Accumulation Rate")
    st.caption("Rate of change of (Cumulative Entries \u2212 Cumulative Exits) over a 14-report window. Positive = backlog growing; negative = backlog clearing.")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["Date"], y=df["Backlog_Accumulation_Rate"], line=dict(color="crimson")))
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig1, width="stretch")

    st.subheader("Sustained Backlog Periods")
    st.caption("Consecutive reports where Pipeline Throughput stays below a threshold (accumulating).")
    throughput_threshold = st.slider("Throughput threshold (below = accumulating)", 0.5, 1.5, 1.0, 0.05)
    min_run = st.slider("Minimum consecutive reports to count as 'sustained'", 3, 15, 5)

    df_bt = df.copy()
    df_bt["Is_Accumulating"] = df_bt["Pipeline_Throughput_Rate"] < throughput_threshold
    df_bt["Accum_Group"] = (df_bt["Is_Accumulating"] != df_bt["Is_Accumulating"].shift()).cumsum()
    runs = df_bt[df_bt["Is_Accumulating"]].groupby("Accum_Group").agg(
        Start_Date=("Date", "min"), End_Date=("Date", "max"), Reports=("Date", "count")
    ).reset_index(drop=True)
    runs = runs[runs["Reports"] >= min_run].sort_values("Reports", ascending=False)
    if len(runs) > 0:
        st.dataframe(runs, width="stretch")
    else:
        st.info("No sustained backlog periods found at this threshold/duration in the selected range.")

    st.subheader("Day-to-Day Statistical Anomalies")
    z_threshold = st.slider("Anomaly sensitivity (|z-score| threshold)", 1.5, 4.0, 2.5, 0.5)
    for col, label in [("Transfer_Efficiency_Ratio", "Transfer Efficiency"), ("Discharge_Effectiveness", "Discharge Effectiveness")]:
        vals = df[col]
        z = (vals - vals.mean()) / vals.std()
        anomalies = df[z.abs() > z_threshold]
        st.markdown(f"**{label}: {len(anomalies)} anomaly day(s)**")
        if len(anomalies) > 0:
            st.dataframe(anomalies[["Date", col, "CBP_Load", "CBP_Outflow", "HHS_Load", "HHS_Outflow"]], width="stretch")

    st.subheader("Prolonged Stagnation Periods")
    st.caption("Consecutive reports in the bottom 10% of total pipeline movement (inflow + outflow + discharges).")
    movement_threshold = df["Total_Movement"].quantile(0.10)
    df_bt["Is_Stagnant"] = df_bt["Total_Movement"] <= movement_threshold
    df_bt["Stagnant_Group"] = (df_bt["Is_Stagnant"] != df_bt["Is_Stagnant"].shift()).cumsum()
    stagnation = df_bt[df_bt["Is_Stagnant"]].groupby("Stagnant_Group").agg(
        Start_Date=("Date", "min"), End_Date=("Date", "max"), Reports=("Date", "count"),
        Avg_Movement=("Total_Movement", "mean"),
    ).reset_index(drop=True)
    stagnation = stagnation[stagnation["Reports"] >= 5].sort_values("Reports", ascending=False)
    st.write(f"Stagnation threshold (bottom 10% of movement): {movement_threshold:.0f}")
    if len(stagnation) > 0:
        st.dataframe(stagnation, width="stretch")
    else:
        st.info("No prolonged stagnation periods found in the selected range.")

# ---------------------------------------------------------------------
# MODULE 4: Outcome Trend Analysis
# ---------------------------------------------------------------------
elif page == "Outcome Trend Analysis":
    st.title("Outcome Trend Analysis")

    col1, col2 = st.columns(2)
    col1.metric("Avg Outcome Stability Score", f"{df['Outcome_Stability_Score'].mean():.3f}")
    col2.metric("Avg Discharge Volume CV", f"{df['Discharge_CV'].mean():.3f}")

    st.subheader("Outcome Stability Score Over Time")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["Date"], y=df["Outcome_Stability_Score"], line=dict(color="#2A9D8F")))
    stability_alert_threshold = st.slider("Alert threshold (flag below this score)", 0.3, 0.9, 0.7, 0.05)
    fig1.add_hline(y=stability_alert_threshold, line_dash="dash", line_color="red", annotation_text="Alert threshold")
    if start_date <= DETECTED_CHANGEPOINT <= end_date:
        fig1.add_vline(x=DETECTED_CHANGEPOINT, line_dash="dash", line_color="gray")
    st.plotly_chart(fig1, width="stretch")

    below_threshold = df[df["Outcome_Stability_Score"] < stability_alert_threshold]
    st.metric("Reports below alert threshold", f"{len(below_threshold)} of {len(df)}")

    st.subheader("Sudden Drops in Discharge Volume")
    drop_pct = st.slider("Sudden-drop threshold (week-over-week % decline)", -90, -10, -50, 5)
    sudden_drops = df[df["Discharge_PctChange"] * 100 <= drop_pct].copy()
    st.metric("Sudden-drop events", len(sudden_drops))
    if len(sudden_drops) > 0:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["Date"], y=df["HHS_Outflow"], name="Discharges", line=dict(color="steelblue")))
        fig2.add_trace(go.Scatter(
            x=sudden_drops["Date"], y=sudden_drops["HHS_Outflow"], mode="markers",
            name="Sudden Drop", marker=dict(color="red", size=9),
        ))
        st.plotly_chart(fig2, width="stretch")
        st.dataframe(sudden_drops[["Date", "HHS_Outflow", "Discharge_PctChange"]], width="stretch")

    st.subheader("Month-over-Month Trend")
    monthly = df.copy()
    monthly["YearMonth"] = monthly["Date"].dt.to_period("M").astype(str)
    monthly_agg = monthly.groupby("YearMonth").agg(
        Total_Discharges=("HHS_Outflow", "sum"), Total_Apprehensions=("CBP_Inflow", "sum"),
    ).reset_index()
    fig3 = px.line(monthly_agg, x="YearMonth", y=["Total_Discharges", "Total_Apprehensions"], markers=True)
    fig3.update_xaxes(tickangle=-45)
    st.plotly_chart(fig3, width="stretch")

# ---------------------------------------------------------------------
# MODULE 5: ML — Regime Detection & Forecasting Diagnostics
# ---------------------------------------------------------------------
elif page == "ML: Regime & Forecasting":
    st.title("ML: Regime Detection & Forecasting Diagnostics")
    st.caption(
        "Summarizes the project's machine learning phase: automated regime-change detection "
        "and why simple forecasting outperforms complex models in the current stable period."
    )

    st.subheader("Automated Changepoint Detection")
    series = df.set_index("Date")["HHS_Load"].dropna()
    if len(series) > 60:
        best_split, best_diff = None, 0
        for i in range(30, len(series) - 30):
            diff = abs(series.iloc[:i].mean() - series.iloc[i:].mean())
            if diff > best_diff:
                best_diff, best_split = diff, series.index[i]

        col1, col2, col3 = st.columns(3)
        col1.metric("Detected changepoint", str(best_split.date()) if best_split is not None else "N/A")
        col2.metric("Mean before", f"{series[:best_split].mean():,.0f}" if best_split is not None else "N/A")
        col3.metric("Mean after", f"{series[best_split:].mean():,.0f}" if best_split is not None else "N/A")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name="HHS Care Load"))
        if best_split is not None:
            fig.add_vline(x=best_split, line_dash="dash", line_color="red")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough reports in this range to run changepoint detection (need 60+).")

    st.markdown("---")
    st.subheader("Forecasting: Naive Persistence vs. Random Forest")
    st.caption(
        "Documented project finding: in the current stable regime, a Random Forest trained on lag/rolling "
        "features fails to beat a simple 'tomorrow = today' baseline, because tree models cannot extrapolate "
        "beyond the value ranges seen in training. Recompute below on the current filtered range."
    )

    if st.button("Run forecasting comparison on this date range"):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error

        fdf = df[["Date", "HHS_Load"]].copy()
        fdf["lag_1"] = fdf["HHS_Load"].shift(1)
        fdf["lag_2"] = fdf["HHS_Load"].shift(2)
        fdf["lag_7"] = fdf["HHS_Load"].shift(7)
        fdf["rolling_mean_7"] = fdf["HHS_Load"].shift(1).rolling(7).mean()
        fdf = fdf.dropna().reset_index(drop=True)

        if len(fdf) > 40:
            split = int(len(fdf) * 0.8)
            train, test = fdf.iloc[:split], fdf.iloc[split:]
            naive_mae = mean_absolute_error(test["HHS_Load"], test["lag_1"])

            feats = ["lag_1", "lag_2", "lag_7", "rolling_mean_7"]
            model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
            model.fit(train[feats], train["HHS_Load"])
            preds = model.predict(test[feats])
            model_mae = mean_absolute_error(test["HHS_Load"], preds)

            col1, col2 = st.columns(2)
            col1.metric("Naive Baseline MAE", f"{naive_mae:.2f}")
            col2.metric("Random Forest MAE", f"{model_mae:.2f}", delta=f"{model_mae - naive_mae:+.2f} vs naive", delta_color="inverse")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test["Date"], y=test["HHS_Load"], name="Actual"))
            fig.add_trace(go.Scatter(x=test["Date"], y=test["lag_1"], name="Naive Prediction"))
            fig.add_trace(go.Scatter(x=test["Date"], y=preds, name="Random Forest Prediction"))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Not enough reports in this range to run the forecasting comparison (need 40+).")
