# UAC Care Pipeline — Transition Efficiency & Placement Outcome Analytics

Interactive Streamlit dashboard analyzing the efficiency, backlog risk, and outcome stability of the UAC (Unaccompanied Alien Children) care pipeline — CBP custody → HHS care → sponsor placement. Combines a Machine Learning phase (forecasting, regime-change detection, anomaly detection) with a Data Analytics phase (pipeline KPIs, temporal patterns, outcome stability). Built as part of a Machine Learning internship project.

## Overview

The UAC Program operates as a multi-stage care and reunification pipeline, not just a capacity-tracking system. While aggregate custody counts are routinely monitored, process efficiency — how fast children move through the pipeline, whether discharges keep pace with intake, and where backlogs form — is largely invisible without structured analysis. This project builds that visibility.

**Approach:** deliberately sequenced Machine-Learning-first, then Data-Analytics-second — forecasting and structural detection were built before descriptive KPIs, and a confirmed structural regime change (see below) directly shaped how the subsequent analysis was framed.

## Key Findings

- **A confirmed structural regime change occurred January 28–30, 2025** (cross-validated by two independent changepoint detection methods), with average HHS custody load falling ~70% (7,686 → 2,302) and volatility dropping sharply.
- **Forecasting models underperform simple naive persistence** in the current stable period — tree-based models cannot extrapolate beyond the value ranges seen in training, causing a -452% result relative to baseline when tested naively across the regime boundary. Operational monitoring is currently more actionable than prediction for this metric.
- **HHS care load changes lag CBP custody load changes by ~9 reports** (roughly 1.5–2 weeks), quantifying how disruptions propagate downstream through the pipeline.
- **The pipeline operated in backlog-clearing mode for nearly the entire 3-year period** — a net 52,011 more children were discharged than newly apprehended. Only one sustained backlog-accumulation episode was found (April–May 2025, immediately after the regime change).
- **Outcome stability collapsed sharply during the transition period** (Feb–Mar 2025 alone accounts for 54% of all sudden discharge drops), before partially recovering to a calmer, if still less stable than pre-2025, pattern.
- **Discharge Effectiveness fell 6–8× in proportional terms** post-regime-change — an open question as to whether this reflects a genuine process slowdown or a change in the composition of the remaining caseload.

Full methodology and results are documented in the accompanying technical report.

## Dashboard Modules

| Page | Contents |
|---|---|
| Care Pipeline Flow Visualization | Dual-axis CBP/HHS load chart, net movement by stage, Sankey flow snapshot |
| Transfer & Discharge Efficiency | Toggle between Transfer Efficiency Ratio, Discharge Effectiveness, and Pipeline Throughput; on-demand CBP→HHS lag calculation |
| Bottleneck Detection | Adjustable backlog-threshold and anomaly-sensitivity sliders, sustained-backlog and stagnation-period detection |
| Outcome Trend Analysis | Outcome Stability Score with adjustable alert threshold, sudden-drop detection, month-over-month trends |
| ML: Regime & Forecasting | Live changepoint detection and naive-vs-Random-Forest forecasting comparison on the selected date range |

All pages respond to a global date-range filter.

## KPIs Tracked

- **Transfer Efficiency Ratio** — Transfers ÷ CBP Custody (measures CBP → HHS speed)
- **Discharge Effectiveness Index** — Discharges ÷ HHS Care (placement success rate)
- **Pipeline Throughput** — Total Exits ÷ Total Entries, rolling (overall system movement)
- **Backlog Accumulation Rate** — rate of change of the entries-minus-exits gap (delay severity)
- **Outcome Stability Score** — 1 ÷ (1 + Discharge Coefficient of Variation), consistency of placements

## Tech Stack

- [Streamlit](https://streamlit.io/) — dashboard framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data processing
- [Plotly](https://plotly.com/python/) — interactive charts, including the Sankey flow diagram
- [scikit-learn](https://scikit-learn.org/) — Random Forest forecasting model
- [SciPy](https://scipy.org/) — cross-correlation for lag detection

## Setup & Local Run

```bash
# clone the repo
git clone https://github.com/AlharathAjlan/Care-Transition-Efficiency-Placement-Outcome-Analytics
cd Care-Transition-Efficiency-Placement-Outcome-Analytics

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the app
python -m streamlit run app.py
```

The app expects a CSV file in the project folder with the following columns:

```
Date, Children apprehended and placed in CBP custody, Children in CBP custody,
Children transferred out of CBP custody, Children in HHS Care,
Children discharged from HHS Care
```

Update the `DATA_PATH` variable near the top of `app.py` to match your filename.

## Data Notes

⚠️ **The raw reporting dataset is not included in this repository** — see `.gitignore`.

**Known data characteristics** (documented and handled in `app.py`, not treated as errors):
- The raw export contains fully-blank filler rows and a comma-formatted numeric column (`Children in HHS Care`, e.g. `"7,546"`) — both are cleaned automatically on load.
- 86% of missing calendar dates fall on a Friday or Saturday, consistent with a structural reporting schedule (the source largely does not publish on those days) rather than data loss. The dashboard treats the series as a report sequence, not a forced daily calendar.
- Custody-load figures do not fully reconcile with the reported apprehension/transfer flow columns (a systematic ~+35/report gap), indicating a definitional difference rather than a data error. Custody load is treated as an independently reported figure throughout.

## Machine Learning Notes

The dashboard's ML module recomputes two things live on the currently filtered date range, rather than showing static pre-computed results:
- **Changepoint detection** — a sliding-window mean-comparison algorithm (dependency-free); the full project additionally cross-validated this with the `ruptures` PELT algorithm (not included in the live dashboard to keep dependencies minimal).
- **Forecasting comparison** — trains a Random Forest on lag/rolling features and compares it against naive persistence, reproducing the project's central finding that simple persistence currently outperforms machine learning for this metric.

## Project Structure

```
Care-Transition-Efficiency-Placement-Outcome-Analytics/
├── app.py              # Streamlit dashboard (5 modules)
├── requirements.txt
├── README.md
├── .gitignore
└── HHS_Unaccompanied_Alien_Children_Program.csv    
```

## Deliverables

- Technical report covering both the Machine Learning and Data Analytics phases (methodology, results, recommendations)
- This Streamlit dashboard (live analytics)
- Executive summary for government stakeholders

## Status

All Core Modules and User Capabilities (date range selection, ratio-based metric toggles, threshold-based visual alerts) from the project requirements are implemented, plus a dedicated ML module reproducing the project's forecasting and regime-detection findings live.
