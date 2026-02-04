import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import INDICATOR_METADATA_FILE

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Optimization Results", layout="wide")
st.title("📊 Optimization Results & Diagnostics")

# =====================================================
# LOAD METADATA (ROBUST ENCODING)
# =====================================================

def load_sp1_metadata():
    for enc in ["cp1252", "utf-8", "latin1", "ISO-8859-1"]:
        try:
            return pd.read_csv(INDICATOR_METADATA_FILE, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("Failed to load metadata file")

sp1_metadata = load_sp1_metadata()

# =====================================================
# CLEAN METADATA
# =====================================================

for col in [
    "Target",
    "Indicator_lower_value_boundary",
    "Indicator_upper_value_boundary"
]:
    sp1_metadata[col] = pd.to_numeric(sp1_metadata[col], errors="coerce")

indicator_meta = (
    sp1_metadata
    .set_index("ExplorationCodeNew")
    .to_dict(orient="index")
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader("📂 Upload dataset", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

# =====================================================
# EXPENDITURE COLUMNS 
# =====================================================

exp_cols = [c for c in df.columns if c.startswith("GVT_")]

# =====================================================
# COUNTRY SELECTION
# =====================================================

country = st.selectbox(
    "🌍 Country",
    sorted(df["Country_name"].unique())
)

df_c = df[df["Country_name"] == country]

# =====================================================
# YEAR SELECTION (DEFAULT = 2022)
# =====================================================

years = sorted(df_c["years"].unique())
default_year = 2022 if 2022 in years else years[-1]

year = st.selectbox(
    "🗓 Year",
    years,
    index=years.index(default_year)
)

df_y = df_c[df_c["years"] == year]

# =====================================================
# 1️⃣ TOTAL EXPENDITURE 
# =====================================================

st.subheader("📈 Total Expenditure (All Years)")

exp_type_total = st.radio(
    "Select expenditure type",
    ["Absolute expenditure", "Per capita expenditure"],
    horizontal=True,
    index=1,
    key="total_exp_type"
)

if exp_type_total == "Per capita expenditure":
    exp_cols_total = [
        c for c in exp_cols if c.endswith("_PC") and c != "GVT_TOTAL_PC"
    ]
else:
    exp_cols_total = [
        c for c in exp_cols if not c.endswith("_PC") and c != "GVT_TOTAL"
    ]

df_total = (
    df_c.groupby("years")[exp_cols_total]
    .sum()
    .reset_index()
)

df_total["Total"] = df_total[exp_cols_total].sum(axis=1)


df_total["HasNegative"] = (df_total[exp_cols_total] < 0).any(axis=1)

marker_colors = np.where(
    df_total["HasNegative"],
    "#d62728",
    "#1f77b4"
)

fig_total = go.Figure()

fig_total.add_trace(
    go.Scatter(
        x=df_total["years"],
        y=df_total["Total"],
        mode="lines",
        line=dict(color="#1f77b4", width=3),
        showlegend=False
    )
)

fig_total.add_trace(
    go.Scatter(
        x=df_total["years"],
        y=df_total["Total"],
        mode="markers",
        marker=dict(size=8, color=marker_colors),
        name="Total expenditure"
    )
)

fig_total.update_layout(
    xaxis_title="Year",
    yaxis_title="Total expenditure",
    height=380
)

st.plotly_chart(fig_total, use_container_width=True)

# =====================================================
# 2️⃣ OPTIMIZED EXPENDITURE ALLOCATION 
# =====================================================

st.subheader("📊 Optimized Expenditure Allocation")

exp_type_alloc = st.radio(
    "Select expenditure type",
    ["Absolute expenditure", "Per capita expenditure"],
    horizontal=True,
    index=1,
    key="alloc_exp_type"
)

if exp_type_alloc == "Per capita expenditure":
    exp_cols_alloc = [
        c for c in exp_cols if c.endswith("_PC") and c != "GVT_TOTAL_PC"
    ]
else:
    exp_cols_alloc = [
        c for c in exp_cols if not c.endswith("_PC") and c != "GVT_TOTAL"
    ]

values = df_y[exp_cols_alloc].iloc[0]

pos_mask = values > 0
neg_mask = values < 0

fig_alloc = go.Figure()

fig_alloc.add_bar(
    x=values.index[pos_mask],
    y=values[pos_mask],
    marker_color="#1f77b4",
    name="Positive expenditure"
)

fig_alloc.add_trace(
    go.Scatter(
        x=values.index[neg_mask],
        y=values[neg_mask],
        mode="markers",
        marker=dict(
            symbol="triangle-down",
            size=14,
            color="#d62728"
        ),
        name="Negative expenditure"
    )
)

fig_alloc.add_hline(y=0, line_width=2, line_color="black")

fig_alloc.update_layout(
    xaxis_title="Expenditure category",
    yaxis_title="Value",
    height=450
)

st.plotly_chart(fig_alloc, use_container_width=True)

# =====================================================
# 3️⃣ INDICATORS VS TARGET
# =====================================================

st.subheader("🎯 Optimized Value vs Target")

indicator_cols = [
    c for c in df_y.columns if c.startswith("X") and c in indicator_meta
]

df_plot = df_y[indicator_cols].T
df_plot.columns = ["Value"]

df_plot["Target"] = df_plot.index.map(lambda x: indicator_meta[x]["Target"])
df_plot["Direction"] = df_plot.index.map(lambda x: indicator_meta[x]["TargetDirection"])
df_plot["LB_meta"] = df_plot.index.map(lambda x: indicator_meta[x]["Indicator_lower_value_boundary"])
df_plot["UB_meta"] = df_plot.index.map(lambda x: indicator_meta[x]["Indicator_upper_value_boundary"])
df_plot["Indicator"] = df_plot.index

fig = go.Figure()

# Bounds
for _, r in df_plot.iterrows():
    if r["Direction"] == "lowerOrEqual":
        lb, ub = r["LB_meta"], r["Target"]
    elif r["Direction"] == "upperOrEqual":
        lb, ub = r["Target"], r["UB_meta"]
    else:
        continue

    if pd.notna(lb) and pd.notna(ub):
        fig.add_trace(
            go.Scatter(
                x=[lb, ub],
                y=[r["Indicator"], r["Indicator"]],
                mode="lines",
                line=dict(color="rgba(180,180,180,0.5)", width=12),
                hoverinfo="skip",
                showlegend=False
            )
        )

# Gap lines
for _, r in df_plot.iterrows():
    fig.add_trace(
        go.Scatter(
            x=[r["Target"], r["Value"]],
            y=[r["Indicator"], r["Indicator"]],
            mode="lines",
            line=dict(color="#7f7f7f", width=2),
            showlegend=False
        )
    )

# Target & Value
fig.add_trace(
    go.Scatter(
        x=df_plot["Target"],
        y=df_plot["Indicator"],
        mode="markers",
        marker=dict(symbol="circle-open", size=10, line=dict(width=2)),
        name="Target"
    )
)

fig.add_trace(
    go.Scatter(
        x=df_plot["Value"],
        y=df_plot["Indicator"],
        mode="markers",
        marker=dict(size=10, color="#2ca02c"),
        name="Optimized value"
    )
)

fig.update_layout(
    xaxis_title="Indicator value",
    yaxis_title="",
    height=600,
    margin=dict(l=120),
    legend=dict(orientation="h", y=1.05)
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 4️⃣ CONSTRAINT STATUS TABLE
# =====================================================

rows = []

for ind in indicator_cols:
    meta = indicator_meta[ind]
    value = float(df_y[ind].iloc[0])
    target = meta["Target"]

    if meta["TargetDirection"] == "lowerOrEqual":
        lb, ub = meta["Indicator_lower_value_boundary"], target
    else:
        lb, ub = target, meta["Indicator_upper_value_boundary"]

    if pd.notna(lb) and value < lb:
        status = "❌ BELOW LB"
    elif pd.notna(ub) and value > ub:
        status = "❌ ABOVE UB"
    else:
        status = "✅ OK"

    rows.append({
        "Indicator": ind,
        "Value": value,
        "Target": target,
        "LB": lb,
        "UB": ub,
        "Status": status
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

