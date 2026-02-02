import streamlit as st
import pandas as pd
import numpy as np

from config import INDICATOR_METADATA_FILE

st.set_page_config(
    page_title="Optimization Results",
    layout="wide"
)

st.title("📊 Optimization Results & Diagnostics")

# =====================================================
# SAFETY CHECKS
# =====================================================

required_keys = [
    "opt_df",
    "indicator_meta",
    "optimized_countries",
    "optimized_years",
    "run_mode"
]

for k in required_keys:
    if k not in st.session_state:
        st.warning("⚠ Please run the optimization first.")
        st.stop()

df = st.session_state.opt_df.copy()
indicator_meta = st.session_state.indicator_meta
countries_run = st.session_state.optimized_countries
years_run = st.session_state.optimized_years
run_mode = st.session_state.run_mode

# =====================================================
# HEADER INFO
# =====================================================

st.info(
    f"**Run mode:** {run_mode}  \n"
    f"**Countries optimized:** {len(countries_run)}  \n"
    f"**Years optimized:** {min(years_run)} → {max(years_run)}"
)

# =====================================================
# FILTER TO OPTIMIZED DATA ONLY
# =====================================================

df = df[
    (df["iso3"].isin(countries_run)) &
    (df["years"].isin(years_run))
].copy()

# =====================================================
# COUNTRY SELECTION (DISPLAY ONLY)
# =====================================================

country_map = (
    df[["iso3", "Country_name"]]
    .drop_duplicates()
    .sort_values("Country_name")
)

selected_country = st.selectbox(
    "🌍 Select country",
    country_map["Country_name"].unique()
)

iso3 = country_map.loc[
    country_map["Country_name"] == selected_country,
    "iso3"
].iloc[0]

df_c = df[df["iso3"] == iso3].copy()

# =====================================================
# 1️⃣ TOTAL EXPENDITURE (OPTIMIZED YEARS ONLY)
# =====================================================

st.subheader("📈 Total Expenditure (Optimized Years)")

exp_cols = [c for c in df_c.columns if c.startswith("GVT_")]

df_total = (
    df_c
    .groupby("years")[exp_cols]
    .sum()
    .sum(axis=1)
    .reset_index(name="Total Expenditure")
    .set_index("years")
)

st.line_chart(df_total)

# =====================================================
# 2️⃣ EXPENDITURE REALLOCATION (LAST YEAR)
# =====================================================

st.subheader("📊 Expenditure Reallocation (Last Optimized Year)")

latest_year = max(years_run)
prev_year = latest_year - 1

df_prev = df_c[df_c["years"] == prev_year]
df_curr = df_c[df_c["years"] == latest_year]

if not df_prev.empty and not df_curr.empty:

    exp_compare = pd.DataFrame({
        "Historical": df_prev[exp_cols].iloc[0],
        "Optimized": df_curr[exp_cols].iloc[0]
    })

    exp_compare["Delta"] = exp_compare["Optimized"] - exp_compare["Historical"]
    exp_compare = exp_compare.sort_values("Optimized", ascending=False).head(10)

    st.bar_chart(exp_compare[["Historical", "Optimized"]])

else:
    st.info("Not enough data to compare expenditures.")

# =====================================================
# 3️⃣ INDICATORS VS TARGETS (OPTIMIZED ONLY)
# =====================================================

st.subheader("🎯 Indicators vs Targets (Last Optimized Year)")

indicator_cols = [
    c for c in df_c.columns
    if c.startswith("X") and c in indicator_meta
]

df_ind = (
    df_c[df_c["years"] == latest_year][indicator_cols]
    .T
    .rename(columns={df_c.index.max(): "Optimized"})
)

df_ind["Target"] = df_ind.index.map(
    lambda x: indicator_meta.get(x, {}).get("target", np.nan)
)

st.bar_chart(df_ind[["Optimized", "Target"]])

# =====================================================
# 4️⃣ INDICATOR CONSTRAINT STATUS
# =====================================================

st.subheader("🚦 Indicator Constraint Status")

rows = []

for ind in indicator_cols:
    value = df_c.loc[df_c["years"] == latest_year, ind].values[0]
    meta = indicator_meta.get(ind, {})

    lb = meta.get("lb")
    ub = meta.get("ub")

    if lb is not None and value < lb:
        status = "❌ BELOW LB"
    elif ub is not None and value > ub:
        status = "❌ ABOVE UB"
    else:
        status = "✅ OK"

    rows.append({
        "Indicator": ind,
        "Value": value,
        "LB": lb,
        "UB": ub,
        "Status": status
    })

df_status = pd.DataFrame(rows)

st.dataframe(df_status, use_container_width=True)

# =====================================================
# 5️⃣ MULTI-COUNTRY SUMMARY (IF APPLICABLE)
# =====================================================

if run_mode == "All countries":

    st.subheader("🌍 Cross-Country Summary (Last Optimized Year)")

    df_multi = (
        df[df["years"] == latest_year]
        .groupby("Country_name")[exp_cols]
        .sum()
        .sum(axis=1)
        .reset_index(name="Total Expenditure")
        .sort_values("Total Expenditure", ascending=False)
    )

    st.bar_chart(
        df_multi.set_index("Country_name")
    )

# =====================================================
# DOWNLOAD
# =====================================================

st.subheader("⬇ Download Optimized Results")

st.download_button(
    "Download optimized dataset",
    df.to_csv(index=False),
    file_name="optimized_results.csv"
)
