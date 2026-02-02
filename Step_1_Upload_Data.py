import streamlit as st
import pandas as pd
from pathlib import Path
import re

from config import (
    DATA_DIR,
    INDICATOR_METADATA_FILE,
    APP_TITLE,
    PAGE_LAYOUT
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout=PAGE_LAYOUT
)

st.title(APP_TITLE)

# =====================================================
# Load SP1 metadata
# =====================================================

@st.cache_data
def load_sp1_metadata():
    return pd.read_csv(
        INDICATOR_METADATA_FILE,
        encoding="cp1252"
    )

sp1_metadata = load_sp1_metadata()

# =====================================================
# Session state initialization
# =====================================================

for key in [
    "step",
    "data",
    "filtered_data",
    "indicator_configs",
    "calib_df",
    "opt_df",
    "rscript_path",
    "uploaded_file_path",
    "calib_running",
    "opt_running"
]:
    if key not in st.session_state:
        st.session_state[key] = None

st.session_state.calib_running = st.session_state.calib_running or False
st.session_state.opt_running = st.session_state.opt_running or False
st.session_state.step = st.session_state.step or 1

if st.session_state.indicator_configs is None:
    st.session_state.indicator_configs = {}

# =====================================================
# STEP 1 — Upload Data
# =====================================================

with st.expander("🧩 Step 1: Upload Data", expanded=st.session_state.step == 1):

    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file:
        try:
            # ensure data directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            uploaded_file_path = DATA_DIR / uploaded_file.name

            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.uploaded_file_path = str(uploaded_file_path)

            # read file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file_path)
            else:
                df = pd.read_excel(uploaded_file_path)

            st.session_state.data = df

            st.success("✅ File uploaded successfully!")
            st.dataframe(df, height=400)

            st.session_state.step = max(st.session_state.step, 2)

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# Safety stop
if st.session_state.data is None:
    st.info("Please upload a CSV or Excel file to proceed.")
    st.stop()

# =====================================================
# STEP 2 — Select Filter Criteria
# =====================================================

with st.expander("🔍 Step 2: Select Filter Criteria", expanded=st.session_state.step == 2):

    df = st.session_state.data

    # criteria columns = non-indicator, non-control, non-expenditure
    criteria_cols = [
        col for col in df.columns
        if not (
            col.startswith("X") or
            col.startswith("CV") or
            col.startswith("GVT")
        )
    ]

    if not criteria_cols:
        st.warning("No criteria columns found. Using full dataset.")
        filtered_df = df.copy()
        st.session_state.filtered_data = filtered_df
        st.session_state.step = max(st.session_state.step, 3)

    else:
        filter_col = st.selectbox(
            "Select a filter column (optional)",
            ["<no filter>"] + criteria_cols
        )

        if filter_col != "<no filter>":
            unique_vals = sorted(df[filter_col].dropna().unique())
            selected_value = st.selectbox(
                f"Select a value for {filter_col}",
                unique_vals
            )

            if st.button("Apply Filter"):
                filtered_df = (
                    df[df[filter_col] == selected_value]
                    .reset_index(drop=True)
                )
                st.session_state.filtered_data = filtered_df
                st.success(
                    f"✅ Filter applied: {filter_col} = {selected_value}"
                )
                st.dataframe(filtered_df, height=400)
                st.session_state.step = max(st.session_state.step, 3)

        else:
            if st.button("Use full dataset"):
                filtered_df = df.copy()
                st.session_state.filtered_data = filtered_df
                st.success("✅ Using full dataset (no filter).")
                st.dataframe(filtered_df, height=400)
                st.session_state.step = max(st.session_state.step, 3)

# Final safety
if st.session_state.filtered_data is None:
    st.stop()
