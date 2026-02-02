# =====================================================
# STEP 3 — OPTIMIZATION
# =====================================================

import time
import tempfile
import streamlit as st
import pandas as pd

from optimization import optimize_country_year as optimize_no_gdp
from optimization_gdp import optimize_country_year as optimize_with_gdp
from data_loader import load_indicator_metadata

from config import (
    DATA_DIR,
    COEF_DIR,
    INDICATOR_METADATA_FILE
)

# =====================================================
# SAFETY CHECKS
# =====================================================

if "filtered_data" not in st.session_state or st.session_state.filtered_data is None:
    st.error("❌ No filtered data found. Please complete Step 1 & 2 first.")
    st.stop()

st.subheader("Step 3 — Optimization")

df = st.session_state.filtered_data.copy()

# =====================================================
# TEMP FILE (USED BY OPTIMIZATION FUNCTIONS)
# =====================================================

tmp_dir = tempfile.mkdtemp()
TMP_DATA_FILE = f"{tmp_dir}/filtered_input.csv"
df.to_csv(TMP_DATA_FILE, index=False)

# =====================================================
# COUNTRY MAPPING
# =====================================================

country_map = (
    df[["iso3", "Country_name"]]
    .dropna()
    .drop_duplicates()
    .sort_values("Country_name")
)

country_name_to_iso3 = dict(
    zip(country_map["Country_name"], country_map["iso3"])
)

# =====================================================
# OPTIMIZATION MODE
# =====================================================

opt_mode = st.radio(
    "Select optimization mode",
    [
        "Without GDP constraint (pure cost minimization)",
        "With GDP constraint"
    ]
)

optimize_fn = (
    optimize_with_gdp
    if "GDP" in opt_mode
    else optimize_no_gdp
)

# =====================================================
# TIME CONFIGURATION
# =====================================================

available_years = sorted(df["years"].dropna().unique())

st.subheader("🕒 Time horizon")

col1, col2 = st.columns(2)

with col1:
    START_YEAR = st.selectbox(
        "Start year (historical base)",
        available_years,
        index=available_years.index(2022) if 2022 in available_years else 0
    )

with col2:
    END_YEAR = st.selectbox(
        "End year (last optimized year)",
        available_years,
        index=available_years.index(2030) if 2030 in available_years else len(available_years) - 1
    )

if END_YEAR <= START_YEAR:
    st.error("❌ End year must be strictly greater than start year")
    st.stop()

# =====================================================
# COUNTRY SELECTION
# =====================================================

st.subheader("🌍 Country selection")

run_mode = st.radio(
    "Run mode",
    ["Single country", "All countries"]
)

if run_mode == "Single country":
    selected_country = st.selectbox(
        "Select country",
        country_name_to_iso3.keys()
    )
    countries = [country_name_to_iso3[selected_country]]
else:
    countries = list(country_name_to_iso3.values())
    st.info(f"{len(countries)} countries selected")

# =====================================================
# CONTROL BUTTONS
# =====================================================

col_run, col_stop = st.columns(2)

with col_run:
    run_clicked = st.button("▶ Run optimization")

with col_stop:
    stop_clicked = st.button("⛔ Stop optimization")

if "stop_optimization" not in st.session_state:
    st.session_state.stop_optimization = False

if stop_clicked:
    st.session_state.stop_optimization = True
    st.warning("⛔ Optimization will stop after current iteration")

# =====================================================
# RUN OPTIMIZATION
# =====================================================

if run_clicked:

    st.session_state.stop_optimization = False

    log_box = st.empty()
    progress_bar = st.progress(0.0)
    time_box = st.empty()

    total_steps = len(countries) * (END_YEAR - START_YEAR)
    step_counter = 0
    start_time = time.time()

    for iso3 in countries:

        country_label = (
            country_map
            .loc[country_map["iso3"] == iso3, "Country_name"]
            .iloc[0]
        )

        log_box.markdown(f"## 🌍 {country_label} ({iso3})")

        for year in range(START_YEAR, END_YEAR):

            if st.session_state.stop_optimization:
                log_box.warning("⛔ Optimization stopped by user")
                break

            log_box.markdown(f"⏳ Optimizing {year} → {year + 1}")

            result = optimize_fn(
                iso3=iso3,
                year=year,
                cv_year=year + 1,
                data_file=TMP_DATA_FILE,
                coef_dir=str(COEF_DIR),
                metadata_file=str(INDICATOR_METADATA_FILE)
            )

            mask = (df["iso3"] == iso3) & (df["years"] == year + 1)

            # Write expenditures
            for k, v in result["expenditures"].items():
                df.loc[mask, k] = v

            # Write indicators
            for k, v in result["indicators"].items():
                df.loc[mask, k] = v

            # ---- Progress & ETA
            step_counter += 1
            progress_bar.progress(step_counter / total_steps)

            elapsed = time.time() - start_time
            avg_time = elapsed / step_counter
            remaining = avg_time * (total_steps - step_counter)

            time_box.info(
                f"⏱ Elapsed: {elapsed/60:.1f} min | "
                f"Estimated remaining: {remaining/60:.1f} min"
            )

        if st.session_state.stop_optimization:
            break

        log_box.success(f"✅ Completed for {country_label}")

    # =====================================================
    # EXPORT RESULT
    # =====================================================

    st.success("🎉 Optimization completed")

    st.session_state.opt_df = df.copy()

    st.download_button(
        "⬇ Download optimized dataset",
        df.to_csv(index=False),
        file_name="optimized_dataset.csv"
    )

# =====================================================
# SAVE RESULTS IN SESSION STATE (FOR NEXT PAGES)
# =====================================================

indicator_meta = load_indicator_metadata(INDICATOR_METADATA_FILE)

st.session_state.indicator_meta = indicator_meta
st.session_state.optimized_countries = countries
st.session_state.optimized_years = list(range(START_YEAR + 1, END_YEAR + 1))
st.session_state.run_mode = run_mode
