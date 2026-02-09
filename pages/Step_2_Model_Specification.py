# =====================================================
# pages/model_spec_calibration.py
# Configure & Run Calibration
# =====================================================

import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path

from config import (
    APP_TITLE,
    PAGE_LAYOUT,
    DATA_DIR,
    COEF_DIR,
    INDICATOR_METADATA_FILE,
    CALIBRATION_SCRIPT,
    RSCRIPT_EXECUTABLE
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Configure & Run Calibration",
    layout=PAGE_LAYOUT
)

st.header("📊 Configure Indicators & Run Calibration")

# =====================================================
# SAFETY CHECKS
# =====================================================

if "filtered_data" not in st.session_state or st.session_state.filtered_data is None:
    st.error("❌ Please upload and filter data first.")
    st.stop()

if "uploaded_file_path" not in st.session_state or st.session_state.uploaded_file_path is None:
    st.error("❌ Please upload the input CSV first.")
    st.stop()

# =====================================================
# LOAD METADATA
# =====================================================

@st.cache_data
def load_sp1_metadata():
    return pd.read_csv(
        INDICATOR_METADATA_FILE,
        encoding="cp1252"
    )

sp1_metadata = load_sp1_metadata()

# TargetDirection lookup
meta_target_direction = (
    sp1_metadata
    .set_index("ExplorationCodeNew")["TargetDirection"]
    .to_dict()
)

# =====================================================
# SESSION STATE INIT
# =====================================================

st.session_state.setdefault("indicator_configs", {})
st.session_state.setdefault("efficiency_configs", {})
st.session_state.setdefault("calib_running", False)

# =====================================================
# EXTRACT METADATA CODES
# =====================================================

indicator_codes = sorted(
    sp1_metadata.loc[
        sp1_metadata["ExplorationCodeNew"].str.startswith("X", na=False),
        "ExplorationCodeNew"
    ].unique()
)

exp_codes = sorted(
    sp1_metadata.loc[
        sp1_metadata["ExplorationCodeNew"].str.startswith("GVT", na=False),
        "ExplorationCodeNew"
    ].unique()
)

ctrl_codes = sorted(
    sp1_metadata.loc[
        sp1_metadata["ExplorationCodeNew"].str.startswith("CV", na=False),
        "ExplorationCodeNew"
    ].unique()
)

# =====================================================
# INDICATOR SELECTION
# =====================================================

selected_indicators = st.multiselect(
    "Select indicator(s)",
    options=indicator_codes,
    default=list(st.session_state.indicator_configs.keys())
)

if not selected_indicators:
    st.info("Please select at least one indicator.")
    st.stop()

# =====================================================
# APPLY-TO-ALL CONTROLS
# =====================================================

common_exp = st.multiselect("Select expenditure variables (GVT*)", exp_codes)
apply_exp_all = st.button("Apply expenditures to all")

common_ctrl = st.multiselect("Select control variables (CV*)", ctrl_codes)
apply_ctrl_all = st.button("Apply controls to all")

common_eff_ctrl = st.multiselect("Select efficiency control variables (CV*)", ctrl_codes)
apply_eff_all = st.button("Apply efficiency controls to all")

model_options = ["Translog"]
common_model_spec = st.selectbox("Model specification", model_options)
apply_model_all = st.button("Apply model spec to all")

# =====================================================
# BUILD CONFIGURATION TABLE
# =====================================================

rows = []

for ind in selected_indicators:

    base_cfg = st.session_state.indicator_configs.get(
        ind, {"exp": [], "ctrl": [], "model_spec": "Translog"}
    )

    eff_cfg = st.session_state.efficiency_configs.get(
        ind,
        {
            "eff_ctrl": [],
            "eff_fe": True,
            "main_fe": False,
            "main_intercept": True,
            "eff_intercept": False
        }
    )

    rows.append({
        "Indicator": ind,
        "Target Direction": meta_target_direction.get(ind),
        "Expenditures (GVT*)": common_exp if apply_exp_all and common_exp else base_cfg["exp"],
        "Controls (CV*)": common_ctrl if apply_ctrl_all and common_ctrl else base_cfg["ctrl"],
        "Efficiency Controls (CV*)": common_eff_ctrl if apply_eff_all and common_eff_ctrl else eff_cfg["eff_ctrl"],
        "Efficiency Fixed Effect": eff_cfg["eff_fe"],
        "Main Fixed Effect": eff_cfg["main_fe"],
        "Main Intercept": eff_cfg["main_intercept"],
        "Efficiency Intercept": eff_cfg["eff_intercept"],
        "Model Specification": common_model_spec if apply_model_all else base_cfg["model_spec"]
    })

config_df = pd.DataFrame(rows)

# =====================================================
# EDITABLE TABLE
# =====================================================

edited_df = st.data_editor(
    config_df,
    use_container_width=True,
    height=450,
    num_rows="fixed",
    column_config={
        "Indicator": st.column_config.TextColumn(disabled=True),
        "Target Direction": st.column_config.TextColumn(disabled=True),
        "Expenditures (GVT*)": st.column_config.ListColumn(),
        "Controls (CV*)": st.column_config.ListColumn(),
        "Efficiency Controls (CV*)": st.column_config.ListColumn(),
        "Efficiency Fixed Effect": st.column_config.CheckboxColumn(),
        "Main Fixed Effect": st.column_config.CheckboxColumn(),
        "Main Intercept": st.column_config.CheckboxColumn(),
        "Efficiency Intercept": st.column_config.CheckboxColumn(),
        "Model Specification": st.column_config.SelectboxColumn(options=model_options)
    }
)

# =====================================================
# SAVE CONFIG TO SESSION STATE
# =====================================================

indicator_cfg = {}
eff_cfg = {}

for _, row in edited_df.iterrows():
    ind = row["Indicator"]

    indicator_cfg[ind] = {
        "exp": row["Expenditures (GVT*)"] or [],
        "ctrl": row["Controls (CV*)"] or [],
        "model_spec": row["Model Specification"],
        "target_direction": row["Target Direction"]
    }

    eff_cfg[ind] = {
        "eff_ctrl": row["Efficiency Controls (CV*)"] or [],
        "eff_fe": bool(row["Efficiency Fixed Effect"]),
        "main_fe": bool(row["Main Fixed Effect"]),
        "main_intercept": bool(row["Main Intercept"]),
        "eff_intercept": bool(row["Efficiency Intercept"])
    }

st.session_state.indicator_configs = indicator_cfg
st.session_state.efficiency_configs = eff_cfg

# =====================================================
# RUN CALIBRATION
# =====================================================

st.markdown("## ▶ Run Calibration")

def run_r_calibration(ind, cfg, eff):

    cmd = [
        RSCRIPT_EXECUTABLE,
        str(CALIBRATION_SCRIPT),

        "--input", st.session_state.uploaded_file_path,
        "--indicator", ind,
        "--expenditures", ",".join(cfg["exp"]),
        "--controls", ",".join(cfg["ctrl"]),
        "--eff_controls", ",".join(eff["eff_ctrl"]),
        "--eff_fe", "1" if eff["eff_fe"] else "0",
        "--main_fe", "1" if eff["main_fe"] else "0",
        "--main_intercept", "1" if eff["main_intercept"] else "0",
        "--eff_intercept", "1" if eff["eff_intercept"] else "0",
        "--model", cfg["model_spec"],
        "--target_direction", cfg.get("target_direction") or "none",
        "--output", str(DATA_DIR / f"sfa_coefficients_{ind}.csv")

    ]

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

# =====================================================
# EXECUTION
# =====================================================

if st.button("🚀 Run Calibration") and not st.session_state.calib_running:

    st.session_state.calib_running = True
    results = []
    errors = []

    COEF_DIR.mkdir(parents=True, exist_ok=True)

    with st.spinner("Running calibration in R..."):

        for ind, cfg in indicator_cfg.items():
            eff = eff_cfg[ind]

            res = run_r_calibration(ind, cfg, eff)

            if res.returncode != 0:
                errors.append((ind, res.stderr))
                continue

            coef_file = COEF_DIR / f"sfa_coefficients_{ind}.csv"

            if not coef_file.exists():
                errors.append((ind, f"Coefficient file not found: {coef_file}"))
                continue

            df_out = pd.read_csv(coef_file)
            df_out["Indicator"] = ind
            df_out["TargetDirection"] = cfg.get("target_direction")
            results.append(df_out)

    st.session_state.calib_running = False

    if results:
        final_df = pd.concat(results, ignore_index=True)
        st.session_state.calib_df = final_df

        st.success("✅ Calibration completed successfully")
        st.dataframe(final_df, height=500)

        st.download_button(
            "⬇ Download calibration results",
            final_df.to_csv(index=False),
            "calibration_results_all_indicators.csv"
        )

    if errors:
        st.warning("⚠️ Some indicators failed:")
        for ind, msg in errors:
            st.text(f"{ind}\n{msg}\n")