import streamlit as st
import pandas as pd
from config import SUMMARY_DIR

st.set_page_config(page_title="Calibration Summaries")

st.header("📑 Calibration Summary Viewer")

summary_files = sorted(SUMMARY_DIR.glob("sfa_summary_table_*.csv"))

if not summary_files:
    st.info("No summary files found.")
    st.stop()

indicator_map = {
    f.stem.replace("sfa_summary_table_", ""): f
    for f in summary_files
}

selected_ind = st.selectbox(
    "Select indicator",
    list(indicator_map.keys())
)

df = pd.read_csv(indicator_map[selected_ind])
st.dataframe(df, use_container_width=True, height=600)

st.download_button(
    "⬇ Download summary",
    df.to_csv(index=False),
    file_name=indicator_map[selected_ind].name
)
