import streamlit as st

# =====================================================
# SIDEBAR RENDER FUNCTION
# =====================================================

def render_sidebar():

    with st.sidebar:

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------
        st.title("🌍 SP Costing")
        st.caption("Calibration & Optimization Engine")

        st.markdown("---")

        # -------------------------------------------------
        # NAVIGATION
        # -------------------------------------------------
        st.header("📂 Navigation")

        st.page_link("Step_1_Upload_Data.py", label="📥 Data Upload & Filter")
        st.page_link("pages/Step_2_Model_Specification.py", label="📊 Calibration")
        st.page_link("pages/Step_3_Optimization.py", label="🚀 Optimization")
        st.page_link("pages/Step_4_Charts.py", label="📈 Results & Diagnostics")

        st.markdown("---")

        # -------------------------------------------------
        # FOOTER
        # -------------------------------------------------
        st.caption(
            "UN SDG Costing Framework  \n"
            "Stochastic Frontier Analysis + Optimization"
        )
