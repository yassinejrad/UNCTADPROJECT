import os
import subprocess
import tempfile

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="CSV Viewer & SDG Methodology",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'upload'


def run_r_script(uploaded_file):
    """Save uploaded file, run R script, return processed CSV path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Path to Rscript executable (adjust if needed)
    r_exec = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"
    r_script = r"C:\Users\jrady\Desktop\SDG Costing\Script\Testing.R"

    try:
        subprocess.run([r_exec, r_script, tmp_path], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Error while running R script: {e}")

    # Look for processed file (your R script must create this)
    processed_path = tmp_path.replace(".csv", "_processed.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Expected output file not found: {processed_path}")

    return processed_path


def show_upload_page():
    """Upload CSV page"""
    st.title("📤 Upload CSV File")
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None and st.button("▶️ Run R Script & View Data"):
        try:
            processed_file = run_r_script(uploaded_file)
            st.success("✅ R script finished successfully!")

            # Load processed file into session state
            st.session_state.csv_data = pd.read_csv(processed_file)

            # Go to table page
            st.session_state.current_page = 'table'
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error while running R script: {e}")

    st.markdown("---")
    if st.button("📘 View Methodology"):
        st.session_state.current_page = 'methodology'
        st.rerun()


def show_table_page():
    """Show CSV data in a table and plots"""
    st.title("📋 Processed CSV Data")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Back to Upload"):
            st.session_state.current_page = 'upload'
            st.rerun()
    with col2:
        if st.button("📘 View Methodology"):
            st.session_state.current_page = 'methodology'
            st.rerun()

    st.markdown("---")

    if 'csv_data' not in st.session_state:
        st.warning("⚠️ No data available. Please upload a file first.")
        return

    df = st.session_state.csv_data

    st.dataframe(df, use_container_width=True, height=500)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Processed CSV",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("📊 Plots")

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found for plotting.")
        return

    # 1. Histogram
    st.markdown(f"**Histogram of {numeric_cols[0]}**")
    fig_hist = px.histogram(df, x=numeric_cols[0], nbins=20)
    st.plotly_chart(fig_hist, use_container_width=True)

    # 2. Scatter plot
    if len(numeric_cols) > 1:
        st.markdown(f"**Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}**")
        fig_scatter = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Box plot
    st.markdown(f"**Box plot of {numeric_cols[0]}**")
    fig_box = px.box(df, y=numeric_cols[0])
    st.plotly_chart(fig_box, use_container_width=True)




def show_methodology_page():
    st.title("📘 Methodology: Translog-SFA for SDG Investment Needs")

    # 1. Framework Overview
    with st.expander("1️⃣ Framework Overview", expanded=True):
        st.markdown("""
        This methodology estimates public investment needs to achieve Sustainable Development Goals (SDGs)
        using a **three-stage approach**:
        1. Production Function Estimation (Translog-Stochastic Frontier Analysis)
        2. Efficiency Adjustment
        3. Dynamic Investment Pathway Modeling
        """)

    # 2. Stage 1
    with st.expander("2️⃣ Stage 1: SDG Production Function (Translog-SFA)"):
        st.markdown(r"""
        We model the SDG outcome \(Y_t\) as a function of multiple public investments \(I_{i,t}\) using a **Translog Stochastic Frontier**:

        $$
        \ln(Y_t) = \alpha_0 + \sum_{i=1} \alpha_i \ln(X_i) 
        + \sum_i \sum_j \beta_{ij} \ln(I_{i,t}) \ln(I_{j,t}) 
        + v_t - u_t
        $$

        **Where:**
            - $\alpha_i$ = sector-specific elasticities  
            - $\beta_{ij}$ = interaction terms (synergies/overlaps)  
            - $v_t \sim N(0, \sigma_v^2)$ = random noise  
            - $u_t \ge 0$ = inefficiency term
        """)

    # 3. Stage 2
    with st.expander("3️⃣ Stage 2: Adjusted Investment Requirement"):
        st.markdown("""
        - Frontier Requirement (\(I^*\)) = ideal spending at 100% efficiency (u=0)  
        - Adjusted Requirement (\(I_{adj}\)) = accounts for inefficiency (TE)

        Example:  
        - SDG Target: 95% universal healthcare  
        - Frontier Requirement: 4.0% GDP  
        - TE = 0.80 → Adjusted Requirement = 4.0 / 0.80 = 5.0% GDP
        """)

        # Interactive slider for TE
        te = st.slider("Technical Efficiency (TE)", min_value=0.5, max_value=1.0, value=0.8, step=0.01)
        frontier_req = 4.0
        adj_req = frontier_req / te
        st.markdown(f"**Adjusted Requirement:** {adj_req:.2f}% GDP")

    # 4. Stage 3
    with st.expander("4️⃣ Stage 3: Dynamic Investment Pathway Modeling"):
        st.markdown("""
        Optimize annual investment paths to meet Y_2030 = 95%, adjusted for TE.

        Example parameters:
        - αE = 0.30 (Education)  
        - αH = 0.25 (Health)  
        - αS = 0.20 (Social Protection)  
        - βEH = 0.05 (synergy effect)
        """)

        # Simulated investment path plot
        years = list(range(2021, 2031))
        frontier_edu = [6.53,7.03,7.65,8.41,9.34,10.45,11.77,13.32,13.35,13.38]
        adj_edu = [x/te for x in frontier_edu]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=frontier_edu, mode='lines+markers', name='Frontier Education (%)'))
        fig.add_trace(go.Scatter(x=years, y=adj_edu, mode='lines+markers', name=f'Adjusted Education (%) TE={te:.2f}'))
        fig.update_layout(title="Investment Pathway (Education)",
                          xaxis_title="Year", yaxis_title="% GDP")
        st.plotly_chart(fig, use_container_width=True)

    # 5. Policy Implications
    with st.expander("5️⃣ Interpretations & Policy Implications"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Increase Funding", "Raise spending to TE-adjusted levels")
        col2.metric("Improve Efficiency", "Reforms to increase TE, e.g., 0.80 → 0.90")
        col3.metric("Mixed Approach", "Combine partial funding increase with efficiency gains")
        st.markdown("""
        - Spending Type vs % GDP:  
            - Frontier Requirement = ideal (no inefficiency)  
            - Efficiency-Adjusted = real-world (accounts for TE)
        """)

    # Back button
    if st.button("⬅️ Back"):
        st.session_state.current_page = 'upload'
        st.rerun()



def main():
    if st.session_state.current_page == 'upload':
        show_upload_page()
    elif st.session_state.current_page == 'table':
        show_table_page()
    elif st.session_state.current_page == 'methodology':
        show_methodology_page()


if __name__ == "__main__":
    main()
