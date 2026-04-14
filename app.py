import streamlit as st
import pandas as pd
from analysis import (
    fairness_metrics,
    is_binary_label,
    get_ai_summary, plot_group_rates,
)
import os

# get the key
api_key = os.getenv("KEY")

st.set_page_config(
    page_title="Bias Audit Tool",
    layout="wide",
)

st.title("Bias Audit Tool")

st.caption(
    "Upload a dataset to detect demographic imbalance, underrepresented groups, "
    "and fairness gaps in raw data"
)

# get the uploaded file if there is one
uploaded = st.file_uploader("Upload your CSV", type="csv")

# else default to the sample data provided
if not uploaded:
    if st.button("Use sample data (Adult Income Dataset)"):
        df_sample = pd.read_csv("sample_data/adult.csv")
        st.session_state["sample_df"] = df_sample

if "sample_df" in st.session_state and not uploaded:
    df = st.session_state["sample_df"]
    st.success("Using sample data. Try demographic: `sex` or `race`, outcome: `income`")
elif uploaded:
    df = pd.read_csv(uploaded)
    st.session_state.pop("sample_df", None)
else:
    df = None

if df is not None:
    col1, col2 = st.columns(2)
    with col1:
        demo_col = st.selectbox("Demographic column", df.columns)
    with col2:
        label_col = st.selectbox("Outcome label column", df.columns)

    # warnings
    if pd.api.types.is_numeric_dtype(df[demo_col]) and df[demo_col].nunique() > 10:
        st.warning(
            f" '{demo_col}' has {df[demo_col].nunique()} unique values, try a categorical column like `sex` or `race`")

    if not is_binary_label(df[label_col]):
        unique_vals = df[label_col].dropna().unique().tolist()
        st.error(
            f" '{label_col}' has {df[label_col].nunique()} unique values {unique_vals[:6]}{'...' if len(unique_vals) > 6 else ''}. The audit requires a binary label so pick a column with exactly 2 outcomes (e.g. hired/not hired, approved/denied, 0/1).")

    nan_demo = df[demo_col].isna().sum()
    nan_label = df[label_col].isna().sum()
    if nan_demo > 0 or nan_label > 0:
        st.info(
            f"ℹ️ {nan_demo} missing values in '{demo_col}', {nan_label} in '{label_col}'. These rows will be excluded from the audit.")

    if st.button("Run Audit", type="primary"):

        if not is_binary_label(df[label_col]):
            st.error("Please select a binary label column before running the audit.")
            st.stop()

        metrics = fairness_metrics(df, demo_col, label_col)

        if metrics is None:
            st.error("Need at least 2 groups in the demographic column.")
            st.stop()

        st.pyplot(plot_group_rates(metrics["all_rates"], demo_col, label_col))

        # ai summary
        if api_key:
            st.subheader("Analysis")
            with st.spinner("Generating summary..."):
                try:
                    st.info(get_ai_summary(metrics, demo_col, label_col))
                except Exception as e:
                    st.warning(f"AI summary failed: {e}")
        else:
            st.warning("OpenAI key not found. Make sure your .env file has KEY=your_key_here")