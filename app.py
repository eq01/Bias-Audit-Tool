import streamlit as st
import pandas as pd
import os
from analysis import (
    fairness_metrics,
    is_binary_label,
    get_ai_summary, plot_group_rates,
)

st.set_page_config(
    page_title="Bias Audit Tool",
    layout="wide",
)
api_key = os.environ.get("OPENAI_API_KEY", "")

st.title("Bias Audit Tool")

st.caption(
    "Upload a dataset to detect demographic imbalance, underrepresented groups, "
    "and fairness gaps in raw data and in a trained model's predictions."
)

uploaded = st.file_uploader("Upload your CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(5), use_container_width=True)
    st.caption(f"{len(df):,} rows × {len(df.columns)} columns")

    col1, col2 = st.columns(2)
    with col1:
        demo_col = st.selectbox("Demographic column", df.columns)
    with col2:
        label_col = st.selectbox("Outcome label column", df.columns)

    # Warnings
    if pd.api.types.is_numeric_dtype(df[demo_col]) and df[demo_col].nunique() > 10:
        st.warning(f"⚠️ '{demo_col}' has {df[demo_col].nunique()} unique numeric values — try a categorical column like `sex` or `race`, or bin it into groups first.")

    if not is_binary_label(df[label_col]):
        unique_vals = df[label_col].dropna().unique().tolist()
        st.error(f"⚠️ '{label_col}' has {df[label_col].nunique()} unique values {unique_vals[:6]}{'...' if len(unique_vals) > 6 else ''}. The audit requires a binary label — pick a column with exactly 2 outcomes (e.g. hired/not hired, approved/denied, 0/1).")

    nan_demo = df[demo_col].isna().sum()
    nan_label = df[label_col].isna().sum()
    if nan_demo > 0 or nan_label > 0:
        st.info(f"ℹ️ {nan_demo} missing values in '{demo_col}', {nan_label} in '{label_col}'. These rows will be excluded from the audit.")

    if st.button("Run Audit", type="primary"):

        if not is_binary_label(df[label_col]):
            st.error("Please select a binary label column before running the audit.")
            st.stop()

        metrics = fairness_metrics(df, demo_col, label_col)

        if metrics is None:
            st.error("Need at least 2 groups in the demographic column.")
            st.stop()

        # Metrics
        st.subheader("Fairness Metrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Demographic Parity Diff", metrics["demographic_parity_diff"],
                      help="Difference in positive outcome rates between the two largest groups. Flag if > 0.1")
            st.error("Above 0.1 — disparity detected") if metrics["dp_flag"] else st.success("Below 0.1 — looks fair")
        with c2:
            st.metric("Disparate Impact Ratio", metrics["disparate_impact_ratio"],
                      help="Ratio of positive rates between groups. Flag if < 0.8 (80% rule)")
            st.error("Below 0.8 — 80% rule violated") if metrics["di_flag"] else st.success("Above 0.8 — within threshold")
        with c3:
            st.metric("Group Size Ratio", metrics["size_ratio"],
                      help="Balance between the two group sizes. Flag if < 0.5")
            st.warning("Imbalanced groups — interpret with caution") if metrics["size_flag"] else st.success("Groups are balanced")

        st.pyplot(plot_group_rates(metrics["all_rates"], demo_col, label_col))

        # AI summary
        if api_key:
            st.subheader("Analysis")
            with st.spinner("Generating summary..."):
                try:
                    st.info(get_ai_summary(metrics, demo_col, label_col))
                except Exception as e:
                    st.warning(f"AI summary failed: {e}")
        else:
            st.warning("OpenAI key not found. Make sure your .env file has KEY=your_key_here")