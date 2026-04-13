import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis import (
    group_representation,
    label_rate_by_group,
    fairness_metrics,
    train_and_audit,
    generate_findings,
)

st.set_page_config(
    page_title="Bias Audit Tool",
    layout="wide",
)

st.title("Bias Audit Tool")
st.caption(
    "Upload a dataset to detect demographic imbalance, underrepresented groups, "
    "and fairness gaps — in raw data and in a trained model's predictions."
)

with st.sidebar:
    st.header("1. Upload your data")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    use_sample = st.checkbox("Use sample dataset instead")

    if use_sample:
        try:
            df = pd.read_csv("sample_data/adult.csv")
            st.success(f"Loaded dataset: {len(df):,} rows")
        except FileNotFoundError:
            st.error("sample_data/adult.csv not found.")
            st.stop()
    elif uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {uploaded.name} — {len(df):,} rows")
    else:
        st.info("Upload a CSV or use the sample dataset to get started.")
        st.stop()

    st.header("2. Select columns")
    demo_col = st.selectbox(
        "Demographic / sensitive attribute",
        options=df.columns.tolist(),
        help="The column that identifies a demographic group (e.g. race, gender).",
    )
    label_col = st.selectbox(
        "Outcome / label column",
        options=["(none)"] + df.columns.tolist(),
        help="The target variable or outcome (e.g. hired, recidivated, approved).",
    )
    label_col = None if label_col == "(none)" else label_col

    run_model = st.checkbox(
        "Train a model and audit its predictions",
        value=bool(label_col),
        disabled=not label_col,
        help="Requires an outcome column. Trains logistic regression and checks if predictions are biased.",
    )

    st.divider()
    min_rows = st.slider(
        "Minimum rows per group (underrepresentation threshold)",
        min_value=5, max_value=200, value=30,
    )

st.subheader("Dataset overview")

rep_df = group_representation(df, demo_col)
n_groups = len(rep_df)
n_underrep = rep_df["underrepresented"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total rows", f"{len(df):,}")
col2.metric("Demographic groups", n_groups)
col3.metric("Underrepresented groups", n_underrep,
            delta=f"{n_underrep} flagged" if n_underrep else None,
            delta_color="inverse")
if label_col:
    pos_rate = df[label_col].astype(str).value_counts(normalize=True).iloc[0]
    col4.metric("Majority label rate", f"{pos_rate:.1%}")

st.divider()

st.subheader("Group representation")
st.caption("How much of the dataset each group accounts for. The dashed line marks equal representation.")

rep_df["color"] = rep_df["underrepresented"].map(
    {True: "Underrepresented", False: "Within threshold"}
)

fig_rep = px.bar(
    rep_df,
    x="pct",
    y="group",
    orientation="h",
    color="color",
    color_discrete_map={
        "Underrepresented": "#D85A30",
        "Within threshold": "#378ADD",
    },
    labels={"pct": "Share of dataset (%)", "group": ""},
    text=rep_df["pct"].round(1).astype(str) + "%",
)

# Add equal-share reference line
equal_share = 100 / n_groups
fig_rep.add_vline(
    x=equal_share,
    line_dash="dash",
    line_color="gray",
    annotation_text=f"Equal share ({equal_share:.1f}%)",
    annotation_position="top right",
)

fig_rep.update_layout(
    showlegend=True,
    legend_title_text="",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=max(300, n_groups * 44),
    margin=dict(l=0, r=20, t=20, b=20),
)
fig_rep.update_traces(textposition="outside")
st.plotly_chart(fig_rep, use_container_width=True)


if label_col:
    st.divider()
    st.subheader("Outcome rate by group")
    st.caption(
        "Positive label rate per group. Large gaps suggest the outcome is not distributed equally."
    )

    rate_df = label_rate_by_group(df, demo_col, label_col)

    fig_rate = px.bar(
        rate_df,
        x="group",
        y="positive_rate",
        color="positive_rate",
        color_continuous_scale=["#378ADD", "#D85A30"],
        labels={"positive_rate": "Positive label rate", "group": ""},
        text=rate_df["positive_rate"].round(3).astype(str),
    )
    fig_rate.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=20),
    )
    st.plotly_chart(fig_rate, use_container_width=True)


if label_col:
    st.divider()
    st.subheader("Fairness metrics")

    fm = fairness_metrics(df, demo_col, label_col)

    if fm:
        st.caption(
            f"Comparing **{fm['group_a']}** (larger group) vs. **{fm['group_b']}** (smaller group)"
        )

        m1, m2, m3 = st.columns(3)

        def metric_status(flagged):
            return "🔴 Flagged" if flagged else "🟢 Passed"

        m1.metric(
            "Demographic parity difference",
            fm["demographic_parity_diff"],
            delta=metric_status(fm["dp_flag"]),
            delta_color="inverse" if fm["dp_flag"] else "normal",
            help="Difference in positive outcome rates between groups. Flag threshold: > 0.1",
        )
        m2.metric(
            "Disparate impact ratio",
            fm["disparate_impact_ratio"],
            delta=metric_status(fm["di_flag"]),
            delta_color="inverse" if fm["di_flag"] else "normal",
            help="Smaller group rate / larger group rate. Flag threshold: < 0.8 (the '80% rule')",
        )
        m3.metric(
            "Group size ratio",
            fm["size_ratio"],
            delta=metric_status(fm["size_flag"]),
            delta_color="inverse" if fm["size_flag"] else "normal",
            help="Smaller group count / larger group count. Flag threshold: < 0.5",
        )

if label_col and run_model:
    st.divider()
    st.subheader("Model audit")
    st.caption(
        "A logistic regression was trained with the given dataset. The chart below shows whether "
        "the model amplified bias beyond the raw data."
    )

    with st.spinner("Training model..."):
        model_fm, raw_fm, accuracy = train_and_audit(df, demo_col, label_col)

    if model_fm and raw_fm:
        st.success(f"Model trained — accuracy: {accuracy}%")

        compare = pd.DataFrame({
            "Metric": ["Demographic parity diff", "Disparate impact ratio", "Group size ratio"],
            "Raw data": [
                raw_fm["demographic_parity_diff"],
                raw_fm["disparate_impact_ratio"],
                raw_fm["size_ratio"],
            ],
            "Model predictions": [
                model_fm["demographic_parity_diff"],
                model_fm["disparate_impact_ratio"],
                model_fm["size_ratio"],
            ],
        })

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name="Raw data", x=compare["Metric"], y=compare["Raw data"],
            marker_color="#378ADD",
        ))
        fig_compare.add_trace(go.Bar(
            name="Model predictions", x=compare["Metric"], y=compare["Model predictions"],
            marker_color="#D85A30",
        ))
        fig_compare.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
            margin=dict(l=0, r=0, t=20, b=20),
        )
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.warning("Not enough data or feature columns to train a model properly.")

st.divider()
st.subheader("Audit findings")

fm_for_findings = fairness_metrics(df, demo_col, label_col) if label_col else None
model_fm_for_findings = model_fm if (label_col and run_model and "model_fm" in dir()) else None

findings = generate_findings(rep_df, fm_for_findings, model_fm_for_findings)

for status, text in findings:
    if status == "warning":
        st.warning(text)
    else:
        st.success(text)
