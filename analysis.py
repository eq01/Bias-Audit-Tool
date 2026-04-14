import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY", "")

def label_rate_by_group(df, demo_col, label_col):
    le = LabelEncoder()
    df = df.copy()
    df["_label"] = le.fit_transform(df[label_col].astype(str))
    rates = df.groupby(demo_col)["_label"].mean().reset_index()
    rates.columns = ["group", "positive_rate"]
    return rates, le.classes_


def is_binary_label(series):
    return series.dropna().nunique() == 2


def fairness_metrics(df, demo_col, label_col):
    # drop nan rows in either column
    df = df.dropna(subset=[demo_col, label_col])

    rates, classes = label_rate_by_group(df, demo_col, label_col)
    counts = df[demo_col].value_counts()
    top2 = counts.nlargest(2).index.tolist()
    if len(top2) < 2:
        return None

    r = rates.set_index("group")["positive_rate"]
    r1, r2 = r.get(top2[0], 0), r.get(top2[1], 0)
    c1, c2 = counts[top2[0]], counts[top2[1]]

    dp_diff = abs(r1 - r2)
    di_ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 1.0
    size_ratio = min(c1, c2) / max(c1, c2)

    return {
        "group_a": top2[0], "group_b": top2[1],
        "rate_a": round(r1, 4), "rate_b": round(r2, 4),
        "demographic_parity_diff": round(dp_diff, 4),
        "disparate_impact_ratio": round(di_ratio, 4),
        "size_ratio": round(size_ratio, 4),
        "dp_flag": dp_diff > 0.1, "di_flag": di_ratio < 0.8, "size_flag": size_ratio < 0.5,
        "all_rates": rates, "label_classes": classes.tolist(),
    }


def get_ai_summary(metrics, demo_col, label_col):
    client = OpenAI(api_key=os.getenv("KEY"))
    prompt = f"""You are a fairness and bias analyst. Write a clear 3-paragraph summary for a writing class presentation.

Dataset context:
- Demographic column: "{demo_col}"
- Outcome label: "{label_col}"
- Group A: {metrics['group_a']} (positive rate: {metrics['rate_a']:.1%})
- Group B: {metrics['group_b']} (positive rate: {metrics['rate_b']:.1%})
- Demographic Parity Difference: {metrics['demographic_parity_diff']} (concerning if > 0.1)
- Disparate Impact Ratio: {metrics['disparate_impact_ratio']} (concerning if < 0.8, this is the legal 80% rule)

Cover: what the disparity means in plain English, how serious it is in real-world terms, and one mitigation recommendation.
Write in paragraphs. No bullet points. No headers. Accessible to a non-technical audience."""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def plot_group_rates(rates, demo_col, label_col):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(rates["group"].astype(str), rates["positive_rate"], color="#4c72b0", height=0.5)
    ax.axvline(x=0.8 * rates["positive_rate"].max(), color="red", linestyle="--",
               linewidth=1.2, label="80% of max rate")
    ax.set_xlabel(f"Positive rate in '{label_col}'")
    ax.set_title(f"Outcome rate by {demo_col}")
    ax.legend()
    for i, val in enumerate(rates["positive_rate"]):
        ax.text(val + 0.005, i, f"{val:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, rates["positive_rate"].max() * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig
