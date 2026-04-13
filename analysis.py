from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# check for group representation, is each group represented roughly equally?
def group_representation(df, demo_col):

    counts = df[demo_col].value_counts().reset_index()
    counts.columns = ["group", "count"]
    counts["pct"] = counts["count"] / len(df) * 100

    equal_share = 100 / len(counts)
    counts["underrepresented"] = counts["pct"] < (equal_share * 0.5)
    counts["expected_pct"] = equal_share

    return counts

# what demographics are missing
def missing_groups_report(df, demo_col, min_rows=30):

    counts = df[demo_col].value_counts()
    flagged = counts[counts < min_rows]
    return flagged.reset_index().rename(columns={"index": "group", demo_col: "count"})


# what fraction of each group has a pos label?
def label_rate_by_group(df, demo_col, label_col):

    le = LabelEncoder()
    df = df.copy()
    df["_label"] = le.fit_transform(df[label_col].astype(str))

    rates = df.groupby(demo_col)["_label"].mean().reset_index()
    rates.columns = ["group", "positive_rate"]
    return rates


# metrics for fairness within the dataset
def fairness_metrics(df, demo_col, label_col):

    rates = label_rate_by_group(df, demo_col, label_col)
    counts = df[demo_col].value_counts()

    # compares top two groups
    top2 = counts.nlargest(2).index.tolist()
    if len(top2) < 2:
        return None

    r = rates.set_index("group")["positive_rate"]
    r1, r2 = r.get(top2[0], 0), r.get(top2[1], 0)
    c1, c2 = counts[top2[0]], counts[top2[1]]

    # group parity
    dp_diff = abs(r1 - r2)
    # impact
    di_ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 1.0
    size_ratio = min(c1, c2) / max(c1, c2)

    return {
        "group_a": top2[0],
        "group_b": top2[1],
        "demographic_parity_diff": round(dp_diff, 4),
        "disparate_impact_ratio": round(di_ratio, 4),
        "size_ratio": round(size_ratio, 4),
        "dp_flag": dp_diff > 0.1,
        "di_flag": di_ratio < 0.8,
        "size_flag": size_ratio < 0.5,
    }

# perform the logistic regression
def train_and_audit(df, demo_col, label_col):
    # drops nans
    df = df.copy().dropna()

    le_label = LabelEncoder()
    df["_label"] = le_label.fit_transform(df[label_col].astype(str))

    feature_cols = [c for c in df.columns if c not in [demo_col, label_col, "_label"]]
    if not feature_cols:
        return None, None, None

    # define X and Y
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)

    y = df["_label"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.3, random_state=42
    )

    # intialize model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    test_df = df.loc[idx_test].copy()
    test_df["_pred"] = model.predict(X_test)

    raw_metrics = fairness_metrics(test_df, demo_col, label_col)

    test_df["_pred_str"] = test_df["_pred"].astype(str)
    model_metrics = fairness_metrics(test_df, demo_col, "_pred_str")

    return model_metrics, raw_metrics, round(accuracy * 100, 1)


# based on the logistic regression performed, output the findings
def generate_findings(rep_df, fm, model_fm=None):
    # list of findings
    findings = []
    # get list of underrep groups
    underrep = rep_df[rep_df["underrepresented"] == True]

    # output
    if not underrep.empty:
        groups = ", ".join(underrep["group"].astype(str).tolist())
        findings.append(("warning", f"Underrepresented groups detected: {groups}. "
                         "These groups appear at less than half the rate expected in a balanced dataset."))
    else:
        findings.append(("ok", "All groups appear at roughly equal rates in the dataset."))

    if fm:
        if fm["dp_flag"]:
            findings.append(("warning",
                f"Demographic parity difference of {fm['demographic_parity_diff']} detected between "
                f"'{fm['group_a']}' and '{fm['group_b']}'. "
                "A difference above 0.1 suggests the outcome is not distributed equally across groups."))
        else:
            findings.append(("ok",
                f"Demographic parity looks acceptable (difference: {fm['demographic_parity_diff']})."))

        if fm["di_flag"]:
            findings.append(("warning",
                f"Disparate impact ratio of {fm['disparate_impact_ratio']} is below the 0.8 threshold. "))
        else:
            findings.append(("ok",
                f"Disparate impact ratio of {fm['disparate_impact_ratio']} is above the 0.8 threshold."))

    if model_fm and fm:
        raw_di = fm["disparate_impact_ratio"]
        model_di = model_fm["disparate_impact_ratio"]
        if model_di < raw_di:
            findings.append(("warning",
                f"The model amplified bias: disparate impact dropped from {raw_di} (raw data) "
                f"to {model_di} (model predictions). Training on biased data made the outcome less fair."))
        else:
            findings.append(("ok",
                f"The model did not amplify bias: disparate impact stayed at {model_di} "
                f"(raw data was {raw_di})."))

    return findings