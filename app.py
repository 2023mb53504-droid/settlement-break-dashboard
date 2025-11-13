# -----------------------------------------------------------
# 💹 Settlement Break Prediction Dashboard (Final Version)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier

st.set_page_config(page_title="Settlement Break Prediction", layout="wide")

# --------------------------------------------------
# Load model and sample dataset
# --------------------------------------------------
st.title("💹 Settlement Break Prediction Dashboard")
st.markdown("""
This dashboard predicts the probability of **settlement failure** in OTC bond trades.  
You can upload a new trade dataset or explore the sample synthetic data below.
""")

try:
    model = joblib.load("best_xgb_model.pkl")
    sample_data = pd.read_csv("synthetic_settlement_data.csv")
    st.success("✅ Model and sample dataset loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading model or data: {e}")
    st.stop()

# --------------------------------------------------
# File Upload Section
# --------------------------------------------------
st.sidebar.header("📂 Upload Trade File")
uploaded_file = st.sidebar.file_uploader("Upload your trade CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.info("Using uploaded trade data.")
else:
    data = sample_data.copy()
    st.info("Using sample synthetic dataset.")

st.write("### Preview of Current Data")
st.dataframe(data.head())

# --------------------------------------------------
# Preprocessing Function (same as model training)
# --------------------------------------------------
def preprocess_for_model(df):
    df = df.copy()

    # Date conversions
    if "trade_date" in df.columns and "settlement_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
        df["trade_age_days"] = (pd.Timestamp("today") - df["trade_date"]).dt.days
        df["time_to_settle_days"] = (df["settlement_date"] - df["trade_date"]).dt.days
        df["trade_month"] = df["trade_date"].dt.month
        df["trade_dayofweek"] = df["trade_date"].dt.dayofweek

    # Drop unused columns
    drop_cols = ["isin", "fail_reason", "trade_date", "settlement_date"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    # One-hot encoding (same as training)
    categorical_cols = [
        "counterparty", "region", "instrument_type",
        "currency", "settlement_method", "trade_channel"
    ]
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = "Unknown"

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align with model’s feature names
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    return df

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.write("### 🔍 Predict Settlement Failure Probability")

try:
    X_features = preprocess_for_model(data)
    pred_probs = model.predict_proba(X_features)[:, 1]
    data["Fail_Probability"] = np.round(pred_probs, 3)
    st.success("✅ Predictions generated successfully!")
except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# Display predictions
st.write("### Predicted Probabilities (Top 10 Trades)")
st.dataframe(data[["counterparty", "notional_value", "Fail_Probability"]].head(10))
# --------------------------------------------------
# 🧩 Identify Probable Failure Reasons
# --------------------------------------------------
def identify_fail_reason(row):
    reasons = []
    if row.get("previous_settlement_fails", 0) > 0:
        reasons.append("Counterparty has prior settlement fails")
    if row.get("notional_value", 0) > 5000000:
        reasons.append("High notional trade – possible funding mismatch")
    if row.get("time_to_settle_days", 0) < 2:
        reasons.append("Short settlement window – timing risk")
    if "FOP" in str(row.get("settlement_method", "")):
        reasons.append("Free of Payment method – missing funds risk")
    if row.get("region_EMEA", 0) == 1:
        reasons.append("Cross-border trade – regional instruction risk")
    return ", ".join(reasons) if reasons else "No apparent issue"

data["Probable_Reason"] = data.apply(identify_fail_reason, axis=1)
st.write("### Predicted Probabilities with Probable Failure Reasons")
st.dataframe(data[["counterparty", "notional_value", "Fail_Probability", "Probable_Reason"]].head(10))

# --------------------------------------------------
# Visualization – Counterparty Risk
# --------------------------------------------------
st.write("### 📊 Counterparty-wise Average Failure Probability")
try:
    avg_risk = data.groupby("counterparty")["Fail_Probability"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_risk.plot(kind="barh", color="teal", ax=ax)
    ax.set_xlabel("Average Failure Probability")
    ax.set_ylabel("Counterparty")
    ax.set_title("Counterparty Risk Heatmap")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not plot chart: {e}")

# --------------------------------------------------
# High-Risk Alert Section
# --------------------------------------------------
st.write("### ⚠️ High-Risk Trades (Fail Probability > 0.7)")
high_risk = data[data["Fail_Probability"] > 0.7]

if high_risk.empty:
    st.success("✅ No high-risk trades detected.")
else:
    st.dataframe(high_risk[["counterparty", "notional_value", "Fail_Probability"]])

st.markdown("---")
st.caption("Developed as part of an MBA FinTech project – Settlement Break Prediction & Root Cause Analysis.")

# --------------------------------------------------
# 🏦 Operational Recommendations
# --------------------------------------------------
st.write("### 🏦 Suggested Actions for Operations Team")

if high_risk.empty:
    st.success("All trades appear healthy. Continue routine monitoring.")
else:
    st.warning("High-risk trades detected. Recommended actions:")
    st.markdown("""
    - **Validate Counterparty Instructions:** Verify standing settlement instructions (SSI) for flagged trades.  
    - **Confirm Funding Availability:** Ensure funds are pre-positioned for high notional or short-window trades.  
    - **Escalate Counterparty Discrepancies:** Communicate early with counterparties showing repeated settlement fails.  
    - **Prioritize Manual Review:** Assign an operations analyst to manually verify all trades with fail probability > 0.7.  
    - **Monitor Cross-Border Settlements:** For EMEA or APAC regions, confirm time zone cut-offs and clearing timelines.
    """)

