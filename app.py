# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier

# --------------------------------------------------
# Load Model and Sample Data
# --------------------------------------------------
model = joblib.load("best_xgb_model.pkl")
sample_data = pd.read_csv("synthetic_settlement_data.csv")

st.title("💹 Settlement Break Prediction Dashboard")
st.write("""
This dashboard predicts the probability of settlement failure for OTC bond trades.
Upload trade data or explore the sample dataset below.
""")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload a new trade CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = sample_data.copy()

st.write("### Sample of Current Data")
st.dataframe(data.head())

# --------------------------------------------------
# Run Predictions
# --------------------------------------------------
st.write("### Predict Settlement Failures")

# Prepare data (drop non-numeric columns if any remain)
features = data.select_dtypes(include=['number'])
pred_probs = model.predict_proba(features)[:, 1]
data["Fail_Probability"] = pred_probs

st.write("#### Predicted Probabilities")
st.dataframe(data[["counterparty", "notional_value", "Fail_Probability"]].head(10))

# --------------------------------------------------
# Counterparty Risk Visualization
# --------------------------------------------------
st.write("### Counterparty-wise Average Failure Probability")

avg_risk = data.groupby("counterparty")["Fail_Probability"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
avg_risk.plot(kind="barh", color="teal", ax=ax)
ax.set_xlabel("Average Failure Probability")
ax.set_ylabel("Counterparty")
st.pyplot(fig)

# --------------------------------------------------
# High-Risk Alerts
# --------------------------------------------------
high_risk = data[data["Fail_Probability"] > 0.7]
st.write("### ⚠️ High-Risk Trades")
if high_risk.empty:
    st.success("No high-risk trades detected.")
else:
    st.dataframe(high_risk[["counterparty", "notional_value", "Fail_Probability"]])
