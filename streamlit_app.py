import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO


# ============================
# LOAD ARTIFACTS
# ============================

@st.cache_resource
def load_artifacts():
    models = joblib.load("all_models.joblib")
    features = joblib.load("features_list.joblib")
    fraud_rate = joblib.load("fraud_rate.joblib")
    return models, features, fraud_rate

model_dict, feature_list, fraud_rate_global = load_artifacts()

# ============================
# MODEL SELECTION ENGINE
# ============================

def recommend_model(
    fraud_type: str,
    primary_goal: str,
    latency_ms: int,
    interpretability: str,
    data_size: str,
    fraud_rate: float
):
    fraud_type = fraud_type.lower()
    primary_goal = primary_goal.lower()
    interpretability = interpretability.lower()
    data_size = data_size.lower()

    # 1. Candidates based on fraud type
    candidates = []
    if "velocity" in fraud_type:
        candidates.extend(["XGBoost", "RandomForest"])
    elif "account takeover" in fraud_type or "ato" in fraud_type:
        candidates.extend(["XGBoost", "MLP"])
    elif "chargeback" in fraud_type or "dispute" in fraud_type:
        candidates.extend(["LogisticRegression", "GradientBoosting"])
    else:
        candidates.extend(["XGBoost", "RandomForest", "LogisticRegression"])

    # 2. Goal preference
    if "maximize recall" in primary_goal or "catch more fraud" in primary_goal:
        priority = ["XGBoost", "RandomForest", "GradientBoosting", "MLP", "LogisticRegression"]
    elif "reduce false positives" in primary_goal or "customer experience" in primary_goal:
        priority = ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost", "MLP"]
    else:
        priority = ["XGBoost", "RandomForest", "LogisticRegression", "GradientBoosting", "MLP"]

    # 3. Latency
    if latency_ms < 50:
        low_latency = ["LogisticRegression", "XGBoost", "RandomForest"]
    else:
        low_latency = ["XGBoost", "RandomForest", "MLP", "GradientBoosting", "LogisticRegression"]

    # 4. Interpretability
    if interpretability == "high":
        interpretable = ["LogisticRegression", "RandomForest", "GradientBoosting"]
    else:
        interpretable = ["XGBoost", "MLP", "RandomForest", "GradientBoosting", "LogisticRegression"]

    # 5. Data size
    if data_size == "small":
        size_pref = ["LogisticRegression", "RandomForest", "GradientBoosting"]
    else:
        size_pref = ["XGBoost", "RandomForest", "MLP", "GradientBoosting", "LogisticRegression"]

    # 6. Fraud imbalance
    if fraud_rate < 0.03:
        imbalance = ["XGBoost", "RandomForest", "GradientBoosting", "LogisticRegression"]
    else:
        imbalance = ["RandomForest", "LogisticRegression", "XGBoost", "GradientBoosting"]

    models = ["LogisticRegression", "RandomForest", "GradientBoosting", "XGBoost", "MLP"]
    scores = {m: 0 for m in models}

    # Base candidate boost
    for m in candidates:
        scores[m] += 2

    # Weighted preferences
    for weight, pref in [
        (3, priority),
        (2, low_latency),
        (2, interpretable),
        (2, size_pref),
        (2, imbalance),
    ]:
        for rank, m in enumerate(pref):
            scores[m] += weight * (len(pref) - rank)

    best_model = max(scores, key=scores.get)
    return {
        "recommended_model": best_model,
        "scores": scores
    }

# ============================
# BASIC STYLING
# ============================

st.set_page_config(
    page_title="AI CNP Fraud Detection",
    layout="wide",
    page_icon="💳"
)

st.markdown(
    """
    <style>
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: rgba(0,0,0,0.08) 0px 4px 10px;
        margin-bottom: 20px;
    }
    .title {
        font-size: 22px;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 14px;
        color: #7f8c8d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# HEADER
# ============================

st.markdown(
    """
    <div class="card">
      <div class="title">AI-based Fraud Detection for Card-Not-Present (CNP) Payments</div>
      <div class="subtitle">
        MBA Dissertation – Evaluating AI-based Fraud Detection Algorithms using the IEEE-CIS Dataset.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================
# SIDEBAR – MODEL SELECTION CONTEXT
# ============================

st.sidebar.header("⚙️ Model Selection – Business & Fraud Context")

fraud_type = st.sidebar.selectbox(
    "Fraud Type",
    ["Generic CNP Fraud", "High-Velocity Fraud", "Account Takeover (ATO)", "Chargeback / Dispute"]
)

primary_goal = st.sidebar.selectbox(
    "Primary Business Goal",
    [
        "Balance fraud catch and false positives",
        "Maximize recall, catch more fraud",
        "Reduce false positives, protect customer experience"
    ]
)

latency_ms = st.sidebar.slider("Latency Requirement (ms)", 10, 500, 100, step=10)

interpretability = st.sidebar.selectbox(
    "Interpretability Need (Regulatory)",
    ["High", "Medium"]
)

data_size = st.sidebar.selectbox(
    "Portfolio Data Size",
    ["Large", "Small"]
)

fraud_rate_slider = st.sidebar.slider(
    "Approximate Fraud Rate (%)",
    min_value=0.1,
    max_value=5.0,
    value=float(round(fraud_rate_global * 100, 2)),
    step=0.1
) / 100.0

selection = recommend_model(
    fraud_type=fraud_type,
    primary_goal=primary_goal,
    latency_ms=latency_ms,
    interpretability=interpretability,
    data_size=data_size,
    fraud_rate=fraud_rate_slider
)

recommended_model_name = selection["recommended_model"]

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Selected Model")
st.sidebar.success(recommended_model_name)
st.sidebar.caption("Chosen using the AI Model Selection Engine based on fraud type + business constraints.")

# ============================
# MAIN LAYOUT
# ============================

col_left, col_right = st.columns([1.1, 0.9])

# ---------- LEFT: Transaction Input ----------
with col_left:
    st.markdown('<div class="card"><div class="title">📝 Transaction Simulation</div>', unsafe_allow_html=True)

    input_data = {}

    if "TransactionAmt" in feature_list:
        input_data["TransactionAmt"] = st.number_input(
            "Transaction Amount (₹)",
            min_value=0.0,
            value=2500.0,
            step=50.0
        )

    if "ProductCD" in feature_list:
        input_data["ProductCD"] = st.selectbox(
            "Product Code (ProductCD)",
            options=["W", "C", "H", "R", "S"]
        )

    if "card1" in feature_list:
        input_data["card1"] = st.number_input(
            "card1 (Numeric card ID)",
            min_value=1000,
            max_value=2000,
            value=1500,
            step=1
        )

    if "card2" in feature_list:
        input_data["card2"] = st.number_input(
            "card2",
            min_value=0,
            max_value=600,
            value=200,
            step=1
        )

    if "card3" in feature_list:
        input_data["card3"] = st.number_input(
            "card3",
            min_value=0,
            max_value=200,
            value=150,
            step=1
        )

    if "card4" in feature_list:
        input_data["card4"] = st.selectbox(
            "card4 (Card Type)",
            options=["visa", "mastercard", "discover", "american express"]
        )

    if "card5" in feature_list:
        input_data["card5"] = st.number_input(
            "card5",
            min_value=0,
            max_value=300,
            value=200,
            step=1
        )

    if "card6" in feature_list:
        input_data["card6"] = st.selectbox(
            "card6 (Card Category)",
            options=["debit", "credit", "charge card", "debit or credit"]
        )

    if "addr1" in feature_list:
        input_data["addr1"] = st.number_input(
            "addr1 (Billing Address Code)",
            min_value=0,
            max_value=500,
            value=200,
            step=1
        )

    if "addr2" in feature_list:
        input_data["addr2"] = st.number_input(
            "addr2 (Address Code 2)",
            min_value=0,
            max_value=1000,
            value=300,
            step=1
        )

    if "dist1" in feature_list:
        input_data["dist1"] = st.number_input(
            "dist1 (Distance Metric)",
            min_value=0.0,
            max_value=5000.0,
            value=10.0,
            step=1.0
        )

    if "P_emaildomain" in feature_list:
        input_data["P_emaildomain"] = st.selectbox(
            "Purchaser Email Domain (P_emaildomain)",
            options=["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com"]
        )

    if "R_emaildomain" in feature_list:
        input_data["R_emaildomain"] = st.selectbox(
            "Recipient Email Domain (R_emaildomain)",
            options=["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com"]
        )

    if "DeviceType" in feature_list:
        input_data["DeviceType"] = st.selectbox(
            "Device Type",
            options=["desktop", "mobile"]
        )

    if "DeviceInfo" in feature_list:
        input_data["DeviceInfo"] = st.text_input(
            "Device Info",
            value="Windows"
        )

    # TransactionDT not shown to user – use dummy value if needed
    if "TransactionDT" in feature_list:
        input_data["TransactionDT"] = 24 * 3600  # 1-day equivalent

    threshold = st.slider(
        "Fraud Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05
    )

    run_prediction = st.button("🚀 Run Fraud Check")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: Result, Model, SHAP ----------
with col_right:
    st.markdown('<div class="card"><div class="title">📊 Fraud Prediction & Insights</div>', unsafe_allow_html=True)

    if run_prediction:
        # Build DataFrame from input
        tx_df = pd.DataFrame([input_data])

        # For engineered features used in training, we can approximate or set defaults
        if "TransactionAmt_log" in feature_list and "TransactionAmt" in tx_df.columns:
            tx_df["TransactionAmt_log"] = np.log1p(tx_df["TransactionAmt"])

        if "TransactionDT_hours" in feature_list and "TransactionDT" in tx_df.columns:
            tx_df["TransactionDT_hours"] = tx_df["TransactionDT"] / 3600

        if "TransactionHour" in feature_list and "TransactionDT_hours" in tx_df.columns:
            tx_df["TransactionHour"] = (tx_df["TransactionDT_hours"] % 24).astype(int)

        if "TransactionDay" in feature_list and "TransactionDT_hours" in tx_df.columns:
            tx_df["TransactionDay"] = (tx_df["TransactionDT_hours"] // 24 % 7).astype(int)

        # Simple defaults for frequency-encoded engineered features
        if "card1_count" in feature_list and "card1" in tx_df.columns:
            tx_df["card1_count"] = 100  # approx average

        if "P_emaildomain_freq" in feature_list and "P_emaildomain" in tx_df.columns:
            tx_df["P_emaildomain_freq"] = 100

        if "DeviceInfo_freq" in feature_list and "DeviceInfo" in tx_df.columns:
            tx_df["DeviceInfo_freq"] = 100

        # Ensure all features exist
        for col in feature_list:
            if col not in tx_df.columns:
                tx_df[col] = 0

        tx_df = tx_df[feature_list]

        chosen_model = model_dict[recommended_model_name]
        proba = chosen_model.predict_proba(tx_df)[0, 1]
        is_fraud = int(proba >= threshold)

        # ---- Prediction Card ----
        if is_fraud:
            st.error(f"🚨 Transaction flagged as FRAUD\n\nFraud Probability: **{proba:.3f}**")
        else:
            st.success(f"✅ Transaction classified as GENUINE\n\nFraud Probability: **{proba:.3f}**")

        st.write(f"**Model Used:** {recommended_model_name}")
        st.write(f"**Threshold:** {threshold:.2f}")

        # ---- Model Selection Scores ----
        st.markdown("#### 🤖 Model Selection Engine Scores")
        st.json(selection["scores"])

        # ---- SHAP Global Explanation ----
        st.markdown("#### 🔍 SHAP Global Feature Importance")
        st.caption("Shows which features are most important overall for the fraud model (IEEE-CIS training data).")
        try:
            st.image("shap_summary.png", use_column_width=True)
        except Exception:
            st.warning("SHAP summary image not found. Make sure `shap_summary.png` is in the same folder.")
    else:
        st.info("Fill in the transaction details on the left and click **Run Fraud Check**.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# FOOTER
# ============================

st.markdown(
    """
    <div class="card">
      <div class="subtitle">
        End-to-end framework: Model Selection Engine → Fraud Scoring → Explainability (SHAP) → Business & Regulatory Readiness.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)