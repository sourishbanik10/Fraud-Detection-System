import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ---------------- DARK THEME STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.big-font {
    font-size:30px !important;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------
model = joblib.load("models/fraud_model.pkl")
threshold = joblib.load("models/threshold.pkl")
features = joblib.load("models/features.pkl")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>💳 Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-Powered Real-Time Risk Analysis</p>", unsafe_allow_html=True)
st.write("---")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Transaction Details")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0)
    transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
    merchant_category = st.selectbox(
        "Merchant Category",
        ["Electronics", "Food", "Grocery", "Travel"]
    )

with col2:
    st.subheader("👤 Cardholder Information")
    foreign_transaction = st.selectbox("Foreign Transaction?", ["No", "Yes"])
    location_mismatch = st.selectbox("Location Mismatch?", ["No", "Yes"])
    device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.5)
    velocity_last_24h = st.number_input("Transactions in Last 24 Hours", min_value=0)
    cardholder_age = st.number_input("Cardholder Age", min_value=18, max_value=100, value=30)

foreign_transaction = 1 if foreign_transaction == "Yes" else 0
location_mismatch = 1 if location_mismatch == "Yes" else 0

# ---------------- FEATURE ENGINEERING ----------------
is_night = 1 if transaction_hour >= 22 or transaction_hour <= 4 else 0
log_amount = np.log1p(amount)

input_dict = {
    'amount': amount,
    'transaction_hour': transaction_hour,
    'foreign_transaction': foreign_transaction,
    'location_mismatch': location_mismatch,
    'device_trust_score': device_trust_score,
    'velocity_last_24h': velocity_last_24h,
    'cardholder_age': cardholder_age,
    'is_night': is_night,
    'log_amount': log_amount
}

for category in ["Electronics", "Food", "Grocery", "Travel"]:
    col_name = f"merchant_category_{category}"
    input_dict[col_name] = 1 if merchant_category == category else 0

input_data = pd.DataFrame([input_dict])
input_data = input_data[features]

st.write("---")

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Transaction", use_container_width=True):

    probability = model.predict_proba(input_data)[0][1]

    st.markdown(
        f"<p class='big-font' style='text-align:center;'>Fraud Probability: {probability:.2%}</p>",
        unsafe_allow_html=True
    )

    st.progress(float(probability))

    # 🎯 Risk Badge Logic
    if probability < 0.30:
        st.success("🟢 LOW RISK")
    elif probability < 0.70:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK - FRAUD ALERT!")