import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Dashboard", layout="wide")

st.title("📊 Fraud Statistics Dashboard")

# Load dataset
df = pd.read_csv("../data/processed/transactions.csv")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", len(df))

with col2:
    st.metric("Total Fraud Cases", df['is_fraud'].sum())

with col3:
    fraud_rate = df['is_fraud'].mean() * 100
    st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}%")

st.write("---")

# Fraud Distribution
st.subheader("Fraud vs Non-Fraud Distribution")

fig, ax = plt.subplots()
df['is_fraud'].value_counts().plot(kind='bar', ax=ax)
ax.set_xticklabels(["Legitimate", "Fraud"], rotation=0)
st.pyplot(fig)

# Fraud by Merchant Category
st.subheader("Fraud by Merchant Category")

fig2, ax2 = plt.subplots()
df.groupby("merchant_category")["is_fraud"].sum().plot(kind='bar', ax=ax2)
st.pyplot(fig2)