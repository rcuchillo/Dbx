
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Monitoring Dashboard", layout="wide")

st.title("📊 Real-Time Fraud Monitoring Dashboard")
st.caption("Live metrics from transactions_stream.csv (auto-refreshes every 10 seconds)")

@st.cache_data(ttl=10)
def load_data():
    try:
        df = pd.read_csv("/mnt/data/transactions_stream.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("Waiting for data...")
    st.stop()

# Metrics
total = len(df)
fraud_total = df["is_fraud"].sum()
genuine_total = total - fraud_total
scored_tx = df[df["score"].notnull()]
missed_fraud = df[(df["is_fraud"] == 1) & (df["score"] < 500)]

st.metric("Total Transactions", f"{total:,}")
st.metric("Fraud Transactions", f"{fraud_total:,}")
st.metric("Missed Frauds (<500)", f"{len(missed_fraud):,}")

# Score thresholds
st.subheader("🚦 Fraud Detection Rate by Score Threshold")
thresholds = [900, 850, 800, 750, 700]
rows = []
for threshold in thresholds:
    flagged = df[df["score"] > threshold]
    fraud_flagged = flagged[flagged["is_fraud"] == 1]
    value_flagged = fraud_flagged["amount"].sum()
    total_fraud_value = df[df["is_fraud"] == 1]["amount"].sum()
    detection_rate = len(fraud_flagged) / fraud_total if fraud_total else 0
    value_rate = value_flagged / total_fraud_value if total_fraud_value else 0
    rows.append((threshold, len(flagged), detection_rate, value_rate))

score_df = pd.DataFrame(rows, columns=["Score >", "Flagged Tx", "Fraud Detection Rate", "Fraud Value Rate"])
st.dataframe(score_df.style.format({"Fraud Detection Rate": "{:.1%}", "Fraud Value Rate": "{:.1%}"}))

# Score Distribution
st.subheader("📉 Score Distribution")
fig, ax = plt.subplots(figsize=(8, 3))
df["score"].hist(bins=50, ax=ax, color='skyblue')
ax.set_title("Score Distribution")
ax.set_xlabel("Score")
ax.set_ylabel("Transaction Count")
st.pyplot(fig)

# Live table (last 50 records)
st.subheader("🕒 Recent Transactions")
st.dataframe(df.sort_values("timestamp", ascending=False).head(50), use_container_width=True)
