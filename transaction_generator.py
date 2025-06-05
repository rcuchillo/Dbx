
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime
import os

# ---- Configurable Parameters ----
output_path = "/dbfs/tmp/streaming_transactions"  # Update for local run if needed
os.makedirs(output_path, exist_ok=True)

rows_per_batch = 50
interval_seconds = 10
fraud_rate = 0.04
high_risk_merchants = [101, 202, 303]
merchant_ids = list(range(100, 110))
channels = ["online", "instore"]
currency = "GBP"

def generate_transaction(index):
    merchant = random.choice(merchant_ids)
    amount = np.random.exponential(scale=300)
    channel = random.choice(channels)

    # Risk logic
    is_high_risk_merchant = merchant in high_risk_merchants
    high_amount = amount > 800
    channel_risk = channel == "online"

    fraud_probability = fraud_rate
    if is_high_risk_merchant:
        fraud_probability += 0.05
    if high_amount:
        fraud_probability += 0.02
    if channel_risk:
        fraud_probability += 0.01

    fraud = int(random.random() < min(fraud_probability, 0.99))

    return {
        "transaction_id": f"T{int(time.time())}_{index}",
        "timestamp": datetime.now().isoformat(),
        "merchant_id": merchant,
        "amount": round(amount, 2),
        "channel": channel,
        "currency": currency,
        "fraud": fraud
    }

print("Starting transaction generator...")

batch_num = 0
while True:
    batch = [generate_transaction(i) for i in range(rows_per_batch)]
    df = pd.DataFrame(batch)
    file_name = f"transactions_batch_{batch_num}.json"
    df.to_json(f"{output_path}/{file_name}", orient="records", lines=True)
    print(f"Wrote {file_name} with {rows_per_batch} rows.")
    batch_num += 1
    time.sleep(interval_seconds)
