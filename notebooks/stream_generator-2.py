
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# Output path
output_path = Path("/mnt/data/transactions_stream.csv")

# Ensure the CSV has headers
if not output_path.exists():
    df_init = pd.DataFrame(columns=["transaction_id", "score", "amount", "is_fraud", "timestamp"])
    df_init.to_csv(output_path, index=False)

def generate_transaction_batch(batch_size=200):
    fraud_ratio = 0.01
    num_fraud = int(batch_size * fraud_ratio)
    num_genuine = batch_size - num_fraud

    # Fraud scores
    fraud_scores_high = np.random.uniform(901, 1000, size=num_fraud // 2)
    fraud_scores_tail = np.random.exponential(scale=200, size=num_fraud - num_fraud // 2)
    fraud_scores_tail = np.clip(fraud_scores_tail, 300, 899)
    fraud_scores = np.concatenate([fraud_scores_high, fraud_scores_tail])

    # Genuine scores (exponential with tight cap)
    genuine_scores = np.random.exponential(scale=150, size=num_genuine)
    genuine_scores = np.clip(genuine_scores, 0, 925)

    scores = np.concatenate([fraud_scores, genuine_scores])
    is_fraud = np.array([1] * num_fraud + [0] * num_genuine)
    amounts = np.round(np.random.uniform(10, 1000, size=batch_size), 2)
    tx_ids = [f"txn_{int(time.time()*1000)%1_000_000}_{i}" for i in range(batch_size)]
    timestamp = [datetime.now().isoformat()] * batch_size

    df = pd.DataFrame({
        "transaction_id": tx_ids,
        "score": scores,
        "amount": amounts,
        "is_fraud": is_fraud,
        "timestamp": timestamp
    })

    df = df.sample(frac=1).reset_index(drop=True)
    return df

# Append every 10 seconds
print("Starting transaction stream (exponential score distribution)...")
try:
    while True:
        new_batch = generate_transaction_batch()
        new_batch.to_csv(output_path, mode="a", header=False, index=False)
        print(f"Appended {len(new_batch)} transactions at {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(10)
except KeyboardInterrupt:
    print("Stream stopped.")
