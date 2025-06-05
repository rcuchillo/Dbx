
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
currencies = ["GBP", "EUR", "USD"]
countries = ["UK", "DE", "FR", "US"]
device_types = ["mobile", "desktop", "tablet"]
card_types = ["debit", "credit", "prepaid"]
customer_segments = ["standard", "premium", "vip"]

def generate_transaction(index):
    merchant = random.choice(merchant_ids)
    amount = np.random.exponential(scale=300)
    channel = random.choice(channels)
    country = random.choice(countries)
    currency = random.choice(currencies)
    device = random.choice(device_types)
    card = random.choice(card_types)
    segment = random.choice(customer_segments)
    customer_id = f"C{random.randint(10000, 99999)}"
    merchant_category = random.choice(["grocery", "electronics", "travel", "fashion", "gaming"])

    # Risk logic
    is_high_risk_merchant = merchant in high_risk_merchants
    high_amount = amount > 800
    channel_risk = channel == "online"
    country_risk = country in ["US"]
    segment_risk = segment == "vip"

    fraud_probability = fraud_rate
    if is_high_risk_merchant:
        fraud_probability += 0.05
    if high_amount:
        fraud_probability += 0.02
    if channel_risk:
        fraud_probability += 0.01
    if country_risk:
        fraud_probability += 0.02
    if segment_risk:
        fraud_probability += 0.01

    fraud = int(random.random() < min(fraud_probability, 0.99))

    return {
        "transaction_id": f"T{int(time.time())}_{index}",
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "merchant_id": merchant,
        "merchant_category": merchant_category,
        "amount": round(amount, 2),
        "currency": currency,
        "channel": channel,
        "country": country,
        "device_type": device,
        "card_type": card,
        "customer_segment": segment,
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
