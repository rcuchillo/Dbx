import pandas as pd
import numpy as np
import random
import string
import time
from datetime import datetime
import threading
import os

# -------------------------------
# Configuration
# -------------------------------
BATCH_SIZE = 50
INTERVAL_SECONDS = 10
ITERATIONS = 10
OUTPUT_FILE = 'uk_synthetic_transactions.csv'

# -------------------------------
# Reference Data
# -------------------------------
MCC_LIST = ['5411', '5812', '5732', '5999', '4111']
MERCHANT_NAMES = ['Tesco', 'Sainsbury’s', 'Argos', 'Boots', 'EasyJet']
UK_CITIES = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 'Bristol', 'Liverpool']
CITY_TO_IP = {
    'London': '51.5074,-0.1278',
    'Manchester': '53.4808,-2.2426',
    'Birmingham': '52.4862,-1.8904',
    'Leeds': '53.8008,-1.5491',
    'Glasgow': '55.8642,-4.2518',
    'Bristol': '51.4545,-2.5879',
    'Liverpool': '53.4084,-2.9916'
}
CARD_TYPES = ['credit', 'debit']
ENTRY_MODES = ['chip', 'swipe', 'tap', 'manual', 'online']
AUTH_CODES = ['00', '05', '12', '14', '51', '91']
DEVICE_TYPES = ['mobile', 'desktop', 'tablet']
CURRENCY = 'GBP'
COUNTRY = 'GB'


# -------------------------------
# Transaction Generator
# -------------------------------
def generate_transaction(transaction_id):
    city = random.choices(UK_CITIES, weights=[0.35 if c == 'London' else 0.1 for c in UK_CITIES])[0]
    amount = round(random.uniform(5.0, 500.0), 2)
    entry_mode = random.choice(ENTRY_MODES)
    mcc = random.choice(MCC_LIST)
    merchant = random.choice(MERCHANT_NAMES)
    auth_code = random.choices(AUTH_CODES, weights=[0.8, 0.05, 0.05, 0.05, 0.03, 0.02])[0]
    card_number = ''.join(random.choices(string.digits, k=16))
    expiry = f"{random.randint(1, 12):02d}/{random.randint(25, 30)}"
    email = f"user{random.randint(1000, 9999)}@example.co.uk"
    device = random.choice(DEVICE_TYPES)
    geo_latlon = CITY_TO_IP[city]
    ip_address = f"UK-{city} ({geo_latlon})"
    merchant_location = f"{city}, {COUNTRY}"

    # Fraud rule with London bias
    is_fraud = (
            entry_mode == 'manual' or
            amount > 300 or
            mcc not in ['5411', '5812'] or
            (city == 'London' and random.random() < 0.2)
    )

    return {
        'transaction_id': transaction_id,
        'timestamp': datetime.utcnow().isoformat(),
        'amount': amount,
        'currency': CURRENCY,
        'mcc': mcc,
        'merchant_name': merchant,
        'merchant_location': merchant_location,
        'entry_mode': entry_mode,
        'auth_code': auth_code,
        'card_number': card_number,
        'card_expiry': expiry,
        'card_type': random.choice(CARD_TYPES),
        'issuer_country': COUNTRY,
        'email': email,
        'ip_address': ip_address,
        'device_type': device,
        'is_fraud': int(is_fraud),
        'city': city,
        'lat_lon': geo_latlon
    }


# -------------------------------
# Batch Generator
# -------------------------------
def generate_batch(batch_size, transaction_id_base, file_path, write_header=False):
    data = [generate_transaction(transaction_id_base + i) for i in range(batch_size)]
    df = pd.DataFrame(data)

    df.to_csv(file_path, mode='a', header=write_header, index=False)


# -------------------------------
# Background Runner
# -------------------------------
def run_background_generation(batch_size=50, interval_sec=10, iterations=10,
                              output_file='uk_synthetic_transactions.csv'):
    transaction_id_base = int(time.time() * 1000)

    # Ensure file is deleted before fresh run
    if os.path.exists(output_file):
        os.remove(output_file)

    for i in range(iterations):
        print(f"[{datetime.utcnow().isoformat()}] Generating batch {i + 1}/{iterations}...")
        write_header = i == 0  # Only first batch writes header
        generate_batch(batch_size, transaction_id_base + i * batch_size, output_file, write_header)
        time.sleep(interval_sec)

    print("Data generation complete.")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("Starting UK synthetic transaction generator...")
    thread = threading.Thread(
        target=run_background_generation,
        kwargs={
            'batch_size': BATCH_SIZE,
            'interval_sec': INTERVAL_SECONDS,
            'iterations': ITERATIONS,
            'output_file': OUTPUT_FILE
        }
    )
    thread.start()
