import pandas as pd
import numpy as np
import random
import uuid
from faker import Faker
from datetime import datetime, timedelta
import time

# ------------------- Configuration -------------------
config = {
    "start_date": "2023-01-01",
    "end_date": "2023-01-03",
    "num_customers": 2,
    "max_cards_per_customer": 2,
    "max_txn_per_day": 5,
    "min_txn_per_day": 1,
    "min_txn_per_card": 10,
    "max_txn_per_card": 200,
    "card_type_distribution": {"debit": 0.7, "credit": 0.3},
    "entry_method_distribution": {
        "chip": 0.2, "contactless": 0.3, "online": 0.4, "swipe": 0.1
    },
    "fraud_rate": {"debit": 0.01, "credit": 0.02},
    "fraud_mcc_high_risk": ["5732", "5812", "5094"],
    "fraud_mcc_low_risk": ["5411", "5541"],
    "high_risk_txn_range": (100, 500),
    "high_risk_age_range": (20, 30),
    "online_txn_home_rate": 0.95,
    "geo_radius_miles": 100,
    "wait_per_day_seconds": 13,
    "output_file_path": "daily_transactions.csv",
    "append_mode": False
}

# ------------------- Static UK City Coordinates -------------------
city_geo = {
    "London": (51.5074, -0.1278), "Birmingham": (52.4862, -1.8904),
    "Manchester": (53.4808, -2.2426), "Glasgow": (55.8642, -4.2518),
    "Leeds": (53.8008, -1.5491), "Liverpool": (53.4084, -2.9916),
    "Bristol": (51.4545, -2.5879), "Sheffield": (53.3811, -1.4701),
    "Cardiff": (51.4816, -3.1791), "Edinburgh": (55.9533, -3.1883),
    "Leicester": (52.6369, -1.1398), "Coventry": (52.4068, -1.5197),
    "Hull": (53.7676, -0.3274), "Bradford": (53.7939, -1.7521),
    "Stoke-on-Trent": (53.0027, -2.1794), "Wolverhampton": (52.5862, -2.1286),
    "Nottingham": (52.9548, -1.1581), "Plymouth": (50.3755, -4.1427),
    "Southampton": (50.9097, -1.4044), "Reading": (51.4543, -0.9781)
}

# ------------------- MCC to Category Mapping (40+ codes) -------------------
mcc_category_map = {
    "0742": "Veterinary Services", "1520": "General Contractors", "1711": "Heating and Plumbing",
    "1731": "Electrical Contractors", "1740": "Masonry & Plastering", "1750": "Carpentry",
    "1761": "Roofing", "1771": "Concrete Work", "1799": "Special Trade Contractors",
    "2741": "Misc Publishing", "3000": "Airlines", "3351": "Railroads", "3501": "Car Rentals",
    "4111": "Commuter Transport", "4112": "Passenger Rail", "4121": "Taxi & Limo",
    "4789": "Transportation Services", "4812": "Telecom Equipment", "4814": "Telecom Services",
    "4900": "Utilities", "5045": "Computers", "5046": "Commercial Equipment",
    "5047": "Medical Equipment", "5065": "Electronic Parts", "5094": "Jewelry, Clocks",
    "5172": "Petroleum", "5200": "Home Supply Stores", "5211": "Lumber & Materials",
    "5251": "Hardware Stores", "5261": "Garden Supply", "5300": "Wholesale Clubs",
    "5311": "Department Stores", "5331": "Variety Stores", "5399": "General Merchandise",
    "5411": "Grocery Stores", "5541": "Fuel Stations", "5611": "Men’s Clothing",
    "5621": "Women’s Clothing", "5631": "Accessory Shops", "5651": "Family Clothing",
    "5661": "Shoe Stores", "5691": "Clothing Stores", "5732": "Electronics Stores",
    "5812": "Restaurants", "5814": "Fast Food", "5912": "Pharmacies", "5921": "Liquor Stores",
    "5941": "Sporting Goods", "5942": "Book Stores", "5944": "Jewelry Stores"
}

# ------------------- Generate Customers -------------------
fake = Faker("en_GB")
city_weights = np.linspace(1, 2, len(city_geo))
city_weights /= city_weights.sum()
customers = []

for _ in range(config["num_customers"]):
    name = fake.name().split()
    city = random.choices(list(city_geo.keys()), weights=city_weights)[0]
    location = city_geo[city]
    age = random.randint(18, 75)
    customers.append({
        "customer_id": str(uuid.uuid4()),
        "first_name": name[0],
        "last_name": name[1],
        "age": age,
        "city": city,
        "location": location,
        "num_products": random.randint(1, 5)
    })

customers_df = pd.DataFrame(customers)

# ------------------- Generate Cards -------------------
cards = []
for _, cust in customers_df.iterrows():
    for _ in range(random.randint(1, config["max_cards_per_customer"])):
        cards.append({
            "card_id": str(uuid.uuid4()),
            "customer_id": cust["customer_id"],
            "card_type": random.choices(["debit", "credit"],
                weights=[config["card_type_distribution"]["debit"],
                         config["card_type_distribution"]["credit"]])[0]
        })
cards_df = pd.DataFrame(cards)

# ------------------- Generate Transactions -------------------
def generate_transaction(cust, card, date):
    entry_type = random.choices(
        list(config["entry_method_distribution"].keys()),
        weights=list(config["entry_method_distribution"].values()))[0]

    amount = round(np.random.exponential(scale=50) + 1, 2)
    mcc = random.choice(list(mcc_category_map.keys()))
    category = mcc_category_map.get(mcc, "Unknown")
    merchant = fake.company()

    if entry_type in ["chip", "contactless", "swipe"]:
        if random.random() < 0.8:
            txn_location = cust["location"]
        else:
            offset_lat = np.random.uniform(-1, 1)
            offset_lon = np.random.uniform(-1, 1)
            txn_location = (
                cust["location"][0] + offset_lat,
                cust["location"][1] + offset_lon
            )
    else:
        txn_location = cust["location"] if random.random() < config["online_txn_home_rate"] else (
            np.random.uniform(50.0, 58.0), np.random.uniform(-5.0, 2.0)
        )

    fraud = False
    if ((card["card_type"] == "credit" and random.random() < config["fraud_rate"]["credit"]) or
        (card["card_type"] == "debit" and random.random() < config["fraud_rate"]["debit"])):
        if (
            mcc in config["fraud_mcc_high_risk"]
            and config["high_risk_txn_range"][0] < amount < config["high_risk_txn_range"][1]
            and config["high_risk_age_range"][0] <= cust["age"] <= config["high_risk_age_range"][1]
        ):
            fraud = True

    return {
        "transaction_id": str(uuid.uuid4()),
        "customer_id": cust["customer_id"],
        "customer_first_name": cust["first_name"],
        "customer_last_name": cust["last_name"],
        "customer_age": cust["age"],
        "card_id": card["card_id"],
        "card_type": card["card_type"],
        "entry_type": entry_type,
        "amount": amount,
        "txn_lat": txn_location[0],
        "txn_lon": txn_location[1],
        "city": cust["city"],
        "merchant": merchant,
        "mcc": mcc,
        "merchant_category": category,
        "date": date,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "fraud": fraud
    }

# ------------------- Generate Daily Transactions (No header duplication) -------------------
start = datetime.strptime(config["start_date"], "%Y-%m-%d")
end = datetime.strptime(config["end_date"], "%Y-%m-%d")
date_range = pd.date_range(start, end)
first_write = not config["append_mode"]

for date in date_range:
    print(f"Generating data for {date.date()}")
    day_txns = []

    for _, card in cards_df.iterrows():
        cust = customers_df[customers_df.customer_id == card.customer_id].iloc[0].to_dict()
        for _ in range(random.randint(config["min_txn_per_day"], config["max_txn_per_day"])):
            day_txns.append(generate_transaction(cust, card, date.date()))

    pd.DataFrame(day_txns).to_csv(
        config["output_file_path"],
        mode='a',
        header=first_write,
        index=False
    )
    first_write = False  # Prevent repeated headers
    time.sleep(config["wait_per_day_seconds"])
