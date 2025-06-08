# -------------------------------
# Load Libraries
# -------------------------------
from pyspark.sql.functions import *
from databricks.feature_store import FeatureStoreClient, FeatureLookup
from pyspark.sql.window import Window

# -------------------------------
# Load Data
# -------------------------------
df = spark.read.csv('/dbfs/FileStore/uk_synthetic_transactions.csv', header=True, inferSchema=True)
df = df.withColumn("timestamp", to_timestamp("timestamp"))
df = df.dropna(subset=["card_number", "amount", "timestamp", "mcc"])

# -------------------------------
# Feature 1: Fraud Ratio Per MCC
# -------------------------------
fraud_ratio_per_mcc = df.groupBy("mcc").agg(
    (sum("is_fraud") / count("*")).alias("fraud_ratio_per_mcc")
)

# Join to base DF
df = df.join(fraud_ratio_per_mcc, on="mcc", how="left")

# -------------------------------
# Feature 2: Time-Windowed Features per Card
# -------------------------------
window_specs = {
    "7d": Window.partitionBy("card_number").orderBy(col("timestamp")).rangeBetween(-7 * 86400, 0),
    "30d": Window.partitionBy("card_number").orderBy(col("timestamp")).rangeBetween(-30 * 86400, 0),
    "60d": Window.partitionBy("card_number").orderBy(col("timestamp")).rangeBetween(-60 * 86400, 0),
}

for days, w in window_specs.items():
    df = df.withColumn(f"avg_amount_{days}", avg("amount").over(w))

# Optional: use latest transaction per card for snapshot
latest_tx = df.withColumn("rn", row_number().over(Window.partitionBy("card_number").orderBy(desc("timestamp")))) \
              .filter("rn = 1") \
              .drop("rn")

# -------------------------------
# Aggregate Features per Card
# -------------------------------
features_df = latest_tx.groupBy("card_number").agg(
    count("*").alias("transaction_count"),
    avg("amount").alias("avg_amount"),
    expr("percentile_approx(amount, 0.95)").alias("p95_amount"),
    max("amount").alias("max_amount"),
    countDistinct("merchant_name").alias("unique_merchants"),
    sum("is_fraud").alias("fraud_tx_count"),
    mean("is_fraud").alias("fraud_ratio"),
    first("fraud_ratio_per_mcc").alias("fraud_ratio_per_mcc"),
    first("avg_amount_7d").alias("avg_amount_7d"),
    first("avg_amount_30d").alias("avg_amount_30d"),
    first("avg_amount_60d").alias("avg_amount_60d")
)

# -------------------------------
# Register in Feature Store
# -------------------------------
fs = FeatureStoreClient()

fs.create_table(
    name="demo_fraud_features.card_behavior_summary",
    primary_keys=["card_number"],
    df=features_df,
    description="Extended features including fraud ratio per MCC and time-windowed averages"
)

# -------------------------------
# Create Training Set (Optional)
# -------------------------------
feature_lookups = [
    FeatureLookup("demo_fraud_features.card_behavior_summary", "transaction_count", "card_number"),
    FeatureLookup("demo_fraud_features.card_behavior_summary", "avg_amount", "card_number"),
    FeatureLookup("demo_fraud_features.card_behavior_summary", "p95_amount", "card_number"),
    FeatureLookup("demo_fraud_features.card_behavior_summary", "fraud_ratio", "card_number"),
    FeatureLookup("demo_fraud_features.card_behavior_summary", "fraud_ratio_per_mcc", "card_number"),
    FeatureLookup("demo_fraud_features.card_behavior_summary", "avg_amount_7d", "card_number"),
    FeatureLookup("demo_fraud_features.card_behavior_summary", "avg_amount_30d", "card_number"),
    FeatureLookup("demo_fraud_features.card_behavior_summary", "avg_amount_60d", "card_number")
]

training_df = fs.create_training_set(
    df=df.select("card_number", "is_fraud").dropna(),
    feature_lookups=feature_lookups,
    label="is_fraud",
    exclude_columns=["card_number"]
).load_df()

display(training_df)
