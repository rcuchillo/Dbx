# Databricks Notebook: Scalable Feature Engineering & Feature Store Reusability

"""
This notebook showcases how to implement scalable and reusable feature engineering pipelines using Databricks Feature Store.

## 🎯 Goal
Demonstrate how to:
- Build 30+ customer-level and merchant-level features (e.g., average transaction amount, count of transactions in past 30 days)
- Reuse logic using loops and abstraction
- Register these features in Databricks Feature Store for reuse across models

We use a realistic sample of daily transaction data.
"""

# COMMAND ----------
# 📁 Load and explore the transaction data sample
import pandas as pd
sample_path = "/mnt/data/daily_transactions.csv"
df_pd = pd.read_csv(sample_path)
df_pd.head()

# COMMAND ----------
# Convert to Spark DataFrame for large-scale processing
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(df_pd)
df = df.withColumn("transaction_date", to_date(col("transaction_date")))
df.display()

# COMMAND ----------
# 📊 Example Feature 1: Average transaction amount per customer over last 30 days
from pyspark.sql.window import Window
from pyspark.sql.functions import avg, count, sum, lit, current_date, expr

lookback_days = 30
filtered_df = df.filter(col("transaction_date") >= expr(f"date_sub(current_date(), {lookback_days})"))

# COMMAND ----------
# 🔁 Define common feature expressions
from pyspark.sql.functions import countDistinct

features = [
    ("avg_amount_30d", avg("amount")),
    ("tx_count_30d", count("transaction_id")),
    ("unique_merchants_30d", countDistinct("merchant")),
    ("total_amount_30d", sum("amount"))
]

# COMMAND ----------
# 🧠 Group by customer and compute features
grouped_features = filtered_df.groupBy("customer_id").agg(*[f.alias(name) for name, f in features])
grouped_features.display()

# COMMAND ----------
# ✨ Feature Factory: Loop to generate customer-level transaction count per N days
from pyspark.sql.functions import date_sub

feature_defs = {}

for window in [7, 14, 30]:
    for metric, func in [("avg_amount", avg("amount")), ("tx_count", count("transaction_id"))]:
        name = f"{metric}_{window}d"
        temp_df = df.filter(col("transaction_date") >= expr(f"date_sub(current_date(), {window})"))
        feature_df = temp_df.groupBy("customer_id").agg(func.alias(name))
        feature_defs[name] = feature_df

# Merge all features into one table
from functools import reduce
from pyspark.sql import DataFrame

final_features = reduce(
    lambda left, right: left.join(right, on="customer_id", how="outer"),
    feature_defs.values()
)

final_features.display()

# COMMAND ----------
# 🏷️ Register features to Feature Store (if available)

# Add more window-based features (7, 60, 90 days)
for window in [60, 90]:
    for metric, func in [
        ("avg_amount", avg("amount")),
        ("tx_count", count("transaction_id")),
        ("unique_merchants", countDistinct("merchant")),
        ("total_amount", sum("amount"))
    ]:
        name = f"{metric}_{window}d"
        temp_df = df.filter(col("transaction_date") >= expr(f"date_sub(current_date(), {window})"))
        feature_df = temp_df.groupBy("customer_id").agg(func.alias(name))
        feature_defs[name] = feature_df

# Define merchant_agg before using it in feature merging
merchant_features = df.filter(col("transaction_date") >= expr(f"date_sub(current_date(), {lookback_days})")) \
    .groupBy("customer_id", "merchant") \
    .agg(
        count("transaction_id").alias("merchant_tx_count_30d"),
        sum("amount").alias("merchant_total_amount_30d"),
        avg("amount").alias("merchant_avg_amount_30d")
    )

merchant_agg = merchant_features.groupBy("customer_id").agg(
    avg("merchant_tx_count_30d").alias("avg_tx_per_merchant_30d"),
    sum("merchant_total_amount_30d").alias("total_amount_all_merchants_30d")
)

# Merge expanded features
final_features = reduce(
    lambda left, right: left.join(right, on="customer_id", how="outer"),
    list(feature_defs.values()) + [merchant_agg]
)

final_features.display()

# Add tags and description
# Define feature metadata manually since FeatureMetadata module is unavailable
metadata = type("Metadata", (), {})()
metadata.description = "Automated features for fraud detection: customer-level metrics over multiple windows, merchant interactions."
metadata.tags = {
    "use_case": "fraud_detection",
    "owner": "ml_team@datacompany.com",
    "window_sizes": "7,14,30,60,90",
    "source": "transaction_logs"
}

from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

fs.create_table(
    name="fraud_demo.customer_features",
    primary_keys=["customer_id"],
    df=final_features,
    description=metadata.description,
    tags=metadata.tags
)

# COMMAND ----------
# 📦 Additional: Merchant-level Features and More
# 🔁 Define and compute merchant-level features
merchant_features = df.filter(col("transaction_date") >= expr(f"date_sub(current_date(), {lookback_days})")) \
    .groupBy("customer_id", "merchant") \
    .agg(
        count("transaction_id").alias("merchant_tx_count_30d"),
        sum("amount").alias("merchant_total_amount_30d"),
        avg("amount").alias("merchant_avg_amount_30d")
    )

merchant_agg = merchant_features.groupBy("customer_id").agg(
    avg("merchant_tx_count_30d").alias("avg_tx_per_merchant_30d"),
    sum("merchant_total_amount_30d").alias("total_amount_all_merchants_30d")
)

# Merge into final feature set
final_features = final_features.join(merchant_agg, on="customer_id", how="outer")

final_features.display()

from graphviz import Source
from IPython.display import display, SVG

diagram = """
digraph {
  rankdir=LR;
  "Raw Transactions" -> "Filtered by Date Window";
  "Filtered by Date Window" -> "Customer-level Features";
  "Filtered by Date Window" -> "Merchant-level Features";
  "Customer-level Features" -> "Merged Feature Table";
  "Merchant-level Features" -> "Merged Feature Table";
  "Merged Feature Table" -> "Feature Store";
}
"""

# Render safely as SVG
display(SVG(Source(diagram).pipe(format="svg")))
# 🔍 Feature Store Preview

# View schema from Feature Store
fs.get_table("fraud_demo.customer_features").to_df().display()

# ✅ Final Thoughts
"""
- You can easily loop through time windows or metrics to generate hundreds of features
- Feature Store allows these features to be reused consistently across training, inference, and monitoring
- This ensures reliability and reproducibility in production ML pipelines

Next steps:
- Add merchant-level features
- Link features to models using MLflow and Feature Store lookups
"""