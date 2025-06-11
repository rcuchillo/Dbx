# Databricks Notebook: Fraud Label Maturity & Time Travel Demo

# COMMAND ----------
# Import libraries
from pyspark.sql.functions import col
from datetime import datetime
import plotly.express as px

# Paths
base_path = "/mnt/fraud_demo/transactions"
csv_path = "/mnt/fraud_demo/transactions.csv"

# COMMAND ----------
# Load CSV (Day 0 data)
df_day0 = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)
df_day0 = df_day0.withColumn("is_fraud", col("is_fraud").cast("boolean"))
df_day0.write.format("delta").mode("overwrite").save(base_path)

# COMMAND ----------
# Check initial version
spark.read.format("delta").load(base_path).groupBy("is_fraud").count().show()

# COMMAND ----------
# Simulate Day 30: Load updated CSV (external) or sample here
df_day30 = df_day0.sample(False, 1.0, seed=42)
fraud_ids_30 = [row.transaction_id for row in df_day30.sample(fraction=0.05, seed=1).collect()]
df_day30 = df_day30.withColumn("is_fraud", col("transaction_id").isin(fraud_ids_30))
df_day30.createOrReplaceTempView("updates")

spark.sql(f"""
MERGE INTO delta.`{base_path}` AS target
USING updates AS source
ON target.transaction_id = source.transaction_id
WHEN MATCHED THEN UPDATE SET target.is_fraud = source.is_fraud
""")

# COMMAND ----------
# Create a version checkpoint
spark.sql(f"DESCRIBE HISTORY delta.`{base_path}`").show(5, False)

# COMMAND ----------
# Simulate further updates (Day 60, Day 90)
for day, fraction in zip(["day_60", "day_90"], [0.10, 0.15]):
    df_update = df_day0.sample(False, 1.0, seed=42)
    fraud_ids = [row.transaction_id for row in df_update.sample(fraction=fraction, seed=1).collect()]
    df_update = df_update.withColumn("is_fraud", col("transaction_id").isin(fraud_ids))
    df_update.createOrReplaceTempView("updates")
    spark.sql(f"""
    MERGE INTO delta.`{base_path}` AS target
    USING updates AS source
    ON target.transaction_id = source.transaction_id
    WHEN MATCHED THEN UPDATE SET target.is_fraud = source.is_fraud
    """)

# COMMAND ----------
# Show version history
spark.sql(f"DESCRIBE HISTORY delta.`{base_path}`").select("version", "timestamp", "operation").show()

# COMMAND ----------
# Time travel snapshots
version_stats = []
for v in range(4):
    df = spark.read.format("delta").option("versionAsOf", v).load(base_path)
    fraud_count = df.filter("is_fraud = true").count()
    total_count = df.count()
    percent = (fraud_count / total_count) * 100 if total_count > 0 else 0
    version_stats.append({"version": v, "% fraud known": round(percent, 2)})

# COMMAND ----------
# Plotly dashboard
import pandas as pd
import plotly.express as px

pdf = pd.DataFrame(version_stats)
fig = px.line(pdf, x="version", y="% fraud known", markers=True, title="Fraud Label Maturity Over Time")
fig.show()

# COMMAND ----------
# Optional: Data quality check
latest_df = spark.read.format("delta").load(base_path)
bad_records = latest_df.filter((col("amount") <= 0) | (col("card_type").isNull()) | (col("merchant").isNull()))
bad_records.count()

# COMMAND ----------
# Display sample data with fraud flag
latest_df.select("transaction_id", "amount", "card_type", "merchant", "is_fraud").show(10, False)
