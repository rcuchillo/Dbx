# 🚀 Fraud Detection with LightGBM, MLflow, and Unity Registry
# Uses previously engineered features from dbfs:/tmp/customer_features_table

from pyspark.sql.functions import col
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK

# Load feature table
df = spark.read.format("delta").load("dbfs:/tmp/customer_features_table")
df = df.dropna()

# Convert 'fraud' to 0/1 explicitly
df = df.withColumn("fraud", col("fraud").cast("int"))

# Define features
all_columns = df.columns
exclude_cols = ["customer_id", "fraud", "amount"]
features = [c for c in all_columns if c not in exclude_cols]

# Split into pandas for LightGBM
df_pd = df.select("fraud", "amount", *features).toPandas()
X = df_pd[features]
y = df_pd["fraud"]
amounts = df_pd["amount"]

# Identify feature types
categorical_features = [col for col in features if "type" in col or "merchant" in col]
continuous_features = [col for col in features if col not in categorical_features]

# Visual: Fraud label distribution
y.value_counts().plot(kind='bar', title='Fraud Distribution (0 = Genuine, 1 = Fraud)', figsize=(5,3), color="skyblue")
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
    X, y, amounts, test_size=0.3, random_state=42, stratify=y
)

# Custom evaluation at 5% decline
def evaluate_at_decline(y_true, y_pred_proba, amounts, decline_rate=0.05):
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true.iloc[sorted_idx].reset_index(drop=True)
    amounts_sorted = amounts.iloc[sorted_idx].reset_index(drop=True)
    num_to_decline = int(len(y_true) * decline_rate)
    y_declined = y_true_sorted.iloc[:num_to_decline]
    amt_declined = amounts_sorted.iloc[:num_to_decline]
    fraud_detected = y_declined.sum()
    value_detected = amt_declined[y_declined == 1].sum()
    total_fraud = y_true.sum()
    total_value = amounts[y_true == 1].sum()
    return {
        "fraud_detection_rate": fraud_detected / total_fraud if total_fraud else 0,
        "fraud_value_detection_rate": value_detected / total_value if total_value else 0
    }

# MLflow tuning
mlflow.set_experiment("/Shared/fraud_model_lgbm")

def objective(params):
    with mlflow.start_run(nested=True):
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, categorical_feature=categorical_features)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        metrics = evaluate_at_decline(y_test, preds, amt_test)
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("fraud_detection_rate", metrics["fraud_detection_rate"])
        mlflow.log_metric("fraud_value_detection_rate", metrics["fraud_value_detection_rate"])
        return {"loss": -metrics["fraud_value_detection_rate"], "status": STATUS_OK}

search_space = {
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
    "num_leaves": hp.choice("num_leaves", [15, 31, 63]),
    "max_depth": hp.choice("max_depth", [4, 6, 8, 10])
}

best_result = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=10)

# Final model training + Unity Registry
best_params = {
    "learning_rate": best_result["learning_rate"],
    "num_leaves": [15, 31, 63][best_result["num_leaves"]],
    "max_depth": [4, 6, 8, 10][best_result["max_depth"]],
}

with mlflow.start_run(run_name="final_model") as run:
    final_model = LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train, categorical_feature=categorical_features)
    preds = final_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_at_decline(y_test, preds, amt_test)
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(final_model, artifact_path="model")
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "models:/main.fraud_demo.fraud_detector")

# Visual: Feature importances
importances = final_model.feature_importances_
pd.Series(importances, index=X.columns).sort_values(ascending=False).head(15).plot(
    kind='barh', title='Top 15 Feature Importances', figsize=(8,6), color="green"
)
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()

# Summary
print(\"\"\"\\n
✅ MLflow tracks every model run, parameter, and metric
✅ Unity Model Registry version-controls your model
✅ Fraud detection rate: {:.2%}
✅ Fraud value detection rate: {:.2%}
\"\"\".format(metrics["fraud_detection_rate"], metrics["fraud_value_detection_rate"]))
