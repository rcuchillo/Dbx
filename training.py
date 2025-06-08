# -------------------------------
# Libraries
# -------------------------------
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd

# -------------------------------
# Load Data from Feature Store
# -------------------------------
pandas_df = training_df.toPandas()
X = pandas_df.drop(columns=["is_fraud"])
y = pandas_df["is_fraud"]

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -------------------------------
# Set Up MLflow Experiment
# -------------------------------
mlflow.set_experiment("/Users/your.name@company.com/fraud_lgbm_demo")  # adjust path

with mlflow.start_run(run_name="LightGBM_Fraud_Model"):

    # Enable auto logging
    mlflow.lightgbm.autolog()

    # Define parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "random_state": 42,
        "verbose": -1
    }

    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=100,
        early_stopping_rounds=10
    )

    # Predict & Evaluate
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))

    # Log extra metrics
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("Accuracy", acc)

    # Register the model
    mlflow.lightgbm.log_model(model, "model", registered_model_name="Fraud_Detection_LGBM")
