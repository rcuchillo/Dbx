# -------------------------------
# Libraries
# -------------------------------
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

# -------------------------------
# Load Data
# -------------------------------
pandas_df = training_df.toPandas()
X = pandas_df.drop(columns=["is_fraud"])
y = pandas_df["is_fraud"]
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -------------------------------
# Define Hyperopt Search Space
# -------------------------------
search_space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'num_leaves': hp.choice('num_leaves', [15, 31, 63, 127]),
    'max_depth': hp.choice('max_depth', [3, 5, 7, 10]),
    'min_child_weight': hp.uniform('min_child_weight', 1, 10),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'verbose': -1
}

# -------------------------------
# Track Results
# -------------------------------
mlflow.set_experiment("/Users/your.name@company.com/fraud_lgbm_hyperopt")
results = []
best_model = None
best_auc = -np.inf

# -------------------------------
# Define Objective Function
# -------------------------------
def objective(params):
    global best_model, best_auc

    with mlflow.start_run(nested=True):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            early_stopping_rounds=10,
            verbose_eval=False
        )

        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)

        # Log metrics and parameters
        mlflow.log_params({k: float(v) if isinstance(v, np.generic) else v for k, v in params.items()})
        mlflow.log_metric("val_auc", auc)

        results.append({**params, "val_auc": auc})

        # Track best model
        if auc > best_auc:
            best_auc = auc
            best_model = model

        return {'loss': -auc, 'status': STATUS_OK}

# -------------------------------
# Run Hyperopt
# -------------------------------
with mlflow.start_run(run_name="Hyperopt_LGBM_Tuning") as parent_run:
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,
        trials=trials
    )

    # Log best params and metrics to parent run
    mlflow.log_params(best_result)
    mlflow.log_metric("best_val_auc", best_auc)

    # -------------------------------
    # Register Best Model Automatically
    # -------------------------------
    mlflow.lightgbm.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="Fraud_Detection_LGBM_Optimized"
    )

# -------------------------------
# Export Summary as Report
# -------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("val_auc", ascending=False)
results_df.to_csv("/dbfs/FileStore/hyperopt_lgbm_results.csv", index=False)
display(results_df.head(10))
