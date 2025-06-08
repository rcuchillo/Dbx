# -------------------------------
# Libraries
# -------------------------------
import mlflow
import mlflow.lightgbm
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np

# -------------------------------
# Prepare Data
# -------------------------------
pandas_df = training_df.toPandas()
X = pandas_df.drop(columns=["is_fraud"])
y = pandas_df["is_fraud"]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -------------------------------
# Search Space for LightGBM
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
# Objective Function
# -------------------------------
def objective(params):
    with mlflow.start_run(nested=True):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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

        mlflow.log_params({k: v for k, v in params.items() if isinstance(v, (int, float))})
        mlflow.log_metric("val_auc", auc)

        return {'loss': -auc, 'status': STATUS_OK}

# -------------------------------
# Run Hyperopt
# -------------------------------
mlflow.set_experiment("/Users/your.name@company.com/fraud_lgbm_hyperopt")

with mlflow.start_run(run_name="Hyperopt_LGBM_Tuning"):
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,
        trials=trials
    )
    mlflow.log_params(best_result)
