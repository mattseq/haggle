import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
from catboost import CatBoostRegressor
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X_val, y_val):
    predictions = np.expm1(model.predict(X_val)).clip(0)

    score = root_mean_squared_log_error(y_val, predictions)

    return score

def create_model():
    return CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=4,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=100,
        random_seed=42,
        l2_leaf_reg=5,
        random_strength=1
    )

def train_model(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train,
        np.log1p(y_train),
        eval_set=(X_val, np.log1p(y_val)),
        early_stopping_rounds=200,
    )
    return model

df = pd.read_csv("data/train.csv")


X = df.drop(columns=["hours_spent", "log_hours", "hours_bucket", "project_id"])
X = df[["active_commit_days", "description_word_count", "lines_added_total", "median_lines_per_commit", "project_span_days", "lines_of_code", "pct_code_in_first_commit", "num_source_files", "pct_commits_weekend", "avg_file_length", "num_commits", "lines_deleted_total", "num_dependencies"]]
y = df["hours_spent"]

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X = X.drop(columns=categorical_features)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

model = create_model()

model = train_model(model, X_train, y_train, X_val, y_val)

importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

print(importance)

train_score = evaluate_model(model, X_train, y_train)
val_score = evaluate_model(model, X_val, y_val)

print("Train RMSLE:", train_score)
print("Validation RMSLE:", val_score)

baseline = np.full(len(y_val), y_train.median())

print("Baseline:", root_mean_squared_log_error(y_val, baseline))