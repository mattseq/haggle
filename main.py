import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
from catboost import CatBoostRegressor

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val).clip(0)

    score = root_mean_squared_log_error(y_val, predictions)

    return score

def create_model():
    return CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        verbose=100
    )

def train_model(model, X_train, y_train, categorical_features):
    # categorical_features = X_train.select_dtypes(
    #     include=["object", "category"]
    # ).columns.tolist()
    

    model.fit(X_train, y_train, cat_features=categorical_features)
    return model

df = pd.read_csv("data/train.csv")


X = df.drop(columns=["hours_spent", "log_hours", "hours_bucket", "project_id"])
y = df["hours_spent"]

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X[categorical_features] = df[categorical_features].fillna("missing")


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model()

model = train_model(model, X_train, y_train, categorical_features)

score = evaluate_model(model, X_val, y_val)

print(f"RMSLE: {score}")