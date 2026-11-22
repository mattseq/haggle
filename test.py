import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error


df = pd.read_csv("train.csv")

X = df.drop(columns=["hours_spent", "log_hours", "hours_bucket"])
y = df["hours_spent"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_val.tail())

predictions = [y_train.mean()] * len(y_val)

score = root_mean_squared_log_error(y_val, predictions)

print(f"RMSLE: {score}")