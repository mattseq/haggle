import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error


df = pd.read_csv("data/train.csv")

X = df.drop(columns=["hours_spent", "log_hours", "hours_bucket", "project_id"])
y = df["hours_spent"]

categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print(categorical_features)