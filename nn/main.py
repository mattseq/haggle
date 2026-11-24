import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LR = 0.0001
WEIGHT_DECAY = 0.0001
# DROPOUT = 0.4
BATCH_SIZE = 64
EPOCHS = 300

class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

def rmsle(predictions, targets):
    predictions = torch.clamp(predictions, min=0)
    targets = torch.clamp(targets, min=0)

    return torch.sqrt(torch.mean(
        (torch.log1p(predictions) - torch.log1p(targets))**2
    ))

def train(model, X_train, y_train, X_val, y_val):

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr = LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        model.train()

        # used to get average loss over epoch
        # epoch_loss_sum = 0
        # num_batches = 0

        for X_batch, y_batch in train_loader:

            # lowk dont know if i need this but i dont feel like lighting my cpu on fire
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = rmsle(outputs, y_batch)

            # epoch_loss_sum += loss.item()

            # num_batches += 1

            loss.backward()
            optimizer.step()

        # epoch_loss_avg = epoch_loss_sum / num_batches

        train_loss = evaluate(model, X_train, y_train)
        val_loss = evaluate(model, X_val, y_val)

        if epoch % 10 == 0:
            # print(f"Epoch {1 + epoch}/{EPOCHS}: {epoch_loss_avg:.4f}")
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print(f" - Train: {train_loss:.4f}")
            print(f" - Val: {val_loss:.4f}")

    return model

def evaluate(model, X_val, y_val):
    model.eval()

    with torch.no_grad():
        outputs = model(X_val)
        loss = rmsle(outputs, y_val)

    return loss.item()


df = pd.read_csv("data/train.csv")


X = df.drop(columns=["hours_spent", "log_hours", "hours_bucket", "project_id"])
y = df["hours_spent"]

features = [
    "active_commit_days",
    "project_span_days",
    "num_commits",
    "num_commit_authors",
    "num_merge_commits",
    "lines_added_total",
    "lines_deleted_total",
    "lines_of_code",
    "num_source_files",
    "num_directories",
    "max_dir_depth",
    "max_lines_in_one_commit",
    "pct_generic_commit_msgs",
    # "churn_ratio",
    "readme_word_count",
]

X = df[features]

print("nans in X:", X.isna().sum())
print("nans in y:", y.isna().sum())

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_val = scalar.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)

model = Model(len(X.columns)).to(device)

model = train(model, X_train, y_train, X_val, y_val)