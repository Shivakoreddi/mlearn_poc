import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# === Define MLP Model ===
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# === Parameter Counting ===
def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters      : {total_params}")
    print(f"Trainable parameters  : {trainable_params}")
    print(f"Approx. memory usage  : {trainable_params * 4 / 1024:.2f} KB")  # float32 = 4 bytes

# === Main function ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv("synthetic_patient_wait_time.csv")

    input_features = ['Department', 'Urgency_level', 'Queue_length', 'Doctor_Availability', 'Patient_Arrival_hour']
    target = 'Expected_wait_time'

    X_df = df[input_features]
    y = df[target].values

    X_encoded = pd.get_dummies(X_df, columns=['Department'], drop_first=True)
    X_encoded = X_encoded.astype(float)
    X = X_encoded.values

    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    X = X / row_norms

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32).to(device)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1).to(device)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = MLP(input_dim=X_train.shape[1]).to(device)
    count_params(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test)
        print(f"\nFinal Test MSE Loss: {test_loss.item():.4f}")
        print("First 5 Test Predictions:", y_test_pred[:5].cpu().numpy().flatten())
        print("First 5 Test Labels:", y_test[:5].cpu().numpy().flatten())

if __name__ == "__main__":
    main()
