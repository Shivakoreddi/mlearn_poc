import torch
import torch.nn as nn
import torch.optim as optim
import time

# Use smaller data
X = torch.randn(100000, 1)
y = 3 * X + 2 + 0.1 * torch.randn(100000, 1)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train(device):
    model = LinearModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    X_d = X.to(device)
    y_d = y.to(device)

    start = time.time()
    for _ in range(5):
        optimizer.zero_grad()
        preds = model(X_d)
        loss = criterion(preds, y_d)
        loss.backward()
        optimizer.step()
    end = time.time()

    return end - start, loss.item()

# Test CPU
cpu_time, cpu_loss = train(torch.device("cpu"))

# Test GPU (if available)
if torch.cuda.is_available():
    gpu_time, gpu_loss = train(torch.device("cuda"))
else:
    gpu_time, gpu_loss = None, None

print("CPU Time:", cpu_time, "Loss:", cpu_loss)
if gpu_time:
    print("GPU Time:", gpu_time, "Loss:", gpu_loss)
else:
    print("GPU not available.")
