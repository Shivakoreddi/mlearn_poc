import numpy as np
import pandas as pd
import time

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Feedforward Neural Network Layers
def feedforward_nn(X, w1, b1, w2, b2):
    # Layer 1: Hidden layer with ReLU
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)

    # Layer 2: Output layer (Linear)
    z2 = np.dot(a1, w2) + b2
    return z1, a1, z2  # return hidden layer activations for backprop

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradients(X, y, z1, a1, y_pred, w2):
    m = X.shape[0]
    error = y_pred - y.reshape(-1, 1)

    dz2 = error / m
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2)

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
    dw1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0)

    return dw1, db1, dw2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, lr):
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    return w1, b1, w2, b2

def main():
    start_time = time.time()

    data = pd.read_csv('synthetic_hospital_bed_forcast.csv')
    input_features = ["Department", "day", "occupancy", "discharges", "admissions", "holiday", "flu"]
    target = "beds_needed"

    X_df = data[input_features]
    Y = data[target].values

    X_encoded = pd.get_dummies(X_df, columns=['Department', 'day'], drop_first=True)
    X = X_encoded.astype(float).values
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    X = X / row_norms

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    X_train, y_train = X[train_idx], Y[train_idx]
    X_test, y_test = X[test_idx], Y[test_idx]

    num_features = X_train.shape[1]
    hidden_units = 16  # Size of hidden layer

    # Initialize parameters
    w1 = np.random.randn(num_features, hidden_units) * 0.01
    b1 = np.zeros(hidden_units)
    w2 = np.random.randn(hidden_units, 1) * 0.01
    b2 = 0.0

    lr = 0.0001
    epochs = 100000

    for epoch in range(epochs):
        z1, a1, y_pred = feedforward_nn(X_train, w1, b1, w2, b2)
        loss = mse_loss(y_train, y_pred.flatten())

        dw1, db1, dw2, db2 = gradients(X_train, y_train, z1, a1, y_pred, w2)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, lr)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} - Loss: {loss:.4f}")

    # Evaluate on test set
    _, _, y_test_pred = feedforward_nn(X_test, w1, b1, w2, b2)
    test_loss = mse_loss(y_test, y_test_pred.flatten())
    print(f"\nFinal Test MSE: {test_loss:.4f}")
    print(f"Execution Time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
