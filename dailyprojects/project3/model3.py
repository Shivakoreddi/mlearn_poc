import numpy as np
import pandas as pd
import time

# Activation functions and derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def feedforward(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    return Z1, A1, Z2, A2, Z3  # Z3 is the final output (linear)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradients(X, y_true, Z1, A1, Z2, A2, y_pred, W2, W3):
    m = X.shape[0]
    error = y_pred - y_true.reshape(-1, 1)

    dZ3 = error  # Linear output layer
    dW3 = (1 / m) * np.dot(A2.T, dZ3)
    db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def main():
    start_time = time.time()
    data = pd.read_csv('synthetic_hospital_bed_forcast.csv')
    input_features = ["Department", "day", "occupancy", "discharges", "admissions", "holiday", "flu"]
    target = "beds_needed"

    X_df = data[input_features]
    Y = data[target].values.reshape(-1, 1)
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

    input_size = X_train.shape[1]
    hidden1_size = 16
    hidden2_size = 8
    output_size = 1

    W1 = np.random.randn(input_size, hidden1_size) * 0.01
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
    b2 = np.zeros((1, hidden2_size))
    W3 = np.random.randn(hidden2_size, output_size) * 0.01
    b3 = np.zeros((1, output_size))

    learning_rate = 0.0001
    epochs = 100000

    for epoch in range(epochs):
        Z1, A1, Z2, A2, y_pred = feedforward(X_train, W1, b1, W2, b2, W3, b3)
        loss = mse_loss(y_train, y_pred)

        dW1, db1, dW2, db2, dW3, db3 = gradients(X_train, y_train, Z1, A1, Z2, A2, y_pred, W2, W3)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    _, _, _, _, test_pred = feedforward(X_test, W1, b1, W2, b2, W3, b3)
    test_loss = mse_loss(y_test, test_pred)
    print(f"\nTest MSE Loss: {test_loss:.4f}")
    print("Sample Predictions:", test_pred[:5].flatten())
    print("True Labels:       ", y_test[:5].flatten())

    print("\nExecution Time: {:.2f} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    main()
