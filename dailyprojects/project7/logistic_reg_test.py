import numpy as np
import time
from sklearn.metrics import confusion_matrix,classification_report


def relu(z):
    return np.maximum(0, z)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def feedforward(X, w1, b1, w2, b2):
    z1 = np.dot(X, w1) + b1          # shape: (m, hidden)
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2         # shape: (m, 1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradients(X, y_true, z1, z2, a1, y_pred, w1, w2):
    m = X.shape[0]
    error = y_pred - y_true.reshape(-1, 1)
    dz2 = error
    dw2 = (1 / m) * np.dot(a1.T, dz2)
    db2 = (1 / m) * np.sum(dz2)

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * relu_derivative(z1)
    dw1 = (1 / m) * np.dot(X.T, dz1)
    db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

    return dw1, db1, dw2, db2

def main():
    start_time = time.time()

    X = np.array([
        [1, 1], [2, 1.5], [3, 2], [4, 3], [5, 3.5],
        [6, 4], [7, 5], [8, 6], [9, 6.5], [10, 7],
        [1, 5], [2, 6], [3, 6.5], [4, 7], [5, 8]
    ])
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])

    # Corrected shapes for weights and biases
    w1 = np.random.randn(2, 1) * 0.01
    b1 = np.zeros((1, 1))  # fixed: now (1, 1) not (1,)
    w2 = np.random.randn(1, 1) * 0.01
    b2 = 0.0               # scalar is fine for output bias

    learning_rate = 0.001
    epochs = 10000

    for epoch in range(epochs):
        z1, a1, z2, y_pred = feedforward(X, w1, b1, w2, b2)
        # Convert y_pred probabilities to class labels using 0.5 threshold
        ##y_pred_class = (y_pred > 0.5).astype(int)
        for thresh in [0.3, 0.5, 0.7]:
            preds = (y_pred > thresh).astype(int)
            print(f"\nThreshold: {thresh}")
            print(confusion_matrix(y_true, preds))
        loss = binary_cross_entropy(y_true, y_pred)
        dw1, db1, dw2, db2 = gradients(X, y_true, z1, z2, a1, y_pred, w1, w2)



        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1  # fixed shape match
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2  # scalar update



        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    print(f"\nTest BCE Loss: {loss:.4f}")
    print("Sample Predictions:", y_pred[:5].flatten())
    print("True Labels:       ", y_true[:5])
    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred_class))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_true, y_pred_class))
    print("\nExecution Time: {:.2f} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    main()
