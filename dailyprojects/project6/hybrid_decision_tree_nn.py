import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Activation functions
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Neural Network
def feedforward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradients(X, y_true, Z1, A1, Z2, A2, W2):
    m = X.shape[0]
    error = A2 - y_true.reshape(-1, 1)

    dZ2 = error
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

def hybrid_decision_tree_nn(data, input_features, target_col, num_trees=10, epochs=10000, lr=0.001):
    X = data[input_features].astype(float).values
    y = data[target_col].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train multiple small decision trees
    trees = []
    for i in range(num_trees):
        sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        tree = DecisionTreeClassifier(max_depth=3, random_state=i)
        tree.fit(X_train[sample_idx], y_train[sample_idx])
        trees.append(tree)

    # Create new feature space from tree outputs
    def tree_features(X, trees):
        return np.array([tree.predict(X) for tree in trees]).T

    X_train_hybrid = tree_features(X_train, trees)
    X_test_hybrid = tree_features(X_test, trees)

    # Standardize (important for NN)
    scaler = StandardScaler()
    X_train_hybrid = scaler.fit_transform(X_train_hybrid)
    X_test_hybrid = scaler.transform(X_test_hybrid)

    # NN dimensions
    input_dim = X_train_hybrid.shape[1]
    hidden_dim = 8
    output_dim = 1

    # Init weights
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros((1, output_dim))

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = feedforward(X_train_hybrid, W1, b1, W2, b2)
        loss = binary_cross_entropy(y_train, A2)

        dW1, db1, dW2, db2 = gradients(X_train_hybrid, y_train, Z1, A1, Z2, A2, W2)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    _, _, _, test_pred = feedforward(X_test_hybrid, W1, b1, W2, b2)
    test_loss = binary_cross_entropy(y_test, test_pred)
    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print("Predictions (rounded):", (test_pred[:5] > 0.5).astype(int).flatten())
    print("Actual:", y_test[:5])

# ==== Load your dataset and run ====
if __name__ == "__main__":
    df = pd.read_csv("synthetic_email_spam.csv")

    features = ['num_links', 'num_special_chars', 'contains_offer', 'contains_free']

    hybrid_decision_tree_nn(df, features, target_col="spam")
