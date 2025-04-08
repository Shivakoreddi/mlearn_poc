import numpy as np
import time
import pandas as pd

# Sigmoid activation
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

def sigmoid(z):
    z = np.clip(z, -500, 500)  # prevent overflow in exp
    return 1 / (1 + np.exp(-z))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def feedforward(X, W1, b1, W2, b2, W3, b3):
    # Layer 1 (Input → Hidden1)
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # Layer 2 (Hidden1 → Hidden2)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    # Layer 3 (Hidden2 → Output)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)   # Apply sigmoid here for classification 🔥

    return Z1, A1, Z2, A2, A3

# Binary Cross Entropy loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Gradients for BCE loss with sigmoid
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

# Update weights
def updated_parameters(w, b, grad_w, grad_b, lr):
    w -= lr * grad_w
    b -= lr * grad_b
    return w, b

def main():
    start_time = time.time()
    data = pd.read_csv('synthetic_email_spam.csv')

    # Feature engineering...
    n = len(data)
    contains_viagra = np.random.choice([1, 0], n)
    contains_click = np.random.choice([1, 0], n)

    data['contains_viagra'] = contains_viagra
    data['contains_click'] = contains_click
    data['char_count'] = data['num_special_chars'] + np.random.randint(10, 1000, n)
    data['word_count'] = data['char_count'] - np.random.randint(1, 10, n)

    data['suspicious_keywords'] = data[['contains_viagra', 'contains_free', 'contains_click', 'contains_offer']].sum(axis=1)
    data['special_char_ratio'] = data['num_special_chars'] / (data['char_count'] + 1)
    data['link_density'] = data['num_links'] / (data['word_count'] + 1)

    data['log_char_count'] = np.log1p(data['char_count'])
    data['is_very_long'] = (data['char_count'] > 500).astype(int)
    data['log_word_count'] = np.log1p(data['word_count'])

    data['spam_score'] = data['num_links'] * data['suspicious_keywords']
    data['special_link_interaction'] = data['link_density'] * data['special_char_ratio']

    # input_features = ['num_links', 'num_special_chars', 'contains_offer', 'contains_free',
    #                   'email_length', 'contains_viagra', 'contains_click', 'char_count',
    #                   'word_count', 'suspicious_keywords', 'special_char_ratio', 'link_density',
    #                   'log_char_count', 'is_very_long', 'log_word_count', 'spam_score',
    #                   'special_link_interaction']
    input_features = ['num_links', 'num_special_chars', 'contains_offer', 'contains_free','contains_viagra', 'contains_click',
                      'link_density','log_word_count', 'spam_score']

    target = "spam"
    X = data[input_features].astype(float).values
    Y = data[target].values.reshape(-1, 1)  # Ensure column vector

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, y_train = X[train_idx], Y[train_idx]
    X_test, y_test = X[test_idx], Y[test_idx]

    num_features = X_train.shape[1]
    # w = np.random.randn(num_features, 1) * 0.01
    # b = 0.0
    #
    # learning_rate = 0.01
    # epochs = 100000
    # losses = []

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

    learning_rate = 0.001
    epochs = 1000000

    for epoch in range(epochs):
        Z1, A1, Z2, A2, y_pred = feedforward(X_train, W1, b1, W2, b2, W3, b3)
        loss = binary_cross_entropy(y_train, y_pred)

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
    test_loss = binary_cross_entropy(y_test, test_pred)
    print(f"\nTest BCE Loss: {test_loss:.4f}")
    print("Sample Predictions:", test_pred[:5].flatten())
    print("True Labels:       ", y_test[:5].flatten())

    print("\nExecution Time: {:.2f} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    main()
