import numpy as np
import pandas as pd
import time


# ==== ACTIVATIONS ====
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# ==== INITIALIZATION ====
def initialize_parameters(layer_dims):
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters[f"W{i}"] = np.random.randn(layer_dims[i - 1], layer_dims[i]) * 0.01
        parameters[f"b{i}"] = np.zeros((1, layer_dims[i]))
    return parameters


# ==== FORWARD PROPAGATION ====
def feedforward_deep(X, parameters):
    cache = {"A0": X}
    L = len(parameters) // 2
    for l in range(1, L):
        Z = np.dot(cache[f"A{l - 1}"], parameters[f"W{l}"]) + parameters[f"b{l}"]
        A = relu(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    # Final layer with sigmoid
    ZL = np.dot(cache[f"A{L - 1}"], parameters[f"W{L}"]) + parameters[f"b{L}"]
    AL = sigmoid(ZL)
    cache[f"Z{L}"] = ZL
    cache[f"A{L}"] = AL
    return AL, cache


# ==== LOSS ====
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ==== BACKWARD PROPAGATION ====
def backprop_deep(X, y_true, parameters, cache):
    grads = {}
    m = X.shape[0]
    L = len(parameters) // 2

    dZ = cache[f"A{L}"] - y_true.reshape(-1, 1)
    for l in reversed(range(1, L + 1)):
        grads[f"dW{l}"] = (1 / m) * np.dot(cache[f"A{l - 1}"].T, dZ)
        grads[f"db{l}"] = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

        if l > 1:
            dA_prev = np.dot(dZ, parameters[f"W{l}"].T)
            dZ = dA_prev * relu_derivative(cache[f"Z{l - 1}"])

    return grads


# ==== UPDATE PARAMETERS ====
def update_parameters(parameters, grads, lr):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= lr * grads[f"dW{l}"]
        parameters[f"b{l}"] -= lr * grads[f"db{l}"]
    return parameters


# ==== MAIN FUNCTION ====
def main():
    start = time.time()
    data = pd.read_csv('synthetic_email_spam.csv')
    n = len(data)
    data['contains_viagra'] = np.random.choice([1, 0], n)
    data['contains_click'] = np.random.choice([1, 0], n)
    data['char_count'] = data['num_special_chars'] + np.random.randint(10, 1000, n)
    data['word_count'] = data['char_count'] - np.random.randint(1, 10, n)
    data['suspicious_keywords'] = data[['contains_viagra', 'contains_free', 'contains_click', 'contains_offer']].sum(
        axis=1)
    data['special_char_ratio'] = data['num_special_chars'] / (data['char_count'] + 1)
    data['link_density'] = data['num_links'] / (data['word_count'] + 1)
    data['log_char_count'] = np.log1p(data['char_count'])
    data['is_very_long'] = (data['char_count'] > 500).astype(int)
    data['log_word_count'] = np.log1p(data['word_count'])
    data['spam_score'] = data['num_links'] * data['suspicious_keywords']
    data['special_link_interaction'] = data['link_density'] * data['special_char_ratio']

    input_features = ['num_links', 'num_special_chars', 'contains_offer', 'contains_free',
                      'contains_viagra', 'contains_click', 'link_density',
                      'log_word_count', 'spam_score']
    target = "spam"


    X = data[input_features].astype(float).values
    y = data[target].values.reshape(-1, 1)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_idx = indices[:int(0.8 * len(X))]
    test_idx = indices[int(0.8 * len(X)):]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Define architecture
    input_size = X_train.shape[1]
    layer_dims = [input_size] + [16] * 99 + [1]  # 100-layer network

    parameters = initialize_parameters(layer_dims)
    learning_rate = 0.0001
    epochs = 10000

    for epoch in range(epochs):
        y_pred, cache = feedforward_deep(X_train, parameters)
        loss = binary_cross_entropy(y_train, y_pred)

        grads = backprop_deep(X_train, y_train, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    # Evaluate
    y_test_pred, _ = feedforward_deep(X_test, parameters)
    test_loss = binary_cross_entropy(y_test, y_test_pred)
    print(f"\nTest Loss: {test_loss:.4f}")
    print("Sample predictions:", y_test_pred[:5].flatten())
    print("True labels:", y_test[:5].flatten())
    print("Execution Time: {:.2f}s".format(time.time() - start))


if __name__ == '__main__':
    main()
