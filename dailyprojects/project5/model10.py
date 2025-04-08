import numpy as np
import pandas as pd
import time

# Activation Functions
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Feedforward for 10-layer FFNN
def feedforward(X, weights, biases):
    activations = [X]
    Zs = []
    for i in range(len(weights) - 1):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        A = relu(Z)
        Zs.append(Z)
        activations.append(A)
    # Output layer
    Z_final = np.dot(activations[-1], weights[-1]) + biases[-1]
    A_final = sigmoid(Z_final)
    Zs.append(Z_final)
    activations.append(A_final)
    return Zs, activations

# Loss

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Gradients for 10-layer FFNN

def gradients(X, y_true, weights, biases, Zs, activations):
    m = X.shape[0]
    grads_w = [0] * len(weights)
    grads_b = [0] * len(biases)

    dZ = activations[-1] - y_true
    for i in reversed(range(len(weights))):
        grads_w[i] = np.dot(activations[i].T, dZ) / m
        grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m
        if i > 0:
            dA = np.dot(dZ, weights[i].T)
            dZ = dA * relu_derivative(Zs[i - 1])
    return grads_w, grads_b

# Update Weights
def update_params(weights, biases, grads_w, grads_b, lr):
    for i in range(len(weights)):
        weights[i] -= lr * grads_w[i]
        biases[i] -= lr * grads_b[i]
    return weights, biases


def main():
    start_time = time.time()
    data = pd.read_csv('synthetic_email_spam.csv')
    n = len(data)
    data['contains_viagra'] = np.random.choice([1, 0], n)
    data['contains_click'] = np.random.choice([1, 0], n)
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

    input_features = ['num_links', 'num_special_chars', 'contains_offer', 'contains_free',
                      'contains_viagra', 'contains_click', 'link_density',
                      'log_word_count', 'spam_score']
    target = "spam"

    X = data[input_features].astype(float).values
    Y = data[target].values.reshape(-1, 1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, y_train = X[train_idx], Y[train_idx]
    X_test, y_test = X[test_idx], Y[test_idx]

    # Initialize 10-layer NN
    layer_sizes = [X_train.shape[1], 32, 64, 32, 16, 8, 16, 8, 4, 2, 1]
    weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]) for i in range(10)]
    biases = [np.zeros((1, layer_sizes[i+1])) for i in range(10)]

    learning_rate = 0.1
    epochs = 10000
    for epoch in range(epochs):
        Zs, activations = feedforward(X_train, weights, biases)
        loss = binary_cross_entropy(y_train, activations[-1])
        grads_w, grads_b = gradients(X_train, y_train, weights, biases, Zs, activations)
        weights, biases = update_params(weights, biases, grads_w, grads_b, learning_rate)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    _, activations_test = feedforward(X_test, weights, biases)
    test_loss = binary_cross_entropy(y_test, activations_test[-1])
    print(f"\nTest BCE Loss: {test_loss:.6f}")
    print("Sample Predictions:", activations_test[-1][:5].flatten())
    print("True Labels:       ", y_test[:5].flatten())
    print("\nExecution Time: {:.2f} sec".format(time.time() - start_time))

if __name__ == "__main__":
    main()
