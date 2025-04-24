import numpy as np
import time
import pandas as pd

# Sigmoid activation
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

def sigmoid(z):
    z = np.clip(z, -500, 500)  # prevent overflow in exp
    return 1 / (1 + np.exp(-z))

# Feedforward with linear layer and sigmoid output
def feedforward(X, w, b):
    return sigmoid(np.dot(X, w) + b)

# Binary Cross Entropy loss
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Gradients for BCE loss with sigmoid
def gradients(X, y_true, y_pred):
    m = X.shape[0]
    error = y_pred - y_true.reshape(-1, 1)  # Ensure shape (m, 1)
    dw = (1 / m) * np.dot(X.T, error)
    db = (1 / m) * np.sum(error)
    return dw, db

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
    w = np.random.randn(num_features, 1) * 0.01
    b = 0.0

    learning_rate = 0.01
    epochs = 1000000
    losses = []

    for epoch in range(epochs):
        y_pred = feedforward(X_train, w, b)
        loss = binary_cross_entropy(y_train, y_pred)
        losses.append(loss)

        dw, db = gradients(X_train, y_train, y_pred)
        w, b = updated_parameters(w, b, dw, db, learning_rate)

        if epoch % 1000 == 0:
            print(f"\nEpoch {epoch + 1}")
            print(f"Train Loss: {loss:.4f}")
            print("First 5 Predictions:", y_pred[:5].flatten())
            print("First 5 True Labels:", y_train[:5].flatten())

            y_test_pred = feedforward(X_test, w, b)
            test_loss = binary_cross_entropy(y_test, y_test_pred)
            print(f"Test Loss: {test_loss:.4f}")
            print("Test Predictions:", y_test_pred[:5].flatten())
            print("Test Labels:     ", y_test[:5].flatten())

    print("\n Total Training Time: {:.2f} sec".format(time.time() - start_time))

if __name__ == "__main__":
    main()
