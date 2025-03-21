import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Initialize weights and biases
input_size = 3
hidden_size = 5
output_size = 5

W1 = np.random.rand(input_size, hidden_size)  # Input to Hidden Weights
b1 = np.random.rand(1, hidden_size)  # Hidden Bias
W2 = np.random.rand(hidden_size, output_size)  # Hidden to Output Weights
b2 = np.random.rand(1, output_size)  # Output Bias

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float64)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def feedforward(x, W1, b1, W2, b2):
    z1 = np.matmul(x, W1) + b1
    a1 = relu(z1)
    z2 = np.matmul(a1, W2) + b2
    a2 = softmax(z2)
    return a2

def cross_entropy_loss(y, y_cap):
    return -np.mean(np.sum(y * np.log(y_cap + 1e-9), axis=1))

def gradient(x, y, W1, b1, W2, b2):
    z1 = np.matmul(x, W1) + b1
    a1 = relu(z1)
    z2 = np.matmul(a1, W2) + b2
    a2 = softmax(z2)

    dL_da2 = a2 - y
    dZ2 = dL_da2
    dW2 = np.matmul(a1.T, dZ2) / x.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / x.shape[0]

    dA1 = np.matmul(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(z1)
    dW1 = np.matmul(x.T, dZ1) / x.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / x.shape[0]

    return dW1, db1, dW2, db2

def updated_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= dW1 * learning_rate
    b1 -= db1 * learning_rate
    W2 -= dW2 * learning_rate
    b2 -= db2 * learning_rate
    return W1, b1, W2, b2

def one_hot_encoding(labels):
    unique_labels = np.unique(labels)
    label_to_onehot = {label: np.eye(len(unique_labels))[i] for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_onehot[label] for label in labels])
    return encoded_labels
def main():
    global W1, b1, W2, b2
    learning_rate = 0.001
    x = np.array([[0, 1, 1], [.2, 0, .5], [1, 0, 1], [1, 1, 1], [0, 0, 0.1]])
    y = np.array(['car', 'bike', 'train', 'aeroplane', 'cycle'])
    y_one_hot = one_hot_encoding(y)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    epochs = 80000
    losses = []
    for epoch in range(epochs):
        y_cap = feedforward(x, W1, b1, W2, b2)
        current_loss = cross_entropy_loss(y_one_hot, y_cap)
        losses.append(current_loss)
        dW1, db1, dW2, db2 = gradient(x, y_one_hot, W1, b1, W2, b2)
        W1, b1, W2, b2 = updated_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss}")
            #print(f"Actual y (one-hot):\n{y_one_hot}")
            #print(f"Calculated y_cap:\n{y_cap}")
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

    # Display Predictions
    y_cap = feedforward(x, W1, b1, W2, b2)
    predictions = np.argmax(y_cap, axis=1)
    unique_labels = np.unique(y)
    predicted_labels = [unique_labels[i] for i in predictions]
    print("\nPredictions:")
    for i in range(len(x)):
        print(f"Input: {x[i]}, Actual: {y[i]}, Predicted: {predicted_labels[i]}")

if __name__ == "__main__":
    main()