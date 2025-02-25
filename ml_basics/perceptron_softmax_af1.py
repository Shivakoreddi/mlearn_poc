import numpy as np
import matplotlib.pyplot as plt


##in todays exercise we will experiment the softmax activation function with
# sample dataset with simple parameters and few train records
# and then we will track the loss and gradients for the inputs passed and try to
# understand the softmax function,
##initialize weight and bias
weights = np.random.rand(2, 3)  # Random weights
bias = np.random.rand(1, 3)  # Random bias


##here is as difference between both activation functions -
# ReLU is used in hidden layers to introduce non-linearity and help the network learn complex patterns.
# Softmax is used in the output layer of classification models to convert logits into probabilities.
#
# Purpose of Softmax: Purpose: Converts logits into probabilities, making them interpretable as class probabilities.
# Usage: Typically used in the output layer of classification models.
# Behavior: Normalizes the input values into a probability distribution where the sum of all probabilities is 1.
# Advantages:
# Probabilistic Interpretation: Outputs can be directly interpreted as probabilities.
# Normalization: Ensures that the outputs are between 0 and 1.
# Disadvantages:
# Exponentiation: Can lead to numerical instability if the input values are very large.,


##Logits:
# Key Points:
# Definition: Logits are the direct output of the last linear layer in a neural network.
# Range: They can take any real value, positive or negative.
# Purpose: Logits are used as inputs to activation functions (e.g., softmax) to produce probabilities.,
def softmax(x):
    # Subtract the max value from x for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def loss(y, y_cap):
    l = np.mean((y_cap - y) ** 2)
    return l

def feedforward(x,weights,bias):
    y = np.matmul(x, weights) + bias

    ##print(f"y-->{y[:5]}")
    return softmax(y)



def gradient(x, y, weights, bias):
    y_cap = feedforward(x,weights,bias)
    # Compute the gradient of the loss with respect to y_cap
    dL_dy_cap = 2 * (y_cap - y) / y.size

    # Compute the gradient of the softmax function
    dy_cap_dy = y_cap * (1 - y_cap)

    # Chain rule to get the gradient of the loss with respect to y
    dL_dy = dL_dy_cap * dy_cap_dy

    # Compute the gradient of the loss with respect to weights and bias
    grad_weights = np.matmul(x.T, dL_dy)
    grad_bias = np.sum(dL_dy, axis=0, keepdims=True)

    return grad_weights, grad_bias

def updated_parameters(weights,bias,grad_weights,grad_bias,learning_rate):
    weights -= learning_rate*grad_weights
    bias -= learning_rate*grad_bias
    return weights,bias

def read_values_from_file(filename):
    with open(filename, 'r') as file:
        # Skip the header line
        next(file)
        # Read the next line with the actual values
        lines = file.readlines()
        data = [list(map(float, line.strip().split(','))) for line in lines]
    return data

##One-hot encoding is a technique used to convert categorical data into a
# numerical format that can be used by machine learning algorithms.
# It is particularly useful for representing categorical variables as binary vectors.

##How It Works:
##Categories: Each unique category in the data is represented by a binary vector.
##Binary Vector: The vector has a length equal to the number of unique categories. Each position in the vector corresponds to a specific category.
##Encoding: For a given category, the position corresponding to that category is set to 1, and all other positions are set to 0.,


##there are other encodings as well -
# 1. label encoding - non binary, also ordinal
# 2. one -hot encoding - binary, non ordinal
# 3. ordinal encoding,
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def main():
    global weights, bias

    learning_rate = 0.000001
    # Example usage
    x = np.array([[1, 2], [3, 4], [5, 6]])  # Input data
    y = np.array([0, 1, 0])  # True labels
    y_one_hot = one_hot_encode(y, 3)  # One-hot encode the true labels

    epochs = 1000
    losses = []

    for epoch in range(epochs):
        # Forward pass
        y_cap = feedforward(x, weights, bias)

        # Compute loss
        current_loss = loss(y_one_hot, y_cap)
        losses.append(current_loss)

        # Compute gradients
        grad_weights, grad_bias = gradient(x, y_one_hot, weights, bias)

        # Update parameters
        weights, bias = updated_parameters(weights, bias, grad_weights, grad_bias, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss}")
            print(f"Actual y (one-hot):\n{y_one_hot}")
            print(f"Calculated y_cap:\n{y_cap}")

    # Plot the loss over epochs
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()


if __name__ == "__main__":
    main()


##this program contains few mistakes as listed -
##1. the loss function which is taken is Mean Squared Error function ,but for any classification model(like current program)
# with softmax as activation function the use of MSE is not a good choice, instead
# we can use cross-entropy loss(as known as log loss)
# Tmmorw we will implement with cross entropy loss function
# ,