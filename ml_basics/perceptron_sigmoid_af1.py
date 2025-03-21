import numpy as np
import matplotlib.pyplot as plt

##sigmoid is used for 2 class classification model
# we can call it as binary classification usage
# whereas softmax can be used in more than 2 classes or multi class classification models ,

def sigmoid(x):
    y_cap = 1/(1+np.exp(-x))
    return y_cap

#
# x_values = np.linspace(-10, 10, 100)  # Generate a range of x values for smooth curve
# y_values = sigmoid(x_values)
#
# plt.plot(x_values, y_values)
# plt.xlabel('x')
# plt.ylabel('sigmoid(x)')
# plt.title('Sigmoid Function')
# plt.grid(True)  # Add grid lines for better visualization
# plt.show()
#
# x = [2,4,5,-1,-2,-3,-4,6]
# for i in x:
#     print(sigmoid(i))


###*************************************************##

##For sigmoid activation functions, particularly when used in binary classification, the most common and appropriate loss function is binary cross-entropy loss (also known as logistic loss).  

##Here's why:

##Binary Cross-Entropy Loss:

##Purpose: Measures the difference between the predicted probabilities (output of the sigmoid) and the true binary labels (0 or 1).  
##Formula:
##If the true label (y) is 1: Loss = -log(p)
##If the true label (y) is 0: Loss = -log(1 - p)
##Where 'p' is the predicted probability from the sigmoid function.
##Or, in a combined form: Loss = -[y * log(p) + (1 - y) * log(1 - p)]

##Advantages:
##It's specifically designed for probabilistic outputs from the sigmoid function.
##It penalizes incorrect predictions more severely when the model is very confident (i.e., when p is close to 0 or 1).  
##It aligns well with the gradient characteristics of the sigmoid function, leading to stable and efficient training.,

def feedforward(x,weights,bias):
    y = np.dot(x,weights)+bias
    return sigmoid(y)

def binary_cross_loss(y,y_cap):
    epsilon = 1e-15
    y_cap = np.clip(y_cap, epsilon, 1 - epsilon)
    return - (y * np.log(y_cap) + (1 - y) * np.log(1 - y_cap)).mean()


def gradients(x, y, weights, bias):
    linear_output = np.dot(x, weights) + bias
    predicted_probabilities = sigmoid(linear_output)

    dL_dp = - (y / predicted_probabilities) + ((1 - y) / (1 - predicted_probabilities))
    dp_dz = predicted_probabilities * (1 - predicted_probabilities)
    dz_dw = x

    dL_dw = np.dot(dz_dw.T, (dL_dp * dp_dz).reshape(-1, 1)).reshape(weights.shape) / len(x) #fix
    dL_db = np.mean(dL_dp * dp_dz)

    return dL_dw, dL_db

def updated_parameters(weights,bias,grad_weights,grad_bias,learning_rate):
    weights -= grad_weights*learning_rate
    bias -= grad_bias*learning_rate
    return weights,bias

def main():
    # Sample Weights and Bias (for a simple linear model)
    # You would typically train these weights and bias using gradient descent
    weights = np.array([0.01, 0.005, 0.02])  # Example weights
    bias = -2.0  # Example bias

    # Realistic Sample Data (Medical Diagnosis)
    # Features: Blood Pressure, Cholesterol Level, Age
    features = np.array([
        [120, 200, 55],  # Patient 1
        [140, 240, 62],  # Patient 2
        [110, 180, 48],  # Patient 3
        [160, 280, 70],  # Patient 4
        [130, 220, 58],  # Patient 5
        [100, 170, 45],  # Patient 6
        [150, 260, 65],  # Patient 7
        [125, 210, 56],  # Patient 8
        [170, 300, 75],  # Patient 9
        [115, 190, 50]  # Patient 10
    ])

    # Labels: 1 (Disease Present), 0 (Disease Absent)
    labels = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    epochs = 100000  # Increased epochs to get better results
    learning_rate = 0.00001
    losses = []
    for epoch in range(epochs):
        predicted_probabilities = feedforward(features, weights, bias)
        loss = binary_cross_loss(labels, predicted_probabilities)
        losses.append(loss)

        grad_weights, grad_bias = gradients(features, labels, weights, bias)
        weights, bias = updated_parameters(weights, bias, grad_weights, grad_bias, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # Display Results
    predicted_probabilities = feedforward(features, weights, bias)
    binary_predictions = (predicted_probabilities > 0.5).astype(int)

    print("\nPredictions (Probabilities):\n", predicted_probabilities)
    print("\nBinary Predictions:\n", binary_predictions)
    print("\nActual Labels:\n", labels)

    # Plot Loss
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Binary Cross-Entropy Loss over Epochs')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()