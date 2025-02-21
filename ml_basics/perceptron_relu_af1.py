import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
##today we will introduce activation function and will learn each function using each perceptron
##first develop using sample small set of data
##then later develop again with real world dataset from online resources


##till now we have taken equation explicitly to develop feedforward perceptron function
# but given a data,we need to identify which function will suit the requirement to fit into feedforward network
# then we apply that function to the data set, when there re multiple layers in neural network, we could also choose
# different activation functions for different layers as per our need to train the model.
# Today we will pick one activation function and apply one any given sample dataset,(but we will learn in later exercises to understand dataset aswell and then decide the activation function)
# as of now for an understanding purpose we will walkthrough and develop relu function in our single perceptron network model,

##different activation functions
##activation function
##relu - rectified linear unit - peacewise linear  -
##sigmoid  - binary classification
##tanh
##leakyrelu
##parametric relu
##softmax
##elu
##softplus
##swish



##relu activation function simply applys max function to pick max of 0and input variable, if its less then zero then it picks zero else its straight input value
def af_relu(x):
    relu = np.maximum(0,x)
    return relu

##network with relu
##a-->relu -->output

##Simplicity: ReLU is computationally efficient because it involves simple thresholding at zero.
##Non-linearity: Despite its simplicity, ReLU introduces non-linearity, which helps neural networks learn complex patterns.
##Sparse Activation: ReLU can lead to sparse activation, meaning that only a subset of neurons are activated at any given time, which can improve the efficiency of the network.
##Avoids Vanishing Gradient Problem: Unlike sigmoid or tanh functions, ReLU helps mitigate the vanishing gradient problem, making it easier to train deep networks.

weights = 0.5
bias = 0

def forward(a,weights,bias):
    d = weights*a + bias
    return af_relu(d)

def loss(d_cap,d):
    l = np.mean((d_cap-d)**2)
    return l

def back_prop(a,d_cap,d):
    grad_l = 2*(d_cap-d)
    grad_weights = grad_l*a
    grad_bias=grad_l
    return grad_weights,grad_bias

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

def feature_scaling(data):
    data = np.array(data)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val)
    print(scaled_data)
    return scaled_data


def main():
    global weights, bias
    data = feature_scaling(read_values_from_file('file4.txt'))
    learning_rate = 0.0000001
    total_loss = 0
    # Lists to store values for plotting
    d_values = []
    d_cap_values = []
    loss_values = []
    for epoch in range(100):
        total_loss = 0
        for a, d_cap in data:
            d = forward(a, weights, bias)
            l = loss(d_cap, d)
            grad_weights, grad_bias = back_prop(a, d_cap, d)
            weights, bias = updated_parameters(weights, bias, grad_weights, grad_bias, learning_rate)
            # Store values for plotting
            d_values.append(d)
            d_cap_values.append(d_cap)
            loss_values.append(l)
            total_loss += l
        if epoch % 2 == 0:  # Adjusted to print every 10 epochs for better visibility
            print(f"Epoch {epoch}: total_loss: {total_loss}")
    # Plotting the values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(d_values[:len(data)], label='Predicted Output (d)')
    plt.plot(d_cap_values[:len(data)], label='Actual Output (d_cap)')
    plt.xlabel('Epoch')
    plt.ylabel('Output Value')
    plt.title('Predicted vs Actual Output')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_values[:len(data)], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


def test():
    global weights, bias
    # Sample dataset for testing
    sample_data = [
        [0.1, 0.2],
        [0.4, 0.5],
        [0.7, 0.8],
        [1.0, 1.1]
    ]

    # Feature scaling the sample data
    scaled_sample_data = feature_scaling(sample_data)

    # Predicting values using the trained model
    predictions = []

    for a in scaled_sample_data:
        prediction = forward(a[0], weights, bias)
        predictions.append(prediction)

    print("Predictions for the sample dataset:")
    for i in range(len(sample_data)):
        print(f"Input: {sample_data[i][0]}, Actual: {sample_data[i][1]}, Predicted: {predictions[i]}")


if __name__=="__main__":
    main()

test()


