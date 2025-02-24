import numpy as np
import matplotlib.pyplot as plt
##from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler

##todays agends : -
##train relu function w/ feedforward on real time dataset
##understand the changes in dataset values at each iteration during training. and also tract the gradient values
##of parameters


##Purpose of ReLU function :
# Purpose: Introduces non-linearity into the model, allowing it to learn complex patterns.
# Usage: Commonly used in hidden layers of neural networks.
# Behavior: Outputs the input directly if it is positive; otherwise, it outputs zero.
# Advantages:
# Simplicity: Easy to compute and implement.
# Efficiency: Helps mitigate the vanishing gradient problem, allowing for faster training.
# Disadvantages:
# Dying ReLU Problem: Neurons can become inactive and output zero for all inputs if they fall into the negative region.,

##Initialize weights and bias
weights = np.random.rand(64, 1)  # 64 features, 1 output
bias = 0.0

##relu activation function is simple ,but it helps solve vanishing gradient problems in DNN,CNN etc.
def relu(x):
    relu = np.maximum(0,x)
    print(f"relu-->{relu[:5]}")
    return relu


def feedforward(x_train,weights,bias):
    y = np.matmul(x_train,weights) + bias
    ##print(f"y-->{y[:5]}")
    return relu(y)

# Loss function (mean squared error)
def loss(y, y_cap):
    l = np.mean((y_cap - y.reshape(-1, 1)) ** 2)
    return l


def backprop(x_train,y,y_cap):
    grad_y_cap = 2 * (y_cap - y.reshape(-1, 1)) / y.size
    ##grad calculation
    grad_relu = grad_y_cap * (y_cap > 0)
    grad_weights = np.dot(x_train.T,grad_relu)
    grad_bias = np.sum(grad_relu)

    return grad_weights,grad_bias

def updated_parameters(weights,bias,grad_weights,grad_bias,learning_rate):
    weights -= learning_rate*grad_weights
    bias -= learning_rate*grad_bias
    return weights,bias




def main():
    # Load the MNIST dataset
    # Load the dataset from CSV
    global weights,bias
    learning_rate=0.01
    train_data = np.loadtxt('MNIST.csv', delimiter=',', skiprows=1)
    test_data = np.loadtxt('MNIST.csv', delimiter=',', skiprows=1)
    # Split into features and labels
    # Split into features and labels
    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    # Split into features and labels
    x_test, y_test = train_data[:, :-1], train_data[:, -1]
    # print(x_train.shape)  ##size of the training data file
    # print(y_train.shape) ##number of records or rows in file

    # Normalize the images to a range of 0 to 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # print(x_train[0])  ##each value in the train data is divided by 255.0 to make all the values under the range of 0 to 1
    # Apply Standardization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    # Print the shapes of the datasets
    print(f'Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}')
    print(f'Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}')

    # Training loop
    epochs = 5000
    for epoch in range(epochs):
        y_cap = feedforward(x_train, weights, bias)
        print(f"x_train[0] --> {x_train[0]},y_cap--->{y_cap[0]}")
        current_loss = loss(y_train, y_cap)
        #print(f"current_loss--->{current_loss}")
        grad_weights, grad_bias = backprop(x_train, y_train, y_cap)
        #print(f"grad_weights-->{grad_weights},grad_bias--->{grad_bias}")
        weights, bias = updated_parameters(weights, bias, grad_weights, grad_bias, learning_rate)
        #print(f"weights-->{grad_weights},bias--->{grad_bias}")
        #print(f'Actual_Value-->{y_train},Predicted_Value-->{y_cap},Epoch {epoch + 1}/{epochs}, Loss: {current_loss}')
        # Print the first 5 predicted and actual values
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {current_loss}')
        print(f'Predicted values: {y_cap[:20].flatten()}')
        print(f'Actual values: {y_train[:20]}')
if __name__=="__main__":
    main()