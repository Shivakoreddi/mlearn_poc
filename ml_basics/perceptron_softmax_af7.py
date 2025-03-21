import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


##on march17

##today we will revisit ml model basic structure using softmax activation function

##initialize weight and bias
weights = np.random.rand(3, 5)  # Random weights
bias = np.random.rand(1, 5)  # Random bias


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def feedforward(x,weight,bias):
    y = np.matmul(x,weight)+bias
    return softmax(y)

##loss function
def cross_entropy_loss(y,y_cap):
    return -np.mean(np.sum(y * np.log(y_cap + 1e-9), axis=1))

def gradient(x,y,weight,bias):
    y_cap = feedforward(x, weights, bias)
    dL_dy_cap = y_cap - y
    grad_weights = np.matmul(x.T, dL_dy_cap) / x.shape[0]
    grad_bias = np.sum(dL_dy_cap, axis=0, keepdims=True) / x.shape[0]
    return grad_weights,grad_bias




def updated_parameters(weights,bias,grad_weights,grad_bias,learning_rate):
    weights -= grad_weights*learning_rate
    bias -= grad_bias*learning_rate
    return weights,bias

def one_hot_encoding(labels):
    ##get unique labels
    unique_labels = np.unique(labels)
    ##create dictionary to map onehotencoded vectors
    label_to_onehot = {label: np.eye(len(unique_labels))[i] for i, label in enumerate(unique_labels)}
    ##transform the labels to one hot encoded vectors
    encoded_labels = np.array([label_to_onehot[label] for label in labels])
    return encoded_labels


def main():
    global weights, bias
    learning_rate = 0.001
    ##goal is to pass dataset with number feature and categorical feature
    ##lets take x_train with few column values joined together and gives a classification of data on column y
    x = np.array([[0, 1, 1], [.2, 0, .5], [1, 0, 1], [1, 1, 1], [0, 0, 0.1]])
    y = np.array(['car', 'bike', 'train', 'aeroplane', 'cycle'])
    y_one_hot = one_hot_encoding(y)
    # Normalize and scale the input features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    epochs = 80000
    losses = []
    for epoch in range(epochs):
        y_cap = feedforward(x,weights,bias)
        # Compute loss
        current_loss = cross_entropy_loss(y_one_hot, y_cap)
        losses.append(current_loss)
        ##compute the gradients
        grad_weights, grad_bias = gradient(x, y_one_hot, weights, bias)
        weights, bias = updated_parameters(weights, bias, grad_weights, grad_bias, learning_rate)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss}")
            print(f"Actual y (one-hot):\n{y_one_hot}")
            print(f"Calculated y_cap:\n{y_cap}")
        # Plot the loss over epochs
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

if __name__=="__main__":
    main()

