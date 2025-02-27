import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


###todays agenda: -
##1. use softmax with all its necessary classification functions
##2. use sample test dataset with numeric and categorical feature set
##3. apply some normalization and scalling on top of datasets before doing any encoding on data
##4. after the model is prepared try some performance optimization like -applied sklearn.preprocessing - standardscaler

##initialize weight and bias
weights = np.random.rand(4, 5)  # Random weights
bias = np.random.rand(1, 5)  # Random bias

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y,y_cap):
    return -np.mean(np.sum(y * np.log(y_cap + 1e-9), axis=1))

def feedforward(x,weights,bias):
    y = np.matmul(x,weights)+bias
    return softmax(y)

def gradient(x,y,weights,bias):
    y_cap = feedforward(x,weights,bias)
    dL_dy_cap = y_cap - y
    grad_weights = np.matmul(x.T, dL_dy_cap) / x.shape[0]
    grad_bias = np.sum(dL_dy_cap, axis=0, keepdims=True) / x.shape[0]
    return grad_weights, grad_bias


def updated_parameters(weights,bias,grad_weights,grad_bias,learning_rate):
    weights -= learning_rate*grad_weights
    bias -= learning_rate*grad_bias
    return weights,bias

##One-hot encoding is a technique used to convert categorical data into a
# numerical format that can be used by machine learning algorithms.
# It is particularly useful for representing categorical variables as binary vectors.

##How It Works:
##Categories: Each unique category in the data is represented by a binary vector.
##Binary Vector: The vector has a length equal to the number of unique categories. Each position in the vector corresponds to a specific category.
##Encoding: For a given category, the position corresponding to that category is set to 1, and all other positions are set to 0.,

def one_hot_encoding(labels):
    ##get unique labels
    unique_labels = np.unique(labels)

    ##create dictionary to map onehotencoded vectors
    label_to_onehot = {label:np.eye(len(unique_labels))[i] for i,label in enumerate(unique_labels)}
    ##transform the labels to one hot encoded vectors
    encoded_labels = np.array([label_to_onehot[label] for label in labels])
    return encoded_labels

# def main():
#     global  weights,bias
#     learning_rate = 0.001
#     ##goal is to pass dataset with number feature and categorical feature
#     ##lets take x_train with few column values joined together and gives a classification of data on column y
#     x = np.array([[0,1,1,.4],[.2,0,.5,1],[1,0,1,.2],[1,1,1,0],[0,0,0.1,1]])
#     y = np.array(['apple','banana','grape','berry','orange'])
#     y_one_hot = one_hot_encoding(y)
#     # Normalize and scale the input features
#     scaler = StandardScaler()
#     x = scaler.fit_transform(x)
#     epochs = 80000
#     losses = []
#     for epoch in range(epochs):
#         y_cap = feedforward(x,weights,bias)
#         # Compute loss
#         current_loss = cross_entropy_loss(y_one_hot, y_cap)
#         losses.append(current_loss)
#
#         ##compute the gradients
#         grad_weights,grad_bias = gradient(x,y_one_hot,weights,bias)
#         weights,bias = updated_parameters(weights,bias,grad_weights,grad_bias,learning_rate)
#         if epoch % 10 == 0:
#             print(f"Epoch {epoch}, Loss: {current_loss}")
#             print(f"Actual y (one-hot):\n{y_one_hot}")
#             print(f"Calculated y_cap:\n{y_cap}")
#     # Plot the loss over epochs
#     plt.plot(losses)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Loss over Epochs')
#     plt.show()

def feature_evaluation(X_train, y_train, X_test, y_test, weights, bias):
    """
    Evaluates the feature set by training and testing the model.
    """
    y_pred_one_hot = feedforward(X_test, weights, bias)
    y_pred = np.argmax(y_pred_one_hot, axis=1)
    y_test_encoded = np.argmax(one_hot_encoding(y_test),axis=1)

    print("Feature Evaluation:")
    print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
    print("Classification Report:\n", classification_report(y_test_encoded, y_pred, target_names=np.unique(y_test)))

def main():
    global weights, bias
    learning_rate = 0.001

    x = np.array([[0, 1, 1, .4], [.2, 0, .5, 1], [1, 0, 1, .2], [1, 1, 1, 0], [0, 0, 0.1, 1]])
    y = np.array(['apple', 'banana', 'grape', 'berry', 'orange'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    y_train_one_hot = one_hot_encoding(y_train)
    num_classes = y_train_one_hot.shape[1]  # get the number of unique classes.

    # Initialize weight and bias with correct number of classes.
    global weights, bias
    weights = np.random.rand(X_train.shape[1], num_classes)
    bias = np.random.rand(1, num_classes)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    epochs = 80000
    losses = []
    for epoch in range(epochs):
        y_cap = feedforward(X_train_scaled, weights, bias)
        current_loss = cross_entropy_loss(y_train_one_hot, y_cap)
        losses.append(current_loss)
        grad_weights, grad_bias = gradient(X_train_scaled, y_train_one_hot, weights, bias)
        weights, bias = updated_parameters(weights, bias, grad_weights, grad_bias, learning_rate)
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Loss: {current_loss}")

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

    feature_evaluation(X_train_scaled, y_train, X_test_scaled, y_test, weights, bias)


if __name__ == "__main__":
    main()