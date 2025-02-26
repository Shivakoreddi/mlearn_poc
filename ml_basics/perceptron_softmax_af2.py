import numpy as np
import matplotlib.pyplot as plt


##todays agenda is to experiment the softmax activation function with MNIST dataset and predict the
# classes of images
# Also apply one-hot encoding at the end ,

##data preparation -
##feature identify, selection,cleaning/normalization, encoding
##initialize weight and bias

# Initialize weight and bias
weights = np.random.rand(64, 10)  # Random weights
bias = np.random.rand(1, 10)  # Random bias

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.astype(int)]

##classification model's loss function is cross-entropy
def cross_entropy_loss(y, y_cap):
    return -np.mean(np.sum(y * np.log(y_cap + 1e-9), axis=1))


def feedforward(x,weights,bias):
    y = np.matmul(x,weights)+bias
    return softmax(y)

def gradient(x,y,weights,bias):
    y_cap = feedforward(x, weights, bias)
    dL_dy_cap = y_cap - y
    grad_weights = np.matmul(x.T, dL_dy_cap) / x.shape[0]
    grad_bias = np.sum(dL_dy_cap, axis=0, keepdims=True) / x.shape[0]
    return grad_weights, grad_bias

def updated_parameters(weights,bias,grad_weights,grad_bias,learning_rate):
    weights -= learning_rate*grad_weights
    bias -= learning_rate*grad_bias
    return weights,bias

def main():
    global weights, bias
    learning_rate = 0.0001
    train_data = np.loadtxt('MNIST.csv', delimiter=',', skiprows=1)
    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    # Normalize the images to a range of 0 to 1
    x_train = x_train / 255.0

    # Print the shapes of the datasets
    print(f'Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}')
    # Training loop

    y_one_hot = one_hot_encode(y_train, 10)  # One-hot encode the true labels
    epochs = 10000
    losses = []

    for epoch in range(epochs):
        # Forward pass
        y_cap = feedforward(x_train, weights, bias)
        # Compute loss
        current_loss = cross_entropy_loss(y_one_hot, y_cap)
        losses.append(current_loss)


        # Compute gradients
        grad_weights, grad_bias = gradient(x_train, y_one_hot, weights, bias)

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



if __name__=="__main__":
    main()



##try to understnd the features in the above dataset,before moving forward
# this is to make sure we are understanidng the real time dataset before training the model
# otherwise we may not be able to create or identify performance/optimization improvment measures
# there are different types of features like
# 1. numerical features - age,marks etc
# 2. categorical featurs - like subject,class ,grade etc.
#
# othertopics to handle these features
# 1. feature crossing
# 2. hashing
# 3. embedding - powrfull way to represent large vocabolary
#
# 1. Numerical Features
# Statistical Measures: Mean, median, standard deviation, variance, etc.
# Aggregations: Sum, count, min, max, etc.
# Ratios: Ratios between different numerical features.

# 2. Categorical Features
# Frequency Encoding: Count of each category.
# One-Hot Encoding: Binary columns for each category.
# Label Encoding: Assigning a unique integer to each category.

# 3. Text Features
# Bag of Words: Count of each word in the text.
# TF-IDF: Term frequency-inverse document frequency.
# Word Embeddings: Vector representations of words (e.g., Word2Vec, GloVe).
# N-grams: Sequences of n words or characters.

# 4. Date and Time Features
# Date Components: Year, month, day, hour, minute, second.
# Day of the Week: Monday, Tuesday, etc.
# Week of the Year: Week number within the year.
# Season: Winter, spring, summer, fall.

# 5. Image Features
# Pixel Values: Raw pixel values.
# Color Histograms: Distribution of colors.
# Edge Detection: Features based on edges in the image.
# Convolutional Features: Features extracted using convolutional neural networks (CNNs).

# 6. Audio Features
# Spectrogram: Visual representation of the spectrum of frequencies.
# Mel-Frequency Cepstral Coefficients (MFCCs): Features representing the short-term power spectrum.
# Chroma Features: Representing the 12 different pitch classes.

# 7. Geospatial Features
# Latitude and Longitude: Coordinates.
# Distance Measures: Distance to a specific point or between points.
# Region-Based Features: Features based on regions or zones.

# 8. Interaction Features
# Polynomial Features: Combinations of existing features (e.g.,
# Cross Features: Interaction between categorical features.

# 9. Aggregated Features
# Rolling Statistics: Moving averages, moving sums, etc.
# Window-Based Features: Features calculated over a sliding window.,
