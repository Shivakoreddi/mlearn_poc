import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
sea.set()


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        if i % 300 == 0:
            print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss: # Updating weight
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss: # Updating weight
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss: # Updating bias
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss: # Updating bias
            b -= lr
        else:
            return w, b

    raise Exception("Couldn't converge within %d iterations" % iterations)


# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))