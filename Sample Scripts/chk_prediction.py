import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
sea.set()
plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)




sea.set()
plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)
plt.xticks(fontsize=14)                                  # set x axis ticks
plt.yticks(fontsize=14)                                  # set y axis ticks
plt.xlabel("Tuesdays", fontsize=14)                  # set x axis label
plt.ylabel("Eat_Chk", fontsize=14)                        # set y axis label
X, Y = np.loadtxt("chk.txt", skiprows=1, unpack=True)  # load data
plt.plot(X, Y, "bo")
plt.show()
def predict(X, w, b):
    return (X * w  + b)


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
X, Y = np.loadtxt("chk.txt", skiprows=1, unpack=True)

# Train the system
w, b = train(X, Y, iterations=1000000, lr=0.001)
print("\nw=%.3f, b=%.3f" % (w, b))

# Predict the number of pizzas
print("Prediction: x=%d , y=%.2f" % (35,predict(35, w, b)))