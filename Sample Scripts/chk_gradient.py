import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
sea.set()
plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)



#
# sea.set()
# plt.axis([0, 50, 0, 50])                                 # scale axes (0 to 50)
# plt.xticks(fontsize=14)                                  # set x axis ticks
# plt.yticks(fontsize=14)                                  # set y axis ticks
# plt.xlabel("Tuesdays", fontsize=14)                  # set x axis label
# plt.ylabel("Eat_Chk", fontsize=14)                        # set y axis label
# X, Y = np.loadtxt("chk.txt", skiprows=1, unpack=True)  # load data
# plt.plot(X, Y, "bo")
# plt.show()
def predict(X, w, b):
    return (X * w  + b)

def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

def gradient(X, Y, w):
  return 2 * np.average(X * (predict(X, w, 0) - Y))
def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, 0)))
        w -= gradient(X, Y, w) * lr
    return w

# Import the dataset
X, Y = np.loadtxt("chk.txt", skiprows=1, unpack=True)

# Train the system
w = train(X, Y, iterations=100, lr=0.001)
print("\nw=%.10f" % w)

