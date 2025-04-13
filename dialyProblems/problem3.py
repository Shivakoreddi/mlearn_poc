import numpy as np
import matplotlib.pyplot as plt

# Define x range
x = np.linspace(-10, 10, 200)

# Linear parameters
w = 1.2
b = 0.5

# Polynomial parameters
w_poly = [1.0, -0.5, 0.1, 0.01]  # w0, w1, w2, w3

# Linear Function
y_linear = w * x + b

# Linear + ReLU
y_relu = np.maximum(0, y_linear)

# Linear + Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
y_sigmoid = sigmoid(y_linear)

# Polynomial Function: y = w0 + w1*x + w2*x^2 + w3*x^3
y_poly = w_poly[0] + w_poly[1]*x + w_poly[2]*x**2 + w_poly[3]*x**3

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(x, y_linear, label="Linear", linestyle='--')
plt.plot(x, y_relu, label="Linear + ReLU")
plt.plot(x, y_sigmoid, label="Linear + Sigmoid")
plt.plot(x, y_poly, label="Polynomial", linestyle='-.')
plt.axhline(0, color='gray', linestyle=':')
plt.axvline(0, color='gray', linestyle=':')

plt.title("Function Comparisons")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
