import numpy as np

# Difference between Polynomial vs Linear + ReLU:
# Both introduce non-linearity into the model:
# - Polynomial adds non-linearity via feature transformation (x, x², x³,...)
# - ReLU adds non-linearity via piecewise activation after a linear transformation

# Polynomial Equation: y = w0 + w1*x + w2*x^2 + w3*x^3 + b
def poly(x, w, b):
    y = w[0] * x + w[1] * x**2 + w[2] * x**3 + b
    return y

# ReLU Activation Function
def relu(y):
    return np.maximum(0, y)

# Linear model followed by ReLU activation
def linear_relu(x, w, b):
    y = np.dot(w, x) + b
    return relu(y)

def main():
    x = np.array([1.2, 1.2, 1.2])  # same x repeated for x, x^2, x^3
    w_poly = [0.8, 9.2, 0.9]  # for x, x^2, x^3
    b = 0.2

    # Polynomial regression output
    y_poly = poly(x[0], w_poly, b)
    print("Polynomial Output:", y_poly)

    # Linear + ReLU (simulating a 3-feature input)
    w_linear = [0.8, 9.2, 0.9]
    y_relu = linear_relu(x, w_linear, b)
    print("Linear + ReLU Output:", y_relu)

if __name__ == "__main__":
    main()
