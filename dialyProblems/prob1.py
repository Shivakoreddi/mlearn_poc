##1. Custom Linear Regression with Gradient Descent
##Goal:
##1. Build your own linear regression model using only NumPy, and train it with manual gradients.
##2. Use dot product and projection to understand fitting
##3. Derive gradients from MSE
##4. Apply gradient descent and plot updates
##Bonus: Add 2nd-order update using Hessian for Newton's method!

import numpy as np
import matplotlib.pyplot as plt


def feedforward(x,w,b):
    y_cap = w*x+b
    return y_cap

def loss(y,y_cap):
    return np.mean((y - y_cap) ** 2)


def gradient(x, y, w, b):
    x = np.array(x)
    y = np.array(y)
    y_cap = feedforward(x, w, b)
    n = len(y)

    grad_w = (-2 / n) * np.sum(x * (y - y_cap))
    grad_b = (-2 / n) * np.sum(y - y_cap)

    return grad_w, grad_b


def updated_parameters(w,b,grad_w,grad_b,lr):
    w -= grad_w*lr
    b -= grad_b*lr
    return w,b
def projection_solution(x, y):
    x = np.array(x)
    y = np.array(y)

    # Add bias term as first column (X becomes 2D)
    X = np.c_[np.ones_like(x), x]  # shape: (n, 2)

    # Projection formula: y_hat = X (XᵀX)⁻¹ Xᵀ y
    XTX_inv = np.linalg.inv(X.T @ X)
    w_proj = XTX_inv @ X.T @ y
    y_proj = X @ w_proj

    print("\nProjection solution (analytical):")
    print("  Weights (bias and slope):", np.round(w_proj, 4))
    print("  Projected predictions (y_cap):", np.round(y_proj, 3))

    return w_proj, y_proj


def main():
    ##initialize weights
    w = 0.5
    b = 0
    x = [1,1.1,1.2,1.3,2,2.1,2.6]
    y = [1,1,1,1,2,2,3]
    lr = 0.01
    epochs = 1000
    loss_values=[]
    print("\n--- Projection Approach ---")
    w_proj, y_proj = projection_solution(x, y)
    print(f"w_proj-{w_proj},y_proj-{y_proj}")
    plt.figure(figsize=(12, 6))
    for epoch in range(epochs):
        y_cap = feedforward(np.array(x), w, b)
        current_loss = loss(np.array(y), y_cap)
        grad_w,grad_b = gradient(x, y, w, b)
        w,b = updated_parameters(w,b,grad_w,grad_b,lr)
        loss_values.append(current_loss)
        print(f"\nEpoch {epoch + 1}")
        print(f"  Weights (w): {w:.4f}")
        print(f"  Bias (b): {b:.4f}")
        print(f"  Loss: {current_loss:.4f}")
        print(f"  Gradient w.r.t w: {grad_w:.4f}")
        print(f"  Gradient w.r.t b: {grad_b:.4f}")
        print(f"  Predictions (y_cap): {np.round(y_cap, 3)}")
        # Add this inside main(), after your gradient descent

        # Plot fit every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            plt.subplot(1, 2, 1)
            plt.scatter(x, y, label="Data")
            line_x = np.linspace(min(x), max(x), 100)
            line_y = feedforward(line_x, w, b)
            plt.plot(line_x, line_y, label=f"Epoch {epoch + 1}")
            plt.title("Linear Regression Fit")
            plt.xlabel("x")
            plt.ylabel("y")

        # Plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), loss_values, marker='o')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.tight_layout()
    plt.legend()
    plt.show()




if __name__=="__main__":
    main()