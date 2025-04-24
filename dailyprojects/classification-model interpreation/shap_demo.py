import numpy as np
import shap
# X = np.array([
#         [1, 1], [2, 1.5], [3, 2], [4, 3], [5, 3.5],
#         [6, 4], [7, 5], [8, 6], [9, 6.5], [10, 7],
#         [1, 5], [2, 6], [3, 6.5], [4, 7], [5, 8]
#     ])
# y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])

x = np.array([[1,2,3],[1.2,1.3,2.3],[2.3,1.2,1.4]])
y_true = np.array([0,0,1])

# Corrected shapes for weights and biases
w = np.random.randn(3, 1) * 0.1
b = np.zeros((1, 1))  # fixed: now (1, 1) not (1,)

def model(X):
    y_pred = np.matmul(X,w)+b
    return y_pred


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


##x,w,b-->equation --> loss y-y_cap

def gradient(X, y_true, y_pred,w,b):
    m = X.shape[0]
    error = y_true - y_pred
    dw = (-2 / m) * np.dot(X.T, error)       # shape: (n, 1)
    db = (-2 / m) * np.sum(error)            # scalar
    return dw, db

##apply them to model

learning_rate = 0.001
epochs = 100

for epoch in range(epochs):
    y_pred = model(x)
    loss = mse_loss(y_true,y_pred)
    w,b = gradient(x,y_true,y_pred,w,b)
    w -= w*learning_rate
    b -= b*learning_rate
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


explainer = shap.Explainer(model,x)
shap_values = explainer(x)
shap.summary_plot(shap_values, x)









