##patient wait time


import numpy as np
import matplotlib.pyplot as plt
import openpyxl as xl
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import seaborn as sns



##now create functions for model




def loss_mse(y_true, y_pred):
    """
    Compute Mean Squared Error loss.

    Parameters:
        y_true : np.ndarray → actual target values
        y_pred : np.ndarray → predicted values

    Returns:
        mse : float → mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)

def feedforward(X, w, b):
    """
    Perform feedforward prediction for linear regression.

    Parameters:
        X : np.ndarray   → shape (m, n) — input features
        w : np.ndarray   → shape (n,)   — weights
        b : float        → scalar       — bias

    Returns:
        y_hat : np.ndarray → shape (m,) — predicted values
    """
    return np.dot(X, w) + b

def gradients(X, y_true, y_pred):
    """
    Compute gradients of MSE loss w.r.t weights and bias.

    Parameters:
        X : np.ndarray        → shape (m, n)
        y_true : np.ndarray   → shape (m,)
        y_pred : np.ndarray   → shape (m,)

    Returns:
        dw : np.ndarray → gradient w.r.t weights
        db : float      → gradient w.r.t bias
    """
    m = X.shape[0]  # number of samples

    error = y_pred - y_true

    dw = (2 / m) * np.dot(X.T, error)
    db = (2 / m) * np.sum(error)

    return dw, db

def updated_parameters(w,b,grad_w,grad_b,lr):
    w -= grad_w*lr
    b -= grad_b*lr
    return w,b


def projections(X, y):
    """
    Apply least squares projection to get weights using normal equation:
    w_proj = (X^T X)^(-1) X^T y

    Parameters:
        X : np.ndarray → shape (m, n)
        y : np.ndarray → shape (m,)

    Returns:
        w_proj : np.ndarray → weights from projection
        y_proj : np.ndarray → predicted values using projected weights
    """
    # Calculate projection weights using normal equation
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)

    w_proj = np.linalg.inv(XTX).dot(XTy)
    y_proj = X.dot(w_proj)

    return w_proj, y_proj


def main():
    ##prepare the dataset,understand the business and requirement to develop model

    # Load your dataset
    df = pd.read_csv("synthetic_patient_wait_time.csv")

    # Visualize numeric features vs wait time
    numeric_cols = ['Urgency_level', 'Queue_length', 'Doctor_Availability', 'Patient_Arrival_hour']

    # for col in numeric_cols:
    #     plt.figure(figsize=(6, 4))
    #     sns.scatterplot(data=df, x=col, y='Expected_wait_time')
    #     plt.title(f"{col} vs Expected Wait Time")
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    ##X = [
    ##  [Urgency_Level₁, Queue_Length₁, Doctor_Available₁, ...],
    ##  [Urgency_Level₂, Queue_Length₂, Doctor_Available₂, ...],
    ##  ...
    ##]

    ##Step 1
    ##Define input features
    ##['Urgency_level', 'Queue_length', 'Doctor_Availability', 'Patient_Arrival_hour']
    input_features = ['Department','Urgency_level', 'Queue_length', 'Doctor_Availability', 'Patient_Arrival_hour']
    target = 'Expected_wait_time'

    ##Split features and target
    X_df = df[input_features]
    y = df[target].values

    ##Step 2 - handle categorical columns

    X_encoded = pd.get_dummies(X_df, columns=['Department'], drop_first=True)

    ##Step 3 - convert to numpy or normalize

    X = X_encoded.values  # Convert to NumPy
    X = X_encoded.astype(float).values  # Ensure all values are floats

    # Optional: Normalize using L2 norm
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    X = X / row_norms

    ##now split the train,test datasets
    # Now do the train-test split
    np.random.seed(42)
    indices = np.random.permutation(len(X))

    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx,:]
    y_train = y[train_idx]  # ✅ Use .iloc for position-based slicing

    X_test = X[test_idx,:]
    y_test = y[test_idx]

    num_features = X_train.shape[1]  # number of input columns

    # Initialize weights and bias
    w = np.random.randn(num_features) * 0.01
    b = 0.0

    # X: input feature matrix
    # w: weight vector
    # b: bias
    # Step 1: Get predictions from feedforward
    y_cap = feedforward(X_train, w, b)
    print("Predictions:", y_cap)
    # Step 2: Compute MSE loss
    loss = loss_mse(y_train, y_cap)
    print(f"Loss (MSE): {loss:.4f}")

    # Apply projections (closed-form solution)
    w_proj, y_proj = projections(X_train, y_train)

    print("Projection Weights:\n", w_proj[:5])
    print("First 5 Projected Predictions:\n", y_proj[:5])

    # Compare projection vs gradient-descent predictions
    mse_proj = loss_mse(y_train, y_proj)
    print(f"MSE from projection solution: {mse_proj:.4f}")

    # Step 2: Hyperparameters
    learning_rate = 0.001
    epochs = 200000



    # Track loss values
    losses = []

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        y_pred = feedforward(X_train, w, b)

        # Loss
        loss = loss_mse(y_train, y_pred)
        losses.append(loss)

        # Gradients
        dw, db = gradients(X_train, y_train, y_pred)

        # Update parameters
        w, b = updated_parameters(w, b, dw, db, learning_rate)

        # Print all info at each epoch
        print(f"\nEpoch {epoch + 1}")
        print(f"Loss: {loss:.4f}")
        print("First 5 Predictions: ", y_pred[:5])
        print("First 5 True Labels: ", y_train[:5])
        print("First 5 Weights:     ", w[:5])
        print("Bias:                ", b)
        # === Inference on Test Set ===
        print("\n=== Testing on Test Set ===")

        y_test_pred = feedforward(X_test, w, b)

        test_loss = loss_mse(y_test, y_test_pred)
        print(f"Test MSE Loss: {test_loss:.4f}")

        print("First 5 Test Predictions:", y_test_pred[:5])
        print("First 5 Test Labels:     ", y_test[:5])
        import joblib

        # After training loop ends
        joblib.dump({'weights': w, 'bias': b, 'columns': X_encoded.columns.tolist()}, 'wait_time_model.pkl')


if __name__=="__main__":
    main()
