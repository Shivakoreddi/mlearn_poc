##Predicting Delivery Delay Using Weather + Distance

import numpy as np
import matplotlib.pyplot as plt
import openpyxl as xl
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import seaborn as sns


##preparing dataset
##first create method for calculating distance and adding it as column in dataframe
from math import radians, sin, cos, sqrt, atan2

import numpy as np

def distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth in km

    # Use numpy instead of math
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c



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

def norm_l2(x_df):
    X_encoded = pd.get_dummies(x_df, columns=['Type_of_vehicle'], drop_first=True)
    X_encoded = X_encoded.astype(float)  # 🔥 Ensures all values are numeric
    X = X_encoded.values

    # Apply L2 normalization
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    X = X / row_norms
    return X

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


def plot(df):
    # sns.lineplot(data=df, x=df['Delivery_person_Age'], y=df['Delivery_person_Ratings'])
    # plt.title("Ratings Over Time per Person")
    # plt.show()

    df_melt = df.melt(id_vars='Delivery_person_Age', value_vars=['Delivery_person_Ratings'])

    sns.lineplot(data=df_melt, x='Delivery_person_Age', y='value', hue='variable')
    plt.title("Multiple Y Lines on Shared X")
    plt.show()


def remove_outliers_zscore(df, feature_cols, threshold=3):
    """
    Remove rows from the DataFrame where any of the feature columns
    have Z-score > threshold (absolute).

    Parameters:
        df : pandas DataFrame
        feature_cols : list of column names (only numerical ones)
        threshold : float, Z-score limit (commonly 3)

    Returns:
        df_cleaned : filtered DataFrame
    """
    from scipy.stats import zscore

    # Compute z-scores only for selected columns
    z_scores = np.abs(zscore(df[feature_cols]))

    # Keep rows where all features are within the threshold
    mask = (z_scores < threshold).all(axis=1)
    return df[mask]



def main():
    data_path = r"C:\Users\shiva\PycharmProjects\mlearn_poc\dailyprojects\project1\food _delivery_time.xlsx"
    data = pd.read_excel(data_path)

    print(data.columns)
    data['distance'] = distance(data['Restaurant_latitude'],data['Restaurant_longitude'],data['Delivery_location_latitude'],data['Delivery_location_longitude'])
    cleaned_data = data.drop(columns=['ID','Delivery_person_ID','Type_of_order','Restaurant_latitude', 'Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude'])
    ##rename column time

    cleaned_data.rename(columns={'Time_taken(min)': 'Time_taken'}, inplace=True)
    print(cleaned_data.columns)
    ##check distance
    # Remove outliers using Z-score on numeric columns
    # Apply Z-score outlier removal FIRST
    numeric_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance', 'Time_taken']
    cleaned_data = remove_outliers_zscore(cleaned_data, numeric_cols, threshold=3)

    # Prepare features and target from cleaned data
    input_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Type_of_vehicle', 'distance']
    target_feature = 'Time_taken'

    x_df = cleaned_data[input_features]
    y = cleaned_data[target_feature]

    # One-hot encode
    X_encoded = pd.get_dummies(x_df, columns=['Type_of_vehicle'], drop_first=True)
    X_encoded = X_encoded.astype(float)
    X = X_encoded.values

    # Normalize X (optional)
    from numpy.linalg import norm
    row_norms = norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    X = X / row_norms

    # Now do the train-test split
    np.random.seed(42)
    indices = np.random.permutation(len(X))

    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx]
    y_train = y.iloc[train_idx]  # ✅ Use .iloc for position-based slicing

    X_test = X[test_idx]
    y_test = y.iloc[test_idx].values

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
    learning_rate = 0.01
    epochs = 100000

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
        print("First 5 True Labels: ", y_train.values[:5])
        print("First 5 Weights:     ", w[:5])
        print("Bias:                ", b)
        # === Inference on Test Set ===
        print("\n=== Testing on Test Set ===")

        y_test_pred = feedforward(X_test, w, b)

        test_loss = loss_mse(y_test, y_test_pred)
        print(f"Test MSE Loss: {test_loss:.4f}")

        print("First 5 Test Predictions:", y_test_pred[:5])
        print("First 5 Test Labels:     ", y_test[:5])



if __name__=="__main__":
    main()

