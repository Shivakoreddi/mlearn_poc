import numpy as np
import pandas as pd
import time
##hospital bed predictions
##below is the dataset which will tell us, whats the attributes we will represent to identify -
# - hospital  beds needed for next 7days

# df = pd.DataFrame({
#     "Department":department,
#     "day":day_of_week,
#     "occupancy":current_occupancy,
#     "discharges":discharge_3days,
#     "admissions":admission_3days,
#     "holiday":holiday_flag,
#     "flu":flu_season,
#     "beds_needed":beds_needed.round(0)
#})

##Insights of dataset: -
##1. the dataset would predict the upcoming bed availability based on current admissions/discharges
##2. if there are 10 patients admitted for emergency then on day2 we can predict that most of these patients can discharge becoz of dept
##3. similarly ,when there are 5 patients joined for surgery dept, then there is high chances that discharge may not be immediate ,it may take time
##this makes that previous admissions/discharges would effect the current/future beds needed
##current occupancy tells us - how many beds are being occupied based on till date admissions/discharges
##beds needed tells us - how many remaining we need to fill the admissions


def feedforward(x,w,b):
    y_cap = np.dot(x,w)+b
    return y_cap

def mse_loss(y_train,y_cap):
    return np.mean((y_train - y_cap) ** 2)

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


def one_hot_encoding():
    pass

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
    start_time = time.time()  # Start timer

    # === Your code starts here ===
    data = pd.read_csv("synthetic_hospital_bed_forcast.csv")
    input_features = ["Department", "day", "occupancy", "discharges", "admissions", "holiday", "flu"]
    target = "beds_needed"
    X_df = data[input_features]
    Y = data[target].values
    X_encoded = pd.get_dummies(X_df, columns=['Department', 'day'], drop_first=True)
    X = X_encoded.astype(float).values
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    X = X / row_norms

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    X_train, y_train = X[train_idx], Y[train_idx]
    X_test, y_test = X[test_idx], Y[test_idx]

    num_features = X_train.shape[1]
    w = np.random.randn(num_features) * 0.01
    b = 0.0

    y_cap = feedforward(X_train, w, b)
    print("Predictions:", y_cap)
    loss = mse_loss(y_train, y_cap)
    print(f"Loss (MSE): {loss:.4f}")

    w_proj, y_proj = projections(X_train, y_train)
    print("Projection Weights:\n", w_proj[:5])
    print("First 5 Projected Predictions:\n", y_proj[:5])
    mse_proj = mse_loss(y_train, y_proj)
    print(f"MSE from projection solution: {mse_proj:.4f}")

    learning_rate = 0.0001
    epochs = 100000
    losses = []

    for epoch in range(epochs):
        y_pred = feedforward(X_train, w, b)
        loss = mse_loss(y_train, y_pred)
        losses.append(loss)
        dw, db = gradients(X_train, y_train, y_pred)
        w, b = updated_parameters(w, b, dw, db, learning_rate)

        if epoch % 1000 == 0:  # 🔥 reduce print frequency
            print(f"\nEpoch {epoch + 1}")
            print(f"Loss: {loss:.4f}")
            print("First 5 Predictions: ", y_pred[:5])
            print("First 5 True Labels: ", y_train[:5])
            print("First 5 Weights:     ", w[:5])
            print("Bias:                ", b)

            y_test_pred = feedforward(X_test, w, b)
            test_loss = mse_loss(y_test, y_test_pred)
            print(f"Test MSE Loss: {test_loss:.4f}")
            print("First 5 Test Predictions:", y_test_pred[:5])
            print("First 5 Test Labels:     ", y_test[:5])

    # Save model
    import joblib
    joblib.dump({'weights': w, 'bias': b, 'columns': X_encoded.columns.tolist()}, 'wait_time_model.pkl')

    # === End timer and print time taken ===
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✅ Total Execution Time: {total_time:.2f} seconds")

if __name__=="__main__":
    main()







