import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Generate synthetic dataset
np.random.seed(0)
n_samples, n_features = 100, 20

# Only first 3 features are useful
X = np.random.randn(n_samples, n_features)
true_weights = np.zeros(n_features)
true_weights[:3] = [5, -3, 2]  # only 3 are relevant

y = X @ true_weights + np.random.randn(n_samples) * 0.5  # add noise

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. Train different models
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

# 4. Print coefficients
print("True Weights:  ", true_weights)
print("Linear Coefs: ", lr.coef_)
print("Ridge Coefs:  ", ridge.coef_)
print("Lasso Coefs:  ", lasso.coef_)

# 5. Compare performance
print("\nMSE (Linear):", mean_squared_error(y_test, lr.predict(X_test)))
print("MSE (Ridge): ", mean_squared_error(y_test, ridge.predict(X_test)))
print("MSE (Lasso): ", mean_squared_error(y_test, lasso.predict(X_test)))

# 6. Plot coefficients
plt.figure(figsize=(12, 5))
plt.plot(true_weights, label='True weights', linewidth=3)
plt.plot(lr.coef_, label='Linear Regression')
plt.plot(ridge.coef_, label='Ridge')
plt.plot(lasso.coef_, label='Lasso')
plt.legend()
plt.title("Comparison of Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.show()
