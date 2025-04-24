from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Example dataset
X = pd.DataFrame({
    'visits': [1, 2, 3, 4, 5, 6, 7],
    'time': [1, 2, 2.5, 3, 3.5, 4, 5],
})
y = [0, 0, 0, 1, 1, 1, 1]

# Train a model
model = RandomForestClassifier()
model.fit(X, y)

# PDP for feature 'time'
# PartialDependenceDisplay.from_estimator(model, X, features=['time'])
# plt.show()

##multiple feature pdp
PartialDependenceDisplay.from_estimator(model, X, features=[('time', 'visits')])
plt.show()
