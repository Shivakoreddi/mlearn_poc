import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Sample categorical feature and binary target
x = np.array(['ICU', 'SURGERY', 'PEDIA', 'ICU', 'SURGERY', 'ORTHO', 'ICU', 'CARDIO', 'SURGERY'])
y = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0])

# Convert to DataFrame and apply one-hot encoding for the categorical feature
df = pd.DataFrame({'Department': x})
X_encoded = pd.get_dummies(df)

# Fit decision tree classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=2)
clf.fit(X_encoded, y)

# Show the structure of the tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X_encoded.columns, class_names=["Not Admitted", "Admitted"], filled=True)
plt.title("Decision Tree Split Based on Department (Gini Criterion)")
plt.show()
