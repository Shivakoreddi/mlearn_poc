import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Sample data
data = {
    'age': [25, 70, 45, 60, 30, 80, 50, 40, 65, 55],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'F'],
    'blood_pressure': ['high', 'low', 'normal', 'normal', 'high', 'low', 'normal', 'high', 'low', 'normal'],
    'diabetes': [1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    'num_previous_admissions': [2, 5, 0, 1, 1, 4, 0, 2, 3, 0],
    'discharged_within_5days': [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    'readmitted_30_days': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# print(df)
#
# df = pd.get_dummies('gender','blood_pressure')


##create decision tree


# Encode
df_encoded = pd.get_dummies(df, drop_first=True)

# Split
X = df_encoded.drop('readmitted_30_days', axis=1)
y = df_encoded['readmitted_30_days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Sample data
data = {
    'age': [25, 70, 45, 60, 30, 80, 50, 40, 65, 55],
    'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'F'],
    'blood_pressure': ['high', 'low', 'normal', 'normal', 'high', 'low', 'normal', 'high', 'low', 'normal'],
    'diabetes': [1, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    'num_previous_admissions': [2, 5, 0, 1, 1, 4, 0, 2, 3, 0],
    'discharged_within_5days': [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    'readmitted_30_days': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# print(df)
#
# df = pd.get_dummies('gender','blood_pressure')


##create decision tree


# Encode
df_encoded = pd.get_dummies(df, drop_first=True)

# Split
X = df_encoded.drop('readmitted_30_days', axis=1)
y = df_encoded['readmitted_30_days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()
