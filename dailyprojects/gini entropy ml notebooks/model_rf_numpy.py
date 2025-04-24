import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# Gini impurity
def gini(y):
    counts = Counter(y)
    impurity = 1.0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / len(y)
        impurity -= prob_of_lbl ** 2
    return impurity

# Split dataset
def split(X_column, threshold):
    left_idx = np.argwhere(X_column <= threshold).flatten()
    right_idx = np.argwhere(X_column > threshold).flatten()
    return left_idx, right_idx

# Best split
def best_split(X, y):
    best_gini = float('inf')
    best_idx, best_thresh = None, None

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for t in thresholds:
            left_idx, right_idx = split(X[:, feature_idx], t)
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            y_left, y_right = y[left_idx], y[right_idx]
            gini_split = (len(y_left) / len(y)) * gini(y_left) + \
                         (len(y_right) / len(y)) * gini(y_right)
            if gini_split < best_gini:
                best_gini = gini_split
                best_idx = feature_idx
                best_thresh = t
    return best_idx, best_thresh

# Tree node class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Build tree recursively
def build_tree(X, y, depth=0, max_depth=5):
    if len(set(y)) == 1 or depth >= max_depth:
        most_common = Counter(y).most_common(1)[0][0]
        return Node(value=most_common)

    feature, threshold = best_split(X, y)
    if feature is None:
        most_common = Counter(y).most_common(1)[0][0]
        return Node(value=most_common)

    left_idx, right_idx = split(X[:, feature], threshold)
    left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)
    return Node(feature, threshold, left, right)

# Predict single sample
def predict_tree(sample, node):
    if node.value is not None:
        return node.value
    if sample[node.feature] <= node.threshold:
        return predict_tree(sample, node.left)
    else:
        return predict_tree(sample, node.right)

# Random Forest
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), self.sample_size or len(X), replace=True)
            tree = build_tree(X[idxs], y[idxs], max_depth=self.max_depth)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([[predict_tree(x, tree) for tree in self.trees] for x in X])
        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])

# Load your spam dataset
df = pd.read_csv("synthetic_email_spam.csv")
features = ['num_links', 'num_special_chars', 'contains_offer', 'contains_free'
            ]
X = df[features].values
y = df['spam'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest
rf = RandomForest(n_trees=100, max_depth=5, sample_size=len(X_train) // 2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate
accuracy = np.mean(y_pred == y_test)
print(f"\nRandom Forest Accuracy: {accuracy * 100:.2f}%")
