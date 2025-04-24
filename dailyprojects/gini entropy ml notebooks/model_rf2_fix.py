import numpy as np
import pandas as pd

# Seed for reproducibility
np.random.seed(42)
n = 100  # Number of records

# Generate synthetic hospital data
data = pd.DataFrame({
    'Department': np.random.choice(['ICU', 'SURGERY', 'ORTHO', 'PEDIA', 'CARDIO'], size=n),
    'Bed_Occupancy': np.random.randint(0, 100, size=n),
    'Emergency_Admissions': np.random.randint(0, 50, size=n),
    'Weekend': np.random.choice([0, 1], size=n),
})

# Define the target: whether more beds are required
data['beds_required'] = ((data['Department'] == 'ICU') & (data['Emergency_Admissions'] > 10)).astype(int)

# Define target column and features
y_col = 'beds_required'
features = [col for col in data.columns if col != y_col]

# Function to calculate Gini Impurity
def gini_impurity(y_group):
    m = len(y_group)
    if m == 0:
        return 0
    p1 = np.sum(y_group == 0) / m
    p2 = np.sum(y_group == 1) / m
    return 1 - p1**2 - p2**2

# Function to calculate weighted Gini for a split
def weighted_gini(data, feature, threshold):
    if data[feature].dtype == 'object':
        left_mask = data[feature] == threshold
        right_mask = data[feature] != threshold
    else:
        left_mask = data[feature] <= threshold
        right_mask = data[feature] > threshold

    y_left = data[y_col][left_mask]
    y_right = data[y_col][right_mask]

    nl, nr = len(y_left), len(y_right)
    gini_left = gini_impurity(y_left)
    gini_right = gini_impurity(y_right)
    return (nl / n) * gini_left + (nr / n) * gini_right

# Build a "forest" with one decision tree (stump) per feature
def build_forest_from_features(data):
    forest = []
    for feature in features:
        best_split = None
        best_gini = float('inf')
        values = np.unique(data[feature])
        for val in values:
            gini = weighted_gini(data, feature, val)
            if gini < best_gini:
                best_gini = gini
                best_split = val
        forest.append({'feature': feature, 'best_split': best_split, 'gini': best_gini})
    return pd.DataFrame(forest)

# Build the simulated forest
forest_results = build_forest_from_features(data)
print(forest_results)
