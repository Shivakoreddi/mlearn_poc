##test1
import numpy as np
##gini test runs
x = np.array([1,9,3,4,2,5,7,8,2,2])
##x2 = np.array([0.11,0.21,0.42,0.21,0.31,0.2,0.3,0.2,.25,.13])

##our labeled value
y = (x > 4).astype(int)
print("Labels:", y)

#3samples
n = 10

threshold = 10/2

left_mask = (x<=threshold)
right_mask = x>threshold

print(left_mask)
print(right_mask)
x_left, y_left = x[left_mask], y[left_mask]
x_right, y_right = x[right_mask], y[right_mask]

nl = len(x_left)
print(nl)
nr = len(x_right)
print(nr)
# Gini Impurity function
def gini_impurity(y_group):
    m = len(y_group)
    if m == 0:
        return 0
    p1 = np.sum(y_group == 0) / m
    p2 = np.sum(y_group == 1) / m
    return 1 - p1**2 - p2**2

# Calculate impurities
gini_left = gini_impurity(y_left)
gini_right = gini_impurity(y_right)

# Weighted Gini
weighted_gini = (nl / n) * gini_left + (nr / n) * gini_right

# Output
print("\nGini Left:", gini_left)
print("Gini Right:", gini_right)
print("Weighted Gini:", weighted_gini)