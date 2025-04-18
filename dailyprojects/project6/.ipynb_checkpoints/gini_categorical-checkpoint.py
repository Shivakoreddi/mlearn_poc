import numpy as np

x = np.array(['ICU','ICU','PEDIA','ICU','SURGERY','ORTHO','ICU','ICU','SURGERY'])
y = np.array([1,1,0,1,0,0,1,1,0])



left_mask=x=='ICU'
right_mask=x!='ICU'

print(left_mask)
print(right_mask)
x_left, y_left = x[left_mask], y[left_mask]
x_right, y_right = x[right_mask], y[right_mask]
print(x_left)
print(y_left)
print(x_right)
print(y_right)
nl = len(x_left)
print(nl)
nr = len(x_right)
print(nr)
n=9

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