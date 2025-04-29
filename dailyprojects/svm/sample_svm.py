import numpy as np

##given a dataset indeitfy below and see if the values fall under these categorieis
##decision bounderies/ margins

import numpy as np
import pandas as pd

# Small dataset (manually created)
X = np.array([
    [2, 2],
    [4, 4],
    [4, 0],
    [0, 0],
    [3, 1],
    [1, 3]
])

y = np.array([1, 1, 0, 0, 0, 1])
print(X[2][1])

# Visualize
import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.coolwarm, s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Small 2D Dataset for SVM Practice')
plt.show()

##decision boundry - hyperplane

##wt*x+b =0

##inititlze w
w1 = 1
w2 = 0
b = 1


##datasample1
# 2*w1+2*w2+b>=1

def svm_classification(w1, w2, b, X, y):
    flag = []
    for i in range(len(X)):
        margin = y[i] * X[i][0] * w1 + X[i][1] * w2 + b

        print(f'{X[i]}:{margin}')
        flag.append(margin >= 1)

    print(flag)
    if False in flag:
        return "Misclassified"
    else:
        return "Classified"


svm_classification(w1, w2, b, X, y)

