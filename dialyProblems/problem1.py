##daily calculations

import numpy as np


x = np.array([[1.2,1.5,1.6],[2,2.1,2.2]])
# y = np.array([[0.4,.2,.3],[1.1,1.2]])

w = np.array([[1.1,1.2],[1.8,1.5]])
b = np.array([.1,.2])

##we will form an equation for y = xw+b

##multiple linear

# y = np.dot(w,x) + b

print(x.shape)
print(w.shape)

y = np.dot(x.T,w) + b

print(y)

