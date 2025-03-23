import numpy as np
import matplotlib.pyplot as plt

from ml_basics.perceptron_sigmoid_af1 import sigmoid

input = [4.12,0.12,0.4]

exp_value = np.exp(input)
norm = sum(exp_value)
norm_values=[]
print(exp_value)
print(norm)

for value in exp_value:
    norm_values.append(value/norm)

print(norm_values)
print(sum(norm_values))  