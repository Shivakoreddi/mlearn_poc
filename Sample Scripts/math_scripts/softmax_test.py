import numpy as np

# Applying softmax function
def softmax(logits):
    exponentials = np.exp(logits)
    print("Exp:")
    print(exponentials)
    print("sumEX:")
    print(np.sum(exponentials,axis=1).reshape(-1,1))
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)

# Taking random arrays and applying softmax function on them
output = np.array([[0.3, 0.8, 0.2], [0.1, 0.9, 0.1]])
print(softmax(output))