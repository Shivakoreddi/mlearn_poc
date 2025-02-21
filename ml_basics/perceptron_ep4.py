import numpy as np

import matplotlib.pyplot as plt

##forward equation for neural network is

## y = w1a*w2b+w3c**2
##l = mean(y_cap-y)**2
##pd_y with l= 2(y_cap-y)
##pd_w1 with y = a*(w2b)
##pd_w2 with y = b*(w1a)
##pd_w3 with y = c**2
##gradient of w1 with l = 2(y_cap-y)*a(w2b)
##gradient of w2 with l = 2(y_cap-y)*b(w1a)
##gradient of w3 with l = 2(y_cap-y)*c**2


weights = np.array([2,1,2.5])
bias = 0
def forward(a,b,c,weights,bias):
    y = weights[0]*a*weights[1]*b+weights[2]*c**2
    return y+bias

def loss(y_cap,y):
    l = np.mean((y_cap-y)**2)
    return l

def gradient(a,b,c,y_cap,y,weights):
    grad_y = 2*(y_cap-y)
    grad_w1 = grad_y*a*(weights[1]*b)
    grad_w2 = grad_y*b*(weights[0]*a)
    grad_w3 = grad_y*c**2
    grad_weights = grad_w1,grad_w2,grad_w3
    grad_bias=1.0
    return grad_weights,grad_bias

def update_parameters(weights,bias,grad_weights,grad_bias,learning_rate):
    weights[0] -= learning_rate*grad_weights[0]
    weights[1] -= learning_rate*grad_weights[1]
    weights[2] -= learning_rate*grad_weights[2]
    bias -= learning_rate*grad_bias
    return weights,bias

def read_values_from_file(filename):
    with open(filename, 'r') as file:
        # Skip the header line
        next(file)
        # Read the next line with the actual values
        lines = file.readlines()
        data = [list(map(float, line.strip().split(','))) for line in lines]
    return data

# Feature scaling (Min-Max Scaling)
def feature_scaling(data):
    data = np.array(data)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data



def main():
    global weights, bias
    filename = 'file3.txt'
    data = read_values_from_file(filename)
    # Apply feature scaling to the data
    scaled_data = feature_scaling(data)
    learning_rate = 0.000001
    total_loss = 0
    # Lists to store values for plotting
    d_values = []
    d_cap_values = []
    loss_values = []
    for epoch in range(100):
        total_loss = 0
        for a, b, c, d_cap in scaled_data:
            d = forward(a, b, c, weights, bias)
            l = loss(d_cap, d)
            grad_weights, grad_bias = gradient(a, b, c, d_cap, d,weights)
            weights, bias = update_parameters(weights, bias, grad_weights, grad_bias, learning_rate)
            # Store values for plotting
            d_values.append(d)
            d_cap_values.append(d_cap)
            loss_values.append(l)
            total_loss += l
        if epoch % 2 == 0:  # Adjusted to print every 10 epochs for better visibility
            print(f"Epoch {epoch}: total_loss: {total_loss}")

    # Plotting the values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(d_values[:len(scaled_data)], label='Predicted Output (d)')
    plt.plot(d_cap_values[:len(scaled_data)], label='Actual Output (d_cap)')
    plt.xlabel('Epoch')
    plt.ylabel('Output Value')
    plt.title('Predicted vs Actual Output')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_values[:len(scaled_data)], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()

