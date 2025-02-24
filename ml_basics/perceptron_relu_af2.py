import numpy as np
import matplotlib.pyplot as plt

### today task is to use relu and understand its more internal dynamics ,with each change in data form
# today will take simple data set and observe the changes
# later we will
#we will do in two step process
#first we will develop perceptron with input parameters send it to relu
# second we will then add feedforward later,

##initialize weight and bias
w = 1.013
bias = 0.19

def relu(w,a,bias):
    relu = np.maximum(0,w*a+bias)
    #print(f"a-->{a},b_cap-->{b_cap},relu-->{relu}")
    return relu

def leaky_relu(w,a,bias, alpha=0.01):
    x = w*a+bias
    return np.where(x > 0, x, alpha * x)

def loss(b,b_cap):
    l = np.mean((b_cap - b)**2)
    print(f"loss-->{l}")
    return l

def backprop(a,b,b_cap):
    grad_l = 2*(b_cap-b)
    grad_w = grad_l*a
    grad_b = grad_l
    print(f"grad_w-->{grad_w},grad_b--->{grad_b}")
    return grad_w,grad_b

def updated_parameters(w,b,grad_w,grad_b,learning_rate):
    w -= learning_rate*grad_w
    b -= learning_rate*grad_b
    print(f"new w-->{w},new bias-->{b}")
    return w,b

def read_values_from_file(filename):
    with open(filename, 'r') as file:
        # Skip the header line
        next(file)
        # Read the next line with the actual values
        lines = file.readlines()
        data = [list(map(float, line.strip().split(','))) for line in lines]
    return data



def main():
    global w, bias
    data = read_values_from_file('file5.txt')
    learning_rate = 0.000001
    total_loss = 0
    d_values = []
    d_cap_values = []
    loss_values = []
    for epoch in range(100):
        total_loss = 0
        for a, d_cap in data:
            print(f"loop start w-->{w},bias-->{bias}")
            d = leaky_relu( w,a, bias,alpha=0.01)
            l = loss(d_cap, d)
            grad_weights, grad_bias = backprop(a, d, d_cap)
            w, bias = updated_parameters(w, bias, grad_weights, grad_bias, learning_rate)
            test_b = w*a+bias
            print(f"after applying new weights,bias test_b-->{test_b}")
            # Store values for plotting
            d_values.append(d)
            d_cap_values.append(d_cap)
            loss_values.append(l)
            total_loss += l
        if epoch % 1 == 0:  # Adjusted to print every 10 epochs for better visibility
            print(f"Epoch {epoch}: total_loss: {total_loss}")
    # Plotting the values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(d_values[:len(data)], label='Predicted Output (d)')
    plt.plot(d_cap_values[:len(data)], label='Actual Output (d_cap)')
    plt.xlabel('Epoch')
    plt.ylabel('Output Value')
    plt.title('Predicted vs Actual Output')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_values[:len(data)], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


def test():

    global w, bias
    print(f"w-->{w},bias-->{bias}")
    # Sample dataset for testing
    sample_data = [
        [0.1, 0.2],
        [0.4, 0.5],
        [0.7, 0.8],
        [1.0, 1.1]
    ]

    # Feature scaling the sample data
    # scaled_sample_data = feature_scaling(sample_data)

    # Predicting values using the trained model
    predictions = []

    for a in sample_data:
        prediction = leaky_relu( w,a[0], bias,alpha=0.01)
        predictions.append(prediction)

    print("Predictions for the sample dataset:")
    for i in range(len(sample_data)):
        print(f"Input: {sample_data[i][0]}, Actual: {sample_data[i][1]}, Predicted: {predictions[i]}")


if __name__=="__main__":
    main()

test()