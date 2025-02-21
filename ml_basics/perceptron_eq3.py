import numpy as np
import matplotlib.pyplot as plt



##create network of formulated layers

##a,b,c --> d = 3a+2b/c


##initialize weights

weights = np.array([3.0,2.0])
bias = 0.0

def forward(a,b,c,weights,bias):
    ##try1
    ##d = (3*a+2*b)/c ##there is small limitation or glitch here

    ##try2
    ##we will not have weights known before it self,this needs to be measure as well.thats were neural network and learning comes in
    ##d = (w1 * a + w2 * b) / c
    d = (weights[0]*a + weights[1]*b)/c

    ##by adding some bias we are trying to reduce some loss in the function
    ##but problem with below constant value is ,it doesnt know what is predicted value, instead it will always keep adding below value to output,
    ##so sometimes this value may give near loss or less loss value or sometimes it can be huge than actual predicted value
    ##to avoid such situation ,we need more measures to calculate bias as well in runtime

    return d+bias

##now can i have function to calculate my bias based on loss and inputs

##create end loss function to calculate difference b/w real output and predicted output(formula output)
##create any loss function,here in this case we create mean squared error function becoz its an single linear regression equation


def loss(d_cap,d):
    l = np.mean((d_cap - d)**2)
    return l

##now we will try to get the gradient values based on inputs,output d and error using loss function

def gradients(a,b,c,d_cap,d):
    ##gradient of loss with respect to weights and bias
    ##partial derivative of loss_pd,w1,w2
    loss_pd = 2*(d-d_cap)
    grad_weights = np.array([loss_pd*a/c,loss_pd*b/c])
    grad_bias = 1.0
    return grad_weights,grad_bias

##now we need to iteratively pass the gradient values to equation and measure the loss and its predicted output d_cap

def updated_parameters(weights,bias,grad_weights,grad_bias,learning_rate):
    weights -= learning_rate*grad_weights
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

def main():
    global weights, bias
    filename = 'file3.txt'
    data = read_values_from_file(filename)



    ##by only increasing the weights based on output it could lead any of below issues in neural network
    ##overfitting,
    # model unstability,
    # gradient explosion,
    # ignoring other parameters

    learning_rate=0.000001
    total_loss = 0
    # Lists to store values for plotting
    d_values = []
    d_cap_values = []
    loss_values = []

    for epoch in range(10000):
        total_loss = 0
        for a, b, c, d_cap in data:
            d = forward(a, b, c, weights, bias)
            l = loss(d_cap, d)
            grad_weights, grad_bias = gradients(a, b, c, d_cap, d)
            weights, bias = updated_parameters(weights, bias, grad_weights, grad_bias, learning_rate)
            # Store values for plotting
            d_values.append(d)
            d_cap_values.append(d_cap)
            loss_values.append(l)
            total_loss += l
        if epoch % 10 == 0:  # Adjusted to print every 10 epochs for better visibility
            print(f"Epoch {epoch}: total_loss: {total_loss}")

        # Plotting the values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(d_values, label='Predicted Output (d)')
    plt.plot(d_cap_values, label='Actual Output (d_cap)')
    plt.xlabel('Epoch')
    plt.ylabel('Output Value')
    plt.title('Predicted vs Actual Output')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()




