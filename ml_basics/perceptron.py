import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


a=0.2
b = a**2
c = 3*b + 2
d = np.exp(c*3)

##graph
##a --> b --> c --> d


##loss function
##l = f(d_cap - d)

##activation function


def forward(a):
    b = a**2
    c = 3*b+2
    d = c*2
    print(f'a->{a},b->{b},c->{c},d->{d}')
    return d

def loss(d,d_cap):
    mse = np.mean((d_cap-d)**2)
    return mse


##now define backprop
##during backpropagation we measure/cal gradeint of the loss function
##wrt each parameter in network
def backprop(a,d_cap,d):
    grad_d = d_cap-d
    grad_c = grad_d*2
    grad_b = grad_c*3
    grad_a = grad_b*2*a
    print(f'grad_d->{grad_d},grad_c->{grad_c},grad_b->{grad_b},grad_a->{grad_a}')
    return grad_a




##now measure the gradient descent
def updated_parameter(a,grad_a,lr):
    a -= lr*grad_a
    return a

a = 0.001
d_cap = 5
lr = 0.00000001

# Lists to store values for plotting
epochs = []
loss_values = []

# Training loop
for epoch in range(10000):
    d = forward(a)
    l = loss(d, d_cap)
    grad_a = backprop(a, d_cap, d)

    a = updated_parameter(a, grad_a, lr)

    # Store values for plotting
    epochs.append(epoch + 1)
    loss_values.append(l)

    print(f"Epoch {epoch + 1}, Loss: {l}, updated_a(GD): {a}")

print(f"Final output: {forward(a)}")

# Plot the loss over epochs
plt.plot(epochs, loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
#plt.legend()
#plt.legend()



# Create a graph of the neural network with values
G = nx.DiGraph()

# Add nodes with values
G.add_node('a', label=f'a={a:.4f}')
G.add_node('b', label=f'b={b:.4f}')
G.add_node('c', label=f'c={c:.4f}')
G.add_node('d', label=f'd={d:.4f}')

# Add edges to represent the flow of data
G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])

# Draw the graph
pos = nx.spring_layout(G)
labels = nx.get_node_attributes(G, 'label')
plt.subplot(1, 2, 2)
nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
plt.title('Neural Network Graph with Values')

#plt.show()