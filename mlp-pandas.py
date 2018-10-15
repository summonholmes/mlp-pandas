from numpy import exp
from numpy.random import randn
from pandas import DataFrame
# seed(0)
"""MLP Pandas"""

# Create Dataset
"""
Use any numerical dataset with a target,
or process categorical data first
"""
films = DataFrame({
    "Liberalness": [3, 5, 7, 10, 10],
    "Duration": [3, 1, 1.5, 2, 2.5],
    "Amount of Races": [1, 3, 2, 5, 7],
    "Has Female Lead": [0, 1, 0, 1, 1],
    "Has Romantic Subplot": [0, 0, 1, 1, 1],
    "Is Superhero Film": [0, 0, 0, 1, 1],
    "Film Review Score": [0, 0.65, 0.72, 0.93, 1]
})
films.index = ("Sample " + str(i + 1) for i in range(films.shape[0]))
X = films  # Tensor
y = DataFrame(X.pop("Film Review Score"))  # Target

# Define Hypers
"""
These dictate the structure of the ANN
"""
input_nodes = X.shape[1]
hidden_nodes = X.shape[1] + 1
output_nodes = 1
learning_rate = 1  # Dictates weight adjustment
iterations = 500
output_neuron = ["Output Neuron"]  # Just a final column label

# Generate weights
"""
Take random samples from the standard normal
distribution
"""
weights_1 = DataFrame(
    randn(input_nodes, hidden_nodes),
    index=X.columns,
    columns=("Input Synapses " + str(i + 1) for i in range(X.shape[1] + 1)))
weights_2 = DataFrame(
    randn(hidden_nodes, output_nodes),
    index=("HL1-Neuron " + str(i + 1) for i in range(weights_1.shape[1])),
    columns=["Output Synapses"])

# Forward Propagation
"""
1. Matrix multiply dimensional inputs for sample 1 and weights 1
2. Apply the sigmoid activation to (1.)
3. Matrix multiply (2.) by weights 2
4. Apply the sigmoid activation to (3.)
"""


def sigmoid_activation(neurons):
    # Activation for the input, basically 1/(1 + e^-x)
    return 1 / (1 + exp(-neurons))


# Initialize the Hidden Layer
input_dot_weights_1 = X.dot(weights_1)
input_dot_weights_1.columns = weights_2.index

# Perform the activation
activity_2 = sigmoid_activation(input_dot_weights_1)

# Generate the output
activity_2_dot_weights_2 = activity_2.dot(weights_2)
activity_2_dot_weights_2.columns = output_neuron
predicted_output = sigmoid_activation(activity_2_dot_weights_2)

# Backward Propagation - Gradient Descent - Cost Function Prime
"""
1. How wrong are the predictions?  Work backwards.
2. Define a cost_fxn to obtain error - Pretty much everything above
    - Sum of squares error (SSE) is used and multiplied by 0.5
    - SSE = sum(0.5 * (y - predicted_output)**2)
    - Expanding the SSE equation yields:
        SSE = sum(0.5 * (y - sigmoid_activation(
            dot(sigmoid_activation(dot(X, weights_1)),
            weights_2)))^2)
2. Calculate the derivative of the sigmoid function
3. Use the chain rule to solve for the d_SSE_d_weights 1 & 2
4. Adjust the weights
"""


def sigmoid_activation_prime(neurons):
    # Derivative of the sigmoid function above - Power rule & Chain rule
    # Essentially e^-x/(1+e^-x)^2
    return exp(-neurons) / (1 + exp(-neurons))**2


# How much was missed?
miss_amount = predicted_output.values - y

# Perform the sigmoid_activation_prime function on the output layer
# (moving backwards)
sigmoid_prime_3 = sigmoid_activation_prime(activity_2_dot_weights_2)

# Calculate Delta 3
delta_3 = miss_amount * sigmoid_prime_3.values

# Calculate the gradients for weights_2
d_SSE_d_weights_2 = activity_2.T.dot(delta_3)

# Perform the sigmoid_activation_prime on the hidden layer
sigmoid_prime_2 = sigmoid_activation_prime(input_dot_weights_1)

# Calculate Delta 2
delta_2 = (delta_3.dot(weights_2.T.values)).values * sigmoid_prime_2

# Calculate the gradients for weights_1
d_SSE_d_weights_1 = X.T.dot(delta_2)

# Minimize error by updating weights
weights_2 -= d_SSE_d_weights_2.values * learning_rate
weights_1 -= d_SSE_d_weights_1.values * learning_rate

# Training the model
"""
1. Take everything learned from above and simply iterate
1. By adjusting the weights with -=, error is minimized
for every iteration
2. Optionally, a stop condition may be added but this model
will run for the amount of iterations provided
"""

for i in range(iterations):
    # Forward Prop
    input_dot_weights_1 = X.dot(weights_1)
    input_dot_weights_1.columns = weights_2.index
    activity_2 = sigmoid_activation(input_dot_weights_1)
    activity_2_dot_weights_2 = activity_2.dot(weights_2)
    activity_2_dot_weights_2.columns = output_neuron
    predicted_output = sigmoid_activation(activity_2_dot_weights_2)

    # Backward Prop
    miss_amount = predicted_output.values - y
    sigmoid_prime_3 = sigmoid_activation_prime(activity_2_dot_weights_2)
    delta_3 = miss_amount * sigmoid_prime_3.values
    d_SSE_d_weights_2 = activity_2.T.dot(delta_3)
    sigmoid_prime_2 = sigmoid_activation_prime(input_dot_weights_1)
    delta_2 = (delta_3.dot(weights_2.T.values)).values * sigmoid_prime_2
    d_SSE_d_weights_1 = X.T.dot(delta_2)
    weights_2 -= d_SSE_d_weights_2.values * learning_rate
    weights_1 -= d_SSE_d_weights_1.values * learning_rate

# Assign the results back to the original Pandas dataframe
films["Film Review Score"] = y
films["Predicted Score"] = predicted_output
films
