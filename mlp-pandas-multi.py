from numpy import exp
from numpy.random import randn
from pandas import DataFrame
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
hidden_neurons = X.shape[1] + 1
hidden_layers = 1
output_neuron = 1
learning_rate = 1  # Dictates weight adjustment
iterations = 500

# Generate weights
"""
Take random samples from the standard normal
distribution
"""
weights = {
    "Input-HL":
    DataFrame(
        randn(input_nodes, hidden_neurons),
        index=X.columns,
        columns=("Input Synapses " + str(i)
                 for i in range(1, hidden_neurons + 1))),
    "HL-HL": [
        DataFrame(
            randn(hidden_neurons, hidden_neurons),
            index=("HL" + str(i) + "-Neuron " + str(j)
                   for j in range(1, hidden_neurons + 1)),
            columns=("HL" + str(i) + "-Synapse " + str(j)
                     for j in range(1, hidden_neurons + 1)))
        for i in range(1, hidden_layers)  # Offset here
    ],
    "HL-Output":
    DataFrame(
        randn(hidden_neurons, output_neuron),
        index=("HL" + str(hidden_layers) + "-Neuron " + str(i)
               for i in range(1, hidden_neurons + 1)),
        columns=["Output Synapses"])
}

if not weights["HL-HL"]:
    del weights["HL-HL"]

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


# Forward Prop
fwd_neurons = []
if "HL-HL" in weights:
    fwd_neurons.append(X.dot(weights["Input-HL"]))
    fwd_neurons[-1].columns = weights["HL-HL"][0].index
    fwd_neurons[-1] = (fwd_neurons[-1], sigmoid_activation(fwd_neurons[-1]))

    for weight_1, weight_2 in zip(weights["HL-HL"][:-1], weights["HL-HL"][1:]):
        fwd_neurons.append(fwd_neurons[-1][1].dot(weight_1))
        fwd_neurons[-1].columns = weight_2.index
        fwd_neurons[-1] = (fwd_neurons[-1],
                           sigmoid_activation(fwd_neurons[-1]))

    fwd_neurons.append(fwd_neurons[-1][1].dot(weights["HL-HL"][-1]))
    fwd_neurons[-1].columns = weights["HL-Output"].index
    fwd_neurons[-1] = (fwd_neurons[-1], sigmoid_activation(fwd_neurons[-1]))

    fwd_neurons.append(fwd_neurons[-1][1].dot(weights["HL-Output"]))
    fwd_neurons[-1].columns = ["Output Neuron"]
    fwd_neurons[-1] = (fwd_neurons[-1], sigmoid_activation(fwd_neurons[-1]))
else:
    fwd_neurons.append(X.dot(weights["Input-HL"]))
    fwd_neurons[-1].columns = weights["HL-Output"].index
    fwd_neurons[-1] = (fwd_neurons[-1], sigmoid_activation(fwd_neurons[-1]))

    fwd_neurons.append(fwd_neurons[-1][1].dot(weights["HL-Output"]))
    fwd_neurons[-1].columns = ["Output Neuron"]
    fwd_neurons[-1] = (fwd_neurons[-1], sigmoid_activation(fwd_neurons[-1]))

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


# Backprop
bkwd_neurons = []
deltas = []
d_SSE_d_weights = []

miss_amount = fwd_neurons[-1][1].values - y
bkwd_neurons.insert(0, sigmoid_activation_prime(fwd_neurons[-1][0]))
deltas.insert(0, miss_amount * bkwd_neurons[0].values)
d_SSE_d_weights.insert(0, fwd_neurons[-2][0].T.dot(deltas[0]))

bkwd_neurons.insert(0, sigmoid_activation_prime(fwd_neurons[-2][0]))
deltas.insert(
    0, (deltas[0].dot(weights["HL-Output"].T.values)).values * bkwd_neurons[0])
d_SSE_d_weights.insert(0, X.T.dot(deltas[0]))

weights["HL-Output"] -= d_SSE_d_weights[1].values * learning_rate
weights["Input-HL"] -= d_SSE_d_weights[0].values * learning_rate

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
    fwd_neurons = []
    if "HL-HL" in weights:
        fwd_neurons.append(X.dot(weights["Input-HL"]))
        fwd_neurons[-1].columns = weights["HL-HL"][0].index
        fwd_neurons[-1] = (fwd_neurons[-1],
                           sigmoid_activation(fwd_neurons[-1]))

        for weight_1, weight_2 in zip(weights["HL-HL"][:-1],
                                      weights["HL-HL"][1:]):
            fwd_neurons.append(fwd_neurons[-1][1].dot(weight_1))
            fwd_neurons[-1].columns = weight_2.index
            fwd_neurons[-1] = (fwd_neurons[-1],
                               sigmoid_activation(fwd_neurons[-1]))

        fwd_neurons.append(fwd_neurons[-1][1].dot(weights["HL-HL"][-1]))
        fwd_neurons[-1].columns = weights["HL-Output"].index
        fwd_neurons[-1] = (fwd_neurons[-1],
                           sigmoid_activation(fwd_neurons[-1]))

        fwd_neurons.append(fwd_neurons[-1][1].dot(weights["HL-Output"]))
        fwd_neurons[-1].columns = ["Output Neuron"]
        fwd_neurons[-1] = (fwd_neurons[-1],
                           sigmoid_activation(fwd_neurons[-1]))
    else:
        fwd_neurons.append(X.dot(weights["Input-HL"]))
        fwd_neurons[-1].columns = weights["HL-Output"].index
        fwd_neurons[-1] = (fwd_neurons[-1],
                           sigmoid_activation(fwd_neurons[-1]))

        fwd_neurons.append(fwd_neurons[-1][1].dot(weights["HL-Output"]))
        fwd_neurons[-1].columns = ["Output Neuron"]
        fwd_neurons[-1] = (fwd_neurons[-1],
                           sigmoid_activation(fwd_neurons[-1]))

    # Backward Prop
    bkwd_neurons = []
    deltas = []
    d_SSE_d_weights = []

    miss_amount = fwd_neurons[-1][1].values - y
    bkwd_neurons.insert(0, sigmoid_activation_prime(fwd_neurons[-1][0]))
    deltas.insert(0, miss_amount * bkwd_neurons[0].values)
    d_SSE_d_weights.insert(0, fwd_neurons[-2][0].T.dot(deltas[0]))

    bkwd_neurons.insert(0, sigmoid_activation_prime(fwd_neurons[-2][0]))
    deltas.insert(0, (deltas[0].dot(weights["HL-Output"].T.values)).values *
                  bkwd_neurons[0])
    d_SSE_d_weights.insert(0, X.T.dot(deltas[0]))

    weights["HL-Output"] -= d_SSE_d_weights[1].values * learning_rate
    weights["Input-HL"] -= d_SSE_d_weights[0].values * learning_rate

# Assign the results back to the original Pandas dataframe
films["Film Review Score"] = y
films["Predicted Score"] = fwd_neurons[-1][-1]
films
