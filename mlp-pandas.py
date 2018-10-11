from numpy import dot, exp, newaxis, random
from pandas import DataFrame
random.seed(0)
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
X = films  # Tensor
y = X.pop("Film Review Score")  # Target
y = y.values[..., newaxis]
X = X.values

# Define Hypers
"""
These dictate the structure of the ANN
"""
input_nodes = X.shape[1]
hidden_nodes = X.shape[1] + 1
output_nodes = 1
learning_rate = 1  # Dictates weight adjustment
iterations = 20000

# Generate weights
"""
Take random samples from the standard normal
distribution
"""
weights_1 = random.randn(input_nodes, hidden_nodes)  # W1
weights_2 = random.randn(hidden_nodes, output_nodes)  # W2

# Forward Propagation
"""
1. Matrix multiply dimensional inputs for sample 1 and weights 1
2. Apply the sigmoid activation to (1.)
3. Matrix multiply (2.) by weights 2
4. Apply the sigmoid activation to (3.)
"""


# Activation for the input, basically 1/(1 + e^-x)
def sigmoid_activation(dot_result):
    return 1 / (1 + exp(-dot_result))


input_dot_weights_1 = dot(X, weights_1)  # z2
activity_2 = sigmoid_activation(input_dot_weights_1)  # a2
activity_2_dot_weights_2 = dot(activity_2, weights_2)  # z3
predicted_output = sigmoid_activation(activity_2_dot_weights_2)  # yHat

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


# Derivative of the sigmoid function above - Power rule & Chain rule
# Essentially e^-x/(1+e^-x)^2
def sigmoid_activation_prime(dot_result):
    return exp(-dot_result) / (1 + exp(-dot_result))**2


miss_amount = -(y - predicted_output)
sigmoid_prime_3 = sigmoid_activation_prime(activity_2_dot_weights_2)
delta_3 = miss_amount * sigmoid_prime_3
d_SSE_d_weights_2 = dot(activity_2.T, delta_3)
sigmoid_prime_2 = sigmoid_activation_prime(input_dot_weights_1)
delta_2 = dot(delta_3, weights_2.T) * sigmoid_prime_2
d_SSE_d_weights_1 = dot(X.T, delta_2)

# Minimize error by updating weights
weights_2 -= d_SSE_d_weights_2 * learning_rate
weights_1 -= d_SSE_d_weights_1 * learning_rate

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
    input_dot_weights_1 = dot(X, weights_1)
    activity_2 = sigmoid_activation(input_dot_weights_1)
    activity_2_dot_weights_2 = dot(activity_2, weights_2)
    predicted_output = sigmoid_activation(activity_2_dot_weights_2)

    # Backward Prop
    miss_amount = -(y - predicted_output)
    sigmoid_prime_3 = sigmoid_activation_prime(activity_2_dot_weights_2)
    delta_3 = miss_amount * sigmoid_prime_3
    d_SSE_d_weights_2 = dot(activity_2.T, delta_3)
    sigmoid_prime_2 = sigmoid_activation_prime(input_dot_weights_1)
    delta_2 = dot(delta_3, weights_2.T) * sigmoid_prime_2
    d_SSE_d_weights_1 = dot(X.T, delta_2)

    # Minimize error by updating weights
    weights_2 -= d_SSE_d_weights_2 * learning_rate
    weights_1 -= d_SSE_d_weights_1 * learning_rate

# Assign the results back to the original Pandas dataframe
films["Film Review Score"] = y
films["Predicted Score"] = predicted_output
films
