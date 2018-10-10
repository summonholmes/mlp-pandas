from numpy import dot, exp, newaxis, random
from pandas import DataFrame
"""MLP Pandas"""

# Create Dataset
"""
Use any dataset with a target
"""
test_scores = DataFrame({
    "Liberalness": [3, 5, 7, 10, 10],
    "Duration": [3, 1, 1.5, 2, 2.5],
    "Amount of Races": [1, 3, 2, 5, 7],
    "Has Female Lead": [0, 1, 0, 1, 1],
    "Has Romantic Subplot": [0, 0, 1, 1, 1],
    "Is Superhero Film": [0, 0, 0, 1, 1],
    "Film Review Score": [0, 0.65, 0.72, 0.93, 1]
})
X = test_scores.copy()  # Tensor
y = X.pop("Film Review Score")  # Target

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
Take random samples from specific ranged standard normal
distributions
"""
random.seed(0)
weights_1 = random.randn(input_nodes, hidden_nodes)  # W1
weights_2 = random.randn(hidden_nodes, output_nodes)  # W2

# Forward Propagation
"""
1. Matrix multiply dimensional inputs for sample 1 and weights 1
2. Apply the sigmoid activation to (1.)
    - 1 / ln (input_dot_weights_1)
3. Matrix multiply (2.) by weights 2
4. Apply the sigmoid activation to (3.)
"""


# Activation for the input, basically 1/e^x
def sigmoid_activation(dot_result):
    return 1 / (1 + exp(-dot_result))


input_dot_weights_1 = dot(X.values, weights_1)  # z2
activity_2 = sigmoid_activation(input_dot_weights_1)  # a2
activity_2_dot_weights_2 = dot(activity_2, weights_2)  # z3
predicted_output = sigmoid_activation(activity_2_dot_weights_2)  # yHat

# Backward Propagation - Gradient Descent - Cost Function Prime
"""
1. How wrong are the predictions?
2. Define a cost_fxn - Pretty much everything above
    - cost_fxn = sum(0.5 * (y.values[..., np.newaxis] -
        predicted_output)**2)
2. Calculate the derivative of the sigmoid function
3. Use the chain rule to solve for d_SSE_d_W
4. Adjust the weights via Gradient Descent
"""


# Derivative of the sigmoid function above
def sigmoid_activation_prime(dot_result):
    return exp(-dot_result) / ((1 + exp(-dot_result))**2)


miss_amount = -(y.values[..., newaxis] - predicted_output)
sigmoid_prime_3 = sigmoid_activation_prime(activity_2_dot_weights_2)
delta_3 = miss_amount * sigmoid_prime_3
d_SSE_d_weights_2 = dot(activity_2.T, delta_3)
sigmoid_prime_2 = sigmoid_activation_prime(input_dot_weights_1)
delta_2 = dot(delta_3, weights_2.T) * sigmoid_prime_2
d_SSE_d_weights_1 = dot(X.T, delta_2)
weights_2 -= d_SSE_d_weights_2 * learning_rate
weights_1 -= d_SSE_d_weights_1 * learning_rate

# Training the model
"""
1. Take everything learned from above and simply iterate
1. By adjusting the weights with -=, cost is minimized
for every iteration
2. Optionally, a stop condition may be added but this model
will run for the amount of iterations provided
"""
for i in range(iterations):
    # Forward Prop
    input_dot_weights_1 = dot(X.values, weights_1)
    activity_2 = sigmoid_activation(input_dot_weights_1)
    activity_2_dot_weights_2 = dot(activity_2, weights_2)
    predicted_output = sigmoid_activation(activity_2_dot_weights_2)
    # Backward Prop
    miss_amount = -(y.values[..., newaxis] - predicted_output)
    sigmoid_prime_3 = sigmoid_activation_prime(activity_2_dot_weights_2)
    delta_3 = miss_amount * sigmoid_prime_3
    d_SSE_d_weights_2 = dot(activity_2.T, delta_3)
    sigmoid_prime_2 = sigmoid_activation_prime(input_dot_weights_1)
    delta_2 = dot(delta_3, weights_2.T) * sigmoid_prime_2
    d_SSE_d_weights_1 = dot(X.T, delta_2)
    weights_2 -= d_SSE_d_weights_2 * learning_rate
    weights_1 -= d_SSE_d_weights_1 * learning_rate

# View results
X["Film Review Score"] = y
X["Predicted Score"] = predicted_output
X
