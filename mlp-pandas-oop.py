from numpy import exp
from numpy.random import randn
from pandas import DataFrame
# seed(0)
"""MLP Pandas"""


class MLP:
    def __init__(self, X, y):
        # Define Hypers
        """
        These dictate the structure of the ANN
        """
        self.X = X
        self.y = y
        self.input_nodes = X.shape[1]
        self.hidden_nodes = X.shape[1] + 1
        self.output_nodes = 1
        self.learning_rate = 1  # Dictates weight adjustment
        self.iterations = 500
        self.output_neuron = ["Output Neuron"]  # Just a final column label

        # Generate weights
        """
        Take random samples from the standard normal
        distribution
        """
        self.weights_1 = DataFrame(
            randn(self.input_nodes, self.hidden_nodes),
            index=X.columns,
            columns=("Input Synapses " + str(i + 1)
                     for i in range(X.shape[1] + 1)))
        self.weights_2 = DataFrame(
            randn(self.hidden_nodes, self.output_nodes),
            index=("HL1-Neuron " + str(i + 1)
                   for i in range(self.weights_1.shape[1])),
            columns=["Output Synapses"])
        self.output_neuron = ["Output Neuron"]

    def sigmoid_activation(self, neurons):
        # Activation for the input, basically 1/(1 + e^-x)
        return 1 / (1 + exp(-neurons))

    def forward_prop(self):
        # Forward Propagation
        """
        1. Matrix multiply dimensional inputs for sample 1 and weights 1
        2. Apply the sigmoid activation to (1.)
        3. Matrix multiply (2.) by weights 2
        4. Apply the sigmoid activation to (3.)
        """

        # Initialize the Hidden Layer
        self.input_dot_weights_1 = self.X.dot(self.weights_1)
        self.input_dot_weights_1.columns = self.weights_2.index

        # Perform the activation
        self.activity_2 = self.sigmoid_activation(self.input_dot_weights_1)

        # Generate the output
        self.activity_2_dot_weights_2 = self.activity_2.dot(self.weights_2)
        self.activity_2_dot_weights_2.columns = self.output_neuron
        self.predicted_output = self.sigmoid_activation(
            self.activity_2_dot_weights_2)

    def sigmoid_activation_prime(self, neurons):
        # Derivative of the sigmoid function above - Power rule & Chain rule
        # Essentially e^-x/(1+e^-x)^2
        return exp(-neurons) / (1 + exp(-neurons))**2

    def backward_prop(self):
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

        # How much was missed?
        self.miss_amount = self.predicted_output.values - self.y

        # Perform the sigmoid_activation_prime function on the output layer
        # (moving backwards)
        self.sigmoid_prime_3 = self.sigmoid_activation_prime(
            self.activity_2_dot_weights_2)

        # Calculate Delta 3
        self.delta_3 = self.miss_amount * self.sigmoid_prime_3.values

        # Calculate the gradients for weights_2
        self.d_SSE_d_weights_2 = self.activity_2.T.dot(self.delta_3)

        # Perform the sigmoid_activation_prime on the hidden layer
        self.sigmoid_prime_2 = self.sigmoid_activation_prime(
            self.input_dot_weights_1)

        # Calculate Delta 2
        self.delta_2 = (self.delta_3.dot(
            self.weights_2.T.values)).values * self.sigmoid_prime_2

        # Calculate the gradients for weights_1
        self.d_SSE_d_weights_1 = self.X.T.dot(self.delta_2)

    def mlp_train(self):
        # Training the model
        """
        1. Take everything learned from above and simply iterate
        1. By adjusting the weights with -=, error is minimized
        for every iteration
        2. Optionally, a stop condition may be added but this model
        will run for the amount of iterations provided
        """
        for i in range(self.iterations):
            # Simply go back and forth
            self.forward_prop()
            self.backward_prop()

            # Minimize error by updating weights
            self.weights_2 -= self.d_SSE_d_weights_2.values * \
                self.learning_rate
            self.weights_1 -= self.d_SSE_d_weights_1.values * \
                self.learning_rate


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
mlp = MLP(X, y)
mlp.mlp_train()

# Assign the results back to the original Pandas dataframe
films["Film Review Score"] = mlp.y
films["Predicted Score"] = mlp.predicted_output
films
