from matplotlib.pyplot import Circle, subplots
from matplotlib.lines import Line2D
from math import floor
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

    def plot_input(self):
        # Input Layer
        self.x_input = [0.1 for i in range(1, self.X.shape[1] + 1)]
        self.y_input = [0.1 + i * 0.1 for i in range(1, self.X.shape[1] + 1)]
        self.circles_input = (Circle((x, y), 0.025, color='r')
                              for x, y in zip(self.x_input, self.y_input))
        self.labels_input = [
            self.ax.annotate(col, xy=(x, y), fontsize=15, ha="center")
            for x, y, col in zip(self.x_input, self.y_input, self.X.columns)
        ]

    def plot_hidden(self):
        # Hidden Layer
        self.x_hidden = [0.4 for i in range(1, self.X.shape[1] + 2)]
        self.y_hidden = [0.05 + i * 0.1 for i in range(1, self.X.shape[1] + 2)]
        self.circles_hidden = (Circle((x, y), 0.025, color='b')
                               for x, y in zip(self.x_hidden, self.y_hidden))
        self.labels_hidden = [
            self.ax.annotate(
                "HL1-Neuron " + str(i), xy=(x, y), fontsize=15, ha="center")
            for x, y, i in zip(self.x_hidden, self.y_hidden,
                               range(1, self.X.shape[1] + 2))
        ]

    def plot_output(self):
        # Output Layer
        self.hidden_mode = floor(len(self.y_hidden) / 2)
        self.x_out = 0.8
        self.y_out = self.y_hidden[self.hidden_mode]
        self.output_circle = Circle((self.x_out, self.y_out), 0.025, color='g')
        self.label_output = self.ax.annotate(
            "Output Neuron",
            xy=(self.x_out, self.y_out),
            fontsize=15,
            ha="center")

    def plot_circles(self):
        # Plot all circles
        for input_circle in self.circles_input:
            self.ax.add_artist(input_circle)
        for hl_circle in self.circles_hidden:
            self.ax.add_artist(hl_circle)
        self.ax.add_artist(self.output_circle)

    def plot_synapses(self):
        # Synapses
        self.input_synapses = ((y1, y2) for y2 in self.y_hidden
                               for y1 in self.y_input)
        self.output_synapses = ((y, self.y_out) for y in self.y_hidden)

        # Plot all synapses
        for syn in self.input_synapses:
            self.ax.add_line(Line2D((0.1, 0.4), syn, color='y'))
        for syn in self.output_synapses:
            self.ax.add_line(Line2D((0.4, 0.8), syn, color='y'))

    def plot_mlp(self):
        # Graph the neural network
        """
        Plot each layer separately
        """
        self.fig, self.ax = subplots(figsize=(20, 20))
        self.ax.axis("off")
        self.ax.set_xlim((0, 0.9))
        self.ax.set_ylim((0.1, 1))
        self.plot_input()
        self.plot_hidden()
        self.plot_output()
        self.plot_circles()
        self.plot_synapses()


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

# View results
mlp.plot_mlp()
films
