from numpy import dot, exp, newaxis, random
from pandas import DataFrame
"""
MLP Pandas

When adapting the bare Jupyter script
into object-oriented form, the code
is easier to interpret.  The tradeoff
is that abstraction increases debugging difficulty.
"""


class MLP:
    random.seed(0)

    def __init__(self, dataset):
        # Set Hypers & Initialize weights
        """
        These dictate the structure of the ANN
        """
        self.X = dataset.copy()
        self.y = self.X.pop("Film Review Score")
        self.y = self.y.values[..., newaxis]
        self.X = self.X.values
        self.input_nodes = self.X.shape[1]
        self.hidden_nodes = self.X.shape[1] + 1
        self.output_nodes = 1
        self.learning_rate = 1  # Dictates weight adjustment
        self.iterations = 20000
        """
        Take random samples from the standard normal
        distribution
        """
        self.weights_1 = random.randn(self.input_nodes,
                                      self.hidden_nodes)  # W1
        self.weights_2 = random.randn(self.hidden_nodes,
                                      self.output_nodes)  # W2

    def sigmoid_activation(self, dot_result):
        # Activation for the input, basically 1/(1 + e^-x)
        return 1 / (1 + exp(-dot_result))

    def forward_prop(self):
        # Forward Propagation
        """
        1. Matrix multiply dimensional inputs for samples and weights 1
        2. Apply the sigmoid activation to (1.)
        3. Matrix multiply (2.) by weights 2
        4. Apply the sigmoid activation to (3.)
        """
        self.input_dot_weights_1 = dot(self.X, self.weights_1)
        self.activity_2 = self.sigmoid_activation(self.input_dot_weights_1)
        self.activity_2_dot_weights_2 = dot(self.activity_2, self.weights_2)
        self.predicted_output = self.sigmoid_activation(
            self.activity_2_dot_weights_2)

    def sigmoid_activation_prime(self, dot_result):
        # Derivative of the sigmoid function above (Power rule).
        return exp(-dot_result) / (1 + exp(-dot_result))**2

    def backward_prop(self):
        # Backward Propagation - Gradient Descent - Cost Function Prime
        """
        1. How wrong are the predictions?
        2. Define a cost_fxn - Pretty much everything above
            - cost_fxn = sum(0.5 * (y -
                predicted_output)**2)
        2. Calculate the derivative of the sigmoid function
        3. Use the chain rule to solve for d_SSE_d_W
        4. Adjust the weights via Gradient Descent
        """
        self.miss_amount = -(self.y - self.predicted_output)
        self.sigmoid_prime_3 = self.sigmoid_activation_prime(
            self.activity_2_dot_weights_2)
        self.delta_3 = self.miss_amount * self.sigmoid_prime_3
        self.d_SSE_d_weights_2 = dot(self.activity_2.T, self.delta_3)
        self.sigmoid_prime_2 = self.sigmoid_activation_prime(
            self.input_dot_weights_1)
        self.delta_2 = dot(self.delta_3,
                           self.weights_2.T) * self.sigmoid_prime_2
        self.d_SSE_d_weights_1 = dot(self.X.T, self.delta_2)
        self.weights_2 -= self.d_SSE_d_weights_2 * self.learning_rate
        self.weights_1 -= self.d_SSE_d_weights_1 * self.learning_rate

    def mlp_train(self):
        # Training the model
        """
        1. Take everything learned from above and simply iterate
        1. By adjusting the weights with -=, cost is minimized
        for every iteration
        2. Optionally, a stop condition may be added but this model
        will run for the amount of iterations provided
        """
        for i in range(self.iterations):
            self.forward_prop()
            self.backward_prop()


# Create Dataset
"""
Use any dataset with a target
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

# Test the object
mlp = MLP(films)
mlp.mlp_train()
mlp.predicted_output
mlp.y
