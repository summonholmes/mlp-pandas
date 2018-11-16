from gen_weights import gen_weights
from forward_prop import forward_prop
from backward_prop import backward_prop
from plot_mlp import plot_mlp
from pandas import DataFrame
"""MLP Pandas"""


class MLP:
    def __init__(self, X, y, hidden_layers):
        # Define Hyperparameters
        """
        These dictate the structure of the ANN.
        Most are auto-generated but they may be tuned here
        as well.
        """
        self.X = X
        self.y = y
        self.input_nodes = X.shape[1]
        self.hidden_neurons = X.shape[1] + 1
        self.hidden_layers = hidden_layers
        self.output_neuron = 1
        self.learning_rate = 1
        self.iterations = 500
        self.weights = {}
        self.check_hidden_input()
        gen_weights(self)

    def check_hidden_input(self):
        # Check hidden input for errors
        """
        If the number of hidden layers provided is invalid,
        default to 1.
        """
        if isinstance(self.hidden_neurons, int) is False:
            print("Valid integer not provided.  Defaulting to 1")
            self.hidden_layers = 1
        elif isinstance(self.hidden_neurons,
                        int) is True and self.hidden_neurons < 1:
            print("Valid integer not provided.  Defaulting to 1")
            self.hidden_neurons = 1

    def train_mlp(self):
        # The entire process consolidated in one loop
        """
        Alternate forward and backward propatation
        and update weights/synapses in the process.
        """
        for i in range(self.iterations):
            forward_prop(self)
            backward_prop(self)


# Create Dataset
"""
Use any numerical dataset with a target,
or process categorical data first.
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

# Predict
"""
The only three things needed to use the ANN
are the tensor, target, and number of hidden layers.

The only methods the user needs to run are plot_mlp()
and train_mlp().
"""
mlp = MLP(X, y, 1)  # Provide X, y, and # hidden layers
mlp.train_mlp()  # Iterate between forward & backward prop
plot_mlp(mlp)  # Plot the architecture if <= 3

# Reassign results back to original dataframe
"""
The last two columns show the results
of the whole process.
"""
films["Film Review Score"] = y
films["Predicted Score"] = mlp.fwd_neurons[-1][-1]
films
