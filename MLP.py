from gen_weights import gen_weights
from forward_prop import forward_prop
from backward_prop import backward_prop
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
