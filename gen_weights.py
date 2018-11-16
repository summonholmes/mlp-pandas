from numpy.random import randn
from pandas import DataFrame


def gen_weights(self):
    # Generate all weights randomly
    """
    Take random samples from the standard normal
    distribution
    Three sets of weights will be generated for all synapses.
    """

    def gen_input_hl_weights(self):
        # Initialize input to hidden layer synapses
        self.weights["Input-HL"] = DataFrame(
            randn(self.input_nodes, self.hidden_neurons),
            index=self.X.columns,
            columns=("Input Synapses " + str(i)
                     for i in range(1, self.hidden_neurons + 1)))

    def gen_hl_hl_weights(self):
        # Initialize hidden layer to hidden layer synapses
        self.weights["HL-HL"] = [
            DataFrame(
                randn(self.hidden_neurons, self.hidden_neurons),
                index=("HL" + str(i) + "-Neuron " + str(j)
                       for j in range(1, self.hidden_neurons + 1)),
                columns=("HL" + str(i) + "-Synapse " + str(j)
                         for j in range(1, self.hidden_neurons + 1)))
            for i in range(1, self.hidden_layers)  # Offset here
        ]

    def gen_hl_output_weights(self):
        # Initialize hidden layer to output synapses
        self.weights["HL-Output"] = DataFrame(
            randn(self.hidden_neurons, self.output_neuron),
            index=("HL" + str(self.hidden_neurons) + "-Neuron " + str(i)
                   for i in range(1, self.hidden_neurons + 1)),
            columns=["Output Synapses"])

    def check_hl_count(self):
        # If # hidden layers == 1, delete the "HL-HL" key
        if not self.weights["HL-HL"]:
            del self.weights["HL-HL"]

    gen_input_hl_weights(self)
    gen_hl_hl_weights(self)
    gen_hl_output_weights(self)
    check_hl_count(self)
