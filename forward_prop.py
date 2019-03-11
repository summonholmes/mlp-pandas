from numpy import exp


def forward_prop(self):
    # Forward propagation to generate model output
    """
    1. Matrix multiply dimensional inputs for X and with Input-HL
    weights.
    2. Apply the sigmoid activation to (1.).
    3. Matrix multiply (2.) by HL-HL weights and
    loop to complete the inner HL layers.
    4. Apply the sigmoid activation to (3.).
    5. Matrix multiply (4.) by HL-Output Weights.
    6. Apply the sigmoid activation to (5.).
    Sigmoid activations will be tupled with the pre-activated
    tensors to make backpropagation easier.
    """
    self.fwd_neurons = []

    def sigmoid_activation(neurons):
        # Activation for the input, basically 1/(1 + e^-x)
        return 1 / (1 + exp(-neurons))

    def fwd_input_hl(self):
        # Forward to Hidden Layer
        self.fwd_neurons.append(self.X.dot(self.weights["Input-HL"]))
        self.fwd_neurons[-1].columns = self.weights["HL-HL"][0].index
        self.fwd_neurons[-1] = (
            self.fwd_neurons[-1],
            sigmoid_activation(self.fwd_neurons[-1]),
        )

    def fwd_hl_hl(self):
        # Hidden layer to hidden layer
        for weight_1, weight_2 in zip(
                self.weights["HL-HL"][:-1],
                self.weights["HL-HL"][1:],
        ):
            self.fwd_neurons.append(self.fwd_neurons[-1][1].dot(weight_1))
            self.fwd_neurons[-1].columns = weight_2.index
            self.fwd_neurons[-1] = (
                self.fwd_neurons[-1],
                sigmoid_activation(self.fwd_neurons[-1]),
            )

    def fwd_hl_output(self):
        # Hidden layer to output
        self.fwd_neurons.append(self.fwd_neurons[-1][1].dot(
            self.weights["HL-HL"][-1]))
        self.fwd_neurons[-1].columns = self.weights["HL-Output"].index
        self.fwd_neurons[-1] = (
            self.fwd_neurons[-1],
            sigmoid_activation(self.fwd_neurons[-1]),
        )

    def fwd_output(self):
        # Finalize output
        self.fwd_neurons.append(self.fwd_neurons[-1][1].dot(
            self.weights["HL-Output"]))
        self.fwd_neurons[-1].columns = ["Output Neuron"]
        self.fwd_neurons[-1] = (
            self.fwd_neurons[-1],
            sigmoid_activation(self.fwd_neurons[-1]),
        )

    def fwd_input_hl_single(self):
        # If single layer, the weight multiplications finish here
        self.fwd_neurons.append(self.X.dot(self.weights["Input-HL"]))
        self.fwd_neurons[-1].columns = self.weights["HL-Output"].index
        self.fwd_neurons[-1] = (
            self.fwd_neurons[-1],
            sigmoid_activation(self.fwd_neurons[-1]),
        )

    def fwd_hl_output_single(self):
        # If single layer, output is finalized here
        self.fwd_neurons.append(self.fwd_neurons[-1][1].dot(
            self.weights["HL-Output"]))
        self.fwd_neurons[-1].columns = ["Output Neuron"]
        self.fwd_neurons[-1] = (
            self.fwd_neurons[-1],
            sigmoid_activation(self.fwd_neurons[-1]),
        )

    def fwd_check_single(self):
        # Check if single or multilayered and perform forward propagation
        if "HL-HL" in self.weights:
            fwd_input_hl(self)
            fwd_hl_hl(self)
            fwd_hl_output(self)
            fwd_output(self)
        else:
            fwd_input_hl_single(self)
            fwd_hl_output_single(self)

    fwd_check_single(self)
