from numpy import exp


def backward_prop(self):
    # Backward propagation to adjust the model synapses
    """
    1. How wrong are the predictions?  Work backwards.
    2. Define a cost function to obtain the amount of error.
        - Sum of squares error (SSE) is used and multiplied by 0.5
        - SSE = sum(0.5 * (y - predicted_output)**2)
        - Expanding the SSE equation yields:
            SSE = sum(0.5 * (y - sigmoid_activation(
                dot(sigmoid_activation(dot(X, weights["Input-HL"])),
                weights["HL-HL"])))^2)
    2. Calculate the derivative of the sigmoid function
    3. Use the chain rule to solve for d_SSE_d_weights
    4. Adjust the weights
    """
    self.bkwd_neurons = []
    self.deltas = []
    self.d_SSE_d_weights = []

    def sigmoid_activation_prime(neurons):
        # Derivative of the sigmoid function above - Power & Chain
        # Essentially e^-x/(1+e^-x)^2
        return exp(-neurons) / (1 + exp(-neurons))**2

    def calc_miss(self):
        # How well did the model do?  Probably bad at first
        self.miss_amount = self.fwd_neurons[-1][1].values - self.y

    def bkwd_output(self):
        # Output - Now work backwards, preferring insert over append
        self.bkwd_neurons.insert(
            0, sigmoid_activation_prime(self.fwd_neurons[-1][0]))
        self.deltas.insert(0, self.miss_amount * self.bkwd_neurons[0].values)
        self.d_SSE_d_weights.insert(
            0, self.fwd_neurons[-2][0].T.dot(self.deltas[0]))

    def bkwd_output_hl_single(self):
        # Single layer: Output to hidden layers using previous delta
        # Gradient calculaton ends here for single layer ANNs
        self.bkwd_neurons.insert(
            0, sigmoid_activation_prime(self.fwd_neurons[-2][0]))
        self.deltas.insert(0, (self.deltas[0].dot(
            self.weights["HL-Output"].T.values)).values * self.bkwd_neurons[0])
        self.d_SSE_d_weights.insert(0, self.X.T.dot(self.deltas[0]))

    def bkwd_output_hl(self):
        # Unlike the single layer above, continue backpropagating through
        # the unactivated forward neurons, and activate them
        # with sigmoid_activation_prime instead
        self.bkwd_neurons.insert(
            0, sigmoid_activation_prime(self.fwd_neurons[-2][0]))
        self.deltas.insert(0, (self.deltas[0].dot(
            self.weights["HL-Output"].T.values)).values * self.bkwd_neurons[0])
        self.d_SSE_d_weights.insert(
            0, self.fwd_neurons[-3][0].T.dot(self.deltas[0]))

    def bkwd_hl_hl(self):
        # When backpropagating through hidden layers, must
        # look at previous and current layer plus the correct weight.
        # This is accomplished by reversing the forward lists and
        # offsetting
        for fwd_cur, fwd_pre, weight in zip(
                reversed(self.fwd_neurons[1:-2]),
                reversed(self.fwd_neurons[0:-3]),
                reversed(self.weights["HL-HL"][1:])):
            self.bkwd_neurons.insert(0, sigmoid_activation_prime(fwd_cur[0]))
            self.deltas.insert(0, (self.deltas[0].dot(weight.T.values)).values
                               * self.bkwd_neurons[0])
            self.d_SSE_d_weights.insert(0, fwd_pre[0].T.dot(self.deltas[0]))

    def bkwd_hl_input(self):
        # From the hidden layer to input, now the actual inputs are
        # reached and the gradient calculations are complete
        self.bkwd_neurons.insert(
            0, sigmoid_activation_prime(self.fwd_neurons[0][0]))
        self.deltas.insert(0, (self.deltas[0].dot(
            self.weights["HL-HL"][0].T.values)).values * self.bkwd_neurons[0])
        self.d_SSE_d_weights.insert(0, self.X.T.dot(self.deltas[0]))

    def update_weights_single(self):
        # Update weights based on gradient direction
        self.weights[
            "HL-Output"] -= self.d_SSE_d_weights[1].values * self.learning_rate
        self.weights[
            "Input-HL"] -= self.d_SSE_d_weights[0].values * self.learning_rate

    def update_weights(self):
        # Update weights based on gradient direction
        # while looping throught the hidden layers separately
        def update_weights_hl_hl(self):
            for i, weight in enumerate(self.weights["HL-HL"], 1):
                weight -= self.d_SSE_d_weights[i].values * self.learning_rate

        self.weights["HL-Output"] -= self.d_SSE_d_weights[
            len(self.d_SSE_d_weights) - 1].values * self.learning_rate
        update_weights_hl_hl(self)
        self.weights[
            "Input-HL"] -= self.d_SSE_d_weights[0].values * self.learning_rate

    def bkwd_check_if_single(self):
        # Check if single or multilayered and perform backward propagation
        if len(self.fwd_neurons) <= 2:
            bkwd_output_hl_single(self)
            update_weights_single(self)
        else:
            bkwd_output_hl(self)
            bkwd_hl_hl(self)
            bkwd_hl_input(self)
            update_weights(self)

    calc_miss(self)
    bkwd_output(self)
    bkwd_check_if_single(self)
