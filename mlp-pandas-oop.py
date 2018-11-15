from math import floor
from matplotlib.pyplot import Circle, subplots
from matplotlib.lines import Line2D
from numpy import exp
from numpy.random import randn
from pandas import DataFrame


class MLP:
    def __init__(self, X, y, hidden_layers):
        self.input_nodes = X.shape[1]
        self.hidden_neurons = X.shape[1] + 1
        self.hidden_layers = hidden_layers
        self.output_neuron = 1
        self.learning_rate = 1
        self.iterations = 500
        self.weights = {}
        self.check_hidden_input()
        self.gen_weights()

    def check_hidden_input(self):
        if isinstance(self.hidden_neurons, int) is False:
            print("Valid integer not provided.  Defaulting to 1")
            self.hidden_layers = 1
        elif isinstance(self.hidden_neurons,
                        int) is True and self.hidden_neurons < 1:
            print("Valid integer not provided.  Defaulting to 1")
            self.hidden_neurons = 1

    def gen_weights(self):
        def gen_input_hl_weights(self):
            self.weights["Input-HL"] = DataFrame(
                randn(self.input_nodes, self.hidden_neurons),
                index=X.columns,
                columns=("Input Synapses " + str(i)
                         for i in range(1, self.hidden_neurons + 1)))

        def gen_hl_hl_weights(self):
            self.weights["HL-HL"] = [
                DataFrame(
                    randn(self.hidden_neurons, self.hidden_neurons),
                    index=("HL" + str(i) + "-Neuron " + str(j)
                           for j in range(1, self.hidden_neurons + 1)),
                    columns=("HL" + str(i) + "-Synapse " + str(j)
                             for j in range(1, self.hidden_neurons + 1)))
                for i in range(1, self.hidden_neurons)  # Offset here
            ]

        def gen_hl_output_weights(self):
            self.weights["HL-Output"] = DataFrame(
                randn(self.hidden_neurons, self.output_neuron),
                index=("HL" + str(self.hidden_neurons) + "-Neuron " + str(i)
                       for i in range(1, self.hidden_neurons + 1)),
                columns=["Output Synapses"])

        def check_hl_count(self):
            if not self.weights["HL-HL"]:
                del self.weights["HL-HL"]

        gen_input_hl_weights(self)
        gen_hl_hl_weights(self)
        gen_hl_output_weights(self)
        check_hl_count(self)

    def forward_prop(self):
        self.fwd_neurons = []

        def sigmoid_activation(neurons):
            # Activation for the input, basically 1/(1 + e^-x)
            return 1 / (1 + exp(-neurons))

        def fwd_input_hl(self):
            self.fwd_neurons.append(X.dot(self.weights["Input-HL"]))
            self.fwd_neurons[-1].columns = self.weights["HL-HL"][0].index
            self.fwd_neurons[-1] = (self.fwd_neurons[-1],
                                    sigmoid_activation(self.fwd_neurons[-1]))

        def fwd_hl_hl(self):
            for weight_1, weight_2 in zip(self.weights["HL-HL"][:-1],
                                          self.weights["HL-HL"][1:]):
                self.fwd_neurons.append(self.fwd_neurons[-1][1].dot(weight_1))
                self.fwd_neurons[-1].columns = weight_2.index
                self.fwd_neurons[-1] = (self.fwd_neurons[-1],
                                        sigmoid_activation(
                                            self.fwd_neurons[-1]))

        def fwd_hl_output(self):
            self.fwd_neurons.append(self.fwd_neurons[-1][1].dot(
                self.weights["HL-HL"][-1]))
            self.fwd_neurons[-1].columns = self.weights["HL-Output"].index
            self.fwd_neurons[-1] = (self.fwd_neurons[-1],
                                    sigmoid_activation(self.fwd_neurons[-1]))

        def fwd_output(self):
            self.fwd_neurons.append(self.fwd_neurons[-1][1].dot(
                self.weights["HL-Output"]))
            self.fwd_neurons[-1].columns = ["Output Neuron"]
            self.fwd_neurons[-1] = (self.fwd_neurons[-1],
                                    sigmoid_activation(self.fwd_neurons[-1]))

        def fwd_input_hl_single(self):
            self.fwd_neurons.append(X.dot(self.weights["Input-HL"]))
            self.fwd_neurons[-1].columns = self.weights["HL-Output"].index
            self.fwd_neurons[-1] = (self.fwd_neurons[-1],
                                    sigmoid_activation(self.fwd_neurons[-1]))

        def fwd_hl_output_single(self):
            self.fwd_neurons.append(self.fwd_neurons[-1][1].dot(
                self.weights["HL-Output"]))
            self.fwd_neurons[-1].columns = ["Output Neuron"]
            self.fwd_neurons[-1] = (self.fwd_neurons[-1],
                                    sigmoid_activation(self.fwd_neurons[-1]))

        def fwd_check_single(self):
            if "HL-HL" in self.weights:
                fwd_input_hl(self)
                fwd_hl_hl(self)
                fwd_hl_output(self)
                fwd_output(self)
            else:
                fwd_input_hl_single(self)
                fwd_hl_output_single(self)

        fwd_check_single(self)

    def backward_prop(self):
        self.bkwd_neurons = []
        self.deltas = []
        self.d_SSE_d_weights = []

        def sigmoid_activation_prime(neurons):
            # Derivative of the sigmoid function above - Power & Chain
            # Essentially e^-x/(1+e^-x)^2
            return exp(-neurons) / (1 + exp(-neurons))**2

        def calc_miss(self):
            self.miss_amount = self.fwd_neurons[-1][1].values - y

        def bkwd_output(self):
            self.bkwd_neurons.insert(
                0, sigmoid_activation_prime(self.fwd_neurons[-1][0]))
            self.deltas.insert(0,
                               self.miss_amount * self.bkwd_neurons[0].values)
            self.d_SSE_d_weights.insert(
                0, self.fwd_neurons[-2][0].T.dot(self.deltas[0]))

        def bkwd_output_hl_single(self):
            self.bkwd_neurons.insert(
                0, sigmoid_activation_prime(self.fwd_neurons[-2][0]))
            self.deltas.insert(
                0,
                (self.deltas[0].dot(self.weights["HL-Output"].T.values)).values
                * self.bkwd_neurons[0])
            self.d_SSE_d_weights.insert(0, X.T.dot(self.deltas[0]))

        def bkwd_output_hl(self):
            self.bkwd_neurons.insert(
                0, sigmoid_activation_prime(self.fwd_neurons[-2][0]))
            self.deltas.insert(
                0,
                (self.deltas[0].dot(self.weights["HL-Output"].T.values)).values
                * self.bkwd_neurons[0])
            self.d_SSE_d_weights.insert(
                0, self.fwd_neurons[-3][0].T.dot(self.deltas[0]))

        def bkwd_hl_hl(self):
            for fwd_cur, fwd_pre, weight in zip(
                    reversed(self.fwd_neurons[1:-2]),
                    reversed(self.fwd_neurons[0:-3]),
                    reversed(self.weights["HL-HL"][1:])):
                self.bkwd_neurons.insert(0,
                                         sigmoid_activation_prime(fwd_cur[0]))
                self.deltas.insert(0, (self.deltas[0].dot(
                    weight.T.values)).values * self.bkwd_neurons[0])
                self.d_SSE_d_weights.insert(0,
                                            fwd_pre[0].T.dot(self.deltas[0]))

        def bkwd_hl_input(self):
            self.bkwd_neurons.insert(
                0, sigmoid_activation_prime(self.fwd_neurons[0][0]))
            self.deltas.insert(
                0,
                (self.deltas[0].dot(self.weights["HL-HL"][0].T.values)).values
                * self.bkwd_neurons[0])
            self.d_SSE_d_weights.insert(0, X.T.dot(self.deltas[0]))

        def update_weights_single(self):
            self.weights["HL-Output"] -= self.d_SSE_d_weights[
                1].values * self.learning_rate
            self.weights["Input-HL"] -= self.d_SSE_d_weights[
                0].values * self.learning_rate

        def update_weights(self):
            def update_weights_hl_hl(self):
                for i, weight in enumerate(self.weights["HL-HL"], 1):
                    weight -= self.d_SSE_d_weights[
                        i].values * self.learning_rate

            self.weights["HL-Output"] -= self.d_SSE_d_weights[
                len(self.d_SSE_d_weights) - 1].values * self.learning_rate
            update_weights_hl_hl(self)
            self.weights["Input-HL"] -= self.d_SSE_d_weights[
                0].values * self.learning_rate

        def bkwd_check_if_single(self):
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

    def plot_mlp(self):
        if self.hidden_layers > 3:
            print("Cannot plot this architecture")
            return

        self.input_coord = {}
        self.hidden_coord = {}
        self.output_coord = {}
        fig, ax = subplots(figsize=(20, 20))
        ax.axis("off")
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))

        def set_final_syn_pos(self):
            if self.hidden_layers == 3:
                self.final_syn_pos = 0.7
            elif self.hidden_layers == 2:
                self.final_syn_pos = 0.5
            elif self.hidden_layers == 1:
                self.final_syn_pos = 0.3

        def add_input_layer(self):
            def gen_x_y_coords(self):
                self.input_coord["x"] = [0.1 for i in range(1, X.shape[1] + 1)]
                self.input_coord["y"] = [
                    0.1 + i * 0.1 for i in range(1, X.shape[1] + 1)
                ]

            def gen_circles_labels_coords(self):
                self.input_coord["circles"] = (Circle(
                    (x, y), 0.025, color='r') for x, y in zip(
                        self.input_coord["x"], self.input_coord["y"]))
                self.input_coord["labels"] = [
                    ax.annotate(col, xy=(x, y), fontsize=15, ha="center")
                    for x, y, col in zip(self.input_coord["x"], self.
                                         input_coord["y"], X.columns)
                ]

            def add_circles(self):
                for input_circle in self.input_coord["circles"]:
                    ax.add_artist(input_circle)

            gen_x_y_coords(self)
            gen_circles_labels_coords(self)
            add_circles(self)

        def add_hidden_layers(self):
            def gen_x_y_coords(self):
                self.hidden_coord["x"] = [
                    0.3 for i in range(1, X.shape[1] + 2)
                ]
                self.hidden_coord["y"] = [
                    0.05 + i * 0.1 for i in range(1, X.shape[1] + 2)
                ]

            def gen_circles_coords(self):
                self.hidden_coord["circles"] = (Circle(
                    (x, y), 0.025, color='b') for x, y in zip(
                        self.hidden_coord["x"], self.hidden_coord["y"]))

            def gen_labels_coords(self):
                self.hidden_coord["labels"] = [
                    ax.annotate(
                        "HL1-Neuron " + str(i),
                        xy=(x, y),
                        fontsize=15,
                        ha="center") for x, y, i in zip(
                            self.hidden_coord["x"], self.hidden_coord["y"],
                            range(1, X.shape[1] + 2))
                ]

            def gen_synapses_coords(self):
                self.hidden_coord["input-hl-synapses"] = (
                    (y1, y2) for y2 in self.hidden_coord["y"]
                    for y1 in self.input_coord["y"])

            def add_circles(self):
                for hl_circle in self.hidden_coord["circles"]:
                    ax.add_artist(hl_circle)

            def add_synapses(self):
                for syn in self.hidden_coord["input-hl-synapses"]:
                    ax.add_line(Line2D((0.1, 0.3), syn, color='y'))

            def gen_x2_coords(self):
                self.hidden_coord["x2"] = [
                    0.5 for i in range(1, X.shape[1] + 2)
                ]

            def gen_circles2_coords(self):
                self.hidden_coord["circles2"] = (Circle(
                    (x, y), 0.025, color='b') for x, y in zip(
                        self.hidden_coord["x2"], self.hidden_coord["y"]))

            def gen_labels2_coords(self):
                self.hidden_coord["labels2"] = [
                    ax.annotate(
                        "HL2-Neuron " + str(i),
                        xy=(x, y),
                        fontsize=15,
                        ha="center") for x, y, i in zip(
                            self.hidden_coord["x2"], self.hidden_coord["y"],
                            range(1, X.shape[1] + 2))
                ]

            def gen_synapses2_coords(self):
                self.hidden_coord["hl-hl-synapses"] = (
                    (y1, y2) for y2 in self.hidden_coord["y"]
                    for y1 in self.hidden_coord["y"])

            def add_circles2(self):
                for hl_circle in self.hidden_coord["circles2"]:
                    ax.add_artist(hl_circle)

            def add_synapses2(self):
                for syn in self.hidden_coord["hl-hl-synapses"]:
                    ax.add_line(Line2D((0.3, 0.5), syn, color='y'))

            def gen_x3_coords(self):
                self.hidden_coord["x3"] = [
                    0.7 for i in range(1, X.shape[1] + 2)
                ]

            def gen_circles3_coords(self):
                self.hidden_coord["circles3"] = (Circle(
                    (x, y), 0.025, color='b') for x, y in zip(
                        self.hidden_coord["x3"], self.hidden_coord["y"]))

            def gen_labels3_coords(self):
                self.hidden_coord["labels3"] = [
                    ax.annotate(
                        "HL3-Neuron " + str(i),
                        xy=(x, y),
                        fontsize=15,
                        ha="center") for x, y, i in zip(
                            self.hidden_coord["x3"], self.hidden_coord["y"],
                            range(1, X.shape[1] + 2))
                ]

            def gen_synapses3_coords(self):
                self.hidden_coord["hl-hl-synapses2"] = (
                    (y1, y2) for y2 in self.hidden_coord["y"]
                    for y1 in self.hidden_coord["y"])

            def add_circles3(self):
                for hl_circle in self.hidden_coord["circles3"]:
                    ax.add_artist(hl_circle)

            def add_synapses3(self):
                for syn in self.hidden_coord["hl-hl-synapses2"]:
                    ax.add_line(Line2D((0.5, 0.7), syn, color='y'))

            gen_x_y_coords(self)
            gen_circles_coords(self)
            gen_labels_coords(self)
            gen_synapses_coords(self)
            add_circles(self)
            add_synapses(self)

            if self.hidden_layers >= 2:
                gen_x2_coords(self)
                gen_circles2_coords(self)
                gen_labels2_coords(self)
                gen_synapses2_coords(self)
                add_circles2(self)
                add_synapses2(self)
                if self.hidden_layers == 3:
                    gen_x3_coords(self)
                    gen_circles3_coords(self)
                    gen_labels3_coords(self)
                    gen_synapses3_coords(self)
                    add_circles3(self)
                    add_synapses3(self)

        def add_output_layer(self):
            def gen_x_y_coords(self):
                mode_hidden = floor(len(self.hidden_coord["y"]) / 2)
                self.output_coord["x"] = 0.9
                self.output_coord["y"] = self.hidden_coord["y"][mode_hidden]

            def gen_circle_coords(self):
                self.output_coord["circle"] = Circle(
                    (self.output_coord["x"], self.output_coord["y"]),
                    0.025,
                    color='g')

            def gen_label_coords(self):
                self.output_coord["label"] = ax.annotate(
                    "Output Neuron",
                    xy=(self.output_coord["x"], self.output_coord["y"]),
                    fontsize=15,
                    ha="center")

            def gen_synapses_coords(self):
                self.output_coord["hl-output-synapses"] = (
                    (y, self.output_coord["y"]) for y in self.hidden_coord["y"]
                    for y1 in self.hidden_coord["y"])

            def add_circle(self):
                ax.add_artist(self.output_coord["circle"])

            def add_synapses(self):
                for syn in self.output_coord["hl-output-synapses"]:
                    ax.add_line(
                        Line2D((self.final_syn_pos, 0.9), syn, color='y'))

            gen_x_y_coords(self)
            gen_circle_coords(self)
            gen_label_coords(self)
            gen_synapses_coords(self)
            add_circle(self)
            add_synapses(self)

        set_final_syn_pos(self)
        add_input_layer(self)
        add_hidden_layers(self)
        add_output_layer(self)

    def train_mlp(self):
        for i in range(self.iterations):
            self.forward_prop()
            self.backward_prop()


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
mlp = MLP(X, y, 3)  # Provide X, y, and # hidden layers
mlp.gen_weights()
mlp.plot_mlp()  # Plot the architecture if <= 3
mlp.train_mlp()  # Iterate between forward & backward prop

# Reassign results back to original dataframe
films["Film Review Score"] = y
films["Predicted Score"] = mlp.fwd_neurons[-1][-1]
films
