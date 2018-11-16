from math import floor
from matplotlib.pyplot import Circle, subplots
from matplotlib.lines import Line2D


def plot_mlp(self):
    """
    An ANN with up to three layers may be plotted.
    Unfortunately, matplotlib is convoluted and far harder to
    use than it should be.
    """
    if self.hidden_layers > 3:
        # If more than three layers, exit the function
        print("Cannot plot this architecture")
        return
    # Define parameters
    self.input_coord = {}
    self.hidden_coord = {}
    self.output_coord = {}
    fig, ax = subplots(figsize=(20, 20))
    ax.axis("off")
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    def set_final_syn_pos(self):
        # Get final output synapses location ahead of time
        if self.hidden_layers == 3:
            self.final_syn_pos = 0.7
        elif self.hidden_layers == 2:
            self.final_syn_pos = 0.5
        elif self.hidden_layers == 1:
            self.final_syn_pos = 0.3

    def add_input_layer(self):
        # Generate input neuron coordinates
        def gen_x_y_coords(self):
            self.input_coord["x"] = [
                0.1 for i in range(1, self.X.shape[1] + 1)
            ]
            self.input_coord["y"] = [
                0.1 + i * 0.1 for i in range(1, self.X.shape[1] + 1)
            ]

        def gen_circles_coords(self):
            # Add circles to input neuron coordinates
            self.input_coord["circles"] = (
                Circle((x, y), 0.025, color='r')
                for x, y in zip(self.input_coord["x"], self.input_coord["y"]))

        def gen_labels_coords(self):
            # Add labels to input neuron coordinates
            self.input_coord["labels"] = [
                ax.annotate(col, xy=(x, y), fontsize=15, ha="center")
                for x, y, col in zip(self.input_coord["x"], self.
                                     input_coord["y"], self.X.columns)
            ]

        def add_circles(self):
            # Plot the circles
            for input_circle in self.input_coord["circles"]:
                ax.add_artist(input_circle)

        gen_x_y_coords(self)
        gen_circles_coords(self)
        gen_labels_coords(self)
        add_circles(self)

    def add_hidden_layers(self):
        # Create list of tuples to reuse hidden layer plotting functions
        def gen_hl_coord_list(self):
            if self.hidden_layers >= 1:
                self.hl_coord_list = [(0.1, 0.3, "HL1-Neuron ")]
                if self.hidden_layers >= 2:
                    self.hl_coord_list.append((0.3, 0.5, "HL2-Neuron "))
                    if self.hidden_layers == 3:
                        self.hl_coord_list.append((0.5, 0.7, "HL3-Neuron "))

        def gen_x_y_coords(self, coord):
            # Generate hidden neuron coordinates
            self.hidden_coord["x"] = [
                coord[1] for i in range(1, self.X.shape[1] + 2)
            ]
            self.hidden_coord["y"] = [
                0.05 + i * 0.1 for i in range(1, self.X.shape[1] + 2)
            ]

        def gen_circles_coords(self):
            # Add circles to hidden neuron coordinates
            self.hidden_coord["circles"] = (Circle(
                (x, y), 0.025, color='b') for x, y in zip(
                    self.hidden_coord["x"], self.hidden_coord["y"]))

        def gen_labels_coords(self, coord):
            # Add labels to hidden neuron coordinates
            self.hidden_coord["labels"] = [
                ax.annotate(
                    coord[2] + str(i), xy=(x, y), fontsize=15,
                    ha="center") for x, y, i in zip(
                        self.hidden_coord["x"], self.hidden_coord["y"],
                        range(1, self.X.shape[1] + 2))
            ]

        def gen_synapses_coords(self, coord, i):
            # Add synapses to hidden neuron coordinates
            if i == 0:
                self.hidden_coord["hl-synapses"] = (
                    (y1, y2) for y2 in self.hidden_coord["y"]
                    for y1 in self.input_coord["y"])
            else:
                self.hidden_coord["hl-synapses"] = (
                    (y1, y2) for y2 in self.hidden_coord["y"]
                    for y1 in self.hidden_coord["y"])

        def add_circles(self):
            # Plot the circles
            for hl_circle in self.hidden_coord["circles"]:
                ax.add_artist(hl_circle)

        def add_synapses(self, coord):
            # Plot the synapses
            for syn in self.hidden_coord["hl-synapses"]:
                ax.add_line(Line2D((coord[0], coord[1]), syn, color='y'))

        def hl_iter(self):
            # Iterate over the above functions for the appropriate
            # number of hidden layers
            for i, coord in enumerate(self.hl_coord_list):
                gen_x_y_coords(self, coord)
                gen_circles_coords(self)
                gen_labels_coords(self, coord)
                gen_synapses_coords(self, coord, i)
                add_circles(self)
                add_synapses(self, coord)

        gen_hl_coord_list(self)
        hl_iter(self)

    def add_output_layer(self):
        # Generate output neuron coordinates
        def gen_x_y_coords(self):
            # Generate output neuron coordinates
            mode_hidden = floor(len(self.hidden_coord["y"]) / 2)
            self.output_coord["x"] = 0.9
            self.output_coord["y"] = self.hidden_coord["y"][mode_hidden]

        def gen_circle_coords(self):
            # Add circles to output neuron coordinates
            self.output_coord["circle"] = Circle(
                (self.output_coord["x"], self.output_coord["y"]),
                0.025,
                color='g')

        def gen_label_coords(self):
            # Add labels to output neuron coordinates
            self.output_coord["label"] = ax.annotate(
                "Output Neuron",
                xy=(self.output_coord["x"], self.output_coord["y"]),
                fontsize=15,
                ha="center")

        def gen_synapses_coords(self):
            # Add synapses to output neuron coordinates
            self.output_coord["hl-output-synapses"] = (
                (y, self.output_coord["y"]) for y in self.hidden_coord["y"]
                for y1 in self.hidden_coord["y"])

        def add_circle(self):
            # Plot the circle
            ax.add_artist(self.output_coord["circle"])

        def add_synapses(self):
            # Plot the synapses
            for syn in self.output_coord["hl-output-synapses"]:
                ax.add_line(Line2D((self.final_syn_pos, 0.9), syn, color='y'))

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
