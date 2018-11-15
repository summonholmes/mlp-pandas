from math import floor
from matplotlib.pyplot import Circle, subplots
from matplotlib.lines import Line2D
from pandas import DataFrame

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

# Graph the neural network
"""
Perform graphing on each layer separately
"""
fig, ax = subplots(figsize=(20, 20))
ax.axis("off")
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))

# Input Layer
x_input = [0.1 for i in range(1, X.shape[1] + 1)]
y_input = [0.1 + i * 0.1 for i in range(1, X.shape[1] + 1)]
circles_input = (Circle((x, y), 0.025, color='r')
                 for x, y in zip(x_input, y_input))
labels_input = [
    ax.annotate(col, xy=(x, y), fontsize=15, ha="center")
    for x, y, col in zip(x_input, y_input, X.columns)
]

# Hidden Layer
x_hidden = [0.4 for i in range(1, X.shape[1] + 2)]
y_hidden = [0.05 + i * 0.1 for i in range(1, X.shape[1] + 2)]
circles_hidden = (Circle((x, y), 0.025, color='b')
                  for x, y in zip(x_hidden, y_hidden))
labels_hidden = [
    ax.annotate("HL1-Neuron " + str(i), xy=(x, y), fontsize=15, ha="center")
    for x, y, i in zip(x_hidden, y_hidden, range(1, X.shape[1] + 2))
]

# Output Layer
hidden_mode = floor(len(y_hidden) / 2)
x_out = 0.8
y_out = y_hidden[hidden_mode]
output_circle = Circle((x_out, y_out), 0.025, color='g')
label_output = ax.annotate(
    "Output Neuron", xy=(x_out, y_out), fontsize=15, ha="center")

# Synapses
input_synapses = ((y1, y2) for y2 in y_hidden for y1 in y_input)
output_synapses = ((y, y_out) for y in y_hidden)

# Plot all circles
for input_circle in circles_input:
    ax.add_artist(input_circle)
for hl_circle in circles_hidden:
    ax.add_artist(hl_circle)
ax.add_artist(output_circle)

# Plot all synapses
for syn in input_synapses:
    ax.add_line(Line2D((0.1, 0.4), syn, color='y'))
for syn in output_synapses:
    ax.add_line(Line2D((0.4, 0.8), syn, color='y'))
