from MLP import MLP
from plot_mlp import plot_mlp
from pandas import DataFrame

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
