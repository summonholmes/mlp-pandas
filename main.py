# %% Import libraries
"""
THIS IS NOT FOR PRODUCTION PURPOSES
"""
from MLP import MLP
from numpy import where
from plot_mlp import plot_mlp
from pandas import DataFrame, get_dummies
from seaborn import load_dataset

# %% Create Dataset
"""
Use any numerical dataset with a target,
or process categorical data first.
"""
titanic = load_dataset("titanic")
titanic.index = ("Passenger " + str(i + 1) for i in range(titanic.shape[0]))
X = titanic.copy()  # Tensor

# %% One hot encode categorical variables
"""
New features for sex, embark towns, and
children will be added as Booleans.  The original
values will be deleted.
"""
X = get_dummies(X, columns=["sex"])
# X = get_dummies(X, columns=["embark_town"])
X = get_dummies(X, columns=["who"])

# %% Process the data
"""
This is only an example.  Data must contain no NaNs, and
categorical values must be one-hot encoded.
"""
X["adult_male"] = where(True, 1, 0)
X["alone"] = where(True, 1, 0)

# %% Drop any unncessary columns
"""
Some columns are redundant or unnecessary for this demonstration.
"""
X.drop(
    [
        "sibsp",
        "parch",
        "embarked",
        "embark_town",
        "class",
        "adult_male",
        "deck",
        "alive",
        "who_man",
        "who_woman",
    ],
    axis=1,
    inplace=True,
)

# %% Drop NaN
"""
NaN values will break the model and raise too many errors.
Remove them.
"""
X.dropna(inplace=True)

# %% Finally, get the dependent variable
"""
The ANN will predict this variable as a probability
"""
y = DataFrame(X.pop("survived"))  # Target

# %% Predict
"""
The only three things needed to use the ANN
are the tensor, target, and number of hidden layers.

The only methods the user needs to run are plot_mlp()
and train_mlp().
"""
mlp = MLP(X, y, 1)  # Provide X, y, and # hidden layers
mlp.train_mlp()  # Iterate between forward & backward prop
plot_mlp(mlp)  # Plot the architecture if <= 3

# %% Reassign results back to original dataframe
"""
The last two columns show the results
of the whole process.
"""
X["survived"] = y
X["survival_pred"] = mlp.fwd_neurons[-1][-1]
X
