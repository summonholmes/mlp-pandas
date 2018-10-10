# mlp-pandas
The artificial neural network (ANN) made easy with Pandas and plenty of documentation

## ANNs too hard?
With copious amounts of linear algebra and calculus, ANNs are often not for the faint of heart.  
This project was inspired by 
[Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU&index=1)
as well as [A Neural Network in 11 lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/).

## Overview
The ANN in this example is also known as a Multilayer Perceptron (MLP) regressor with a single hidden layer.  
This will be referred as the model throughout this readme.

#### Data Source
A hypothetical data source providing data for film reviews is provided.  This can be any
n-dimensional numerical data.  If the data is categorial, it requires conversion
to boolean or one-hot encoding.

#### Hyperparameters
The hyperparameters will scale according to the dimensionality of the data provided.
This initializes the architecture of the model.

#### Weights/Synapses
These are initialized randomly using ranged standard normal distributions.

#### Forward Propagation
Consisting of only linear algebra, this process simply takes the input and allows the model to ouput a result.

#### Backward Propagation
In addition to linear algebra, the optimization method Stochastic Gradient Descent (SGD) is applied
to indicate the changes needed for the weights/synapses.

#### Training
The process will iterate for a given amount of time so the model can learn.

#### Results
The final results are added back to the original dataframe as the "Predicted Score".
