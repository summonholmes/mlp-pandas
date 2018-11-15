# mlp-pandas
The artificial neural network (ANN) made easy with Pandas, matplotlib, and plenty of documentation

## ANNs too hard?
With copious amounts of linear algebra and calculus, ANNs are often not for the faint of heart.  
This project was inspired by 
[Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU&index=1)
as well as [A Neural Network in 11 lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/).

![alt text](https://raw.githubusercontent.com/summonholmes/mlp-pandas/master/Sample.png)

## Dependencies
* python3-numpy
* python3-pandas
* python3-matplotlib

## Scripts
mlp-pandas.py : Run the ANN and plot the architecture.  Change the parameters as needed.

## Overview
The ANN in this example is also known as a Multilayer Perceptron (MLP) regressor with a adjustable hidden layers.  
This will be referred to as 'the model' throughout this README.

#### Data Source
A hypothetical dataset for film reviews is provided.  This dataset may be replaced with any
n-dimensional numerical dataset.  If any data is categorial, it must be converted
to boolean, or one-hot encoding must be performed.

#### Hyperparameters
Only the number of hidden layers is required.  The hyperparameters will scale according to the dimensionality of the dataset.  The hyperparameters initialize the architecture of the model.  

#### Weights/Synapses
These are initialized randomly using the standard normal distribution.

#### Forward Propagation
Consisting of linear algebra, this process simply takes the input and allows the model to ouput a result.

#### Backward Propagation
In addition to linear algebra, the optimization method Gradient Descent is applied
to indicate the changes needed for the weights/synapses.

#### Training
The Forward Propagation and Backward Propagation will iterate and update the weights using the gradient values directly.  This can be accelerated with the learning rate.

#### Results
The predicted results are added back to the original dataframe as the "Predicted Score".

## Instructions
These scripts are intended for educational purposes.  Use these scripts interactively in an IPython environment.
