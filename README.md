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

## Overview
The ANN in this example is also known as a Multilayer Perceptron (MLP) regressor with a single hidden layer.  
This will be referred to as 'the model' throughout this README.

### The Scripts
1. mlp.py : This is the shortest and most efficient implementation of the process.  However,
there are no index or column labels throughout the process.

2. mlp-oop.py : This is the object oriented form of (1.).  This organization allows
for easier interpretation of the code, at the cost of increased debugging difficulty.

3. mlp-pandas.py : This is (1.) adapted for Pandas with complete index and column labeling.

4. mlp-pandas-oop.py : This is the object oriented form of (3.) with the same
pros and cons in (2.).

#### 1. Data Source
A hypothetical dataset for film reviews is provided.  This dataset may be replaced with any
n-dimensional numerical dataset.  If any data is categorial, it must be converted
to boolean, or one-hot encoding must be performed.

#### 2. Hyperparameters
The hyperparameters will scale according to the dimensionality of the dataset.
The hyperparameters initialize the architecture of the model.

#### 3. Weights/Synapses
These are initialized randomly using the standard normal distribution.

#### 4. Forward Propagation
Consisting of only linear algebra, this process simply takes the input and allows the model to ouput a result.

#### 5. Backward Propagation
In addition to linear algebra, the optimization method Stochastic Gradient Descent (SGD) is applied in
batch-style to indicate the changes needed for the weights/synapses.

#### 6. Training
The Forward Propagation and Backward Propagation will iterate and update the weights using the gradient values directly.

#### 7. Results
The predicted results are added back to the original dataframe as the "Predicted Score".

## Instructions
These scripts are intended for educational purposes.  Use these scripts interactively in an IPython environment.
