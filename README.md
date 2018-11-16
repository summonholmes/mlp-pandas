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
* main.py : The file used to demo the project.  Run in a Jupyter 
notebook environment.
* MLP.py : The MLP class.  Change the parameters as needed.
* forward_prop.py : Perform forward propagation.
* backward_prop.py : Perform backward propagation.
* plot_mlp.py : Plot the model architecture.

## Overview
The ANN in this example is also known as a Multilayer Perceptron (MLP) regressor with adjustable hidden 
layers.  This will be referred to as 'the model' throughout this README.  This basic process involves
sending inputs through the network to generate a predicted output, looking at how off the output 
was from the actual output, then tuning the weights/synapses based to increase performance. 

Starting out as a simple project that was only capable of handling a single hidden layer, 
this project is now capable of handling multiple hidden layers.  If one to three hidden
layers are provided, the resulting architecture may be plotted.

## Components
#### 1. Data Source
A hypothetical dataset for film reviews is provided.  This dataset may be replaced with any
n-dimensional numerical dataset.  If any data is categorial, it must be converted
to Boolean, or one-hot encoding must be performed.

#### 2. Hyperparameters
Only the number of hidden layers is required.  The hyperparameters will scale according to the 
dimensionality of the dataset.  The hyperparameters initialize the architecture of the model.  

#### 3. Weights/Synapses
These are initialized randomly using the standard normal distribution.  The matrix 
dimensionality must be consistently correct for matrix multiplication to be possible.

#### 4. Forward Propagation
Consisting of primarily linear algebra, this process simply takes the input and allows the model to output a result.  
The basic process involves computing a dot product between adjacent layers and performing an activation function on 
the resulting product.  The program stores both -the initial dot product and activated dot product as a tuple with 
the structure:

(initial, activated)

#### 5. Backward Propagation
In addition to linear algebra, the optimization method Gradient Descent is applied
to indicate the changes needed for the weights/synapses.  The process is structurally
similar to forward propagation in reverse, but utilizes the derivative of the 
activation function to reactivate the initial dot products computed during
forward propagation.  In addition, computing deltas and gradients involves
frequent use of the chain rule.  

Example: The initial delta simply utilizes the missed amount, while future deltas
must utilize the previous deltas.  The final gradient will always involve multiplying
the deltas by the transpose of the previous network layer.

#### 6. Training
The Forward Propagation and Backward Propagation will iterate and update the weights using the 
gradient values directly.  This may be accelerated with the learning rate.

#### 7. Results
The predicted results are added back to the original dataframe as the "Predicted Score".

#### 8. Plotting
The model is capable of plotting the ANN architecture if the number of hidden layers is
between 1 and 3.
