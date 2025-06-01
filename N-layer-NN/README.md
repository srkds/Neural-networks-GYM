# Tiny NN

---

This(tinynn.py) is raw implementation of the neural network where you can have n no of layers with n of of nodes. As of it can do classification as it has classification cost function implemented and its derivative. It has also implementation of ReLU and sigmoid activation functions and its differentiations for the backward pass. it can save the trained model to h5 file also load the saved model.

The structure of the neural network is fixed as follows nx inputs to the 1st layer of the neural network, and there can be n of hidden layers with n of nodes followed by output layer which also can have n of nodes.

for example if you want to build neural network of 3 layers and you have input feature size 9 then
the layer_dims will be 9, 4, 4, 1. In addition to it the hidden layers will have ReLU as default activation function and the output layer will have sigmoid for classification. however that canatu be changed by modifying the script :).

## Installation

Requirements:

```
h5py
numpy
matplotlib
```

## Setup

- create a new `py` file
- import `tinynn.py`
- Create a dataset `X` training set of shape $X \in \mathbb{R}^{(nx, m)}$ where $nx$ is feature dimention and $m$ is no of trainig examples
- Create Y that is true labels of shape $(1,m)$.

```py
from tinynn import *


X # (nx, m)
Y # (1, m)

layer_dims = np.array([nx, 5,2,1])

parameters, cost, w1_l, cost_l =  train(X, Y, layer_dims, learning_rate=0.0075, iteration=3000)


# prediction of trained parameters
test # i/p to nn of shape (nx, no of examples)

predict(parameters, test)
```
