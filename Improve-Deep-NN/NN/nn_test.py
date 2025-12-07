import numpy as np
from .src.nn import *

def test_l2_regularization_cost():
    parameters = {}
    parameters["W1"] = np.array([[.4, .9],
        [1.1, .5]])
    parameters["W2"] = np.array([[2, 2, 3],
        [0.2, -1, 0.01],
        [1,1,1 ]])
    parameters["b1"] = 2
    parameters["b2"] = 3
    lambd = 0.2 # panalty value
    m = 10 # no of training samples

    l2_cost =  L2(parameters, m, lambd)
    
    cost = (lambd / (2*m)) * (np.sum(np.square(parameters["W1"])) + np.sum(np.square(parameters["W2"])))
    print(l2_cost)
    assert l2_cost == cost


