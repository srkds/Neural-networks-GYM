import numpy as np
from .src.nn import *

np.random.seed(1)

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


def test_softmax():

    z = np.array([5,2,-1,3])
    z = z.reshape(-1, 1)
    a = Softmax(z)
    #print(a) 
    assert a.all() == np.array([[0.84203357],[0.04192238],[0.00208719],[0.11395685]]).all()
    
    a_sum = np.sum(a)

    assert a_sum == 1.0, "Sum is not 1"

def test_CE():
    z = np.array([5,2,-1,3])
    z = z.reshape(-1, 1)
    a = Softmax(z)
    
    #print("Softmax : ",a)
    y = np.array([[0],[0],[1],[0]])
    loss = CE(a, y)
    
    #print("loss: ", loss)
    #assert a.all() == np.array([[0.84203357],[0.04192238],[0.00208719],[0.11395685]]).all()
    
    #a_sum = np.sum(a)

def test_multisample_CE():
    # creating dummy example of shape (4, 2) where 4 is no of classes and 2 is traininng samples
    z = np.random.rand(4,2)
    a = Softmax(z)

    print("A shape: ", a.shape)

    y = np.array([[0,1],[0,0],[1,0],[0,0]])
    loss = CE(a,y)
    print(loss)

