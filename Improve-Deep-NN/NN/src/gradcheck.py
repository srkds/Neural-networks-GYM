from nn import *

"""
Gradient check, Why?
How we can verify that the backpropagation is calculating gradients correctly? we need to verify that our implementation is correct.


What is gradient? it tells us that how much the output chenge if we nudge the input, and in which direction upwards or downwards.



gradcheck:
    parameters,
    gradients
    X
    Y

here we are comparing the original gradients callulated by our implementation and with this implementaion

1. make a single vector of all the parameters
1. one by one calculate gradients for each parameter


"""


def grad_check(parameters, gradients, X, Y):



    
if __name__ == "__main__":

    np.random.seed(1)
    X = np.random.rand(3,1) # nx, 1
    Y = np.array([[0]]) # 1,1
    nx = X.shape[0]
    
    layer_dims = [nx, 3, 2, 1]
    dropout_size = [0,0,0]
    print("X ", X, " Y ", Y)
    #Neural Net

    parameters = initialize_parameters(layer_dims)
    preds, caches = forward_propagation(parameters, X, dropout_size, False)
    grads = backward_propagation(Y, preds, caches, lambd=0.0)
    
