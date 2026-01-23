import numpy as np


"""
# Basics of NN

- Initialization of parameters
- forward pass
- calculate cost
- backward pass
- train loop

# Regularization

- L2 regularization
- Dropout Regularization
- Softmax and hardmax
- RMS promp
- Adam Optimizer
- Batch norm

"""



# Initialize Parameters
def initialize_parameters(layer_dims):
    """
    Takes
    layer_dims : eg [nx, layer_1, 2, 3]

    Let :math:`L` be a no of layers and :math:`\\text{layer_dims} = [n_0, n_1, ..., n_L]` where :math:`n_l` denotes no of neurons in :math:`l^{th}` layer. then the :math:`W` and :math:`b` are weight and bias matrix, where :math:`W^{[l]} \in R^{n_l \\times n_{l-1}}`, :math:`b^{[l]} \in R^{n_l \\times 1}`.

    And the :math:`parameters = {W^{[1]}, b^{[1]}, ..., W^{[L]}, b^{[L]}}`

    Examples::
        >>> nx = 3
        >>> layer_dims = [nx, 4, 4, 1]
        >>> parameters = initialize_parameters(layer_dims)
        >>> len(parameters)
        6
        >>> parameters["W1"].shape
        (4, 3)
        >>> parameters["W2"].shape
        (4, 4)
        >>> parameters["W3"].shape
        (1, 4)
        >>> parameters["b1"].shape
        (4, 1)
    """
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/ layer_dims[l-1]) # l, l-1
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1)) # l, 1

    return parameters


# Linear quation

def linear(W, X, b):
    """
    This function will apply linear equation
    
    .. math:: 
        Z = W . X + b

    Args:
        W: weight matrix of layer ``l``

        X: input or output of previous layer of neural net X or A[l-1]

        b: bias vector for all the nodes

    Examples::
        >>> W = np.array([[.5, .8],[1, .4]])
        >>> b = np.ones((2,1))
        >>> X = np.array([[1],[2]])
        >>> Z = linear(W, X, b)
        >>> Z.shape
        (2,1)
    """
    #print(f"W {W.shape}, X {X.shape}")
    Z = np.dot(W,X) + b
    cache = (W, X)
    return Z, cache

# Activation Function
def activation(Z, a_name="relu"):
    """
    This function will apply activation on given input. It supports ``ReLU``, ``Sigmoid`` and ``Softmax``.

    Args:
        Z: output matrix of linear function.
        a_name: activation name that you want to apply

    - Relu

    .. math::
        Z = W.X + b

        A = g(Z)

    Here matrix ``Z`` is output of linear function and is input for activation function ``g()``.

    - ReLU
    
    .. math::
        A = max(0,Z)

    - Sigmoid
    
    .. math::
        A = \\frac{1}{1+e^{-Z}}
    """
    A = None
    if a_name == 'relu':
        A = np.maximum(0, Z) 
    if a_name == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))

    if a_name == 'softmax':
        A = Softmax(Z)

    cache = (Z, A, a_name)
    return A, cache

def Softmax(Z):
    """Softmax activation function

    Z : of shape (C, 1) where C is no of classes

    returns
    ----------
    A of shape (C, 1) but the values will be probability and its sum will be 1


    Let say :math:`Z` is matrix of shape :math:`(C, m)` where ``C`` is no of classes and ``m`` is no of training samples. And :math:`z` is a vector of values :math:`[z_1, z_2,...z_c]` then the softmax output of vector ``z`` is following equation.


    .. math::
        A(z_i) = \\frac{e^{z_i}}{\\sum_{i=1}^{C}e^{z_i}}
    """
   # print("Z.shape: ", Z.shape)
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)    
    t = np.exp(Z_shifted)
    t_sum = np.sum(t, axis=0, keepdims=True)
    A = t / t_sum
    return A

def dropout(A, keep_prob=0.8):
    """
    Dropout helps regularize the network by randomly shutting down the units. The intuition is that, any single neuron cannot rely on single feature or input, because, it any time it can go off, and that's why it will be reluctunt to give more weight to any single input or feature.

    Specifically here I have installed ``Inverted Dropout``.
    
    With Dropout the feedforward operation becames.

    .. math::
        D \\sim Bernouli(keepprob)

        Z = W.X + b

        A = g(Z)

        A = \\frac{A*D}{keep_prob}
    """
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    A = A*D
    A = A/keep_prob
    cache = (D, keep_prob)
    #print(A.shape)
    return A, cache

def neuron(W, X, b, a_name="relu", drpout=False, keep_prob=0.5):
    """
    This function can work as a single neuron or a single layer which computes linear->activation function with the help of previously defined atomic function. This takes weight & bias metrix, output of previous layer or input in case of first layer, activation, dropout, and keep_prob. You can directly use this function as a layer where you give input and you get output.

    """
    Z, linear_cache = linear(W, X, b)
    A, activation_cache = activation(Z, a_name)
    dropout_cache = None
    if drpout:
        A, dropout_cache = dropout(A)
        #print(len(dropout_cache))
       # A = dropout_cache[0]

    #A, dropout_cache = dropout(A) if drpout == True else A, None
    #print(A)
    #print(f"A {A.shape}")
    cache = (linear_cache, activation_cache, dropout_cache)
    return A, cache


# Forward Propagation
def forward_propagation(parameters, X, dropout_size, multiclass=False):

    caches = []
    L = len(parameters) // 2

    inp = X.copy()
    for l in range(1, L):
        inp, cache = neuron(parameters["W"+str(l)], inp, parameters["b"+str(l)], drpout=dropout_size[l-1])
        caches.append(cache)

    activation = "sigmoid"    
    if multiclass==True:
        activation = "softmax"
    op, cache = neuron(parameters["W"+str(L)], inp, parameters["b"+str(L)], activation, drpout=dropout_size[L-1])
    caches.append(cache)
    return op, caches


# BCE Binary cross entropy loss function
def BCE(Y, yh):
    """
        This function computes the Binary Cross Entropy cost function.

        .. math::
            J() = - Y log(\\hat{Y}) + (1-Y) log(1-\\hat{Y})
    """
    epsilon = 0.001
    bce_cost = -np.sum(np.multiply(Y,np.log(yh)) + np.multiply((1-Y), np.log(1-yh))) / Y.shape[1]
    return bce_cost

# L2-Regularization
def L2(parameters, m, lambd=0.0):
    #print(lambd)
    #print(m)
    L = len(parameters) // 2 # no of layers
    L2_cost = 0
    for l in range(L):
        L2_cost += np.sum(np.square(parameters["W"+str(l+1)]))
    L2_cost *= (lambd/(2*m)) 

    return L2_cost 


#Compute Cost
def compute_cost(Y, yh, parameters, c_name='BCE', lambd=0.0):
    #print(Y.shape)
    #print(yh.shape)

    #Y = Y
    #yh = yh + epsilon
    # print(Y)
    # print(yh)
    m = Y.shape[1]
    if c_name=='BCE':
        # print(cost)
        bce_cost = BCE(Y, yh)
        L2_cost = L2(parameters,m, lambd)
        cost = bce_cost + L2_cost

    if c_name=="CE":
        cost = CE(yh, Y)
        L2_cost = L2(parameters,m, lambd)
        cost = cost + L2_cost
        
    return cost


# Loss cross entropy
def CE(s, y):
    """Cross entropy loss function for softmax

    .. math:: 

        L(Y,S) = \\sum_{i=1}^{C} Y_i.log(S_i)

        J(W,b) = \\sum_{j=1}^{m}L(Y^j, S^j)

    Here :math:`L` is loss function, calculates how well our model is performing on single training example for all classes :math:`C`. And :math:`J` is cost function that calculates loss for all the samples in training examples
    """
    m = y.shape[1] # no of training examples
    #print("m: ", m)
    #print(s)
    eps = 1e-12
    s = np.clip(s, eps, 1-eps) 
    out = -np.sum(np.sum(np.multiply(y, np.log(s)), axis=0)) / m
    return out

# linear backward
def linear_backward(dZ, cache, lambd=0.0):
    """
    This function will compute backward pass for linear layer, meaning calculating gradients of W, X, and b with respect to Z and multiplying it with derivative of Z for chain rule.
    
    derivative of ``W`` will be its corresponding value in X. for example if c = a*b then der of a is b. then we multiply it with der of Z to apply the chain rule. And following is vectorized version.

    .. math::
        dW = \\frac{1}{m}.dZ.X^T + \\frac{\\lambda}{m}*W

    Now derivative of sum 1 so local gradient will be 1 so what ever the gradient of dZ, will be passed to derivative of ``b``. if the we have only 1 training semple then the shape of dZ will be (l,1) and if there are m training examples then (l, m). for second we need to sum all the dz for first node that is first row, and same for all that's why in implementation you will see the sum on axis 1

    .. math::
        db = \\frac{1}{m}.1*dZ

    And now you calculate derivative of input that is ``X`` which can be output of previous layer.

    .. math::
        dX = W^T.dZ

    
    Args:
        dZ: Gradient matrix of ``Z``, that you got from activation backward

        cache: tuple of matrix ``(W,X)``

        lambd: L2 regularization penalty value. default is 0.0


    """
    W, X = cache
    m = X.shape[1]
    dW = np.dot(dZ, X.T) / m + ((lambd/m)*W)
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dX = np.dot(W.T, dZ)

    return dW, db, dX


# activation backward
def activation_backward(dA, cache):
    Z, A, a_name = cache
    if a_name == 'relu':
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
    if a_name == 'sigmoid':
        dZ = A * (1-A)
        dZ *= dA

    return dZ

# dropout backward
def dropout_backward(dA, D, keep_prob):
    dA = dA*D
    #print(f"dA {dA}, D {D} keep Prob {keep_prob}")
    dA = dA/keep_prob

    return dA


def activation_linear_backward(dA, cache, lambd=0.0):
    linear_cache, activation_cache, dropout_cache = cache
    
    if dropout_cache is not None:
        dA = dropout_backward(dA, dropout_cache[0], dropout_cache[1])
    dZ = activation_backward(dA, activation_cache)
    dW, db, dX = linear_backward(dZ, linear_cache, lambd)

    return dW, db, dX

# Backward Propagation
def backward_propagation(Y, yh, caches, lambd=0.0):
    epsilon = 0.000001
    dA = None 
    grads = {}
    L = len(caches)

    linear_cache, activation_cache, dropout_cache = caches[-1]
    if Y.shape[0] > 1:
        dZ = yh - Y
        dW, db, dX = linear_backward(dZ, linear_cache, lambd)
        grads["dW"+str(L)] = dW
        grads["db"+str(L)] = db
        dA = dX
    else:
        dA = -np.divide(Y, yh+epsilon) + np.divide((1-Y), (1-yh+epsilon))
        dZ = activation_backward(dA, activation_cache)
        dW, db, dX = linear_backward(dZ, linear_cache, lambd)
        
        grads["dW"+str(L)] = dW
        grads["db"+str(L)] = db
        dA = dX

    for l in reversed(range(1, L)):
        dW, db, dA = activation_linear_backward(dA, caches[l-1], lambd)
        grads["dW"+str(l)] = dW
        grads["db"+str(l)] = db
    return grads
        
#def backward_CE(Y, yh, caches, lambd=0.0):

# Update Parameters
def update_parameters(parameters, grads, learning_rate=0.01, optimize="GD"):
    L = len(parameters) // 2
       
    if optimize=="GD":
        for l in range(1, L+1):
            parameters["W"+str(l)] -= learning_rate * grads["dW"+str(l)]
            parameters["b"+str(l)] -= learning_rate * grads["db"+str(l)]

    return parameters

# Train mode
def train(X, Y, test_X, test_Y, layer_dims, dropout_size, learning_rate=0.01, epochs=3, lambd=0.0, multiclass=False):
    meta_data = {"train_cost":[], "test_cost":[]}
    parameters = initialize_parameters(layer_dims)
    c_name = "CE" if multiclass==True else "BCE"
    for i in range(0, epochs):
        preds, caches = forward_propagation(parameters, X, dropout_size, multiclass)
        cost = compute_cost(Y, preds, parameters, c_name, lambd)
        test_cost = test(test_X, test_Y, parameters, dropout_size, c_name, True)
        meta_data["train_cost"].append(cost.item())
        meta_data["test_cost"].append(test_cost.item())
        #print(cost)
        #print(preds.shape)
        grads = backward_propagation(Y, preds, caches, lambd)
    #    print(grads)
        parameters = update_parameters(parameters, grads, learning_rate)
        if (i % 5 == 0):
            print("Iteration: ", i, " Loss: ", cost)
    return parameters, meta_data


def test(test_X, test_Y, parameters, dropout_size, c_name="BCE", multiclass=False):
    preds, _ = forward_propagation(parameters, test_X, dropout_size, multiclass)
    cost = compute_cost(test_Y, preds, parameters, c_name)
    return cost

def predict(parameters, X, multiclass=False):
    dropout_size = np.zeros(len(parameters) // 2)
    logits, _ = forward_propagation(parameters, X, dropout_size, multiclass)
    
    if multiclass==True:
        #print(logits)
       classes = logits.argmax(axis=0) # (C,M) -> (1,M)
    else:
        logits[logits > 0.5] = 1
        logits[logits <= 0.5] = 0
    return classes, logits

if __name__ == "__main__":
    print("Comes here..")
    np.random.seed(1)
    # Dummy Data
    X = np.random.rand(5, 6) # nx, m
    nx = X.shape[0]
    m = X.shape[1]
    Y = np.random.randint(0,2, size=(1, m))

    print("X.shape ", X.shape)
    print("Y ", Y)
    print("nx", nx)

    # Define network
    #layer_dims = [nx, 4, 2, 1]

    # Initialize parameters
    #parameters = initialize_parameters(layer_dims)
    #z, caches = forward_propagation(parameters, X)
    #print(z)
    #cost = compute_cost(Y, z)
    #print(cost)
    #grads = backward_propagation(Y, z, caches)
    #print(grads)
     #print(parameters.keys())
 #
     #assert len(parameters.keys())//2  == len(layer_dims)-1 , "Wrong parameters size"
     #assert parameters["W1"].shape == (4, nx), "Dimentions does not match"
 #
     #z = linear(parameters["W1"], X, parameters["b1"]) # wx+b
     #a = activation(z, sigmoid")
     #print(z)
     #print(a)  
 #



