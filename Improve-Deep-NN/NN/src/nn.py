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

    returns
    --------
    parameters: dictionary {
        "W1": {}, layer 1
        "b1": ,
        ...
    }
    """
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/ layer_dims[l-1]) # l, l-1
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1)) # l, 1

    return parameters


# Linear quation

def linear(W, X, b):
    print(f"W {W.shape}, X {X.shape}")
    Z = np.dot(W,X) + b
    cache = (W, X)
    return Z, cache

# Activation Function
def activation(Z, a_name="relu"):
    A = None
    if a_name == 'relu':
        A = np.maximum(0, Z) 
    if a_name == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))

    cache = (Z, A, a_name)
    return A, cache

def dropout(A, keep_prob=0.5):
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    A = A*D
    A = A/keep_prob
    cache = (D, keep_prob)
    print(A.shape)
    return A, cache

def neuron(W, X, b, a_name="relu", drpout=False, keep_prob=0.5):
    Z, linear_cache = linear(W, X, b)
    A, activation_cache = activation(Z, a_name)
    dropout_cache = None
    if drpout:
        dropout_cache = dropout(A)
        print(len(dropout_cache))
        A = dropout_cache[0]

    #A, dropout_cache = dropout(A) if drpout == True else A, None
    print(A)
    print(f"A {A.shape}")
    cache = (linear_cache, activation_cache, dropout_cache)
    return A, cache


# Forward Propagation
def forward_propagation(parameters, X, dropout_size):

    caches = []
    L = len(parameters) // 2

    inp = X.copy()
    for l in range(1, L):
        inp, cache = neuron(parameters["W"+str(l)], inp, parameters["b"+str(l)], drpout=dropout_size[l-1])
        caches.append(cache)
    op, cache = neuron(parameters["W"+str(L)], inp, parameters["b"+str(L)], "sigmoid", drpout=dropout_size[L-1])
    caches.append(cache)
    return op, caches


# BCE Binary cross entropy loss function
def BCE(Y, yh):
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
    return cost


# linear backward
def linear_backward(dZ, cache, lambd=0.0):
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

def activation_linear_backward(dA, cache, lambd=0.0):
    linear_cache, activation_cache, dropout_cache = cache
    dZ = activation_backward(dA, activation_cache)
    dW, db, dX = linear_backward(dZ, linear_cache, lambd)

    return dW, db, dX

# Backward Propagation
def backward_propagation(Y, yh, caches, lambd=0.0):
    epsilon = 0.000001
    dA = -np.divide(Y, yh) + np.divide((1-Y), (1-yh))
    
    grads = {}
    L = len(caches)
    for l in reversed(range(1, L+1)):
        dW, db, dA = activation_linear_backward(dA, caches[l-1], lambd)
        grads["dW"+str(l)] = dW
        grads["db"+str(l)] = db
    return grads
        

# Update Parameters
def update_parameters(parameters, grads, learning_rate=0.01, optimize="GD"):
    L = len(parameters) // 2
       
    if optimize=="GD":
        for l in range(1, L+1):
            parameters["W"+str(l)] -= learning_rate * grads["dW"+str(l)]
            parameters["b"+str(l)] -= learning_rate * grads["db"+str(l)]

    return parameters

# Train mode
def train(X, Y, layer_dims, dropout_size, learning_rate=0.01, epochs=3, lambd=0.0):
    parameters = initialize_parameters(layer_dims)
    for i in range(0, epochs):
        preds, caches = forward_propagation(parameters, X, dropout_size)
        cost = compute_cost(Y, preds, parameters)
        grads = backward_propagation(Y, preds, caches, lambd)
    #    print(grads)
        parameters = update_parameters(parameters, grads, learning_rate)
        if (i % 100 == 0):
            print(cost)
    return parameters

def predict(parameters, X):
    logits, _ = forward_propagation(parameters, X)
    logits[logits > 0.5] = 1
    logits[logits <= 0.5] = 0
    return logits

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



