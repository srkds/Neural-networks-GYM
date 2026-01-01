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
    #print(f"W {W.shape}, X {X.shape}")
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

    if a_name == 'softmax':
        A = Softmax(Z)

    cache = (Z, A, a_name)
    return A, cache

def Softmax(Z):
    """Softmax activation function

    Z : of shape (C, 1) where C is no of classes

    returns
    ----------
    A
    """
   # print("Z.shape: ", Z.shape)
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)    
    t = np.exp(Z_shifted)
    t_sum = np.sum(t, axis=0, keepdims=True)
    A = t / t_sum
    return A

def dropout(A, keep_prob=0.8):
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    A = A*D
    A = A/keep_prob
    cache = (D, keep_prob)
    #print(A.shape)
    return A, cache

def neuron(W, X, b, a_name="relu", drpout=False, keep_prob=0.5):
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
    """Cross entropy loss function for softmax"""
    m = y.shape[1] # no of training examples
    #print("m: ", m)
    #print(s)
    eps = 1e-12
    s = np.clip(s, eps, 1-eps) 
    out = -np.sum(np.sum(np.multiply(y, np.log(s)), axis=0)) / m
    return out

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



