import numpy as np
import copy

np.random.seed(1)
# Tiny NN
""" 

 - Initializing the neural networks.
 - Forward pass
 - Calculate the cost
 - Backward pass
 - Update the parameters 


"""

def initialize_network(layer_dims):
    # *** input
    # layer_dims - array of no of units in each layer
    # *** output
    # parameters - dictionary of the paramenters of the layers

    np.random.seed(1)
    L = len(layer_dims)
    parameters = {}
    
    parameters['W1'] = np.random.randn(layer_dims[1], layer_dims[0]) / np.sqrt(layer_dims[0]) #*0.01
    parameters['b1'] = np.zeros((layer_dims[1], 1))

    for l in range(2, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])  #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def activation_forward(Z, activation):
    if activation == 'relu':
        A = np.maximum(0, Z)
    else: 
        A = 1 / (1 + np.exp(-Z))
    cache = (Z, activation)
    return A, cache

def linear_forward(W, Ap, b):
    Z = np.dot(W, Ap) + b
    cache = (Ap, W, b)
    return Z, cache

def linear_activation_forward(W, Ap, b, activation):

    Z, linear_cache = linear_forward(W, Ap, b)
    A, activation_cache = activation_forward(Z, activation)
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_pass(X, parameters):
    
    # Zi = WX+b
    # Ai = activation(Zi)
    caches = []
    L = len(parameters)//2

    A, cache = linear_activation_forward(parameters['W1'], X, parameters['b1'], 'relu') 

    caches.append(cache)
    for l in range(1,L-1):
        A, cache = linear_activation_forward(parameters['W'+str(l+1)], A, parameters['b'+str(l+1)], 'relu')

        caches.append(cache)


    AL, cache = linear_activation_forward(parameters['W'+str(L)], A, parameters['b'+str(L)], 'sigmoid')

    caches.append(cache)

    return AL, caches

def calculate_cost(AL, Y):
    m = Y.shape[1]
    J = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))) / m
    J = np.squeeze(J)
    return J

def linear_backward(dZ, cache):
    Ap, W, b = cache
    m = Ap.shape[1]
    dW = np.dot(dZ, Ap.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) /m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def activation_backward(dA, cache):
    # print("cache: ", cache)
    Z, activation = cache

    if activation == 'relu':
        mask = Z <= 0
        dZ = np.array(dA, copy=True)
        dZ[mask] = 0
        # dZ[~mask] = 1
        # dZ = np.multiply(dZ, dA)
    else:
        A = 1 / (1+np.exp(-Z))
        dZ = np.multiply(np.multiply(A, (1 - A)), dA)
    
    return dZ

def linear_activation_backwards(dA, cache):
    linear_cache, activation_cache = cache
    dZ = activation_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def backward_pass(AL, Y, caches):
    grads = {}
    L = len(caches)
    # print(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]
    # print(current_cache)
    # a, b = current_cache
    # print(len(a), len(b))
    # a1, a2 = b
    dA_prev, dW, db =  linear_activation_backwards(dAL, current_cache)

    grads['dW'+str(L)] = dW
    grads['db'+str(L)] = db
    grads['dA'+str(L-1)] = dA_prev

    for l in reversed(range(L-1)):
        current_cache = caches[l]
    
        dA_prev, dW, db =  linear_activation_backwards(dA_prev, current_cache)

        grads['dW'+str(l+1)] = dW
        grads['db'+str(l+1)] = db
        grads['dA'+str(l)] = dA_prev

    return grads

def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - (learning_rate * grads['dW'+str(l+1)])
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - (learning_rate * grads['db'+str(l+1)])

    return parameters

def train(X, Y, layer_dims, learning_rate=0.1, iteration=1000):

    np.random.seed(1)
    parameters = initialize_network(layer_dims=layer_dims)
    costs = []
    w1_grads = []
    all_cost = []
    for i in range(iteration):
        Al, caches = forward_pass(X, parameters=parameters)

        cost = calculate_cost(Al, Y)    

        w1_grads.append(parameters['W1'][0][0])
        all_cost.append(cost)
        grads = backward_pass(Al, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if (i % 100 == 0 or i == iteration - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0:
            costs.append(cost)
    return parameters, costs, w1_grads, all_cost

def hello():
    print("Hello from tiny")

if __name__ == '__main__':

    # DATASET
    # Sample dataset for binary classification 0 or 1 type
    # X - training examples of shape (nx, m) where nx is feature dimention and m is no of training example
    # Y - output variable, ground truth of traning example of shape (1, m) 
    nx, m = 12288, 200 # 5 - features of single example, and total 10 such example
    X = np.random.rand(nx, m)
    Y = np.random.randint(0, 2, size=(1, m))

    print(X.shape)
    print(X)
    print(Y.shape)
    print(Y)

    # Initialize parameters
    layer_dims = np.array([nx, 20, 7, 5, 1]) # [input_feature, # of units in layer1, # of units in layer 2,...., op units]

    parameters, costs = train(X, Y, layer_dims=layer_dims, learning_rate=0.01, iteration=2000)
    # params = initialize_network(layer_dims=layer_dims)
    # print(params)
    # assert len(layer_dims)-1 == len(params)//2 

    # Al, cache = forward_pass(X, parameters=params)
    # print("AL -> ", Al)
    # cost = calculate_cost(Al, Y)
    # print(cost)
    # grads = backward_pass(Al, Y, cache)
    # print("Grads:---", len(grads))
