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


def dictionary_to_vector(dictionary):
    count = 0
    keys = []
    cols = [] # contains column size for each parameter matrix
    el_count = [] # count of no of parameters(elements) in each matrix
    for key in dictionary.keys():
        keys.append(key)
        cols.append(dictionary[key].shape[1])
        new_vec = dictionary[key].reshape(-1,)

        if count == 0:
            param_vector = new_vec
            el_count.append(new_vec.shape[0])
        else:
            el_count.append(new_vec.shape[0])
            param_vector = np.concatenate([param_vector, new_vec])
        count += 1
    return param_vector, keys, cols, el_count

def vector_to_dictionary(vector, keys, cols, el_count):
    parameters_dict = {}
    start = 0
    for i, key in enumerate(keys):
        parameters_dict[key] = vector[start: start+el_count[i]].reshape(-1, cols[i])
        start = start+el_count[i]

    return parameters_dict


def grad_check(parameters, gradients, X, Y, epsilon=1e-7):

    """
    for each parameter calculate gradapprox =  J(theta+epsilon) - J(theta-epsilon) / 2epsilon
    then calculate distance between gradapprox and gradients
    """

    parameter_vec, p_keys, p_cols, pel_count = dictionary_to_vector(parameters)
    gradient_vec, g_keys, g_cols, gel_count = dictionary_to_vector(gradients)

    print("grads", gradients)
    print("grad vector: ", gradient_vec)

    num_params = parameter_vec.shape[0]
    J_plus = np.zeros(parameter_vec.shape)
    J_minus = np.zeros(parameter_vec.shape)
    grad_approx = np.zeros(parameter_vec.shape)

    for i in range(num_params):
        print("i:", i)
        
        # J For theta plus
        param_plus = np.copy(parameter_vec)
        param_plus[i] = param_plus[i] + epsilon
        param_plus_dict = vector_to_dictionary(param_plus, p_keys, p_cols, pel_count) 
        preds_plus, _ = forward_propagation(param_plus_dict, X, [0,0,0], False)
        cost_plus = compute_cost(Y, preds_plus,param_plus_dict)
        J_plus[i] = cost_plus.item()
        
        # J for theta minus
        param_minus = np.copy(parameter_vec)
        param_minus[i] = param_minus[i] - epsilon
        param_minus_dict = vector_to_dictionary(param_minus, p_keys, p_cols, pel_count)
        preds_minus, _ = forward_propagation(param_minus_dict, X, [0, 0, 0], False)
        cost_minus = compute_cost(Y, preds_minus, param_minus_dict)
        J_minus[i] = cost_minus.item()

        print("param_plus: ", param_plus_dict)
        print("param_minus: ", param_minus_dict)
        print("j+", J_plus[i])
        print("J-", J_minus[i])

        # grad approx 
        grad_approx[i] = (J_plus[i] - J_minus[i]) / (2*epsilon) 
        print("grad_approx ", grad_approx[i])
        if i == 10:
            break
    
    print(J_plus[:10])
    print(J_minus[:10])

    numerator = np.linalg.norm(gradient_vec - grad_approx)
    denominator = np.linalg.norm(gradient_vec) + np.linalg.norm(grad_approx)
    difference = numerator/denominator

    print(gradient_vec[:5])
    print(grad_approx[:5])

    if difference > 2e-7:
        print("Mistake in backwardpass, the difference is ", str(difference))
    else:
        print("Backwardpass works perfectly fine!, ", str(difference))
    

    
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
    vec, keys, cols, el_count = dictionary_to_vector(parameters)
    print("Parameters: ", parameters)
    print(vec)
    print(keys)
    print(cols)
    print(el_count)
    param = vector_to_dictionary(vec, keys, cols, el_count)
    print(param)
    preds, caches = forward_propagation(parameters, X, dropout_size, False)
    grads = backward_propagation(Y, preds, caches, lambd=0.0)

    grad_check(parameters, grads, X, Y)
    
