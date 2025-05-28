import h5py
import matplotlib.pyplot as plt

from tinynn import *

np.random.seed(1)

def load_dataset():
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5', 'r')
    test_dataset = h5py.File('./datasets/test_catvnoncat.h5', 'r')

    X_train = np.array(train_dataset['train_set_x'])
    Y_train = np.array(train_dataset['train_set_y']).reshape(1,-1)


    X_test = np.array(test_dataset['test_set_x'])
    Y_test = np.array(test_dataset['test_set_y']).reshape(1,-1)

    return X_train, Y_train, X_test, Y_test


# dataset = h5py.File('./datasets/train_catvnoncat.h5', 'r')
# keys : ['list_classes', 'train_set_x', 'train_set_y']
X, Y, X_test, Y_test = load_dataset()
# X = np.array(dataset['train_set_x'])
# Y = np.array(dataset['train_set_y']).reshape(1,-1)

print("Train X ", X.shape)
print("Train Y ", Y.shape)
print("Test X ", X_test.shape)
print("Test Y ", Y_test.shape)
# Preprocess data
# make feature vector from the image dimention 64,64,3 to 

# m - no of training example
# w, h, rgb - width height and rgb image channel
# reshaping to features by training examples (nx, m)
X_train = X.reshape(-1, X.shape[1]*X.shape[2]*X.shape[3]).T  # (m, w, h, rgb) -> (nx, m) 
X_train = X_train/255.
X_test = X_test.reshape(-1, X_test.shape[1]*X_test.shape[2]*X_test.shape[3]).T  # (m, w, h, rgb) -> (nx, m) 
X_test = X_test/255.
# print(X_train.shape)
nx = X_train.shape[0]
layer_dims = np.array([nx, 20, 7, 5, 1])
parameters, cost, w1_l, cost_l =  train(X_train, Y, layer_dims, learning_rate=0.0075, iteration=3000)

print("Train Accuracy")
get_accuracy(parameters, X_train, Y)

print("Test Accuracy")
get_accuracy(parameters, X_test, Y_test)
# plt.plot(w1_l, cost_l)
# plt.show()
# parameters = initialize_network(layer_dims=layer_dims)
# print(parameters['W1'][0])
