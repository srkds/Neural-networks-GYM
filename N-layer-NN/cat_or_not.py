import h5py
import matplotlib.pyplot as plt

from tinynn import *

np.random.seed(1)

dataset = h5py.File('./datasets/train_catvnoncat.h5', 'r')
# keys : ['list_classes', 'train_set_x', 'train_set_y']

X = np.array(dataset['train_set_x'])
Y = np.array(dataset['train_set_y']).reshape(1,-1)

print(X.shape)
print(Y.shape)
# Preprocess data
# make feature vector from the image dimention 64,64,3 to 

X_train = X.reshape(-1, X.shape[1]*X.shape[2]*X.shape[3]).T
X_train = X_train/255.
# print(X_train.shape)
nx = X_train.shape[0]
layer_dims = np.array([nx, 20, 7, 5, 1])
parameters, cost, w1_l, cost_l =  train(X_train, Y, layer_dims, learning_rate=0.75, iteration=2500)

plt.plot(w1_l, cost_l)
plt.show()
# parameters = initialize_network(layer_dims=layer_dims)
# print(parameters['W1'][0])
