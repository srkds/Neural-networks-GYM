import numpy as np
from src.nn import *

import argparse # cmd arguments

parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--lambd", type=float, default=0.0)

args = parser.parse_args()

#print(initialize_parameters([2,3,1]))
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
layer_dims = [nx, 10, 5, 1]

parameters = train(X, Y, layer_dims, learning_rate=args.lr, epochs=args.epochs, lambd=args.lambd)
preds = predict(parameters, X)

print("Actual: ", Y)
print("Predictions: ", preds)
