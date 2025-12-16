import numpy as np
from src.nn import *

import argparse # cmd arguments

parser = argparse.ArgumentParser()

# python3 train.py --epochs=20
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--lambd", type=float, default=0.0)
parser.add_argument("--dropout", type=bool, default=False)

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

dropout_size = [0,1,0] if args.dropout==True else np.zeros(len(layer_dims)-1)
print(f"drpo size: {dropout_size}")
parameters = train(X, Y, layer_dims, learning_rate=args.lr, epochs=args.epochs, lambd=args.lambd, dropout_size=dropout_size)
preds = predict(parameters, X)

print("Actual: ", Y)
print("Predictions: ", preds)
