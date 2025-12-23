from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np

import argparse

from nn import *

parser = argparse.ArgumentParser()

#python train.py --lr=0.001 --epochs=800
# python3 train.py --epochs=20

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--lambd", type=float, default=0.0)
parser.add_argument("--dropout", type=bool, default=False)


args = parser.parse_args()

images, labels = loadlocal_mnist(
    images_path='dataset/train-images.idx3-ubyte',
    labels_path='dataset/train-labels.idx1-ubyte'
)

print(images.shape, labels.shape)

images = images.reshape(-1, 28, 28)

plt.imshow(images[2])
plt.show()
print(labels[2])
classes = 10
y = np.eye(classes)[labels] # -1, 10
y = y.T # c, no of training examples
print(y.shape)
print(y[:,2])

X = images.reshape(-1, 28*28).T # nx, m
Y = y # c, m

nx =  X.shape[0] # no of features
c = y.shape[0] # classes

layer_dims = [nx, 128, 64, c]

dropout_size = [0,1,0] if args.dropout==True else np.zeros(len(layer_dims)-1)

parameters = train(X[:,:2000], Y[:,:2000], layer_dims, learning_rate=args.lr, epochs=args.epochs, lambd=args.lambd, dropout_size=dropout_size, multiclass=True)
preds = predict(parameters, X[:, 10:20], True)

print("Actual:", Y[:,10:20].argmax(axis=0))
print("predicted: ", preds)
