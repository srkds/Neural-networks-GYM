from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

def showImg(img):
    plt.imshow(img)
    plt.show()

def plotDatasetImageGrid(images, logits):
    nc = 4 # no of columns

    fig, axes = plt.subplots(nrows=2, ncols=nc, figsize=(6,3))

    for i, ax in enumerate(axes.flat):
        
        # image show
        ax.imshow(images[4000+i])
        
        bar_ax = inset_axes(ax, width="100%", height="30%", 
                            loc="lower center", bbox_to_anchor=(0, -0.40, 1, 1), bbox_transform=ax.transAxes, borderpad=0)

        probs = logits[:,i].reshape(-1,) # converting it to row vector (10,1) -> (10,)
        bar_ax.bar(range(10), probs)

        bar_ax.set_ylim(0,1)
        bar_ax.set_xticks(range(10)),
        #bar_ax.xticklabels(range(10), fontsize=6)

        ax.bar([1,2], [1,2])
        #ax.text(0.5, -0.15, f'TL: {str(Y_test[0][i])}, MP: {preds[0][i]}', transform=ax.transAxes, ha='center', va='bottom')
        ax.axis('off')
    plt.show()

def plotLoss(epochs, mdata):
    plt.plot(range(epochs), mdata["train_cost"], label="train_loss")
    plt.plot(range(epochs), mdata["test_cost"], label="test_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()



if __name__ == "__main__":

    # Load dataset
    images, labels = loadlocal_mnist(
        images_path='dataset/train-images.idx3-ubyte',
        labels_path='dataset/train-labels.idx1-ubyte'
    )

    images = images.reshape(-1, 28, 28) # (m,28,28)

    showImg(images[2])

    classes = 10
    y = np.eye(classes)[labels] # one hot encoding for 10 classes -1, 10
    y = y.T # (c,m) (10,6000) no of training examples
    print(y.shape) 
    print(y[:,2])

    X = images.reshape(-1, 28*28).T # nx, m
    Y = y # c, m
    X = X/255

    nx =  X.shape[0] # no of features
    c = y.shape[0] # classes

    layer_dims = [nx, 128, 64, c] # Define network size

    dropout_size = [0,1,0] if args.dropout==True else np.zeros(len(layer_dims)-1)

    parameters, mdata = train(X[:,:4000], Y[:,:4000], X[:,4000:4200], Y[:,4000:4200], layer_dims, learning_rate=args.lr, epochs=args.epochs, lambd=args.lambd, dropout_size=dropout_size, multiclass=True)
    preds, logits = predict(parameters, X[:, 4000:4020], True)

    plotLoss(args.epochs, mdata)
    print(f"Train Loss: {mdata['train_cost'][-1]}, Test Loss: {mdata['test_cost'][-1]}")
    print("Actual:", Y[:,4000:4020].argmax(axis=0))
    print("predicted: ", preds)

    plotDatasetImageGrid(images, logits)
    
   
