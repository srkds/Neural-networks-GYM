# MNIST Iraining

Train handwritten digit classification on feed forward neural network. What you can expect from running this script, you can see all the clear blocks or implementations of neuron(linear equation, activation), Dropout, softmax, loss function (BCE, CE)

```bash
python train.py --lr= --epochs= --dropout= --lambd=
```

Without dropout with following settings `python train.py --lr=0.001 --epochs=300` you can get following results.

`Train Loss` 0.31, `Test Loss` 2.80

<img src="./../../../assets/wo_dropout.jpg" />


Now with dropout with following settings `python train.py --lr=0.001 --epochs=800 --dropout=True` you can get following results.

<img src="./../../../assets/w_dropout.jpg" />

