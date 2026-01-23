Cost Function
=============

My intuition is that the single most import thing in the neural net trainig is the cost function. You will say, why?. Beacause if you know that we calculate the gradients of cost function with respect to each parameters. And that's why we have different cost functions for different usecase. for example ``Binary cross entropy`` is for binary classification(0,1), ``Cross Entropy`` for multiclass classification.

The idea is to design a costfunction in such a way it 

1. Binary Cross Entropy

   .. autofunction:: nn.BCE

2. Cross Entropy

   .. autofunction:: nn.CE
