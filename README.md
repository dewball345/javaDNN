# javaDNN
Work-In-Progress autograd(not sure though) deep learning library made in java. 

# Features

<<<<<<< HEAD
As of 8/8/2021
* Implementation of `layer.DenseLayer`
* Implementation of `losses.MSELoss`
=======
As of 8/13/2021
* Implementation of `DenseLayer`
* Implementation of `MSELoss`
>>>>>>> 15c7bdda716e21be8a816518c7fbee0f2cc6b8ab
* Implementation of `Relu`
* Implementation of `LeakyRelu`
* Implementation of `Sigmoid`
* Batch training loop

Need to add activation functions, different kinds of optimizers, many other layers(like cnn, lstm, etc), and more.
(certainly will have to change my code quite a bit(as my gradient update function won't work for softmax and layers like CNN's and RNN's))

Also have to change my training loop a bit...

This readme will be updated with more information, such as installation/usage/etc.

# Example

main.java contains a simple example of this.

# Want a production-ready deep learning framework in Java?

This is far from done, and certainly not ready for production usage. Try [DL4J](https://deeplearning4j.org/)

# Issues

Perhaps some code was not implemented correctly(this project is a way for me to learn the low-level aspects of deep learning). If that is the case, please report an issue explaining so.
