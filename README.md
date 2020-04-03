# Udacity---AI-Programming-with-Python-Nanodegree

## Summary

With the help of Udacity’s AI Programming with Python Nanodegree program, I built and trained a deep neural network to develop an image classifier for different kinds of flowers.The second portion of the project includes the two files “predict.py” and “train.py” which allow the user to enter hyperparameters and pre-trained networks types into the command line to train and predict a dataset of images. Both parts print training loss, validation loss, and testing accuracy.

### About Project Part I - Image Classifier

In a Jupyter Notebook, a flower dataset of images is downloaded as a training, testing, and validation set. 
Then the datasets are transformed in order to increase accuracy as well as fit the input format for pre-trained networks.
Resizing, cropping, random flipping are a few transformations.
Next, densenet121 is chosen to use as the pre-trained network, and a NeuralNetwork class with a feedforward method is defined.
Both ReLU activation and dropout are used in the the classifier.
After defining hyperparameters, such as number of epochs, the learning rate, etc. the model is trained on the training set. 
Training loss, validation loss, and accuracy are printed. 
A predict and check function are defined which output the top 5 possible flower species for a given image, along with their probabilities in a bar chart.

### About Project Part II - Command Line Application

The second portion of the project includes the 2 python files, “train.py” and “predict.py.”
This application allows people to train a model on a dataset of images and then predict their classes from the command line.
The train file uses the same NeuralNetwork class from Part I, but now the user can choose either vgg16, alexnet, or resnet18 as the pre-trained network.
Other parameters, such as number of epochs, number of hidden layers, etc. can be changed from the user.
This file should output training loss, validation loss, and accuracy; as well as save a checkpoint.
In the predict file, the checkpoint from the train file is loaded and then the top ‘k’ classes and their probabilities are printed.

Built With

-	Python, PyTorch
-	Pandas, Numpy, Matplotlib
-	Jupyter Notebook

Contributing

This project was guided by Udacity’s Nanodegree Program: AI Programming with Python.

Authors

Joshua Sebastian
