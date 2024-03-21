# Neural Network from Scratch
This repository contains a Python implementation of a neural network built from scratch. The network achieves an accuracy of 70% on the spiral dataset, demonstrating its capability to learn and classify non-linear patterns.

## Overview
The neural network is designed with the following components:

** Layer_Dense:**  A fully connected layer that performs the linear transformation of inputs.
Activation_ReLU: The Rectified Linear Unit (ReLU) activation function, used for introducing non-linearity into the network.
Activation_Softmax: The Softmax activation function, used in the output layer for multi-class classification.
Loss_CategoricalCrossentropy: The categorical cross-entropy loss function, used to measure the performance of the model.
Usage
To run the neural network, simply execute the script:

bash
Copy code
python neural_network.py
The script will train the model on the spiral dataset and print the loss and accuracy at every 1000 epochs.

Requirements
Python 3
NumPy
NNFS (Neural Networks from Scratch library)
Acknowledgments
This implementation is based on the concepts and examples from the book "Neural Networks from Scratch" by Sentdex.
