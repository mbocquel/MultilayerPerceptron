import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.max(0, z)


def linear(z):
    return z


def softmax(z):
    exp_vect = np.exp(z)
    return exp_vect/sum(exp_vect)
