import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def linear(z):
    return z


def softmax(z):
    exp_vect = np.exp(z)
    return exp_vect/sum(exp_vect)


def sigmoid_deriv(z):
    return z * (1.0 - z)
     

def relu_deriv(z):
    return np.where(z >= 0, 1, 0)
    

def linear_deriv(z):
    return 1


def softmax_deriv(z):
    s = softmax(z)
    return s * (1 - s)
