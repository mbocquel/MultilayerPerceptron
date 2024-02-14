from abc import ABC
import numpy as np
from typing import Any


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.max(0, z)


def linear(z):
    return z


def softmax(z):
    return


def constantWeightInit(input_units, output_units, value=1):
    return np.ones(input_units, output_units) * value


def uniformWeightInit(input_units, output_units, low=-0.05, high=0.05):
    return np.random.uniform(low, high, size=(input_units, output_units))


def normalWeightInit(input_units, output_units, mean=0.0, std=0.05):
    return np.random.normal(mean, std, size=(input_units, output_units))


def leCunUniformWeightInit(in_units, out_units):
    limit = np.sqrt(3 / float(in_units))
    return np.random.uniform(low=-limit, high=limit, size=(in_units, out_units))


def leCunNormalWeightInit(in_units, out_units):
    limit = np.sqrt(1 / float(in_units))
    return np.random.normal(0.0, limit, size=(in_units, out_units))
    

def xavierUniformWeightInit(in_units, out_units):
    limit = np.sqrt(6 / float(in_units + out_units))
    return np.random.uniform(-limit, limit, size=(in_units, out_units))
    

def xavierNormalWeightInit(in_units, out_units):
    limit = np.sqrt(2 / float(in_units + out_units))
    return np.random.normal(0.0, limit, size=(in_units, out_units))


def heUniformWeightInit(in_units, out_units):
    limit = np.sqrt(6 / float(in_units))
    return np.random.uniform(low=-limit, high=limit, size=(in_units, out_units))


def heNormalWeightInit(in_units, out_units):
    limit = np.sqrt(2 / float(in_units))
    return np.random.normal(0.0, limit, size=(in_units, out_units))


# Besoin de gerer weights_initializer
class DenseLayer(ABC):
    """Class for the Dense Layer:
    nb_Node:int
    input_shape: int, nb of nodes of the previous layer
    activation: str, type of the activation function used
    W: array (input_shape, nb_Node), can be given as argument to load existing values.
    b: array(nb_Node), the bias used
    layername: str, a name so it's easier to organise the neural network
    weights_initializer:str, the method used for weights initialisation
    """
    def __init__(self, nb_Node:int, activation = "relu", W = None, b = None,
                 layerName = "Dense", weights_initializer = 'heUniform') -> None:
        """My Dense Layer constructor"""
        super().__init__()
        self.nb_Node = nb_Node
        self.layerName = layerName
        self.input_shape = nb_Node
        self.input_layer = True
        self.setActivation(activation)
        self.setWeightInit(weights_initializer)
        if W is not None:
            self.input_shape = len(W)
            self.W = W
        else:
            self.W = np.random.rand(self.input_shape, nb_Node)
        if b is not None:
            self.b = b
        else:
            self.b = np.random.rand(nb_Node)
        
    def propag(self, a_in):
        z = self.W.transpose() @ a_in + self.b
        a_out = self.activation(z)
        return a_out
    
    def setActivation(self, activation = "relu"):
        match activation:
            case "sigmoid":
                self.activation = sigmoid
            case "relu":
                self.activation = relu
            case "linear":
                self.activation = linear
            case "softmax":
                self.activation = softmax
            case _:
                raise ValueError("Unknowed activation function")

    def setWeightInit(self, weights_initializer = 'heUniform'):
        match weights_initializer:
            case "constant":
                self.weights_initializer = constantWeightInit
            case "uniform":
                self.weights_initializer = uniformWeightInit
            case "normal":
                self.weights_initializer = normalWeightInit
            case "leCunUniform":
                self.weights_initializer = leCunUniformWeightInit
            case "leCunNormal":
                self.weights_initializer = leCunNormalWeightInit
            case "xavierUniform":
                self.weights_initializer = xavierUniformWeightInit
            case "xavierNormal":
                self.weights_initializer = xavierNormalWeightInit
            case "heUniform":
                self.weights_initializer = heUniformWeightInit
            case "heNormal":
                self.weights_initializer = heNormalWeightInit
            case _:
                raise ValueError("Unknowed weights_initializer function")
            
    def setAsInputLayer(self):
        # for i in len()
        return

    def setInputShape(self, input_shape):
        self.input_shape = input_shape
        # Uniquement si changement de la taille de W besoin de le redefinir
        if self.input_shape != len(self.W):
            self.W = np.random.rand(self.input_shape, self.nb_Node)

    def printLayer(self) -> None:
        print("Layer Name:",self.layerName)
        print("Shape W:",self.W.shape)
        print("W:",self.W)
        print("Shape b:", self.b.shape)
        print("b:", self.b)
        print("activation:", self.activation, "\n")
        print("weights_initializer:", self.weights_initializer, "\n")


class MySequencial(ABC):
    """Class for the Model"""
    def __init__(self, layers) -> None:
        """My Model constructor"""
        super().__init__()
        self.layers = []
        for layer in layers:
            if not isinstance(layer, DenseLayer):
                raise ValueError("MySequencial needs DenseLayer as argument")
            # Besoin de modifier le layer en fonction de ce qu'on a deja recu.
            self.layers.append(layer)
    
    def summary(self) -> None:
        for i in range(len(self.layers)):
            print("----------- Layer", i + 1, "-----------")
            self.layers[i].printLayer()

    def propag(self, x):
        a_in = x
        for layer in self.layers:
            a_out = layer.propag(a_in)
            a_in = a_out
        return a_out

            



