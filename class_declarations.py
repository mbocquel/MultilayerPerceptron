from abc import ABC
import numpy as np
from typing import Any
from activation_functions import sigmoid, softmax, linear, relu
from weightInitialisor import constantOnesWI, constantZerosWI
from weightInitialisor import uniformWI, normalWI
from weightInitialisor import leCunNormalWI, leCunUniformWI
from weightInitialisor import xavierNormalWI, xavierUniformWI
from weightInitialisor import heNormalWI, heUniformWI
from loss_functions import binaryCrossentropyLoss


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
            self.W = self.weights_initializer(self.input_shape, nb_Node)
        if b is not None:
            self.b = b
        else:
            self.b = np.zeros(nb_Node)
        
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
            case "constantZeros":
                self.weights_initializer = constantZerosWI
            case "constantOnes":
                self.weights_initializer = constantOnesWI
            case "uniform":
                self.weights_initializer = uniformWI
            case "normal":
                self.weights_initializer = normalWI
            case "leCunUniform":
                self.weights_initializer = leCunUniformWI
            case "leCunNormal":
                self.weights_initializer = leCunNormalWI
            case "xavierUniform":
                self.weights_initializer = xavierUniformWI
            case "xavierNormal":
                self.weights_initializer = xavierNormalWI
            case "heUniform":
                self.weights_initializer = heUniformWI
            case "heNormal":
                self.weights_initializer = heNormalWI
            case _:
                raise ValueError("Unknowed weights_initializer function")
            
    def setAsInputLayer(self):
        self.W = np.diag(np.ones(self.input_shape))
        return

    def setInputShape(self, input_shape):
        self.input_shape = input_shape
        # Uniquement si changement de la taille de W besoin de le redefinir
        if self.input_shape != len(self.W):
            self.W = self.weights_initializer(self.input_shape, self.nb_Node)

    def printLayer(self) -> None:
        print("Layer Name:",self.layerName)
        print("Shape W:", self.W.shape)
        print("W:", self.W)
        print("Shape b:", self.b.shape)
        print("b:", self.b)
        print("activation:", self.activation)
        print("weights_initializer:", self.weights_initializer, "\n")


class MySequencial(ABC):
    """Class for the Model"""
    def __init__(self, layers) -> None:
        """My Model constructor"""
        super().__init__()
        self.layers = []
        prev_out_size = 0
        for i in range(len(layers)):
            layer = layers[i]
            if not isinstance(layer, DenseLayer):
                raise ValueError("MySequencial needs DenseLayer as argument")
            # Input Layer: 
            if i == 0:
                layer.setAsInputLayer()
            else:
                layer.setInputShape(prev_out_size)
            prev_out_size = layer.nb_Node
            self.layers.append(layer)
    
    def summary(self) -> None:
        for i in range(len(self.layers)):
            print("----------- Layer", i + 1, "-----------")
            self.layers[i].printLayer()

    def predictOneElem(self, x):
        a_in = x
        for layer in self.layers:
            a_out = layer.propag(a_in)
            a_in = a_out
        return a_out
        
    def predict(self, X):
        return np.array([[self.predictOneElem(elem) for elem in X]])
    
    def fit(self, data_train, data_valid, loss="binaryCrossentropy",
            learning_rate=0.0314, batch_size=8, epochs=15):
        return
    
    def cost(self, X, Y):
        m = X.shape[0]
        cost = (1 / m) * sum([binaryCrossentropyLoss(self, x, y) for x, y in zip(X, Y)])
        return cost

    # Finir https://www.coursera.org/learn/advanced-learning-algorithms/lecture/35RQ3/training-details