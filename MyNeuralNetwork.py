from abc import ABC
import numpy as np
from typing import Any
from ActivationFunctions import sigmoid, softmax, linear, relu
from ActivationFunctions import sigmoid_deriv, relu_deriv, linear_deriv, softmax_deriv
from WeightInitialisator import constantOnesWI, constantZerosWI
from WeightInitialisator import uniformWI, normalWI
from WeightInitialisator import leCunNormalWI, leCunUniformWI
from WeightInitialisator import xavierNormalWI, xavierUniformWI
from WeightInitialisator import heNormalWI, heUniformWI
from LossFunctions import binaryCrossentropyLoss


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
                 layerName = None, weights_initializer = 'heUniform') -> None:
        """My Dense Layer constructor"""
        super().__init__()
        self.nb_Node = nb_Node
        self.layerName = layerName
        self.input_shape = nb_Node
        self.input_layer = False
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
            self.b = np.zeros((1, nb_Node))
        self.errors = list()
        self.delta = list()
        return
        
    def forward(self, a_in):
        self.input = a_in
        self.z = a_in @ self.W + self.b
        self.output = [self.activation(z) for z in self.z]
        return self.output
    
    def setErrorLastLayer(self, y):
        expected = np.zeros((len(y),self.nb_Node))
        for i in range(len(y)):
            expected[i][y[i]] = 1
        self.errors = self.output - expected
        self.delta = [diff * self.activation_deriv(output) for diff, output in zip(self.errors, self.output)]
        return
    
    # A faire pour les couches du milieu
    # def setError(self):
    #     return
    
    def setActivation(self, activation = "relu"):
        match activation:
            case "sigmoid":
                self.activation = sigmoid
                self.activation_deriv = sigmoid_deriv
            case "relu":
                self.activation = relu
                self.activation_deriv = relu_deriv
            case "linear":
                self.activation = linear
                self.activation_deriv = linear_deriv
            case "softmax":
                self.activation = softmax
                self.activation_deriv = softmax_deriv
            case _:
                raise ValueError("Unknowed activation function")
        return

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
        self.input_layer = True
        self.W = np.diag(np.ones(self.input_shape))
        return

    def setInputShape(self, input_shape):
        self.input_shape = input_shape
        # Uniquement si changement de la taille de W besoin de le redefinir
        if self.input_shape != len(self.W):
            self.W = self.weights_initializer(self.input_shape, self.nb_Node)

    def setLayerName(self, name):
        self.layerName = name

    def summary(self, full=False) -> None:
        print("Layer Name:",self.layerName)
        print("Input Layer :", self.input_layer)
        print("Shape W:", self.W.shape)
        if (full):
            print("W:", self.W)
        print("Shape b:", self.b.shape)
        if (full):
            print("b:", self.b)
        print("activation:", self.activation.__name__)
        print("weights_initializer:", self.weights_initializer.__name__, "\n")


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
            name = f"Dense {i+1}"
            if i == 0:
                layer.setAsInputLayer()
                name = name + " (input)"
            else:
                layer.setInputShape(prev_out_size)
            if layer.layerName is None :
                layer.setLayerName(name)
            prev_out_size = layer.nb_Node
            self.layers.append(layer)
    
    def summary(self, full=False) -> None:
        for i in range(len(self.layers)):
            print("----------- Layer", i + 1, "-----------")
            self.layers[i].summary(full)
        
    def predict(self, X):
        a_in = X
        for layer in self.layers:
            a_out = layer.forward(a_in)
            a_in = a_out
        return a_out
    
    def back_propag(self, X_batch, y_batch):
        # Last layer
        last_layer = self.layers[-1]
        
        # last_layer.errors = 
        return
    
    def fit(self, data_train, data_valid, loss="binaryCrossentropy",
            alpha=0.0314, batch_size=8, epochs=15):
        
        X_val = data_valid[:, :-1]
        y_val = data_valid[:, -1]
        assert batch_size > 0, "batch_size need to be > 0"
        nb_batch = len(data_train) / batch_size  + 1
        if len(data_train) % batch_size == 0:
            nb_batch = nb_batch - 1
        for epoch in range(epochs):
            print(f"epoch {epoch} :")
            np.random.shuffle(data_train)
            X_train = data_train[:, :-1]
            y_train = data_train[:, -1]
            X_batchs = [X_train[i:i+batch_size] for i in range(0, len(X_train), batch_size)]
            y_batchs = [y_train[i:i+batch_size] for i in range(0, len(y_train), batch_size)]
            num_batch = 0
            for X_batch, y_batch in zip(X_batchs, y_batchs):
                num_batch += 1
                self.back_propag(X_batch, y_batch)
                print(f"{num_batch}/{nb_batch} - accuracy: {self.accuracy(X_train, y_train)}, ", end='', flush=True)
                print(f"loss:{self.cost(X_train, y_train)}, val accuracy: {self.accuracy(X_val, y_val)}, ", end='', flush=True)
                print(f"val loss: {self.cost(X_val, y_val)} \r", end='', flush=True)
        return

    def cost(self, X, Y):
        m = X.shape[0]
        cost = (1 / m) * sum(binaryCrossentropyLoss(self, X, Y))
        return cost[0]
    
    def accuracy(self, X, Y):
        return 0
    
    def update_mini_batch(self, mini_batch, learning_rate):
        return
