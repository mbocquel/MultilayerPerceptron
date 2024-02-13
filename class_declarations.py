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

class DenseLayer(ABC):
    """Class for the Dense Layer"""
    def __init__(self, nb_Node:int, activation:str, node_size = 1, W = None, b = None, layerName = "Dense") -> None:
        """My Dense Layer constructor"""
        super().__init__()
        self.nb_Node = nb_Node
        self.layerName = layerName
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
        if W is not None:
            self.node_size = len(W)
            self.W = W
        else:
            self.node_size = node_size
            self.W = np.random.rand(node_size, nb_Node)
        if b is not None:
            self.b = b
        else:
            self.b = np.random.rand(nb_Node)
        
    def propag(self, a_in):
        z = self.W.transpose() @ a_in + self.b
        a_out = self.activation(z)
        return a_out
    
    def printLayer(self) -> None:
        print("Layer Name:",self.layerName)
        print("Shape W:",self.W.shape)
        print("W:",self.W)
        print("Shape b:", self.b.shape)
        print("b:", self.b)
        print("activation:", self.activation, "\n")


class MySequencial(ABC):
    """Class for the Model"""
    def __init__(self, *args: Any) -> None:
        """My Model constructor"""
        super().__init__()
        self.layers = []
        for arg in args:
            if not isinstance(arg, DenseLayer):
                raise ValueError("MySequencial needs DenseLayer as argument")
            self.layers.append(arg)
    
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

            



