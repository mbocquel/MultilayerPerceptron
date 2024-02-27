from abc import ABC
import numpy as np
from LossFunctions import binaryCrossentropyLoss
from DenseLayer import DenseLayer


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
        # On commence par forward propag
        self.predict(X_batch)
        # Set Error for Last layer
        self.layers[-1].setErrorLastLayer(y_batch)
        previousLayer = self.layers[-1]
        # set Error for Hidden layers
        for i in reversed(range(1, len(self.layers)-1)):
            self.layers[i].setError(previousLayer)
            previousLayer = self.layers[i]
        # Change the params     
            
    
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
