from abc import ABC
import numpy as np
from LossFunctions import setLossFunction
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
    
    def back_propag(self, X_batch, y_batch, alpha):
        # On commence par forward propag
        self.predict(X_batch)
        # Set Error for Last layer
        self.layers[-1].setErrorLastLayer(y_batch)
        previousLayer = self.layers[-1]
        # Set Error for Hidden layers
        for i in reversed(range(1, len(self.layers)-1)):
            self.layers[i].setError(previousLayer)
            previousLayer = self.layers[i]
        # Update the weight
        for layer in self.layers[1:]:
            saveW = []
            saveBias = []
            for input, delta in zip(layer.input, layer.delta):
                W = layer.W.copy()
                bias = layer.b
                for numNode in range(layer.nb_Node):
                    for numFeature in range(len(input)):
                        W[numFeature][numNode] -= alpha * delta[numNode] * input[numFeature]
                    bias[0][numNode] -= alpha * delta[numNode]  
                saveW.append(W)
                saveBias.append(bias)
            layer.W = sum(saveW)/len(saveW)
            layer.b = sum(saveBias) / len(saveBias)

    
    def fit(self, data_train, data_valid, loss="binaryCrossentropy",
            alpha=0.0314, batch_size=8, epochs=15):
        lossFunction = setLossFunction(loss)
        self.lossTrain = []
        self.lossVal = []
        X_val = data_valid[:, :-1]
        y_val = data_valid[:, -1]
        assert batch_size > 0, "batch_size need to be > 0"
        nb_batch = int(len(data_train) / batch_size)  + 1
        if len(data_train) % batch_size == 0:
            nb_batch = nb_batch - 1
        for epoch in range(epochs):
            data = data_train.copy()
            np.random.shuffle(data)
            X_train = data[:, :-1]
            y_train = data[:, -1]
            X_batchs = [X_train[i:i+batch_size] for i in range(0, len(X_train), batch_size)]
            y_batchs = [y_train[i:i+batch_size] for i in range(0, len(y_train), batch_size)]
            num_batch = 0
            for X_batch, y_batch in zip(X_batchs, y_batchs):
                num_batch += 1
                self.back_propag(X_batch, y_batch, alpha)   
                trainloss = lossFunction(self, X_train, y_train)
                print(f"epoch {epoch} : {num_batch}/{nb_batch} - accuracy: {self.accuracy(X_train, y_train)}, ", end='')
                print(f"loss:{trainloss}\r", end='', flush=True)
            valLoss = lossFunction(self, X_val, y_val)
            print(f"epoch {epoch} : {nb_batch}/{nb_batch} - accuracy: {self.accuracy(X_train, y_train)}, ", end='')
            print(f"loss:{trainloss}, val accuracy: {self.accuracy(X_val, y_val)}, ", end='', flush=True)
            print(f"val loss: {valLoss}")
            self.lossTrain.append(trainloss)
            self.lossVal.append(valLoss)
        return
    
    def accuracy(self, X, Y):
        y_prediction = np.argmax(self.predict(X), axis=1)
        correct_predictions = np.sum(Y == y_prediction)
        return correct_predictions / len(Y)
