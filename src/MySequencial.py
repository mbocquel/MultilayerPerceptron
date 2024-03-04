from abc import ABC
import numpy as np
from LossFunctions import setLossFunction
from DenseLayer import DenseLayer
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import copy 


class MySequencial(ABC):
    """Class for the Model"""
    def __init__(self, layers) -> None:
        """My Model constructor"""
        super().__init__()
        self.layers = []
        self.lossTrain = []
        self.lossVal = []
        self.accTrain = []
        self.accVal = []
        prev_out_size = 0
        self.trainMean = None
        self.trainStdDev = None
        self.historyTrain = []
        for i in range(len(layers)):
            layer = layers[i]
            if not isinstance(layer, DenseLayer):
                raise ValueError("MySequencial needs DenseLayer as argument")
            # Input Layer: 
            name = f"Dense {i+1}"
            if i == 0 and layer.input_shape is None:
                raise ValueError("The first layer in MySequencial needs to have an inputshape")
            elif i != 0:
                layer.setInputShape(prev_out_size)
            if layer.layerName is None :
                layer.setLayerName(name)
            prev_out_size = layer.nb_Node
            self.layers.append(layer)
    
    def normaliseData(self, data, training = True):
        df = pd.DataFrame(data)
        if training is True:
            # print(df.describe)
            self.trainMean = df.describe().loc["mean",:len(df.columns)-2].to_numpy()
            self.trainStdDev = df.describe().loc["std",:len(df.columns)-2].to_numpy()
        if self.trainStdDev is None:
            raise ValueError("The model needs to be trained before used for predicting")
        newdata = copy.deepcopy(data)
        newdata[:, :-1] = (data[:, :-1] - self.trainMean) / self.trainStdDev
        return newdata
    
    def getSequentialArchitecture(self):
        text = ""
        i = 0
        for layer in self.layers:
            if i != 0:
                text += " | "
            text += str(layer.W.shape[1]) + " "
            text += layer.activation.__name__ + " "
            text += layer.weights_initializer.__name__
            i += 1
        return text
    
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
        for i in reversed(range(0, len(self.layers)-1)):
            self.layers[i].setError(previousLayer)
            previousLayer = self.layers[i]
        # Update the weight
        for layer in self.layers[0:]:
            saveW = []
            saveBias = []
            for input, delta in zip(layer.input, layer.delta):
                W = copy.deepcopy(layer.W)
                bias = layer.b
                for numNode in range(layer.nb_Node):
                    for numFeature in range(len(input)):
                        W[numFeature][numNode] -= alpha * delta[numNode] * input[numFeature]
                    bias[0][numNode] -= alpha * delta[numNode]  
                saveW.append(W)
                saveBias.append(bias)
            layer.W = sum(saveW)/len(saveW)
            layer.b = sum(saveBias) / len(saveBias)

    
    def fit(self, data_train, data_valid, loss, alpha,
            batch_size, epochs, earlyStop, precisionRecall):
        if loss is None:
            loss = "binaryCrossentropy"
        if alpha is None:
            alpha = 0.0314
        if batch_size is None:
            batch_size = 8
        if epochs is None:
            epochs = 150
        savedModel = []
        # On enregistre ce nouveau train.
        fit_save = {}
        fit_save['loss'] = loss
        fit_save['alpha'] = alpha
        fit_save['batch_size'] = batch_size
        fit_save['epochs'] = epochs
        fit_save['valLoss'] = []
        fit_save['valAcc'] = []
        fit_save['trainLoss'] = []
        fit_save['trainAcc'] = []
        fit_save['precisionTrain'] = []
        fit_save['precisionVal'] = []
        fit_save['recallTrain'] = []
        fit_save['recallVal'] = []
        self.historyTrain.append(fit_save)
        lossFunction = setLossFunction(loss)
        data_train_normalised = self.normaliseData(data_train, training = True)
        data_valid_normalised = self.normaliseData(data_valid, training = False)
        X_val = data_valid_normalised[:, :-1]
        y_val = data_valid[:, -1]
        nb_batch = int(len(data_train_normalised) / batch_size)  + 1

        if len(data_train_normalised) % batch_size == 0:
            nb_batch = nb_batch - 1

        for epoch in range(epochs):
            data = copy.deepcopy(data_train_normalised)
            np.random.shuffle(data)
            X_train = data_train_normalised[:, :-1]
            y_train = data_train_normalised[:, -1]
            X_batchs = [X_train[i:i+batch_size] for i in range(0, len(X_train), batch_size)]
            y_batchs = [y_train[i:i+batch_size] for i in range(0, len(y_train), batch_size)]
            num_batch = 0
            for X_batch, y_batch in zip(X_batchs, y_batchs):
                num_batch += 1
                self.back_propag(X_batch, y_batch, alpha)   
            trainLoss = lossFunction(self, X_train, y_train)
            trainAcc = self.accuracy(X_train, y_train)
            valLoss = lossFunction(self, X_val, y_val)
            valAcc = self.accuracy(X_val, y_val)
            print(f"epoch {epoch+1}/{epochs} : accuracy: {trainAcc}%, ", end='')
            print(f"loss: {round(trainLoss, 3)}, ", end='')
            if precisionRecall:
                precisionTrain, recallTrain = self.precisionAndRecall(X_train, y_train)
                precisionVal, recallVal = self.precisionAndRecall(X_val, y_val)
                print(f"precision: {precisionTrain}%, ", end='')
                print(f"recall: {recallTrain}%, ", end='')
                print(f"val_precision: {precisionVal}%, ", end='')
                print(f"val_recall: {recallVal}%, ", end='')
            print(f"val_accuracy: {valAcc}%, ", end='')
            print(f"val_loss: {round(valLoss, 3)}")

            if earlyStop: 
                if len(self.historyTrain[-1]['valLoss']) >= 5 and self.historyTrain[-1]['valLoss'][-5] <= valLoss:
                    print("Model is doing worse on the validation set. Early stoping the training")
                    self.layers = savedModel[-5]
                    del(self.historyTrain[-1]['valLoss'][-4:])
                    del(self.historyTrain[-1]['valAcc'][-4:])
                    del(self.historyTrain[-1]['trainLoss'][-4:])
                    del(self.historyTrain[-1]['trainAcc'][-4:])
                    del(self.historyTrain[-1]['precisionTrain'][-4:])
                    del(self.historyTrain[-1]['precisionVal'][-4:])
                    del(self.historyTrain[-1]['recallTrain'][-4:])
                    del(self.historyTrain[-1]['recallVal'][-4:])
                    return
                savedModel.append(copy.deepcopy(self.layers))
                del(savedModel[:-5])
            
            self.historyTrain[-1]['valLoss'].append(valLoss)
            self.historyTrain[-1]['valAcc'].append(valAcc)
            self.historyTrain[-1]['trainLoss'].append(trainLoss)
            self.historyTrain[-1]['trainAcc'].append(trainAcc)
            if precisionRecall:
                self.historyTrain[-1]['precisionTrain'].append(precisionTrain)
                self.historyTrain[-1]['precisionVal'].append(precisionVal)
                self.historyTrain[-1]['recallTrain'].append(recallTrain)
                self.historyTrain[-1]['recallVal'].append(recallVal)
        return

    def accuracy(self, X, Y):
        y_prediction = np.argmax(self.predict(X), axis=1)
        correct_predictions = np.sum(Y == y_prediction)
        return round((correct_predictions / len(Y)) * 100, 2)
    
    def precisionAndRecall(self, X, y_val):
        y_prediction = np.argmax(self.predict(X), axis=1)
        truePos = np.sum((y_prediction == 1) & (y_val == 1))
        falsePos = np.sum((y_prediction == 1) & (y_val == 0))
        falseNeg = np.sum((y_prediction == 0) & (y_val == 1))
        precision = 0
        recall = 0
        if (truePos + falsePos) != 0:
            precision = round((truePos / (truePos + falsePos)) * 100, 2)
        if (truePos + falseNeg) != 0:
            recall = round((truePos / (truePos + falseNeg)) * 100, 2)
        return precision, recall

    def save(self, save_name):
        with open(save_name, 'wb') as f:
            pickle.dump(self, f)
        print("\nModel savec in", save_name, "\n")
        return

    def printLastLearningCurve(self):
        if len(self.historyTrain[-1]['trainLoss']) == 0 or len(self.historyTrain[-1]['valLoss']) == 0:
            return
        valLoss = self.historyTrain[-1]['valLoss']
        trainLoss = self.historyTrain[-1]['trainLoss']
        valAcc = self.historyTrain[-1]['valAcc']
        trainAcc = self.historyTrain[-1]['trainAcc']
        loss =  self.historyTrain[-1]['loss']
        alpha =  self.historyTrain[-1]['alpha']
        batch_size =  self.historyTrain[-1]['batch_size']
        nb_epoch = len(valLoss)
        x = range(nb_epoch)
        plt.figure(figsize=(13, 7))
        plt.subplot(1, 2, 1)
        plt.plot(x, valLoss, "b", label="Validation Set")
        plt.plot(x, trainLoss, "r", label="Training Set")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Evolution of the loss \n(alpha={alpha}, loss={loss}, batchSize={batch_size})")

        plt.subplot(1, 2, 2)
        plt.plot(x, valAcc, "b", label="Validation Set")
        plt.plot(x, trainAcc, "r", label="Training Set")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Evolution of the accuracy")

        plt.show()
