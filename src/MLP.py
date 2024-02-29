import matplotlib.pyplot as plt
import numpy as np
from MySequencial import MySequencial
from DenseLayer import DenseLayer
from ActivationFunctions import isActivationFunction
from WeightInitialiser import isWeightInitialiser
from LossFunctions import isLossFunction
import pandas as pd
import argparse
import os
import traceback
from abc import ABC
import pickle


class MPLArgs(ABC):
    """Class for the Model"""
    def __init__(self, **kwargs) -> None:
        """MPLParams constructor"""
        super().__init__()
        self.steps = None
        self.dataset = None
        self.valPart = None
        self.layerRaw = None
        self.layer = None
        self.loss = None
        self.learningRate = None
        self.batchSize = None
        self.epochs = None
        self.dataToPredict = None
        self.data_train = None
        self.data_val = None
        self.myModel = None
        for key, value in kwargs.items():
            if value is not None:
                match key:
                    case 'steps':
                        self.steps = value
                    case 'dataset':
                        self.dataset = value
                    case 'valPart':
                        self.valPart = value
                    case 'layer':
                        self.layerRaw = value
                    case 'loss':
                        self.loss = value
                    case 'learningRate':
                        self.learningRate = value
                    case 'batchSize':
                        self.batchSize = value
                    case 'epochs':
                        self.epochs = value
                    case 'dataToPredict':
                        self.dataToPredict = value
        self.checkInputs()
    
    def checkSteps(self):
        assert self.steps is not None, "You need to provide the steps of the program to execute !"
        self.steps = np.sort(list(set(self.steps)))
        assert (len(np.where((self.steps > 3))[0]) == 0
                and len(np.where((self.steps < 1))[0]) == 0), "Problem with the steps of the program to execute"
    
    def checkStep1(self):
        assert (self.dataset is not None and os.path.isfile(self.dataset) 
                and (self.valPart is None or (self.valPart > 0 and self.valpart < 1))
                ), "Step 1 - dataset missing or invalid"
    
    def checkStep2(self):
        assert (1 in self.steps or 
                (os.path.isfile("data_train.csv") and os.path.isfile("data_val.csv"))
            ),"Step 2 - data_train.csv and or data_val.csv missing from the current directory"
        assert self.layerRaw is not None, "Step 2 - Layer argument is missing"
        layers = []
        for element in self.layerRaw:
            try:
                nb_Node = int(element)
                layers.append([nb_Node, "", ""])
            except ValueError:
                if len(layers) and isActivationFunction(element):
                    layers[-1][1] = element
                elif len(layers) and isWeightInitialiser(element):
                    layers[-1][2] = element
                else:
                    raise AssertionError("Step 2 - Invalid format for Layer argument")
        self.layer = layers
        assert self.loss is None or isLossFunction(self.loss), "Step 2 - Problem with the loss function"
        assert self.learningRate is None or self.learningRate > 0, "Step 2 - Problem with the learning rate"
        assert self.batchSize is None or self.batchSize > 1, "Step 2 - Problem with the batch size"
        assert self.epochs is None or self.epochs > 1, "Step 2 - Problem with the number of epochs" 

    def checkStep3(self):
        assert (1 in self.steps or 
                2 in self.steps or
                (self.dataToPredict is not None and os.path.isfile(self.dataToPredict))
                ), "Step 3 - Invalid or missing dataToPredict argument"
        assert (2 in self.steps or 
                os.path.isfile("saved_model.pkl")
                ), "Step 3 - saved_model.pkl needs to be in the current directory"
        
    def checkInputs(self):
        self.checkSteps()
        if 1 in self.steps:
            self.checkStep1()
        if 2 in self.steps:
            self.checkStep2()
        if 3 in self.steps:
            self.checkStep3()

    def summary(self):
        print("steps",self.steps)
        print("dataset",self.dataset)
        print("valPart",self.valPart)
        print("layerRaw",self.layerRaw)
        print("layer",self.layer)
        print("loss",self.loss)
        print("learningRate",self.learningRate)
        print("batchSize",self.batchSize)
        print("epochs",self.epochs)
        print("dataToPredict",self.dataToPredict)


def splitDataSet(mlp):
    df = pd.read_csv(mlp.dataset, header=None)
    valPart = mlp.valPart
    if valPart is None:
        valPart = 0.2
    df[len(df.columns)] = 0.0
    df.loc[df.iloc[:,1] == 'M', len(df.columns) - 1 ] = 1
    df.drop([0, 1], axis=1, inplace=True)
    df.columns = range(len(df.columns))
    data = df.to_numpy()
    np.random.seed(1)
    np.random.shuffle(data)
    mlp.data_train = data[:int(len(data) * (1 - valPart)),:]
    mlp.data_val = data[int(len(data) * (1 - valPart)):,:]
    X_train = mlp.data_train[:, :-1]
    X_val = mlp.data_val[:, :-1]
    print("Splitting the dataset:")
    print("     x_train shape :", X_train.shape)
    print("     x_valid shape :", X_val.shape,)
    print("Saving training set in data_train.csv")
    print("Saving validation set in data_val.csv\n")
    pd.DataFrame(mlp.data_train).to_csv("data_train.csv", sep=',', header=False, index=False)
    pd.DataFrame(mlp.data_val).to_csv("data_val.csv", sep=',', header=False, index=False)


def trainModel(mlp):
    if 1 not in mlp.steps:
        mlp.data_train = pd.read_csv("data_train.csv", header=None).to_numpy()
        mlp.data_val = pd.read_csv("data_val.csv", header=None).to_numpy()
    X_train = mlp.data_train[:, :-1]
    nb_example, nb_features = X_train.shape

    modelLayers = []
    for i in range(len(mlp.layer)):
        input_shape = None
        if i == 0:
            input_shape = nb_features
        if (len(mlp.layer[i][1]) and len(mlp.layer[i][2])):
            modelLayers.append(
                DenseLayer(mlp.layer[i][0], input_shape=input_shape,
                           activation=mlp.layer[i][1],
                           weights_initializer=mlp.layer[i][2]))
        elif (len(mlp.layer[i][1])):
            modelLayers.append(
                DenseLayer(mlp.layer[i][0], input_shape=input_shape,
                           activation=mlp.layer[i][1]))
        elif (len(mlp.layer[i][2])):
            modelLayers.append(
                DenseLayer(mlp.layer[i][0], input_shape=input_shape,
                           weights_initializer=mlp.layer[i][2]))
        else:
            modelLayers.append(
                DenseLayer(mlp.layer[i][0], input_shape=input_shape))
    mlp.myModel = MySequencial(modelLayers)
    mlp.myModel.fit(mlp.data_train, mlp.data_val, batch_size=mlp.batchSize, epochs=mlp.epochs, loss=mlp.loss, alpha=mlp.learningRate )
    mlp.myModel.printLearningCurve()
    mlp.myModel.save("saved_model.pkl")


def predict(mlp):
    if 2 not in mlp.steps:
        with open("saved_model.pkl", 'rb') as f:
            mlp.myModel = pickle.load(f)
    if mlp.dataToPredict is None and mlp.data_val is None:
        data = pd.read_csv("data_val.csv", header=None).to_numpy()
        data = mlp.myModel.normaliseData(data, False)
    elif mlp.dataToPredict is None:
        data = mlp.data_val
    else :
        data = pd.read_csv(mlp.dataToPredict, header=None).to_numpy()
        data = mlp.myModel.normaliseData(data, False)
    nb_example, nb_cols = data.shape
    
    if nb_cols == 31:
        print("Prediction done to validate the model:")
        X = data[:, :-1]
        y = data[:, -1]
        y_prediction = np.argmax(mlp.myModel.predict(X), axis=1)
        validite = y_prediction == y
        for i in range(nb_example):
            if validite[i]:
                print("\033[32m ", end = "")
            else:
                print("\033[31m ", end = "")
            print(f"{i+1} - Predicted: {'B' if y_prediction[i] == 0 else 'M'} - Expected: {'B' if y[i] == 0 else 'M'}\033[0m")
        
    elif nb_cols == 30:
        print("Prediction without validation")

    else:
        raise ValueError("Wrong number of features in the dataToPredict file")
    return

def main(**kwargs):
    try:
        mlp = MPLArgs(**kwargs)
        if 1 in mlp.steps:
            splitDataSet(mlp)
        if 2 in mlp.steps:
            trainModel(mlp)
        if 3 in mlp.steps:
            predict(mlp)

        return 0
    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="My MLP program")
    parser.add_argument("--steps", "-s", nargs="+", type=int,
                        help="Steps to run")
    parser.add_argument("--dataset", "-ds", type=str,
                        help="Path to the full dataset (training and validation)")
    parser.add_argument("--valPart", "-v", type=float,
                        help="Portion of the data to use for validation")
    parser.add_argument("--layer", "-la", nargs="+",
                        help="Name of the learning saving file")
    parser.add_argument("--loss", "-lo", type=str,
                        help="Loss function to use")
    parser.add_argument("--learningRate", "-lr", type=float,
                        help="Learning Rate to use")
    parser.add_argument("--batchSize", "-bs", type=int,
                        help="Size of the batchs")
    parser.add_argument("--epochs", "-e", type=int,
                        help="Number of epochs")
    parser.add_argument("--dataToPredict", "-dtp", type=str,
                        help="Path to the dataset to predict")
    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)
