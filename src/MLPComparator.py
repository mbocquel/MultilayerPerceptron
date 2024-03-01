from abc import ABC
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class MLPComparator(ABC):
    """Class for Comparing differents MLP Models
    and their performances"""
    def __init__(self):
        super().__init__()
        self.historyTrain = []
    
    def addNewModel(self, mlp):
        newModel = {}
        newModel['architecture'] = mlp.myModel.getSequentialArchitecture()
        newModel['loss'] = mlp.myModel.historyTrain[-1]['loss']
        newModel['alpha'] = mlp.myModel.historyTrain[-1]['alpha']
        newModel['batchSize'] = mlp.myModel.historyTrain[-1]['batch_size']
        newModel['valLoss'] = mlp.myModel.historyTrain[-1]['valLoss']
        newModel['valAcc'] = mlp.myModel.historyTrain[-1]['valAcc']
        newModel['trainLoss'] = mlp.myModel.historyTrain[-1]['trainLoss']
        newModel['trainAcc'] = mlp.myModel.historyTrain[-1]['trainAcc']
        self.historyTrain.append(newModel)

    def printComparisonGraphs(self):
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])
        i = 1
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        modelText = ""
        for model in self.historyTrain:
            valLabel = "Validation set for model " + str(i)
            trainLabel = "Training set for model " + str(i)
            x = range(len(model["valLoss"]))
            ax0.plot(x, model["valLoss"], label=valLabel, linestyle='dashed')
            ax0.plot(x, model["trainLoss"], label=trainLabel)
            ax1.plot(x, model["valAcc"], label=valLabel, linestyle='dashed')
            ax1.plot(x, model["trainAcc"], label=trainLabel)
            modelText += "\033[1mModel\033[0m " + str(i) + ":\n[" + model["architecture"] + "]\nloss: " 
            modelText += str(model["loss"]) + ", alpha: " + str(model["alpha"])
            modelText += ", batch size: " + str(model["batchSize"]) + "\n\n"
            i += 1
        
        ax0.set_xlabel("epoch")
        ax0.set_ylabel("Loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Accuracy")
        ax0.legend()
        ax1.legend()
        ax0.set_title("Evolution of the loss")
        ax1.set_title("Evolution of the accuracy")
        print(modelText)
        plt.subplots_adjust(left=0.02, right=0.98,
                            bottom=0.02, top=0.98, wspace=0.2)
        plt.show()
