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
        newModel['precisionTrain'] = mlp.myModel.historyTrain[-1]['precisionTrain']
        newModel['precisionVal'] = mlp.myModel.historyTrain[-1]['precisionVal']
        newModel['recallTrain'] = mlp.myModel.historyTrain[-1]['recallTrain']
        newModel['recallVal'] = mlp.myModel.historyTrain[-1]['recallVal']
        self.historyTrain.append(newModel)

    def printComparisonGraphs(self):
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1])
        i = 1
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        modelText = ""
        precisionRecall = False
        for model in self.historyTrain:
            valLabel = "Validation set for model " + str(i)
            trainLabel = "Training set for model " + str(i)
            x = range(len(model["valLoss"]))
            ax0.plot(x, model["valLoss"], label=valLabel, linestyle='dashed')
            ax0.plot(x, model["trainLoss"], label=trainLabel)
            ax1.plot(x, model["valAcc"], label=valLabel, linestyle='dashed')
            ax1.plot(x, model["trainAcc"], label=trainLabel)
            if len(model["precisionTrain"]):
                precisionRecall = True
                ax2.plot(x, model["precisionVal"], label=valLabel, linestyle='dashed')
                ax2.plot(x, model["precisionTrain"], label=trainLabel)
                ax3.plot(x, model["recallVal"], label=valLabel, linestyle='dashed')
                ax3.plot(x, model["recallTrain"], label=trainLabel)
            modelText += "\033[1mModel\033[0m " + str(i) + ":\n[" + model["architecture"] + "]\nloss: " 
            modelText += str(model["loss"]) + ", alpha: " + str(model["alpha"])
            modelText += ", batch size: " + str(model["batchSize"]) + "\n\n"
            i += 1
        
        ax0.set_xlabel("epoch")
        ax0.set_ylabel("Loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Accuracy")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("Precision")
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("Recall")
        ax0.legend()
        ax1.legend()
        if precisionRecall:
            ax2.legend()
            ax3.legend()
        ax0.set_title("Loss")
        ax1.set_title("Accuracy")
        ax2.set_title("Precision")
        ax3.set_title("Recall")
        print(modelText)
        plt.subplots_adjust(left=0.02, right=0.98,
                            bottom=0.02, top=0.95, wspace=0.2, hspace=0.2)
        plt.show()