import numpy as np

def binaryCrossentropyLoss(model, x, y):
    epsilon = 1e-15 # Evite les erreurs de division par zéro
    f_tab = model.predict(x)
    y_pred = f_tab[:,1]
    loss = - np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    return loss

def sparseCategoricalCrossEntropyLoss(model, x, y):
    y_pred = model.predict(x)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    m = y.shape[0]
    loss = -1/m * np.sum(np.log(y_pred[np.arange(m), y.astype(int)]))
    return loss


def setLossFunction(lossFunctionName):
    match lossFunctionName:
        case "binaryCrossentropy":
            return binaryCrossentropyLoss
        case "sparseCategoricalCrossEntropyLoss":
            return sparseCategoricalCrossEntropyLoss
        case _:
            raise ValueError("Unknowed loss function")