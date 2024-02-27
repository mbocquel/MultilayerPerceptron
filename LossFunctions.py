import numpy as np

def binaryCrossentropyLoss(model, x, y):
    epsilon = 1e-15 # Evite les erreurs de division par zéro
    f_tab = model.predict(x)
    y_pred = f_tab[:,1]
    loss = - np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
    return loss


def setLossFunction(lossFunctionName):
    match lossFunctionName:
        case "binaryCrossentropy":
            return binaryCrossentropyLoss
            self.activation_deriv = sigmoid_deriv
        case _:
            raise ValueError("Unknowed loss function")