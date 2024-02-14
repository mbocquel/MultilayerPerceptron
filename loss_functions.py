import numpy as np

def binaryCrossentropyLoss(model, x, y):
        f = model.predictOneElem(x)
        return -y * np.log(f) - (1 - y) * np.log(1 - f)