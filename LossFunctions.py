import numpy as np

def binaryCrossentropyLoss(model, x, y):
    f_tab = model.predict(x)
    y = y.reshape(f_tab.shape)
    return -y * np.log(f_tab) - (1 - y) * np.log(1 - f_tab)
