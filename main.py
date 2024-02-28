import matplotlib.pyplot as plt
import numpy as np
from MySequencial import MySequencial
from DenseLayer import DenseLayer
import pandas as pd
import argparse
import os
import traceback


def processArgs(**kwargs):
    epochs = 150
    path = None
    save_name = "Learning"
    train_portion = 0.75
    batch_size = 8
    for key, value in kwargs.items():
        if value is not None:
            match key:
                case 'epochs':
                    epochs = value
                case 'path':
                    path = value
                case 'save_name':
                    save_name = value
                case 'train_portion':
                    train_portion = value
                case 'batch_size':
                    batch_size = value
    return epochs, path, save_name, train_portion, batch_size


def main(**kwargs):
    try:
        epochs, path, save_name, train_portion, batch_size = processArgs(**kwargs)
        assert path is not None, "Please enter a file path as parametter"
        assert os.path.isfile(path), "Please enter a file as a parametter"
        
        df = pd.read_csv(path, header=None)
        df.iloc[:,-1] -= 1
        data = df.to_numpy()
        np.random.shuffle(data)
        np.random.seed(1)
        data_train = data[:int(len(data) * train_portion),:]
        data_val = data[int(len(data) * train_portion):,:]

        myModel = MySequencial([
            DenseLayer(8, activation='sigmoid', input_shape=7),
            DenseLayer(8, activation='sigmoid'),
            DenseLayer(3, activation='softmax')
        ])

        myModel.fit(data_train, data_val, batch_size=batch_size, epochs=epochs, loss="sparseCategoricalCrossEntropyLoss")
        myModel.printLearningCurve()
        myModel.save(save_name)
        return 0
    except Exception as err:
        print(f"Error: {err}\n")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="My MLP program")
    parser.add_argument("--epochs", "-e", type=int,
                        help="Number of epochs for the training")
    parser.add_argument("--path", "-p", type=str,
                        help="Path to the dataset directory")
    parser.add_argument("--save_name", "-s", type=str,
                        help="Name of the learning saving file")
    parser.add_argument("--train_portion", "-t", type=float,
                        help="Portion of the data to use for training")
    parser.add_argument("--batch_size", "-b", type=int,
                        help="Size of the batchs")
    parser.add_argument("--loss", "-l", type=str,
                        help="Name of the loss function")

    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)