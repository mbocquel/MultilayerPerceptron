import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import traceback
import os


def processArgs(**kwargs):
    path = None
    histogram = None
    scatter = None
    for key, value in kwargs.items():
        if value is not None:
            match key:
                case 'histogram':
                    histogram = value
                case 'scatter':
                    scatter = value
                case 'path':
                    path = value
    return path, histogram, scatter


def printHistogram(benin_set, malin_set):
    nb_col_graph = 5
    nb_lignes_graph = 6
    col_names = benin_set.columns
    plt.figure(figsize=(15, 15))
    for i in range(len(col_names)):
        plt.subplot(nb_lignes_graph, nb_col_graph, i + 1)
        plt.hist(benin_set.loc[:,col_names[i]], alpha = 0.5, lw=3, label="benin", color="g")
        plt.hist(malin_set.loc[:,col_names[i]], alpha = 0.5, lw=3, label="malin", color="r")
        plt.legend()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Histogram")
    plt.show()


def testParamsAreOk(value, col_names):
    try:
        result = int(value)
        assert result <= len(col_names)
        assert result >= 0
        return (result - 1)
    except Exception:
        return (None)
    
def printScatter(idFeature, benin_set, malin_set, col_names, nb_col_graph):
    print(f"     \033[32mShowing you the scatter plot for feature {idFeature + 1}\033[0m")
    nb_lignes_graph = int((len(col_names)- 1)/nb_col_graph) + 1
    plt.figure(figsize=(nb_col_graph * 10, nb_lignes_graph * 10))
    j = 0
    for i in range(len(col_names)):
        if (i != idFeature):
            plt.subplot(nb_lignes_graph, nb_col_graph, j + 1)
            plt.plot(benin_set.iloc[:,idFeature], benin_set.iloc[:,col_names[i]], 'o',  label="benin", color="g")
            plt.plot(malin_set.iloc[:,idFeature], malin_set.iloc[:,col_names[i]], 'o',  label="malin", color="r")
            plt.ylabel(str(col_names[i]))
            plt.legend()
            j += 1
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Scatter plot for " + str(idFeature + 1))
    plt.show()


def scatterPlot(benin_set, malin_set):
    nb_col_graph = 5
    col_names = benin_set.columns
    userMainFeature = testParamsAreOk(input("Please enter the number of the feature to compare, or 0 to quit: "), col_names)
    plt.rcParams.update({'font.size': 8})
    while (1):
        if (userMainFeature is None):
            print("     \033[31mPlease enter a correct number\033[0m")
        elif (userMainFeature == -1):
            return 0
        else:
            printScatter(userMainFeature, benin_set, malin_set, col_names, nb_col_graph)
        userMainFeature = testParamsAreOk(input("Please enter the number of the feature to compare, or 0 to quit: "), col_names)


def main(**kwargs):
    try:
        path, histogram, scatter = processArgs(**kwargs)
        assert path is not None, "Please enter a file path as parametter"
        assert os.path.isfile(path), "Please enter a file as a parametter"
        
        df = pd.read_csv(path, header=None)
        df[len(df.columns)] = 0.0
        df.loc[df.iloc[:,1] == 'M', len(df.columns) - 1 ] = 1
        df.drop([0, 1], axis=1, inplace=True)
        df.columns = range(len(df.columns))
        benin_set = df.loc[df.iloc[:,-1] == 0, 0:(len(df.columns)-2)]
        malin_set = df.loc[df.iloc[:,-1] == 1, 0:(len(df.columns)-2)]

        if histogram:
            printHistogram(benin_set, malin_set)
        if scatter:
            scatterPlot(benin_set, malin_set)
            

        return 0
    except Exception as err:
        print(f"Error: {err}\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="My MLP program")
    parser.add_argument("--histogram", "-hist", action="store_true",
                        help="show histogram")
    parser.add_argument("--scatter", "-s", action="store_true",
                        help="show pair plot")
    parser.add_argument("--path", "-p", type=str,
                        help="Path to the dataset directory")
    
    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)