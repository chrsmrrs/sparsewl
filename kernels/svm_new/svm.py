import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from auxiliarymethods.auxiliary_methods import read_lib_svm

def read_classes(ds_name):
    with open("../datasets/" + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return classes


def main():

    path = "../svm/SVM/src/EXP/"

    algorithm = "LWL2P"
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["NCI109", True], ["PROTEINS", True],
               ["PTC_FM", True], ["REDDIT-BINARY", False], ["ENZYMES", True]]

    for d, use_labels in dataset:
        for i in range(0,6):
            gram_matrix, classes = read_lib_svm(path + d + "__" + algorithm + "_" + str(i) + ".gram")
            print(gram_matrix.shape)




if __name__ == "__main__":
    main()
