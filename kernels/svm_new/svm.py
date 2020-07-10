import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation
from auxiliarymethods.auxiliary_methods import read_lib_svm


def read_classes(ds_name):
    with open("../datasets/" + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return classes


def main():

    path = "../svm/SVM/src/EXP/"

    algorithm = "LWLP2"
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["NCI109", True], ["PROTEINS", True],
               ["PTC_FM", True], ["REDDIT-BINARY", False], ["ENZYMES", True]]

    for d, use_labels in dataset:
        gram_matrices = []
        for i in range(0,6):
            gram_matrix, classes = read_lib_svm(path + d + "__" + algorithm + "_" + str(i) + ".gram")
            gram_matrices.append(gram_matrix)

            acc, s_1, s_2 = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=1, all_std=True)
            print(acc)







if __name__ == "__main__":
    main()
