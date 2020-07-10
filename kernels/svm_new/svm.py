import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation, linear_svm_evaluation
from auxiliarymethods.auxiliary_methods import read_lib_svm


def read_classes(ds_name):
    with open("../../../WWW/graphkerneldatasets/DS_all/" + ds_name + "/" + ds_name + "_graph_labels.txt", "r") as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return np.array(classes)


def main():


    # path = "../svm/SVM/src/EXP/"
    # dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["NCI109", True], ["PROTEINS", True],
    #            ["PTC_FM", True], ["REDDIT-BINARY", False], ["ENZYMES", True]]
    # algorithms = ["LWLP2", "LWLP3"]
    #
    # for a in algorithms:
    #     for d, use_labels in dataset:
    #         gram_matrices = []
    #         for i in range(0,6):
    #             gram_matrix, _ = read_lib_svm(path + d + "__" + a + "_" + str(i) + ".gram")
    #             classes = read_classes(d)
    #             gram_matrices.append(gram_matrix)
    #
    #
    #         acc, s_1, s_2 = kernel_svm_evaluation(gram_matrices, classes, num_repetitions=10, all_std=True)
    #         print(acc, s_1, s_2)

    path = "../svm/SVM/src/EXPSPARSE/"
    for name in ["Yeast", "YeastH", "UACC257", "UACC257H", "OVCAR-8", "OVCAR-8H"]:
        # for algorithm in ["WL", "LWL2", "LWLP2"]:
        for algorithm in ["LWLP2"]:

            print(name)
            print(algorithm)

            # Collect feature matrices over all iterations
            all_feature_matrices = []
            classes = read_classes(name)
            for i in range(0, 6):
                # Load feature matrices.
                feature_vector = pd.read_csv(path + name + "__" + algorithm + "_" + str(i), header=1,
                                             delimiter=" ").to_numpy()

                feature_vector = feature_vector.astype(int)
                feature_vector[:, 0] = feature_vector[:, 0] - 1
                feature_vector[:, 1] = feature_vector[:, 1] - 1
                feature_vector[:, 2] = feature_vector[:, 2] + 1

                xmax = int(feature_vector[:, 0].max())
                ymax = int(feature_vector[:, 1].max())

                feature_vector = sp.coo_matrix((feature_vector[:, 2], (feature_vector[:, 0], feature_vector[:, 1])),
                                               shape=(xmax + 1, ymax + 1))
                feature_vector = feature_vector.tocsr()

                all_feature_matrices.append(feature_vector)
                print("data")

                acc, s_1, s_2 = linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=3, all_std=True)
                print(acc, s_1, s_2)









if __name__ == "__main__":
    main()
