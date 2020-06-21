# sparsewl
Code for "Weisfeiler and Leman go sparse: Towards scalable higher-order graph embeddings"

## Requirements
- Python 3.8
- eigen3
- numpy
- pandas
- scipy
- sklearn
- torch 1.5
- torch-geometric
- pybind11
- libsvm

All results in the paper and the appendix can be reproduced by the following the steps below. 

## Reproducing the kernel experiments (Tables 1, 3b, Table 5, Table 5, Table 9, Table 10)
- `cd kernels`
- Download datasets from `www.graphlearning.io`,  and place the unzipped folders into `kernels/datasets`
- Run `g++ main.cpp src/*cpp -std=c++11 -o local -O2`
- Run `./local` (running times will be outputted on the screen, too)

### Setting up the kernel SVM
- `cd svm/SVM/src`
- Adjust the path to libsvm in line 129 of `svm/SVM/src/cli/AccuracyTest.java`
- Run `javac Main`
- Run `java Main`

### Setting up the linear SVM (for larger datasets)
- `cd linear_svm`
- Run `python linear_svm.py`

## Reproducing the neural baselines (for kernel experiments, Table 1, Table 5, Table 6)
- `cd neural baselines`
- Run `python main_gnn.py`

## Reproducing the neural higher-order results (Table 2b, Figure 2abc, 3b, Table 8)
You first need to build the Python package:
- `cd neural_higher_order/preprocessing`
- You might need to adjust the path to `pybind` in `preprocessing.cpp`, then run 
    - MaxOS: c++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix` 
    - Linux: c++ -O3  -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix`

- Run the Python scripts in `Alchemy`, `QM9`, `ZINC` to reproduce the scores and running times
    - For example: `cd Alchemy`, `python local_2_FULL.py` to reproduce the scores for the \delta-2-LGNN on the  `Alchemy` dataset

