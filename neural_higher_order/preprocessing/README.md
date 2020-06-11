# localwl_dev

compile 
''
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  kernel_baselines.cpp src/*cpp -o ../kernel_baselines`python3-config --extension-suffix`
''

c++ -O3  -shared -std=c++11 -fPIC `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../pre`python3-config --extension-suffix`






 c++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix`
