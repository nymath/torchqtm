<img src="https://github.com/nymath/torchquantum/blob/main/src/fig/logo.png" align="right" width="196" />

# torchquantum

TorchQuantum is a backtesting framework that integrates
the structure of PyTorch and WorldQuant's Operator for
efficient quantitative financial analysis.

## Contents

- [Installation](#installation)
- [Features](#features)
- [Contribution](#contribution)


## Installation
for Unix:
```shell
cd /path/to/your/directory
git clone git@github.com:nymath/torchquantum.git
cd ./torchquantum
```
Before running examples, you should compile the cython code.
```shell
python setup.py build_ext --inplace
```
Now you can run examples
```shell
python ./examples/main.py
```


## Features

- High-speed backtesting framework.
- A revised gplearn library that is compatible with Alpha mining.
- CNN and other state of the art models for mining alphas.
- Event Driven backtesting framework will be available.

## Contribution

For more information, we refer to [Documentation](https://nymath.github.io/torchquantum/navigate).



