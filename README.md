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

If you are not downloading the dataset, then you should

```shell
cd ./examples
mkdir largedata
cd ./largedata
wget https://github.com/nymath/torchquantum/releases/download/V0.1/Stocks.pkl.zip
unzip Stocks.pkl.zip
rm Stocks.pkl.zip
cd ../
cd ../
```

## Example

You can easily create an alpha through torchquantum!

```python
import torchqtm.op as op
import torchqtm.op.functional as F
class NeutralizePE(op.Fundamental):
    def __init__(self, env):
        super().__init__(env)
        self.lag = op.Parameter(5, required_optim=False, feasible_region=None)

    def forward(self):
        self.data = F.divide(1, self.env.PE)
        self.data = F.winsorize(self.data, 'std', 4)
        self.data = F.normalize(self.data)
        self.data = F.group_neutralize(self.data, self.env.Sector)
        self.data = F.regression_neut(self.data, self.env.MktVal)
        self.data = F.ts_mean(self.data, self.lag)
        return self.data
```
- `F` is library that contains the operators defined by WorldQuant.
- `op.Fundamental` implies the NeutralizePE belongs to fundamental alpha.
- `self.lag` is the parameter of rolling mean, which can be optimized through grid search.




## Features

- High-speed backtesting framework (most of the operators are implemented through cython)
- A revised gplearn library that is compatible with Alpha mining.
- CNN and other state of the art models for mining alphas.
- Event Driven backtesting framework is available.

## Contribution

For more information, we refer to [Documentation](https://nymath.github.io/torchquantum/navigate).


## Join us

If you are interested in quantitative finance and are committed to devoting 
your life to alpha mining, you can contact me through WeChat at Ny_math.

