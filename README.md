<img src="https://github.com/nymath/torchqtm/blob/main/resources/fig/logo.png" align="right" width="196" />

# torchquantum

TorchQuantum is a backtesting framework that""" integrates
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
git clone git@github.com:nymath/torchqtm.git
cd ./torchqtm
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
wget https://github.com/nymath/torchqtm/releases/download/V0.1/stocks_f64.pkl.zip
unzip stocks_f64.pkl.zip
rm stocks_f64.pkl.zip
cd ../
cd ../
git checkout dev
```
As for the backtesting dataset, we use the bundle provided by [ricequant](https://www.ricequant.com/welcome/).
We have wrapped the code into Makefile, you can just run the following command to download the bundle.
```shell
make rqalpha_download_bundle
```


for windows:
We highly recommend you to use WSL2 to run torchquantum.

## Examples

### alpha mining
You can easily create an alpha through torchquantum!

```python
import torchqtm.op as op
import torchqtm.op.functional as F


class NeutralizePE(op.Fundamental):
    def __init__(self, env):
        super().__init__(env)
        self.lag = op.Parameter(5, requires_optim=False, feasible_region=None)

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

### backtesting
Here we create a buy and hold strategy for illustration.

```python
from torchqtm.edbt.algorithm import TradingAlgorithm
from torchqtm.assets import Equity

class BuyAndHold(TradingAlgorithm):
    def initialize(self):
        self.safe_set_attr("s0", Equity("000001.XSHE"))
        self.safe_set_attr("count", 0)

    def before_trading_start(self):
        pass

    def handle_data(self):
        if self.count == 0:
            self.order(self.s0, 10000)
        self.count += 1

    def analyze(self):
        pass
```

## Features

- High-speed backtesting framework (most of the operators are implemented through cython)
- A revised gplearn library that is compatible with Alpha mining.
- CNN and other state of the art models for mining alphas.
- Event Driven backtesting framework is available.

## Contribution

For more information, we refer to [Documentation](https://nymath.github.io/torchqtm/navigate).

## Join us

If you are interested in quantitative finance and are committed to devoting
your life to alpha mining, you can contact me through WeChat at Ny_math.

## References

[quantopian/alphalens](https://github1s.com/quantopian/alphalens/blob/HEAD/alphalens/performance.py)

[quantopian/zipline](https://github1s.com/quantopian/zipline/blob/HEAD/zipline/performance.py)

