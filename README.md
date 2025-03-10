# xarray-lmfit

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/xarray-lmfit?logo=python&logoColor=white)](https://pypi.org/project/xarray-lmfit/)
[![PyPi](https://img.shields.io/pypi/v/xarray-lmfit?logo=pypi&logoColor=white)](https://pypi.org/project/xarray-lmfit/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/xarray-lmfit?logo=condaforge&logoColor=white)](https://anaconda.org/conda-forge/xarray-lmfit)
[![Workflow Status](https://img.shields.io/github/actions/workflow/status/kmnhan/xarray-lmfit/ci.yml?logo=github&label=tests)](https://github.com/kmnhan/xarray-lmfit/actions/workflows/ci.yml)
[![Documentation Status](https://img.shields.io/readthedocs/xarray-lmfit?logo=readthedocs&logoColor=white)](https://xarray-lmfit.readthedocs.io/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/kmnhan/xarray-lmfit/main.svg)](https://results.pre-commit.ci/latest/github/kmnhan/xarray-lmfit/main)
[![codecov](https://codecov.io/gh/kmnhan/xarray-lmfit/graph/badge.svg?token=B16DX6OZ0Q)](https://codecov.io/gh/kmnhan/xarray-lmfit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

xarray-lmfit is a Python package that bridges the power of [xarray](http://xarray.pydata.org) for handling multi-dimensional labeled arrays with the flexible fitting capabilities of [lmfit](https://lmfit.github.io/lmfit-py/).

With xarray-lmfit, [lmfit models](https://lmfit.github.io/lmfit-py/model.html) can be fit to xarray Datasets and DataArrays, automatically propagating across multiple dimensions. The fit results are stored as xarray Datasets, retaining the original coordinates and dimensions of the input data.

Disclaimer: Please note that this package is independent and not affiliated with the xarray or lmfit projects. If you encounter any issues, please report them on the [xarray-lmfit issue tracker](https://github.com/kmnhan/xarray-lmfit/issues).

## Installation

Install via pip:

```bash
pip install xarray-lmfit
```

Install via conda:

```bash
conda install -c conda-forge xarray-lmfit
```

## Usage

Below is a basic example to demonstrate how to use xarray-lmfit:

```python
import xarray as xr
import numpy as np
from lmfit.models import GaussianModel

import xarray_lmfit as xlm

# Create an example dataset
x = np.linspace(0, 10, 100)
y = 3.0 * np.exp(-((x - 5) ** 2) / (2 * 1.0**2)) + np.random.normal(0, 0.1, x.size)
data = xr.DataArray(y, dims="x", coords={"x": x})

# Define the model to be used
model = GaussianModel()

# Perform the fit
result = data.xlm.modelfit("x", model=model)
```

## Documentation

For more detailed documentation and examples, please visit the [documentation](https://xarray-lmfit.readthedocs.io).

## License

This project is licensed under the [GPL-3.0 License](LICENSE).
