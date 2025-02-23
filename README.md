# xarray-lmfit

[![Documentation Status](https://img.shields.io/readthedocs/xarray-lmfit?logo=readthedocs&logoColor=white)](https://xarray-lmfit.readthedocs.io/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/kmnhan/xarray-lmfit/main.svg)](https://results.pre-commit.ci/latest/github/kmnhan/xarray-lmfit/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

xarray-lmfit is a Python package that bridges the power of [xarray](http://xarray.pydata.org) for handling multi-dimensional labeled arrays with the flexible fitting capabilities of [lmfit](https://lmfit.github.io/lmfit-py/). It is designed to simplify fitting tasks with multi-dimensional data while keeping your data organization intact.

Disclaimer: This package is not affiliated with the xarray or lmfit projects. Any issue that arises from using this package should be reported to the [xarray-lmfit issue tracker](https://github.com/kmnhan/xarray-lmfit/issues), not to the xarray or lmfit projects.

## Installation

Install via pip:

```bash
pip install xarray-lmfit
```

Or build from source by cloning the repository:

```bash
git clone https://github.com/kmnhan/xarray-lmfit.git
cd xarray-lmfit
pip install .
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
