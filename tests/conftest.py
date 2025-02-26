import lmfit
import numpy as np
import pytest
import xarray as xr


def _exp_decay(t, n0, tau=1):
    return n0 * np.exp(-t / tau)


@pytest.fixture
def exp_decay_model():
    return lmfit.Model(_exp_decay)


@pytest.fixture
def fit_test_darr():
    t = np.arange(0, 5, 0.5)
    da = xr.DataArray(
        np.stack([_exp_decay(t, 3, 3), _exp_decay(t, 5, 4), np.nan * t], axis=-1),
        dims=("t", "x"),
        coords={"t": t, "x": [0, 1, 2]},
    )
    da[0, 0] = np.nan
    return da


@pytest.fixture
def fit_expected_darr():
    return xr.DataArray(
        [[3, 3], [5, 4], [np.nan, np.nan]],
        dims=("x", "param"),
        coords={"x": [0, 1, 2], "param": ["n0", "tau"]},
    )
