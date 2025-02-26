import tempfile

import lmfit
import numpy as np
import xarray as xr

from xarray_lmfit import load_fit, save_fit


def test_darr_io() -> None:
    # Generate toy data
    x = np.linspace(0, 10, 50)
    y = -0.1 * x + 2 + 3 * np.exp(-((x - 5) ** 2) / (2 * 1**2))

    # Add some noise with fixed seed for reproducibility
    rng = np.random.default_rng(5)
    yerr = np.full_like(x, 0.3)
    y = rng.normal(y, yerr)

    y_arr = xr.DataArray(y, dims=("x",), coords={"x": x})

    model = lmfit.models.GaussianModel() + lmfit.models.LinearModel()
    result_ds = y_arr.xlm.modelfit(
        "x",
        model=model,
        params=model.make_params(
            slope=-0.1, center=5.0, sigma={"value": 0.1, "min": 0}
        ),
    )

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        save_fit(result_ds, tmp.name)
        # Use load_fit to load and deserialize the model results.
        loaded_ds = load_fit(tmp.name)
        assert isinstance(loaded_ds["modelfit_results"].item(), lmfit.model.ModelResult)
        assert str(loaded_ds["modelfit_results"].item().model) == str(model)

        xr.testing.assert_identical(
            loaded_ds.drop_vars("modelfit_results"),
            result_ds.drop_vars("modelfit_results"),
        )
