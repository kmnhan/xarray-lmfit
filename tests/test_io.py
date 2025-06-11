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


def test_ds_io() -> None:
    # Generate toy data
    x = np.linspace(0, 10, 50)[:, np.newaxis]
    y = np.linspace(0, 2, 3)[np.newaxis, :]
    z = -0.1 * x + y + 3 * np.exp(-((x - 5) ** 2) / (2 * 1**2))

    # Add some noise with fixed seed for reproducibility
    rng = np.random.default_rng(5)
    zerr = np.full_like(z, 0.3)
    z = rng.normal(z, zerr)

    z_ds = xr.DataArray(
        z, dims=("x", "y"), coords={"x": x[:, 0], "y": y[0, :].astype(int)}
    ).to_dataset("y")

    model = lmfit.models.GaussianModel() + lmfit.models.LinearModel()
    result_ds = z_ds.xlm.modelfit(
        "x",
        model=model,
        params=model.make_params(
            slope=-0.1, center=5.0, sigma={"value": 0.1, "min": 0}
        ),
    )
    # Result dataset should contain 0_modelfit_results, 1_modelfit_results,
    # 2_modelfit_results, etc. for each y coordinate.

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        save_fit(result_ds, tmp.name)
        # Use load_fit to load and deserialize the model results.
        loaded_ds = load_fit(tmp.name)
        for i in range(3):
            assert f"{i}_modelfit_results" in loaded_ds.data_vars
            assert isinstance(
                loaded_ds[f"{i}_modelfit_results"].item(), lmfit.model.ModelResult
            )
            assert str(loaded_ds[f"{i}_modelfit_results"].item().model) == str(model)

        xr.testing.assert_identical(
            loaded_ds.drop_vars([f"{i}_modelfit_results" for i in range(3)]),
            result_ds.drop_vars([f"{i}_modelfit_results" for i in range(3)]),
        )
