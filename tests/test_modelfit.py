import contextlib

import lmfit
import numpy as np
import pytest
import xarray as xr

import xarray_lmfit  # noqa: F401


def power(t, a):
    return np.power(t, a)


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_da_modelfit(
    use_dask: bool,
    exp_decay_model: lmfit.Model,
    fit_test_darr: xr.DataArray,
    fit_expected_darr: xr.DataArray,
) -> None:
    # Tests are adapted from xarray's curvefit tests
    if use_dask:
        fit_test_darr = fit_test_darr.chunk({"x": 1})

    # Params as dictionary
    fit = fit_test_darr.xlm.modelfit(
        coords=[fit_test_darr.t],
        model=exp_decay_model,
        params={"n0": 4, "tau": {"min": 2, "max": 6}},
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)

    # Params as lmfit.Parameters
    fit = fit_test_darr.xlm.modelfit(
        coords=[fit_test_darr.t],
        model=exp_decay_model,
        params=lmfit.create_params(n0=4, tau={"min": 2, "max": 6}),
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)

    if use_dask:
        fit_test_darr = fit_test_darr.compute()

    # Test 0dim output
    fit = fit_test_darr.xlm.modelfit(
        coords="t",
        model=lmfit.Model(power),
        reduce_dims="x",
        params={"a": {"value": 0.3, "vary": True}},
    )

    assert "a" in fit.param
    assert fit.modelfit_results.dims == ()


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "serial"])
def test_ds_modelfit(
    use_dask: bool,
    parallel: bool,
    exp_decay_model: lmfit.Model,
    fit_test_darr: xr.DataArray,
    fit_expected_darr: xr.DataArray,
) -> None:
    warn_ctx = (
        pytest.warns(
            UserWarning,
            match="The input Dataset is chunked. "
            "Parallel fitting will not offer any performance benefits.",
        )
        if (use_dask and parallel)
        else contextlib.nullcontext()
    )
    fit_test_ds = xr.Dataset({"test0": fit_test_darr, "test1": fit_test_darr})

    # Tests are adapted from xarray's curvefit tests
    if use_dask:
        fit_test_ds = fit_test_ds.chunk({"x": 1})

    parallel_kw = {} if not parallel else {"n_jobs": 1}

    # Params as dictionary
    with warn_ctx:
        fit = fit_test_ds.xlm.modelfit(
            coords=[fit_test_ds.t],
            model=exp_decay_model,
            params={"n0": 4, "tau": {"min": 2, "max": 6}},
            parallel=parallel,
            parallel_kw=parallel_kw,
        )
    np.testing.assert_allclose(
        fit.test0_modelfit_coefficients, fit_expected_darr, rtol=1e-3
    )
    np.testing.assert_allclose(
        fit.test1_modelfit_coefficients, fit_expected_darr, rtol=1e-3
    )

    # Params as lmfit.Parameters
    with warn_ctx:
        fit = fit_test_ds.xlm.modelfit(
            coords=[fit_test_ds.t],
            model=exp_decay_model,
            params=lmfit.create_params(n0=4, tau={"min": 2, "max": 6}),
            parallel=parallel,
            parallel_kw=parallel_kw,
        )
    np.testing.assert_allclose(
        fit.test0_modelfit_coefficients, fit_expected_darr, rtol=1e-3
    )
    np.testing.assert_allclose(
        fit.test1_modelfit_coefficients, fit_expected_darr, rtol=1e-3
    )

    if use_dask:
        fit_test_ds = fit_test_ds.compute()

    # Test 0dim output
    fit = fit_test_ds.xlm.modelfit(
        coords="t",
        model=lmfit.Model(power),
        reduce_dims="x",
        params={"a": {"value": 0.3, "vary": True}},
        parallel=parallel,
        parallel_kw=parallel_kw,
    )

    assert "a" in fit.param
    assert fit.test0_modelfit_results.dims == ()
    assert fit.test1_modelfit_results.dims == ()


@pytest.mark.parametrize("use_dask", [True, False])
def test_modelfit_params(use_dask: bool) -> None:
    def sine(t, a, f, p):
        return a * np.sin(2 * np.pi * (f * t + p))

    t = np.arange(0, 2, 0.02)
    da = xr.DataArray(
        np.stack([sine(t, 1.0, 2, 0), sine(t, 1.0, 2, 0)]), coords={"x": [0, 1], "t": t}
    )

    expected = xr.DataArray(
        [[1, 2, 0], [-1, 2, 0.5]], coords={"x": [0, 1], "param": ["a", "f", "p"]}
    )

    # Different initial guesses for different values of x
    a_guess = [1.0, -1.0]
    p_guess = [0.0, 0.5]

    if use_dask:
        da = da.chunk({"x": 1})

    # params as DataArray of JSON strings
    params = []
    for a, p, f in zip(
        a_guess, p_guess, np.full_like(da.x, 2, dtype=float), strict=True
    ):
        params.append(lmfit.create_params(a=a, p=p, f=f).dumps())
    params = xr.DataArray(params, coords=[da.x])
    fit = da.xlm.modelfit(coords=[da.t], model=lmfit.Model(sine), params=params)
    np.testing.assert_allclose(fit.modelfit_coefficients, expected)

    # params as mixed dictionary
    fit = da.xlm.modelfit(
        coords=[da.t],
        model=lmfit.Model(sine),
        params={
            "a": xr.DataArray(a_guess, coords=[da.x]),
            "p": xr.DataArray(p_guess, coords=[da.x]),
            "f": 2.0,
        },
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected)

    def sine(t, a, f, p):
        return a * np.sin(2 * np.pi * (f * t + p))

    t = np.arange(0, 2, 0.02)
    da = xr.DataArray(
        np.stack([sine(t, 1.0, 2, 0), sine(t, 1.0, 2, 0)]), coords={"x": [0, 1], "t": t}
    )

    # Fit a sine with different bounds: positive amplitude should result in a fit with
    # phase 0 and negative amplitude should result in phase 0.5 * 2pi.

    expected = xr.DataArray(
        [[1, 2, 0], [-1, 2, 0.5]], coords={"x": [0, 1], "param": ["a", "f", "p"]}
    )

    if use_dask:
        da = da.chunk({"x": 1})

    # params as DataArray of JSON strings
    fit = da.xlm.modelfit(
        coords=[da.t],
        model=lmfit.Model(sine),
        params=xr.DataArray(
            [
                lmfit.create_params(**param_dict).dumps()
                for param_dict in (
                    {"f": 2, "p": 0.25, "a": {"value": 1, "min": 0, "max": 2}},
                    {"f": 2, "p": 0.25, "a": {"value": -1, "min": -2, "max": 0}},
                )
            ],
            coords=[da.x],
        ),
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected, atol=1e-8)

    # params as mixed dictionary
    fit = da.xlm.modelfit(
        coords=[da.t],
        model=lmfit.Model(sine),
        params={
            "f": {"value": 2},
            "p": 0.25,
            "a": {
                "value": xr.DataArray([1, -1], coords=[da.x]),
                "min": xr.DataArray([0, -2], coords=[da.x]),
                "max": xr.DataArray([2, 0], coords=[da.x]),
            },
        },
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected, atol=1e-8)
