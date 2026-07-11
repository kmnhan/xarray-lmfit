import re

import lmfit
import numpy as np
import pytest
import xarray as xr

import xarray_lmfit  # noqa: F401


def lorentzian(x, amplitude, center, sigma):
    return (amplitude / (1 + ((1.0 * x - center) / max(sigma, 1e-15)) ** 2)) / max(
        np.pi * sigma, 1e-15
    )


def power(t, a):
    return np.power(t, a)


def linear(x, slope, intercept):
    return slope * x + intercept


@pytest.mark.parametrize("progress", [True, False], ids=["tqdm", "no_tqdm"])
@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_da_modelfit(
    use_dask: bool,
    progress: bool,
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
        progress=progress,
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)

    # Params as lmfit.Parameters
    fit = fit_test_darr.xlm.modelfit(
        coords=[fit_test_darr.t],
        model=exp_decay_model,
        params=lmfit.create_params(n0=4, tau={"min": 2, "max": 6}),
        progress=progress,
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)

    # Test weights input as DataArray
    fit = fit_test_darr.xlm.modelfit(
        coords="t",
        model=exp_decay_model,
        params={"n0": 4, "tau": {"min": 2, "max": 6}},
        weights=1.0 / np.sqrt(fit_test_darr),
        progress=progress,
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)

    # Test weights input as DataArray (less broadcasted)
    fit = fit_test_darr.xlm.modelfit(
        coords="t",
        model=exp_decay_model,
        params={"n0": 4, "tau": {"min": 2, "max": 6}},
        weights=np.sqrt(fit_test_darr.t),
        progress=progress,
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)

    # Test weights input as ndarray
    fit = fit_test_darr.xlm.modelfit(
        coords="t",
        model=exp_decay_model,
        params={"n0": 4, "tau": {"min": 2, "max": 6}},
        weights=np.sqrt(fit_test_darr.t.values),
        progress=progress,
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, fit_expected_darr, rtol=1e-3)

    # Test weights input as scalar
    fit = fit_test_darr.xlm.modelfit(
        coords="t",
        model=exp_decay_model,
        params={"n0": 4, "tau": {"min": 2, "max": 6}},
        weights=0.1,
        progress=progress,
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
        progress=progress,
    )

    assert "a" in fit.param
    assert fit.modelfit_results.dims == ()


def test_da_modelfit_skipna_false_best_fit() -> None:
    x = np.arange(5, dtype=float)
    da = xr.DataArray(linear(x, 2.0, 1.0), dims="x", coords={"x": x})

    fit = da.xlm.modelfit(
        coords="x",
        model=lmfit.Model(linear),
        params={"slope": 1.0, "intercept": 0.0},
        skipna=False,
    )

    assert np.isfinite(fit.modelfit_best_fit).all()
    np.testing.assert_allclose(fit.modelfit_best_fit, da)


@pytest.mark.parametrize("skipna", [True, False], ids=["skipna", "keep-nan"])
@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no-dask"])
def test_da_modelfit_complete_multidimensional_core(
    skipna: bool, use_dask: bool
) -> None:
    full_coord = np.arange(12.0).reshape(2, 6)
    coord = full_coord[:, ::2]
    da = xr.DataArray(linear(full_coord, 2.0, 1.0), dims=("row", "column")).isel(
        column=slice(None, None, 2)
    )
    if use_dask:
        da = da.chunk({"row": -1, "column": -1})

    fit = da.xlm.modelfit(
        coords=xr.DataArray(coord, dims=da.dims),
        reduce_dims=da.dims,
        model=lmfit.Model(linear),
        params={"slope": 1.0, "intercept": 0.0},
        weights=np.ones_like(coord),
        skipna=skipna,
        output_result=False,
    ).compute()

    np.testing.assert_allclose(fit.modelfit_coefficients, [2.0, 1.0])
    np.testing.assert_allclose(fit.modelfit_best_fit, da)


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no-dask"])
def test_da_modelfit_single_coord_mask_aligns_weights(use_dask: bool) -> None:
    coord = np.arange(7.0)
    values = linear(coord, 2.0, 1.0)
    coord[1] = np.nan
    values[4] = np.nan
    da = xr.DataArray(values, dims="point")
    if use_dask:
        da = da.chunk({"point": -1})

    fit = da.xlm.modelfit(
        coords=xr.DataArray(coord, dims="point"),
        reduce_dims="point",
        model=lmfit.Model(linear),
        params={"slope": 1.0, "intercept": 0.0},
        weights=np.arange(1.0, 8.0),
        output_result=False,
    ).compute()

    np.testing.assert_allclose(fit.modelfit_coefficients, [2.0, 1.0])
    assert fit.modelfit_stats.sel(fit_stat="ndata") == 5
    assert np.isnan(fit.modelfit_best_fit[[1, 4]]).all()
    np.testing.assert_allclose(
        fit.modelfit_best_fit[[0, 2, 3, 5, 6]],
        linear(np.array([0.0, 2.0, 3.0, 5.0, 6.0]), 2.0, 1.0),
    )


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_da_modelfit_integer_best_fit(use_dask: bool) -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.array([0, 1, 5, 9, 13], dtype=np.int64),
        dims="x",
        coords={"x": x},
    )
    if use_dask:
        da = da.chunk({"x": -1})

    fit = da.xlm.modelfit(
        coords="x",
        model=lmfit.Model(linear),
        params={"slope": 1.0, "intercept": 0.0},
        output_result=False,
    ).compute()

    expected = linear(x, *fit.modelfit_coefficients.values)
    assert fit.modelfit_best_fit.dtype == np.float64
    np.testing.assert_allclose(fit.modelfit_best_fit, expected)


@pytest.mark.parametrize(
    "weights",
    [
        np.arange(1.0, 7.0).reshape(2, 3),
        np.arange(1.0, 7.0).reshape(2, 3).tolist(),
        np.array(0.1),
    ],
    ids=["ndarray", "list", "zero-dimensional"],
)
def test_da_modelfit_multidimensional_array_weights(weights) -> None:
    def plane(x, z, slope_x, slope_z, intercept):
        return slope_x * x + slope_z * z + intercept

    x, z = np.meshgrid(np.arange(2.0), np.arange(3.0), indexing="ij")
    values = plane(x, z, 2.0, -0.5, 1.0)
    values[1, 1] = np.nan
    da = xr.DataArray(values, dims=("x", "z"))

    fit = da.xlm.modelfit(
        coords=[xr.DataArray(x, dims=da.dims), xr.DataArray(z, dims=da.dims)],
        reduce_dims=da.dims,
        model=lmfit.Model(plane, independent_vars=["x", "z"]),
        params={"slope_x": 1.0, "slope_z": 0.0, "intercept": 0.0},
        weights=weights,
        output_result=False,
    )

    np.testing.assert_allclose(fit.modelfit_coefficients, [2.0, -0.5, 1.0])


def test_da_modelfit_rejects_wrong_weight_size() -> None:
    x = np.arange(5.0)
    da = xr.DataArray(linear(x, 2.0, 1.0), dims="x", coords={"x": x})

    with pytest.raises(ValueError, match="same size as the data being fit"):
        da.xlm.modelfit(
            "x",
            model=lmfit.Model(linear),
            params={"slope": 1.0, "intercept": 0.0},
            weights=np.ones(3),
        )


def test_da_modelfit_skipna_multiple_coords() -> None:
    def plane(x, z, slope_x, slope_z, intercept):
        return slope_x * x + slope_z * z + intercept

    x = np.arange(6, dtype=float)
    z = np.array([0.0, 2.0, 1.0, 4.0, 3.0, 1.0])
    da = xr.DataArray(plane(x, z, 2.0, -0.5, 1.0), dims="point")

    x_coord = xr.DataArray(x, dims="point")
    z_coord = xr.DataArray(z, dims="point")
    x_coord[2] = np.nan
    z_coord[4] = np.nan

    fit = da.xlm.modelfit(
        coords=[x_coord, z_coord],
        model=lmfit.Model(plane, independent_vars=["x", "z"]),
        reduce_dims="point",
        params={"slope_x": 1.0, "slope_z": 0.0, "intercept": 0.0},
    )

    np.testing.assert_allclose(
        fit.modelfit_coefficients,
        [2.0, -0.5, 1.0],
        rtol=1e-7,
        atol=1e-7,
    )
    assert np.isnan(fit.modelfit_best_fit[[2, 4]]).all()
    np.testing.assert_allclose(
        fit.modelfit_best_fit[[0, 1, 3, 5]],
        da[[0, 1, 3, 5]],
    )


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_da_modelfit_without_model_results(
    use_dask: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dtypes = []
    apply_ufunc = xr.apply_ufunc

    def capture_output_dtypes(*args, **kwargs):
        output_dtypes.append(tuple(kwargs["output_dtypes"]))
        return apply_ufunc(*args, **kwargs)

    monkeypatch.setattr(xr, "apply_ufunc", capture_output_dtypes)

    x = np.arange(5, dtype=float)
    da = xr.DataArray(
        np.stack([linear(x, 2.0, 1.0), np.full_like(x, np.nan)]),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    )
    if use_dask:
        da = da.chunk({"fit": 1})

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params={"slope": 1.0, "intercept": 0.0},
        output_result=False,
    ).compute()

    assert output_dtypes == [(np.float64,) * 5]
    assert "modelfit_results" not in fit
    assert all(variable.dtype != object for variable in fit.data_vars.values())
    np.testing.assert_allclose(fit.modelfit_coefficients[0], [2.0, 1.0])
    assert np.isnan(fit.modelfit_coefficients[1]).all()


def test_modelfit_data_bypasses_fit_graph() -> None:
    fit_calls = 0

    def counted_linear(x, slope, intercept):
        nonlocal fit_calls
        fit_calls += 1
        return linear(x, slope, intercept)

    x = np.arange(5, dtype=float)
    da = xr.DataArray(
        np.stack([linear(x, 2.0, 1.0), linear(x, -1.0, 3.0)]).T.astype(np.int64),
        dims=("x", "fit"),
        coords={"fit": [0, 1], "x": x, "label": ("fit", ["a", "b"])},
        attrs={"description": "input data"},
    ).chunk({"fit": 1, "x": -1})

    with xr.set_options(keep_attrs=False):
        result = da.xlm.modelfit(
            "x",
            model=lmfit.Model(counted_linear),
            params={"slope": 1.0, "intercept": 0.0},
            output_result=False,
        )

    actual = result.modelfit_data.compute(scheduler="single-threaded")
    expected = da.transpose("fit", "x").astype(np.float64).compute()
    expected.name = "modelfit_data"
    expected.attrs = {}

    xr.testing.assert_identical(actual, expected)
    assert fit_calls == 0

    result.modelfit_coefficients.compute(scheduler="single-threaded")
    assert fit_calls > 0

    with xr.set_options(keep_attrs=True):
        result_with_attrs = da.xlm.modelfit(
            "x",
            model=lmfit.Model(counted_linear),
            params={"slope": 1.0, "intercept": 0.0},
            output_result=False,
        )

    assert result_with_attrs.modelfit_data.attrs == da.attrs


def test_modelfit_builds_static_parameter_template_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_params_calls = 0
    x = np.arange(5.0)
    da = xr.DataArray(
        np.stack([linear(x, slope, 1.0) for slope in range(1, 6)]),
        dims=("fit", "x"),
        coords={"fit": np.arange(5), "x": x},
    )
    model = lmfit.Model(linear)
    model.set_param_hint("slope", min=0.0)
    original_make_params = model.make_params

    def counted_make_params(*args, **kwargs):
        nonlocal make_params_calls
        make_params_calls += 1
        return original_make_params(*args, **kwargs)

    monkeypatch.setattr(model, "make_params", counted_make_params)

    fit = da.xlm.modelfit(
        "x",
        model=model,
        params={"slope": 1.0, "intercept": 0.0},
        output_result=False,
    )

    assert make_params_calls == 1
    np.testing.assert_allclose(
        fit.modelfit_coefficients.sel(param="slope"), np.arange(1.0, 6.0)
    )
    assert make_params_calls == 1


def test_failed_results_do_not_share_static_parameter_template() -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.full((2, x.size), np.nan),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    )
    supplied = lmfit.create_params(slope=1.0, intercept=0.0)
    assert supplied["slope"]._expr_eval is supplied._asteval

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params=supplied,
    )
    first, second = fit.modelfit_results.values

    first.params["slope"].value = 10.0
    assert second.params["slope"].value == 1.0
    assert supplied["slope"].value == 1.0
    assert supplied["slope"]._expr_eval is supplied._asteval


@pytest.mark.parametrize("progress", [True, False], ids=["tqdm", "no_tqdm"])
@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_ds_modelfit(
    use_dask: bool,
    progress: bool,
    exp_decay_model: lmfit.Model,
    fit_test_darr: xr.DataArray,
    fit_expected_darr: xr.DataArray,
) -> None:
    fit_test_ds = xr.Dataset({"test0": fit_test_darr, "test1": fit_test_darr})

    # Tests are adapted from xarray's curvefit tests
    if use_dask:
        fit_test_ds = fit_test_ds.chunk({"x": 1})

    # Params as dictionary
    fit = fit_test_ds.xlm.modelfit(
        coords=[fit_test_ds.t],
        model=exp_decay_model,
        params={"n0": 4, "tau": {"min": 2, "max": 6}},
        progress=progress,
    )
    np.testing.assert_allclose(
        fit.test0_modelfit_coefficients, fit_expected_darr, rtol=1e-3
    )
    np.testing.assert_allclose(
        fit.test1_modelfit_coefficients, fit_expected_darr, rtol=1e-3
    )

    # Params as lmfit.Parameters
    fit = fit_test_ds.xlm.modelfit(
        coords=[fit_test_ds.t],
        model=exp_decay_model,
        params=lmfit.create_params(n0=4, tau={"min": 2, "max": 6}),
        progress=progress,
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
        progress=progress,
    )

    assert "a" in fit.param
    assert fit.test0_modelfit_results.dims == ()
    assert fit.test1_modelfit_results.dims == ()


@pytest.mark.parametrize("progress", [True, False], ids=["tqdm", "no_tqdm"])
@pytest.mark.parametrize("use_dask", [True, False])
def test_modelfit_params(use_dask: bool, progress: bool) -> None:
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

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Parameter 'z' was not found in the fit results. "
            "Check the model and parameter names."
        ),
    ):
        da.xlm.modelfit(
            coords=[da.t],
            model=lmfit.Model(sine),
            params=params,
            param_names=["a", "f", "z"],
            progress=progress,
        ).compute()

    # params as mixed dictionary
    fit = da.xlm.modelfit(
        coords=[da.t],
        model=lmfit.Model(sine),
        params={
            "a": xr.DataArray(a_guess, coords=[da.x]),
            "p": xr.DataArray(p_guess, coords=[da.x]),
            "f": 2.0,
        },
        progress=progress,
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
        progress=progress,
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
        progress=progress,
    )
    np.testing.assert_allclose(fit.modelfit_coefficients, expected, atol=1e-8)


@pytest.mark.parametrize("use_dask", [True, False], ids=["dask", "no_dask"])
def test_modelfit_dataarray_dict_params(use_dask: bool) -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.stack([linear(x, 2.0, 1.0), linear(x, -1.0, 3.0)]),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    )
    if use_dask:
        da = da.chunk({"fit": 1})

    param_specs = np.empty(2, dtype=object)
    param_specs[:] = [
        {"slope": {"value": 0.0, "vary": False}, "intercept": 0.0},
        {"slope": {"value": 0.0, "vary": False}, "intercept": 0.0},
    ]
    params = xr.DataArray(param_specs, dims="fit", coords={"fit": da.fit})

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params=params,
        output_result=False,
    ).compute()

    np.testing.assert_allclose(fit.modelfit_coefficients, [[0.0, 5.0], [0.0, 1.0]])
    assert "is_init_value" not in param_specs[0]["slope"]


def test_modelfit_lazy_broadcast_params() -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.stack([linear(x, 2.0, 1.0), linear(x, -1.0, 3.0)]),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    )
    slope_guess = xr.DataArray([0.0, 0.0], dims="fit", coords={"fit": da.fit}).chunk(
        {"fit": 1}
    )

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params={
            "slope": {"value": slope_guess, "vary": False, "expr": None},
            "intercept": np.float32(0.0),
        },
        output_result=False,
    )

    assert fit.modelfit_coefficients.chunks is not None
    np.testing.assert_allclose(
        fit.compute().modelfit_coefficients,
        [[0.0, 5.0], [0.0, 1.0]],
    )


def test_modelfit_broadcast_parameter_chunks_follow_data() -> None:
    x = np.arange(5.0)
    slopes = np.arange(1.0, 7.0)
    da = xr.DataArray(
        np.stack([linear(x, slope, 1.0) for slope in slopes]),
        dims=("fit", "x"),
        coords={"fit": np.arange(slopes.size), "x": x},
    ).chunk({"fit": 3, "x": -1})
    slope_guess = xr.DataArray(
        np.zeros(slopes.size), dims="fit", coords={"fit": da.fit}
    ).chunk({"fit": 1})

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params={"slope": slope_guess, "intercept": 0.0},
        output_result=False,
    )

    assert fit.modelfit_coefficients.chunks == ((3, 3), (2,))
    graph = fit.modelfit_coefficients.data.__dask_graph__()
    assert all(
        len(layer) > 1
        for name, layer in graph.layers.items()
        if name.startswith("rechunk-merge")
    )
    np.testing.assert_allclose(
        fit.compute().modelfit_coefficients.sel(param="slope"), slopes
    )


def test_modelfit_object_parameter_chunks_follow_data() -> None:
    x = np.arange(5.0)
    slopes = np.arange(1.0, 7.0)
    da = xr.DataArray(
        np.stack([linear(x, slope, 1.0) for slope in slopes]),
        dims=("fit", "x"),
        coords={"fit": np.arange(slopes.size), "x": x},
    ).chunk({"fit": 3, "x": -1})
    specs = np.empty(slopes.size, dtype=object)
    specs[:] = [{"slope": 0.0, "intercept": 0.0}] * slopes.size
    params = xr.DataArray(specs, dims="fit", coords={"fit": da.fit}).chunk({"fit": 1})

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params=params,
        output_result=False,
    )

    assert fit.modelfit_coefficients.chunks == ((3, 3), (2,))
    np.testing.assert_allclose(
        fit.compute().modelfit_coefficients.sel(param="slope"), slopes
    )


def test_modelfit_broadcast_params_preserve_outer_alignment() -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.stack([linear(x, 2.0, 10.0)] * 3),
        dims=("fit", "x"),
        coords={"fit": [0, 1, 2], "x": x},
    )
    model = lmfit.Model(linear)
    model.set_param_hint("slope", value=5.0)
    model.set_param_hint("intercept", value=10.0)
    slope = xr.DataArray([2.0, 4.0], dims="fit", coords={"fit": [0, 1]})
    intercept = xr.DataArray([1.0, 3.0], dims="fit", coords={"fit": [1, 2]})

    fit = da.xlm.modelfit(
        "x",
        model=model,
        params={"slope": slope, "intercept": intercept},
    )

    initial_values = np.array(
        [
            [result.init_params["slope"].value, result.init_params["intercept"].value]
            for result in fit.modelfit_results.values
        ]
    )
    np.testing.assert_allclose(initial_values, [[2.0, 10.0], [4.0, 1.0], [5.0, 3.0]])


@pytest.mark.parametrize("guess", [False, True])
def test_modelfit_broadcast_params_omit_static_nan(guess: bool) -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.stack([linear(x, 2.0, 10.0)] * 2),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    )
    model = lmfit.Model(linear)
    model.set_param_hint("intercept", value=10.0)
    slope = xr.DataArray([2.0, 2.0], dims="fit", coords={"fit": da.fit})

    if guess:
        with pytest.warns(UserWarning, match="`guess` is not implemented"):
            fit = da.xlm.modelfit(
                "x",
                model=model,
                params={"slope": slope, "intercept": np.nan},
                guess=True,
            )
    else:
        fit = da.xlm.modelfit(
            "x",
            model=model,
            params={"slope": slope, "intercept": np.nan},
        )

    assert all(
        result.init_params["intercept"].value == 10.0
        for result in fit.modelfit_results.values
    )


def test_modelfit_broadcast_partial_attribute_replaces_model_hint() -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.full((2, x.size), np.nan),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    )
    model = lmfit.Model(linear)
    model.set_param_hint("slope", value=7.0, min=1.0, max=9.0, vary=False)
    minimum = xr.DataArray([3.0, np.nan], dims="fit", coords={"fit": da.fit})

    fit = da.xlm.modelfit(
        "x",
        model=model,
        params={"slope": {"min": minimum}},
    )

    replaced = fit.modelfit_results.values[0].init_params["slope"]
    preserved = fit.modelfit_results.values[1].init_params["slope"]
    assert (replaced.value, replaced.min, replaced.max, replaced.vary) == (
        3.0,
        3.0,
        np.inf,
        True,
    )
    assert (preserved.value, preserved.min, preserved.max, preserved.vary) == (
        7.0,
        1.0,
        9.0,
        False,
    )


def test_modelfit_broadcast_params_preserve_auxiliary_order() -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.full((2, x.size), np.nan),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    )
    dynamic_aux = xr.DataArray([1.0, 2.0], dims="fit", coords={"fit": da.fit})

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params={
            "dynamic_aux": {"value": dynamic_aux, "vary": False},
            "static_aux": {"value": 3.0, "vary": False},
        },
    )

    for result in fit.modelfit_results.values:
        assert list(result.init_params) == [
            "slope",
            "intercept",
            "dynamic_aux",
            "static_aux",
        ]


def test_modelfit_broadcast_auxiliary_expression_parameter() -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.stack([linear(x, 2.0, 3.0), linear(x, 2.0, 4.0)]),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    ).chunk({"fit": 1, "x": -1})
    delta = xr.DataArray([1.0, 2.0], dims="fit", coords={"fit": da.fit}).chunk(
        {"fit": 1}
    )
    params = {
        "slope": 2.0,
        "delta": delta,
        "intercept": {"expr": "slope + delta"},
    }

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params=params,
        param_names=["slope", "intercept", "delta"],
        output_result=False,
    ).compute()

    np.testing.assert_allclose(
        fit.modelfit_coefficients, [[2.0, 3.0, 1.0], [2.0, 4.0, 2.0]]
    )
    assert "is_init_value" not in params["intercept"]


def test_modelfit_broadcast_params_across_dimensions() -> None:
    x = np.arange(5.0)
    slopes = xr.DataArray([1.0, 2.0], dims="row").chunk({"row": 1})
    intercepts = xr.DataArray([10.0, 20.0, 30.0], dims="column").chunk({"column": 1})
    values = (
        slopes.compute().values[:, np.newaxis, np.newaxis] * x
        + intercepts.compute().values[np.newaxis, :, np.newaxis]
    )
    da = xr.DataArray(
        values,
        dims=("row", "column", "x"),
        coords={"x": x},
    ).chunk({"row": 1, "column": 2, "x": -1})

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params={"slope": slopes, "intercept": intercepts},
        output_result=False,
    ).compute()

    expected_slope, expected_intercept = xr.broadcast(
        slopes.compute(), intercepts.compute()
    )
    np.testing.assert_allclose(
        fit.modelfit_coefficients.sel(param="slope"), expected_slope
    )
    np.testing.assert_allclose(
        fit.modelfit_coefficients.sel(param="intercept"), expected_intercept
    )


def test_modelfit_broadcast_params_override_guesses() -> None:
    x = np.arange(5.0)
    da = xr.DataArray(
        np.stack([linear(x, 2.0, 1.0), linear(x, -1.0, 3.0)]),
        dims=("fit", "x"),
        coords={"fit": [0, 1], "x": x},
    )
    slope = xr.DataArray([0.0, 0.0], dims="fit", coords={"fit": da.fit}).chunk(
        {"fit": 1}
    )

    fit = da.xlm.modelfit(
        "x",
        model=lmfit.models.LinearModel(),
        params={"slope": {"value": slope, "vary": False}},
        guess=True,
        output_result=False,
    ).compute()

    np.testing.assert_allclose(fit.modelfit_coefficients, [[0.0, 5.0], [0.0, 1.0]])


def test_modelfit_dataset_with_different_chunk_layouts() -> None:
    x = np.arange(5.0)
    values = np.stack([linear(x, slope, 1.0) for slope in range(1, 5)])
    ds = xr.Dataset(
        {
            "a": xr.DataArray(values, dims=("fit", "x")).chunk({"fit": 1, "x": -1}),
            "b": xr.DataArray(values, dims=("fit", "x")).chunk({"fit": 2, "x": -1}),
        },
        coords={"fit": np.arange(4), "x": x},
    )
    slope_guess = xr.DataArray(np.zeros(4), dims="fit", coords={"fit": ds.fit}).chunk(
        {"fit": 1}
    )

    fit = ds.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params={"slope": slope_guess, "intercept": 0.0},
        output_result=False,
    )

    assert fit.a_modelfit_coefficients.chunks == ((1, 1, 1, 1), (2,))
    assert fit.b_modelfit_coefficients.chunks == ((2, 2), (2,))
    expected = np.arange(1.0, 5.0)
    computed = fit.compute()
    np.testing.assert_allclose(
        computed.a_modelfit_coefficients.sel(param="slope"), expected
    )
    np.testing.assert_allclose(
        computed.b_modelfit_coefficients.sel(param="slope"), expected
    )


def test_modelfit_parameter_dataset_supports_mixed_object_specs() -> None:
    x = np.arange(5.0)
    slopes = np.arange(1.0, 4.0)
    values = np.stack([linear(x, slope, 1.0) for slope in slopes])
    ds = xr.Dataset(
        {
            "a": xr.DataArray(values, dims=("fit", "x")).chunk({"fit": 1, "x": -1}),
            "b": xr.DataArray(values, dims=("fit", "x")).chunk({"fit": 2, "x": -1}),
        },
        coords={"fit": np.arange(slopes.size), "x": x},
    )
    supplied_a = lmfit.create_params(slope=0.0, intercept=0.0)
    supplied_b = lmfit.create_params(slope=0.0, intercept=0.0)
    dict_spec = {"slope": 0.0, "intercept": 0.0}
    json_spec = lmfit.create_params(slope=0.0, intercept=0.0).dumps()
    a_specs = np.empty(slopes.size, dtype=object)
    b_specs = np.empty(slopes.size, dtype=object)
    for i, spec in enumerate((dict_spec, supplied_a, json_spec)):
        a_specs[i] = spec
    for i, spec in enumerate((json_spec, dict_spec, supplied_b)):
        b_specs[i] = spec
    params = xr.Dataset(
        {
            "a": xr.DataArray(a_specs, dims="fit").chunk({"fit": 2}),
            "b": xr.DataArray(b_specs, dims="fit").chunk({"fit": 1}),
        },
        coords={"fit": ds.fit},
    )

    fit = ds.xlm.modelfit(
        "x",
        model=lmfit.Model(linear),
        params=params,
        output_result=False,
    )

    assert fit.a_modelfit_coefficients.chunks == ((1, 1, 1), (2,))
    assert fit.b_modelfit_coefficients.chunks == ((2, 1), (2,))
    computed = fit.compute()
    np.testing.assert_allclose(
        computed.a_modelfit_coefficients.sel(param="slope"), slopes
    )
    np.testing.assert_allclose(
        computed.b_modelfit_coefficients.sel(param="slope"), slopes
    )
    assert supplied_a["slope"]._expr_eval is supplied_a._asteval
    assert supplied_b["slope"]._expr_eval is supplied_b._asteval


def test_modelfit_does_not_deepcopy_parameter_dataarrays() -> None:
    class UncopyableDataArray(xr.DataArray):
        __slots__ = ()

        def __deepcopy__(self, memo):
            raise AssertionError("parameter DataArrays must not be deep-copied")

    def linear_batch(t, slope, intercept):
        return slope * t + intercept

    t = np.linspace(0, 1, 10)
    da = xr.DataArray(
        np.stack([linear_batch(t, 2.0, 1.0), linear_batch(t, -0.5, 3.0)]),
        dims=("fit", "t"),
        coords={"fit": [0, 1], "t": t},
    )
    slope_guess = UncopyableDataArray(
        [1.0, -1.0],
        dims="fit",
        coords={"fit": da["fit"]},
    )

    fit = da.xlm.modelfit(
        "t",
        model=lmfit.Model(linear_batch),
        params={"slope": slope_guess, "intercept": 0.0},
    )

    np.testing.assert_allclose(
        fit.modelfit_coefficients,
        [[2.0, 1.0], [-0.5, 3.0]],
    )


def test_modelfit_expr() -> None:
    # Generate 2 lorentzian peaks on linear bkg and add poisson noise
    xval = np.linspace(-1, 1, 250)

    yval = 2 * xval + 4
    yval += lorentzian(xval, -0.5, 0.05, 10)
    yval += lorentzian(xval, 0.5, 0.05, 10)
    yval /= yval.sum()

    # Add noise
    npts = 100000
    rng = np.random.default_rng(1)
    # yerr = 1 / np.sqrt(npts)
    yval = rng.poisson(yval * npts).astype(float)

    # lmfit model
    model = (
        lmfit.models.LorentzianModel(prefix="p0_")
        + lmfit.models.LorentzianModel(prefix="p1_")
        + lmfit.models.LinearModel()
    )

    darr = xr.DataArray(yval, dims=("x",), coords={"x": xval})

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Parameter 'p01_delta' is a varying "
            "parameter, but is not included in the results. "
            "Consider providing `param_names` manually."
        ),
    ):
        darr.xlm.modelfit(
            coords=[darr.x],
            model=model,
            params={
                "p0_center": {"expr": "p1_center - p01_delta"},
                "p0_sigma": {"value": 0.05, "min": 0},
                "p0_amplitude": {"value": 10, "min": 0},
                "p1_center": 0.5,
                "p1_sigma": {"value": 0.05, "min": 0},
                "p1_amplitude": {"value": 10, "min": 0},
                "p01_delta": {"value": 1, "min": 0},
                "slope": {"value": 2, "min": 0},
                "intercept": 4,
            },
        )

    darr.xlm.modelfit(
        coords=[darr.x],
        model=model,
        params={
            "p0_center": {"expr": "p1_center - p01_delta"},
            "p0_sigma": {"value": 0.05, "min": 0},
            "p0_amplitude": {"value": 10, "min": 0},
            "p1_center": 0.5,
            "p1_sigma": {"value": 0.05, "min": 0},
            "p1_amplitude": {"value": 10, "min": 0},
            "p01_delta": {"value": 1, "min": 0},
            "slope": {"value": 2, "min": 0},
            "intercept": 4,
        },
        param_names=[
            "p0_amplitude",
            "p0_sigma",
            "p1_amplitude",
            "p1_center",
            "p1_sigma",
            "slope",
            "intercept",
            "p01_delta",
        ],
    )


@pytest.mark.parametrize("use_client", [True, False], ids=["client", "no_client"])
@pytest.mark.parametrize("single_param", [True, False], ids=["single", "broadcasted"])
def test_modelfit_parallel_dask(use_client: bool, single_param: bool) -> None:
    xval = np.linspace(-1, 1, 250)[np.newaxis, :]
    num_z = 400

    center_shift = np.linspace(-0.1, 0.1, num_z)[:, np.newaxis]

    # Lorentzian peaks with slightly shifted centers
    test_data = xr.DataArray(
        lorentzian(xval, amplitude=10, center=center_shift, sigma=0.3),
        dims=["z", "x"],
        coords={"z": np.arange(num_z), "x": xval.flatten()},
    )

    if use_client:
        from dask.distributed import Client

        client = Client()

    try:
        # Chunk data for dask parallelization
        test_data = test_data.chunk({"z": 10})

        # Initialize model and parameters
        model = lmfit.models.LorentzianModel()
        params = {
            "amplitude": 9,
            "sigma": 0.3,
        }
        if single_param:
            params["center"] = 0.0
        else:
            params["center"] = xr.DataArray(
                center_shift.flatten(), coords=[test_data.z]
            )

        # Run modelfit
        res = test_data.xlm.modelfit("x", model=model, params=params)

        # Compute in parallel
        res = res.compute()
        assert isinstance(res.modelfit_results[0].item(), lmfit.model.ModelResult)

    finally:
        if use_client:
            client.shutdown()
