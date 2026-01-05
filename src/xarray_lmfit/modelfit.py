"""Defines accessors for curve fitting."""

from __future__ import annotations

import contextlib
import copy
import functools
import typing
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr
from xarray.core.dataarray import _THIS_ARRAY

from xarray_lmfit._utils import (
    XLMDataArrayAccessor,
    XLMDatasetAccessor,
    emit_user_level_warning,
    register_xlm_dataarray_accessor,
    register_xlm_dataset_accessor,
)

if typing.TYPE_CHECKING:
    # Avoid importing until runtime for initial import performance
    import lmfit
else:
    import lazy_loader as _lazy

    lmfit = _lazy.load("lmfit")


def _nested_dict_vals(d) -> Iterable[typing.Any]:
    """Recursively yield all "leaf" values in a nested dictionary."""
    for v in d.values():
        if isinstance(v, Mapping):
            yield from _nested_dict_vals(v)
        else:
            yield v


def _broadcast_dict_values(d: dict[str, typing.Any]) -> dict[str, xr.DataArray]:
    """Broadcast all values in a dictionary to DataArrays with matching dimensions."""
    to_broadcast = {}
    for k, v in d.items():
        if isinstance(v, xr.DataArray | xr.Dataset):
            to_broadcast[k] = v
        else:
            to_broadcast[k] = xr.DataArray(v)

    d = dict(
        zip(to_broadcast.keys(), xr.broadcast(*to_broadcast.values()), strict=True)
    )
    return typing.cast("dict[str, xr.DataArray]", d)


def _concat_along_keys(d: dict[str, xr.DataArray], dim_name: str) -> xr.DataArray:
    return xr.concat(d.values(), d.keys(), coords="minimal").rename(concat_dim=dim_name)


def _parse_params(
    d: dict[str, typing.Any] | lmfit.Parameters,
) -> xr.DataArray | _ParametersWrapper:
    if isinstance(d, lmfit.Parameters):
        # Input to apply_ufunc cannot be a Mapping, so wrap in a class
        return _ParametersWrapper(d)

    # Iterate over all values to check if any DataArrays are present
    for v in _nested_dict_vals(d):
        if isinstance(v, xr.DataArray):
            return _parse_multiple_params(copy.deepcopy(d))

    # Can be treated as a single lmfit.Parameters object for all fits
    return _ParametersWrapper(lmfit.create_params(**d))


def _parse_multiple_params(d: dict[str, typing.Any]) -> xr.DataArray:
    for k in d:
        if isinstance(d[k], int | float | complex | xr.DataArray):
            d[k] = {"value": d[k]}

        d[k] = _concat_along_keys(_broadcast_dict_values(d[k]), "__dict_keys")

    da = _concat_along_keys(_broadcast_dict_values(d), "__param_names")

    pnames = tuple(da["__param_names"].values)
    argnames = tuple(da["__dict_keys"].values)

    def _reduce_to_param(arr, axis=0):
        axes = (axis,) if np.isscalar(axis) else tuple(axis)
        axes = tuple(a if a >= 0 else a + arr.ndim for a in axes)
        axes_set = set(axes)
        out_arr = np.empty(
            tuple(s for i, s in enumerate(arr.shape) if i not in axes_set), dtype=object
        )
        for i in range(out_arr.size):
            out_arr.flat[i] = {}

        for i, par in enumerate(pnames):
            for j, name in enumerate(argnames):
                for k, val in enumerate(arr[i, j].flat):
                    if par not in out_arr.flat[k]:
                        out_arr.flat[k][par] = {}

                    if isinstance(val, str) or np.isfinite(val):
                        out_arr.flat[k][par][name] = val

        for i in range(out_arr.size):
            out_arr.flat[i] = lmfit.create_params(**out_arr.flat[i])

        return out_arr

    return da.reduce(_reduce_to_param, ("__dict_keys", "__param_names"))


class _ParametersWrapper:
    """Wrapper class to pass lmfit.Parameters through xarray.apply_ufunc.

    Since Mapping types (dict and lmfit.Parameters) are not allowed as positional inputs
    to xarray.apply_ufunc, we wrap lmfit.Parameters in this class.
    """

    def __init__(self, params: lmfit.Parameters) -> None:
        self.params = params


def _model_fit_wrapper(
    Y: npt.NDArray,
    *args,
    model: lmfit.Model,
    param_names: Sequence[str],
    stat_names: Sequence[str],
    n_coords: int,
    skipna: bool,
    guess: bool,
    errors: typing.Literal["raise", "ignore"],
    model_fit_kwargs: Mapping[str, typing.Any] | None = None,
):
    """Top-level wrapper for Model.fit to keep dask graphs picklable."""
    n_params = len(param_names)
    n_stats = len(stat_names)

    coords__ = args[:n_coords]
    init_params_ = args[n_coords]

    fit_kwargs = dict(model_fit_kwargs) if model_fit_kwargs is not None else {}
    if len(args) > n_coords + 1:
        fit_kwargs.pop("weights", None)
        weights_ = args[n_coords + 1].ravel()
    else:
        weights_ = fit_kwargs.pop("weights", None)

    initial_params = lmfit.create_params() if guess else model.make_params()

    if isinstance(init_params_, _ParametersWrapper):
        other = init_params_.params

        if isinstance(other, lmfit.Parameters):
            initial_params.update(other)
        else:
            # Instance check may fail in multiprocess, so forcibly set class
            # This no longer triggers in CI in python 3.14+, presumably due to
            # different default forking method
            other.__class__ = lmfit.Parameters

            # Copy behavior of Parameters.update() and set class for each param
            __params = []
            for par in other.values():
                par.__class__ = lmfit.Parameter

                __params.append(par)
                par._delay_asteval = True
                initial_params[par.name] = par

            for para in __params:
                para._delay_asteval = False

            for sym in other._asteval.user_defined_symbols():
                initial_params._asteval.symtable[sym] = other._asteval.symtable[sym]

    elif isinstance(init_params_, str):
        initial_params.update(lmfit.Parameters().loads(init_params_))

    elif isinstance(init_params_, lmfit.Parameters):
        initial_params.update(init_params_)

    popt = np.full([n_params], np.nan)
    perr = np.full([n_params], np.nan)
    pcov = np.full([n_params, n_params], np.nan)
    stats = np.full([n_stats], np.nan)
    data = Y.copy()
    best = np.full_like(data, np.nan)

    x = np.vstack([c.ravel() for c in coords__])
    y: npt.NDArray = Y.ravel()

    if skipna:
        mask = np.all([np.any(~np.isnan(x), axis=0), ~np.isnan(y)], axis=0)
        x = x[:, mask]
        y = y[mask]
        if weights_ is not None and not np.isscalar(weights_):
            weights_ = weights_[mask]
        if not len(y):
            # No data to fit
            modres = lmfit.model.ModelResult(
                model, initial_params, data=y, weights=weights_
            )
            modres.success = False
            return popt, perr, pcov, stats, data, best, modres
    else:
        mask = np.full_like(y, True)

    x = np.squeeze(x)

    if model.independent_vars is not None:
        if n_coords == 1:
            indep_var_kwargs = {model.independent_vars[0]: x}
            if len(model.independent_vars) == 2:
                # Y-dependent data, like background models
                indep_var_kwargs[model.independent_vars[1]] = y
        else:
            indep_var_kwargs = dict(
                zip(model.independent_vars[:n_coords], x, strict=True)
            )
    else:
        raise ValueError("Independent variables not defined in model")

    if guess:
        if isinstance(model, lmfit.model.CompositeModel):
            guessed_params = model.make_params()
            for comp in model.components:
                with contextlib.suppress(NotImplementedError):
                    guessed_params.update(comp.guess(y, **indep_var_kwargs))
            # Given parameters must override guessed parameters
            initial_params = guessed_params.update(initial_params)

        else:
            try:
                initial_params = model.guess(y, **indep_var_kwargs).update(
                    initial_params
                )
            except NotImplementedError:
                emit_user_level_warning(
                    f"`guess` is not implemented for {model}, "
                    "using supplied initial parameters"
                )
                initial_params = model.make_params().update(initial_params)
    try:
        modres = model.fit(
            y,
            **indep_var_kwargs,
            params=initial_params,
            weights=weights_,
            **fit_kwargs,
        )
    except ValueError:
        if errors == "raise":
            raise
        modres = lmfit.model.ModelResult(
            model, initial_params, data=y, weights=weights_
        )
        modres.success = False
        return popt, perr, pcov, stats, data, best, modres
    else:
        if modres.success:
            popt_list, perr_list = [], []
            for name in param_names:
                if name not in modres.params:
                    raise ValueError(
                        f"Parameter '{name}' was not found in the fit results. "
                        "Check the model and parameter names."
                    )
                p: lmfit.model.Parameter = modres.params[name]
                popt_list.append(p.value if p.value is not None else np.nan)
                perr_list.append(p.stderr if p.stderr is not None else np.nan)

            popt, perr = np.array(popt_list), np.array(perr_list)

            stats = np.array(
                [
                    s if s is not None else np.nan
                    for s in [getattr(modres, s) for s in stat_names]
                ]
            )

            # Fill in covariance matrix entries, entries for non-varying
            # parameters are left as NaN
            if modres.covar is not None:
                var_names = modres.var_names
                for vi in range(modres.nvarys):
                    if var_names[vi] not in param_names:
                        emit_user_level_warning(
                            f"Parameter '{var_names[vi]}' is a varying "
                            "parameter, but is not included in the results. "
                            "Consider providing `param_names` manually."
                        )
                    else:
                        i = param_names.index(var_names[vi])
                        for vj in range(modres.nvarys):
                            if var_names[vj] in param_names:
                                j = param_names.index(var_names[vj])
                                pcov[i, j] = modres.covar[vi, vj]

            best.flat[mask] = modres.best_fit

    return popt, perr, pcov, stats, data, best, modres


@register_xlm_dataset_accessor("modelfit")
class ModelFitDatasetAccessor(XLMDatasetAccessor):
    """`xarray.Dataset.modelfit` accessor for fitting lmfit models."""

    def _define_wrapper(
        self,
        model: lmfit.Model,
        param_names: list[str],
        stat_names: list[str],
        n_coords: int,
        skipna: bool,
        guess: bool,
        errors: typing.Literal["raise", "ignore"],
        model_fit_kwargs: Mapping[str, typing.Any],
    ):
        """Define a picklable wrapper for the model fitting."""
        return functools.partial(
            _model_fit_wrapper,
            model=model,
            param_names=param_names,
            stat_names=stat_names,
            n_coords=n_coords,
            skipna=skipna,
            guess=guess,
            errors=errors,
            model_fit_kwargs=model_fit_kwargs,
        )

    def _define_output_wrapper(
        self,
        model: lmfit.Model,
        params: xr.Dataset | xr.DataArray | _ParametersWrapper,
        reduce_dims_: list[Hashable],
        coords_,
        param_names: list[str],
        stat_names: list[str],
        output_result: bool,
        skipna: bool,
        guess: bool,
        is_dask: bool,
        errors: typing.Literal["raise", "ignore"],
        model_fit_kwargs: Mapping[str, typing.Any],
        weight_da: xr.DataArray | None,
    ) -> typing.Callable:
        _wrapper = self._define_wrapper(
            model=model,
            param_names=param_names,
            stat_names=stat_names,
            n_coords=len(coords_),
            skipna=skipna,
            guess=guess,
            errors=errors,
            model_fit_kwargs=model_fit_kwargs,
        )
        n_params = len(param_names)
        n_stats = len(stat_names)
        dim_sizes = {dim: self._obj.coords[dim].size for dim in reduce_dims_}

        def _output_wrapper(name, da, out) -> None:
            name = "" if name is _THIS_ARRAY else f"{name!s}_"

            # Select parameters for this data variable
            if not isinstance(params, xr.Dataset):
                params_to_apply = params
            else:
                try:
                    params_to_apply = params[name.removesuffix("_")]
                except KeyError:
                    params_to_apply = params[float(name.removesuffix("_"))]

            if isinstance(params_to_apply, xr.DataArray) and is_dask:
                # TODO: check whether this works with chunked datasets
                # Since dask cannot auto-rechunk object arrays, rechunk to single chunk
                params_to_apply = params_to_apply.chunk(
                    dict.fromkeys(params_to_apply.dims, -1)
                )

            # Positional arguments to wrapper
            args = [da, *coords_, params_to_apply]

            # Core dims for data & coords
            input_core_dims = [reduce_dims_ for _ in range(len(coords_) + 1)]

            # Core dims for parameters
            input_core_dims.append([])

            if isinstance(weight_da, xr.DataArray):
                weights = weight_da.broadcast_like(da)
                args.append(weights)

                # Core dims for weights
                input_core_dims.append([d for d in reduce_dims_ if d in weights.dims])

            popt, perr, pcov, stats, data, best, modres = xr.apply_ufunc(
                _wrapper,
                *args,
                vectorize=True,
                dask="parallelized",
                input_core_dims=input_core_dims,
                output_core_dims=[
                    ["param"],
                    ["param"],
                    ["cov_i", "cov_j"],
                    ["fit_stat"],
                    reduce_dims_,
                    reduce_dims_,
                    [],
                ],
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "param": n_params,
                        "fit_stat": n_stats,
                        "cov_i": n_params,
                        "cov_j": n_params,
                    }
                    | dim_sizes
                },
                output_dtypes=(
                    np.float64,
                    np.float64,
                    np.float64,
                    np.float64,
                    np.float64,
                    np.float64,
                    lmfit.model.ModelResult,
                ),
                exclude_dims=set(reduce_dims_),
            )

            if output_result:
                out[name + "modelfit_results"] = modres

            out[name + "modelfit_coefficients"] = popt
            out[name + "modelfit_stderr"] = perr
            out[name + "modelfit_covariance"] = pcov
            out[name + "modelfit_stats"] = stats
            out[name + "modelfit_data"] = data
            out[name + "modelfit_best_fit"] = best

        return _output_wrapper

    def __call__(
        self,
        coords: str | xr.DataArray | Iterable[str | xr.DataArray],
        model: lmfit.Model,
        reduce_dims: str | Collection[Hashable] | None = None,
        skipna: bool = True,
        params: lmfit.Parameters
        | dict[str, float | dict[str, typing.Any]]
        | xr.DataArray
        | xr.Dataset
        | _ParametersWrapper
        | None = None,
        guess: bool = False,
        errors: typing.Literal["raise", "ignore"] = "raise",
        progress: bool = False,
        output_result: bool = True,
        param_names: list[str] | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """Curve fitting optimization for arbitrary models.

        Wraps :meth:`lmfit.Model.fit <lmfit.model.Model.fit>` with
        :func:`xarray.apply_ufunc`.

        Parameters
        ----------
        coords : Hashable, xarray.DataArray, or Sequence of Hashable or xarray.DataArray
            Independent coordinate(s) over which to perform the curve fitting. Must
            share at least one dimension with the calling object. When fitting
            multi-dimensional functions, supply `coords` as a sequence in the same order
            as arguments in `func`. To fit along existing dimensions of the calling
            object, `coords` can also be specified as a str or sequence of strs.
        model : lmfit.Model
            A model object to fit to the data. The model must be an *instance* of
            :class:`lmfit.Model <lmfit.model.Model>`.
        reduce_dims : str, Iterable of Hashable or None, optional
            Additional dimension(s) over which to aggregate while fitting. For example,
            calling `ds.xlm.modelfit(coords='time', reduce_dims=['lat', 'lon'], ...)`
            will aggregate all lat and lon points and fit the specified function along
            the time dimension.
        skipna : bool, default: True
            Whether to skip missing values when fitting. Default is True.
        params : lmfit.Parameters, dict-like, DataArray or Dataset, optional
            Optional input parameters to the fit.

            - If a :class:`lmfit.Parameters <lmfit.parameter.Parameters>` object, it
              will be used for all fits.

            - If a dict-like object, it must be structured like keyword arguments to
              :func:`lmfit.create_params <lmfit.parameter.create_params>`.

              Additionally, each value of the dictionary may also be a DataArray, which
              will be broadcasted appropriately.

            - If a DataArray, each entry must be a dict-like object, a
              :class:`lmfit.Parameters <lmfit.parameter.Parameters>` object, or a JSON
              string created with :meth:`lmfit.Parameters.dumps
              <lmfit.parameter.Parameters.dumps>`.

            - If a Dataset, the name of the data variables in the Dataset must match the
              name of the data variables in the calling object, and each data variable
              will be used as the parameters for the corresponding data variable.

            If none or only some parameters are passed, the rest will be assigned
            initial values and bounds with :meth:`lmfit.Model.make_params
            <lmfit.model.Model.make_params>`, or guessed with :meth:`lmfit.Model.guess
            <lmfit.model.Model.guess>` if `guess` is `True`.
        guess : bool, default: `False`
            Whether to guess the initial parameters with :meth:`lmfit.Model.guess
            <lmfit.model.Model.guess>`. For composite models, the parameters will be
            guessed for each component.
        errors : {"raise", "ignore"}, default: "raise"
            If `'raise'`, any errors from the :meth:`lmfit.Model.fit
            <lmfit.model.Model.fit>` optimization will raise an exception. If
            `'ignore'`, the return values for the coordinates where the fitting failed
            will be NaN.
        progress : bool, default: `False`
            Whether to show a progress bar for fitting over data variables. Only useful
            if there are multiple data variables to fit. Requires the ``tqdm`` package
            to be installed.
        output_result : bool, default: `True`
            Whether to include the full :class:`lmfit.model.ModelResult` object in the
            output dataset. If `True`, the result will be stored in a variable named
            `[var]_modelfit_results`.
        param_names : list of str, optional
            List of parameter names to include in the output dataset. If not provided,
            defaults to :attr:`lmfit.Model.param_names <lmfit.model.Model.param_names>`
            (after calling :meth:`lmfit.Model.make_params
            <lmfit.model.Model.make_params>`).
        **kwargs : optional
            Additional keyword arguments to passed to :meth:`lmfit.Model.fit
            <lmfit.model.Model.fit>`.

            The ``weights`` keyword argument may also be provided as a DataArray, which
            will be broadcasted appropriately. The weights must share at least one
            dimension with the calling object.

        Returns
        -------
        xarray.Dataset
            A single dataset which contains:

            [var]_modelfit_results
                The full :class:`lmfit.model.ModelResult` object from the fit. Only
                included if `output_result` is `True`.
            [var]_modelfit_coefficients
                The coefficients of the best fit.
            [var]_modelfit_stderr
                The standard errors of the coefficients.
            [var]_modelfit_covariance
                The covariance matrix of the coefficients. Note that elements
                corresponding to non varying parameters are set to NaN, and the actual
                size of the covariance matrix may be smaller than the array.
            [var]_modelfit_stats
                Statistics from the fit. See :func:`lmfit.minimize
                <lmfit.minimizer.minimize>`.
            [var]_modelfit_data
                Data used for the fit.
            [var]_modelfit_best_fit
                The best fit data of the fit.

        See Also
        --------
        xarray.Dataset.curvefit

        xarray.Dataset.polyfit

        lmfit.model.Model.fit

        scipy.optimize.curve_fit

        """
        # Implementation analogous to xarray.Dataset.curve_fit

        if params is None:
            params = lmfit.create_params()

        is_dask: bool = bool(self._obj.chunks)

        if not isinstance(params, xr.Dataset) and isinstance(params, Mapping):
            # Given as a mapping from str to ...
            # float or DataArray or dict of str to Any (including DataArray of Any)
            params = _parse_params(params)
            # Now, params is either a _ParametersWrapper or a DataArray of Parameters

        reduce_dims_: list[Hashable]
        if not reduce_dims:
            reduce_dims_ = []
        elif isinstance(reduce_dims, str) or not isinstance(reduce_dims, Iterable):
            reduce_dims_ = [reduce_dims]
        else:
            reduce_dims_ = list(reduce_dims)

        if isinstance(coords, str | xr.DataArray) or not isinstance(coords, Iterable):
            coords = [coords]
        coords_: Sequence[xr.DataArray] = [
            self._obj[coord] if isinstance(coord, str) else coord for coord in coords
        ]

        # Determine whether any coords are dims on self._obj
        for coord in coords_:
            reduce_dims_ += [c for c in self._obj.dims if coord.equals(self._obj[c])]
        reduce_dims_ = list(set(reduce_dims_))
        preserved_dims = list(set(self._obj.dims) - set(reduce_dims_))
        if not reduce_dims_:
            raise ValueError(
                "No arguments to `coords` were identified as a dimension on the "
                "calling object, and no dims were supplied to `reduce_dims`. This "
                "would result in fitting on scalar data."
            )

        # Sort dims by their order in original object
        if all(d in self._obj.dims for d in reduce_dims_):
            # apply_ufunc will raise error downstream if this is False
            reduce_dims_ = sorted(
                reduce_dims_, key=lambda dim: list(self._obj.dims).index(dim)
            )

        # Check that initial guess and bounds only contain coords in preserved_dims
        if isinstance(params, xr.DataArray | xr.Dataset):
            unexpected = set(params.dims) - set(preserved_dims)
            if unexpected:
                raise ValueError(
                    f"Parameters object has unexpected dimensions {tuple(unexpected)}. "
                    "It should only have dimensions that are in data dimensions "
                    f"{preserved_dims}."
                )

        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        # Broadcast all coords with each other
        coords_ = xr.broadcast(*coords_)
        coords_ = [
            coord.broadcast_like(self._obj, exclude=preserved_dims) for coord in coords_
        ]

        if param_names is None:
            # Call make_params before getting parameter names as it may add param hints
            model.make_params()

            # Get the parameter names (assume no expressions)
            param_names = model.param_names

        # Define the statistics to extract from the fit result
        stat_names = [
            "nfev",
            "nvarys",
            "ndata",
            "nfree",
            "chisqr",
            "redchi",
            "aic",
            "bic",
            "rsquared",
        ]

        model_fit_kwargs: dict[str, typing.Any] = dict(kwargs)
        weight_da: xr.DataArray | None = None
        if isinstance(model_fit_kwargs.get("weights"), xr.DataArray):
            weight_da = typing.cast("xr.DataArray", model_fit_kwargs.pop("weights"))

        _output_wrapper = self._define_output_wrapper(
            model=model,
            params=params,
            reduce_dims_=reduce_dims_,
            coords_=coords_,
            param_names=param_names,
            stat_names=stat_names,
            output_result=output_result,
            skipna=skipna,
            guess=guess,
            is_dask=is_dask,
            errors=errors,
            model_fit_kwargs=model_fit_kwargs,
            weight_da=weight_da,
        )
        result = xr.Dataset()

        if progress:
            try:
                import tqdm.auto as tqdm
            except ImportError as e:
                raise RuntimeError(
                    "progress bars require the 'tqdm' package. "
                    "Install with: pip install tqdm"
                ) from e
            var_items_iter = tqdm.tqdm(
                self._obj.data_vars.items(),
                desc="Fitting",
                total=len(self._obj.data_vars),
            )
        else:
            var_items_iter = self._obj.data_vars.items()

        for name, da in var_items_iter:
            _output_wrapper(name, da, result)

        result = result.assign_coords(
            {
                "param": param_names,
                "fit_stat": stat_names,
                "cov_i": param_names,
                "cov_j": param_names,
            }
            | {dim: self._obj.coords[dim] for dim in reduce_dims_}
        )
        result.attrs = self._obj.attrs.copy()

        return result


@register_xlm_dataarray_accessor("modelfit")
class ModelFitDataArrayAccessor(XLMDataArrayAccessor):
    """`xarray.DataArray.modelfit` accessor for fitting lmfit models."""

    def __call__(self, *args, **kwargs) -> xr.Dataset:
        return self._obj.to_dataset(name=_THIS_ARRAY).xlm.modelfit(*args, **kwargs)

    __call__.__doc__ = (
        str(ModelFitDatasetAccessor.__call__.__doc__)
        .replace("Dataset.curvefit", "DataArray.curvefit")
        .replace("Dataset.polyfit", "DataArray.polyfit")
        .replace("[var]_", "")
    )
