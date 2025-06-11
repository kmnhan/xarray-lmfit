import os
import typing

import xarray as xr

if typing.TYPE_CHECKING:
    import lmfit
else:
    import lazy_loader as _lazy

    lmfit = _lazy.load("lmfit")


def _dumps_result(result: "lmfit.model.ModelResult") -> str:
    return result.dumps()


def _loads_result(s: str, funcdefs: dict | None = None) -> "lmfit.model.ModelResult":
    return lmfit.model.ModelResult(
        lmfit.Model(lambda x: x, None), lmfit.Parameters()
    ).loads(s, funcdefs=funcdefs)


def save_fit(result_ds: xr.Dataset, path: str | os.PathLike, **kwargs) -> None:
    """Save fit results to a netCDF file.

    This function processes a dataset containing fit results obtained from
    :meth:`xarray.DataArray.xlm.modelfit` or :meth:`xarray.Dataset.xlm.modelfit` and
    saves it to a netCDF file.

    Serialization of :class:`lmfit.model.ModelResult` objects are handled just like
    :func:`lmfit.model.save_modelresult`, and shares the same limitations.

    Parameters
    ----------
    result_ds
        An xarray Dataset containing the fit results.

        Any :class:`lmfit.model.ModelResult` objects in the dataset will be serialized
        before saving.
    path
        Path to which to save the fit result dataset.
    **kwargs
        Additional keyword arguments that are passed to
        :meth:`xarray.Dataset.to_netcdf`.

    Note
    ----
    Storing fit results to a file for an extended period of time is not recommended, as
    the serialization format does not guarantee compatibility between different versions
    of python or packages. For more information, see the `lmfit documentation
    <https://lmfit.github.io/lmfit-py/model.html#saving-and-loading-models>`_.

    See Also
    --------
    :func:`lmfit.model.save_modelresult`
        Corresponding lmfit function that can save a single
        :class:`lmfit.model.ModelResult`.
    :meth:`load_fit <xarray_lmfit.load_fit>`
        Function to load the saved fit results.

    """
    ds = result_ds.copy()
    for var in ds.data_vars:
        if str(var).endswith("modelfit_results"):
            ds[var] = xr.apply_ufunc(
                _dumps_result,
                ds[var],
                vectorize=True,
                output_dtypes=[str],
            )

    ds.to_netcdf(path, **kwargs)


def load_fit(
    path: str | os.PathLike, funcdefs: dict | None = None, **kwargs
) -> xr.Dataset:
    """Load fit results from a netCDF file.

    This function loads a dataset from a netCDF file and deserializes any
    :class:`lmfit.model.ModelResult` objects that were saved.

    The deserialization is performed just like :func:`lmfit.model.load_modelresult`, and
    shares the same limitations.

    Parameters
    ----------
    path
        Path to the netCDF file to load.
    funcdefs : dict, optional
        Dictionary of functions to use when deserializing the fit results. See
        :func:`lmfit.model.load_modelresult` for more information.
    **kwargs
        Additional keyword arguments that are passed to :func:`xarray.load_dataset`.

    Returns
    -------
    xarray.Dataset
        The dataset containing the fit results.

    Note
    ----
    Storing fit results to a file for an extended period of time is not recommended, as
    the serialization format does not guarantee compatibility between different versions
    of python or packages. For more information, see the `lmfit documentation
    <https://lmfit.github.io/lmfit-py/model.html#saving-and-loading-models>`_.

    See Also
    --------
    :meth:`save_fit <xarray_lmfit.save_fit>`
        Save fit results to a netCDF file.

    """
    result_ds = xr.load_dataset(path, **kwargs)
    for var in result_ds.data_vars:
        if str(var).endswith("modelfit_results"):
            result_ds[var] = xr.apply_ufunc(
                lambda s: _loads_result(s, funcdefs),
                result_ds[var],
                vectorize=True,
                output_dtypes=[object],
            )

    return result_ds
