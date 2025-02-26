import os

import lmfit
import xarray as xr


def _dumps_result(result: lmfit.model.ModelResult) -> str:
    return result.dumps()


def _loads_result(s: str, funcdefs: dict | None = None) -> lmfit.model.ModelResult:
    return lmfit.model.ModelResult(
        lmfit.Model(lambda x: x, None), lmfit.Parameters()
    ).loads(s, funcdefs=funcdefs)


def save_fit(result_ds: xr.Dataset, path: str | os.PathLike, **kwargs) -> None:
    """Save fit results to a netCDF file.

    This function processes a dataset resulting from :meth:`modelfit
    <xarray_lmfit.modelfit.__call__>` and saves it to a netCDF file.

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
    :meth:`load_fit <xarray_lmfit.load_fit>` : Load fit results from a netCDF file.

    """
    ds = result_ds.copy()
    if "modelfit_results" in ds:
        ds["modelfit_results"] = xr.apply_ufunc(
            _dumps_result, ds["modelfit_results"], vectorize=True
        )
    ds.to_netcdf(path, **kwargs)


def load_fit(
    path: str | os.PathLike, funcdefs: dict | None = None, **kwargs
) -> xr.Dataset:
    """Load fit results from a netCDF file.

    This function loads a dataset from a netCDF file and deserializes any
    :class:`lmfit.model.ModelResult` objects that were saved.

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

    See Also
    --------
    :meth:`save_fit <xarray_lmfit.save_fit>` : Save fit results to a netCDF file.

    """
    result_ds = xr.load_dataset(path, **kwargs)
    if "modelfit_results" in result_ds:
        result_ds["modelfit_results"] = xr.apply_ufunc(
            lambda s: _loads_result(s, funcdefs),
            result_ds["modelfit_results"],
            vectorize=True,
        )
    return result_ds
