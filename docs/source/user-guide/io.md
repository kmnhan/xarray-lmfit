# Saving and loading fits

Since the fit results are stored in an xarray Dataset, they can be easily saved as
netCDF files by serializing lmfit objects to JSON. This can be done with {func}`xarray_lmfit.save_fit`:

```python
import xarray_lmfit as xlm

xlm.save_fit(result_ds, "fit_results.nc")
```

The saved Dataset can be loaded back with {func}`xarray_lmfit.load_fit`.

```python
result_ds = xlm.load_fit("fit_results.nc")
```

:::{warning}

Saving full model results that includes the model functions can be difficult. Instead of saving the fit results, it is recommended to save the code that can reproduce the fit. See [the relevant lmfit documentation](https://lmfit.github.io/lmfit-py/model.html#saving-and-loading-modelresults) for more information.

:::
