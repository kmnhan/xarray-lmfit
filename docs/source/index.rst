##########################
xarray-lmfit documentation
##########################

xarray-lmfit is a Python package that bridges the power of `xarray <http://xarray.pydata.org>`_ for handling multi-dimensional labeled arrays with the flexible fitting capabilities of `lmfit <https://lmfit.github.io/lmfit-py/>`_.

With xarray-lmfit, `lmfit models <https://lmfit.github.io/lmfit-py/model.html>`_ can be fit to xarray Datasets and DataArrays, automatically propagating across multiple dimensions. The fit results are stored as xarray Datasets, retaining the original coordinates and dimensions of the input data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   user-guide/index
   api
