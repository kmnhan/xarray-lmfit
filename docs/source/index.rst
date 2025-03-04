##########################
xarray-lmfit documentation
##########################

.. only:: format_html

   **Date**: |today|

   .. image:: https://img.shields.io/pypi/pyversions/xarray-lmfit?style=flat-square&logo=python&logoColor=white
       :target: https://pypi.org/project/xarray-lmfit/
       :alt: Supported Python Versions
   .. image:: https://img.shields.io/pypi/v/xarray-lmfit?style=flat-square&logo=pypi&logoColor=white
       :target: https://pypi.org/project/xarray-lmfit/
       :alt: PyPi
   .. image:: https://img.shields.io/github/last-commit/kmnhan/xarray-lmfit?style=flat-square&logo=github&color=lightseagreen
       :target: https://github.com/kmnhan/xarray-lmfit.git
       :alt: Last Commit

xarray-lmfit is a Python package that bridges the power of `xarray <http://xarray.pydata.org>`_ for handling multi-dimensional labeled arrays with the flexible fitting capabilities of `lmfit <https://lmfit.github.io/lmfit-py/>`_.

With xarray-lmfit, `lmfit models <https://lmfit.github.io/lmfit-py/model.html>`_ can be fit to xarray Datasets and DataArrays, automatically propagating across multiple dimensions. The fit results are stored as xarray Datasets, retaining the original coordinates and dimensions of the input data.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   user-guide/index
   api
