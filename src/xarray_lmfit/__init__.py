__all__ = ["load_fit", "save_fit"]

from importlib.metadata import version as _version

from xarray_lmfit import modelfit as modelfit
from xarray_lmfit._io import load_fit, save_fit

try:
    __version__ = _version("xarray_lmfit")
except Exception:
    __version__ = "0.0.0"
