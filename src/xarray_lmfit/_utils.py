import contextlib
import inspect
import pathlib
import sys
import warnings

import xarray as xr


def _find_stack_level() -> int:
    """Find the first place in the stack that is not inside erlab, xarray, or stdlib.

    This is unless the code emanates from a test, in which case we would prefer to see
    the source.

    This function is adapted from xarray.core.utils.find_stack_level.

    Returns
    -------
    stacklevel : int
        First level in the stack that is not part of erlab or stdlib.
    """
    import xarray

    import xarray_lmfit

    xarray_dir = pathlib.Path(xarray.__file__).parent
    pkg_dir = pathlib.Path(xarray_lmfit.__file__).parent.parent.parent
    test_dir = pkg_dir / "tests"

    std_lib_init = sys.modules["os"].__file__
    if std_lib_init is None:
        return 0

    std_lib_dir = pathlib.Path(std_lib_init).parent

    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if (
            (fname.startswith(str(pkg_dir)) and not fname.startswith(str(test_dir)))
            or (
                fname.startswith(str(std_lib_dir))
                and "site-packages" not in fname
                and "dist-packages" not in fname
            )
            or fname.startswith(str(xarray_dir))
        ):
            frame = frame.f_back
            n += 1
        else:
            break
    return n


def emit_user_level_warning(message, category=None) -> None:
    """Emit a warning at the user level by inspecting the stack trace."""
    stacklevel = _find_stack_level()
    return warnings.warn(message, category=category, stacklevel=stacklevel)


@contextlib.contextmanager
def joblib_progress(file=None, **kwargs):
    """Patches joblib to report into a tqdm progress bar."""
    import joblib
    import tqdm

    if file is None:
        file = sys.stdout

    tqdm_object = tqdm.tqdm(iterable=None, file=file, **kwargs)

    def tqdm_print_progress(self) -> None:
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


USE_QUICK = False


class XLMAccessorRegistrationWarning(Warning):
    """Warning for conflicts in accessor registration."""


class _XLMCachedAccessor:
    def __init__(self, name, accessor):
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor

        try:
            cache = obj._cache
        except AttributeError:
            cache = obj._cache = {}

        try:
            return cache[self._name]
        except KeyError:
            pass

        try:
            accessor_obj = self._accessor(obj._obj)
        except AttributeError as e:
            raise RuntimeError(f"error initializing {self._name!r} accessor.") from e

        cache[self._name] = accessor_obj
        return accessor_obj


def _register_xlm_accessor(name, cls):
    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {accessor!r} under name {name!r} for type "
                f"{cls!r} is overriding a preexisting attribute with the same name.",
                XLMAccessorRegistrationWarning,
                stacklevel=2,
            )
        setattr(cls, name, _XLMCachedAccessor(name, accessor))
        return accessor

    return decorator


@xr.register_dataarray_accessor("xlm")
class XLMDataArrayAccessor:
    def __init__(self, data: xr.DataArray) -> None:
        self._obj: xr.DataArray = data


@xr.register_dataset_accessor("xlm")
class XLMDatasetAccessor:
    def __init__(self, data: xr.Dataset) -> None:
        self._obj: xr.Dataset = data


def register_xlm_dataarray_accessor(name):
    return _register_xlm_accessor(name, XLMDataArrayAccessor)


def register_xlm_dataset_accessor(name):
    return _register_xlm_accessor(name, XLMDatasetAccessor)
