# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_autosummary_accessors

import xarray_lmfit

project = "xarray-lmfit"
copyright = "2025, Kimoon Han"  # noqa: A001
author = "Kimoon Han"


release = xarray_lmfit.__version__
version = release


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_autosummary_accessors",
    "nbsphinx",
]

templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]
exclude_patterns = []

master_doc = "index"
suffix = ".rst"
default_role = "obj"

# -- Autosummary and autodoc settings ----------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = False


# -- Napoleon settings -------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray",
    "DataArray": ":class:`DataArray <xarray.DataArray>`",
    "Dataset": ":class:`Dataset <xarray.Dataset>`",
    "DataTree": ":class:`DataTree <xarray.DataTree>`",
    "np.float32": "float32",
    "numpy.float32": "float32",
    "np.float64": "float64",
    "numpy.float64": "float64",
    "array-like": "`array-like <numpy.typing.ArrayLike>`",
    "path-like": "`path-like <os.PathLike>`",
    "lmfit.Parameters": ":class:`lmfit.Parameters <lmfit.parameter.Parameters>`",
    "lmfit.Model": ":class:`lmfit.Model <lmfit.model.Model>`",
}

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py", None),
    # "matplotlib": ("https://matplotlib.org/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    # "pandas": ("https://pandas.pydata.org/docs", None),
    # "ipywidgets": ("https://ipywidgets.readthedocs.io/en/stable", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    # "panel": ("https://panel.holoviz.org", None),
    # "hvplot": ("https://hvplot.holoviz.org", None),
}

# -- nbsphinx options --------------------------------------------------------
nbsphinx_execute_arguments = ["--InlineBackend.figure_formats={'svg', 'pdf'}"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
pygments_dark_style = "monokai"
