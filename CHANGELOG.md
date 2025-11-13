## v0.4.0 (2025-11-13)

### ✨ Features

- **modelfit:** add support for providing weights as DataArrays ([76d13e5](https://github.com/kmnhan/xarray-lmfit/commit/76d13e501a7c9e65c7eeff86496fd8ae7b395411))

  Weights can now be passed as DataArrays to `modelfit`, which are broadcasted to match the data being fitted.

## v0.3.0 (2025-09-07)

### 💥 Breaking Changes

- Using `modelfit` with `progress=True` now requires the package to be installed with the `progress` optional dependency group, like `pip install xarray-lmfit[progress]`. ([d7324c9](https://github.com/kmnhan/xarray-lmfit/commit/d7324c94b483527bd4540c1328b32d9f4054d2b4))

- While adding dask support, this release drops support for rudimentary joblib-based parallelization across multiple data variables; this removes the `parallel` and `parallel_kw` arguments to `modelfit`. Use dask arrays as an alternative. ([d3f90df](https://github.com/kmnhan/xarray-lmfit/commit/d3f90dffb226fab71e96309f41da35e0a929adc5))

### ✨ Features

- **modelfit:** Add `rsquared` to `modelfit_stats` by @newton-per-sqm (#19) ([e5a8a1e](https://github.com/kmnhan/xarray-lmfit/commit/e5a8a1e8a515627a3b5b6dd2c8fad83b9d15c3d7))

  Co-authored-by: Pascal Muster <Pascal.Muster@infineon.com>

- **modelfit:** properly support dask and drop support for joblib-based parallelization ([d3f90df](https://github.com/kmnhan/xarray-lmfit/commit/d3f90dffb226fab71e96309f41da35e0a929adc5))

  `modelfit` now supports dask arrays properly with minimal serialization overhead.

### ♻️ Code Refactor

- **modelfit:** make `tqdm` an optional dependency (#20) ([d7324c9](https://github.com/kmnhan/xarray-lmfit/commit/d7324c94b483527bd4540c1328b32d9f4054d2b4))

  The `tqdm` package which provides the progress bar when `progress=True` is now an optional dependency. If not installed, passing `progress=True` to `modelfit` will now result in an error.

## v0.2.3 (2025-06-11)

### 🐞 Bug Fixes

- sort dimension order in output to maintain original order ([c2a683f](https://github.com/kmnhan/xarray-lmfit/commit/c2a683f4f166f986dc40042ba8b0cacd5162d857))

- **io:** correctly handle saving and loading fit results output from `Dataset.xlm.modelfit()` ([77acadf](https://github.com/kmnhan/xarray-lmfit/commit/77acadfc6dabf8529e3f8eb7cc5f256ed83627d3))

## v0.2.2 (2025-04-28)

### ♻️ Code Refactor

- add quotes to type hints for deferred loading (#10) ([e0b515c](https://github.com/kmnhan/xarray-lmfit/commit/e0b515c42f5703680c05acb3040d4152069fc00a))

## v0.2.1 (2025-04-14)

### ⚡️ Performance

- delay importing lmfit until needed (#9) ([6773d03](https://github.com/kmnhan/xarray-lmfit/commit/6773d03393057c1b866929724b02798186eedb0b))

  This improves initial import time.

## v0.2.0 (2025-04-08)

### ✨ Features

- **modelfit:** allow the user to manually specify parameters to include in the fit result ([8e6f1a6](https://github.com/kmnhan/xarray-lmfit/commit/8e6f1a66ac0ab6aa4dc425cc37c234b4c61409fc))

  This also allows for complex models with many parameters given as expressions.

## v0.1.3 (2025-03-10)

### 🐞 Bug Fixes

- allow lower versions of dependencies ([139df09](https://github.com/kmnhan/xarray-lmfit/commit/139df09c938795c9af69ddb1e15db7eba7f2f112))

## v0.1.2 (2025-03-08)

### 🐞 Bug Fixes

- lower numpy min version to 1.26.0 ([a9b4928](https://github.com/kmnhan/xarray-lmfit/commit/a9b492847445eac3bfe4a206eb60d06213111dba))

## v0.1.1 (2025-02-27)

### 🐞 Bug Fixes

- avoid modifying the original dataset in `save_fit` ([a3157c0](https://github.com/kmnhan/xarray-lmfit/commit/a3157c067abc479ab56db3e2bbe07d21005912ea))

### ♻️ Code Refactor

- organize imports ([b06251b](https://github.com/kmnhan/xarray-lmfit/commit/b06251ba96f9ac10abbc7b4ad14b649e9a8c88ed))
