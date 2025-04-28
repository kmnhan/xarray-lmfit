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
