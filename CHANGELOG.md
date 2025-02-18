<!-- markdownlint-disable MD024 -->
<!-- markdownlint-disable MD013 -->
<!-- prettier-ignore-start -->
# Changelog

Changelog for `thermoextrap`

## Unreleased

[changelog.d]: https://github.com/usnistgov/thermoextrap/tree/main/changelog.d

See the fragment files in [changelog.d]

<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD013 -->

<!-- scriv-insert-here -->

## v0.6.0 — 2025-02-18

### Changed

- Project now setup to use [uv](https://github.com/astral-sh/uv) with lock file.
- Updated code to use latest version of
  [cmomy](https://github.com/usnistgov/cmomy)
- Initial work for adding typing to code.

## v0.5.0 — 2024-03-15

### Removed

- Scaling of GPR inputs (`x_scale_fac` argument in `HeteroscedasticGPR`)
- Left `x_scale_fac` as object attribute with value 1.0 for back-compatibility

### Added

- Support for multidimensional inputs for GPRs
- Testing around basic multiD input GPRs
- Updated `make_rbf_expr` in `active_utils` (old 1D in `make_rbf_expr_old`)
- Updated `DerivativeKernel`, `HetGaussianDeriv`, `HeteroscedasticGPR` in
  `gpr_models`

### Changed

- Updates to match with newer versions of GPflow
- `HetGaussianDeriv` likelihood now accepts `X` (input data) argument for all
  methods
- `HetGaussianDeriv` init now takes `obs_dims` argument instead of `d_order`
- `build_scaled_cov_mat` method now takes `X`, which includes derivative orders
- all mean functions inherit from gpflow.functions.MeanFunction (same behavior)

- Changed structure of the repo to better support some third party tools.
- Moved nox environments from `.nox` to `.nox/{project-name}/envs`. This fixes
  issues with ipykernel giving odd names for locally installed environments.
- Moved repo specific dot files to the `config` directory (e.g.,
  `.noxconfig.toml` to `config/userconfig.toml`). This cleans up the top level
  of the repo.
- added some support for using `nbqa` to run mypy/pyright on notebooks.
- Added ability to bootstrap development environment using pipx. This should
  simplify initial setup. See Contributing for more info.

## v0.4.0 — 2023-06-15

### Added

- Package now available on conda-forge

- Now support python3.11
- Bumped pymbar version to pymbar>=4.0

### Changed

- Switched from tox to nox for testing.

### Deprecated

- No longer support pymbar < 4.0

## v0.3.0 — 2023-05-03

### Changed

- New linters via pre-commit
- Development env now handled by tox

- Moved `models, data, idealgas` from `thermoextrap.core` to `thermoextrap`.
  These were imported at top level anyway. This fixes issues with doing things
  like `from thermoextrap.data import ...`, etc.
- Moved `core._docstrings_` to `docstrings`.
- Now using `cmomy.docstrings` instead of repeating them here.

Full set of changes:
[`v0.2.2...v0.3.0`](https://github.com/usnistgov/thermoextrap/compare/v0.2.2...0.3.0)

## v0.2.2 - 2023-04-05

Full set of changes:
[`v0.2.1...v0.2.2`](https://github.com/usnistgov/thermoextrap/compare/v0.2.1...v0.2.2)

## v0.2.1 - 2023-03-30

Full set of changes:
[`v0.2.0...v0.2.1`](https://github.com/usnistgov/thermoextrap/compare/v0.2.0...v0.2.1)

## v0.2.0 - 2023-03-28

Full set of changes:
[`v0.1.9...v0.2.0`](https://github.com/usnistgov/thermoextrap/compare/v0.1.9...v0.2.0)

## v0.1.9 - 2023-02-15

Full set of changes:
[`v0.1.8...v0.1.9`](https://github.com/usnistgov/thermoextrap/compare/v0.1.8...v0.1.9)

## v0.1.8 - 2023-02-15

Full set of changes:
[`v0.1.7...v0.1.8`](https://github.com/usnistgov/thermoextrap/compare/v0.1.7...v0.1.8)

## v0.1.7 - 2023-02-14
