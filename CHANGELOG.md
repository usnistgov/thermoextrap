# Changelog

Changelog for `thermoextrap`

## Unreleased

See the fragment files in
[changelog.d](https://github.com/usnistgov/thermoextrap)

<!-- scriv-insert-here -->

## v0.3.0 — 2023-05-03

### Changed

- New linters via pre-commit
- Development env now handled by tox

- Moved `modesl, data, idealgas` from `thermoextrap.core` to `thermoextrap`.
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
