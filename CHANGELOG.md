# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Reader API consolidation: `read_odim` is now the canonical public API for both
  standard reads and read/write workflows.
- `read_odim` supports `mode` and `return_handle` to replace separate read/write
  entry points.
- Top-level package exports no longer include `read_write_odim`; use
  `pyodim.pyodim.read_write_odim` only for temporary compatibility during migration.

### Deprecated
- `read_write_odim` is deprecated and emits `DeprecationWarning`.
- Migration path:
  - Old: `read_write_odim(path, read_write=False, ...)`
  - New: `read_odim(path, mode="r", return_handle=True, ...)`
  - Old: `read_write_odim(path, read_write=True, ...)`
  - New: `read_odim(path, mode="r+", return_handle=True, ...)`

### Planned Removal Timeline
- Next release (`N`): `read_write_odim` remains available with deprecation warning.
- Following release (`N+1`): documentation and examples use `read_odim` only.
- Next major release after `N+1`: `read_write_odim` planned for removal.
