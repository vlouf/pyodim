# Changelog

All notable changes to this project will be documented in this file.

## [v0.7.0] 9 September 2026

### Fixed
- `z` (and the ground distance behind `x`, `y`) now use the 4/3 effective-Earth-radius
  refraction model instead of a flat Earth. At 300 km range and 0.5° elevation the old
  `z` was 5.3 km too low; at 100 km, 0.6 km too low.
- The azimuth grid is uniform for any `astart` (it was compressed whenever
  `astart != -0.5 * 360/nrays`, i.e. for every file without `how/astart`). Per-ray
  `how/startazA`/`stopazA` take precedence when present.
- Fields are `float32` again (NumPy 2 promoted them to `float64`, doubling memory).
- `undetect` gates are `NaN` (they used to decode to a real value, e.g. -31.9 dBZ in
  `DBZH_CLEAN`). Use `mask_undetect=False` for the old behaviour.
- Datasets can be written with `to_netcdf()`: attributes no longer contain bytes,
  `None` or arrays. `prt` is now a per-ray variable `prt(azimuth)` on every sweep.
- ODIM string attributes stored as `str` (variable-length) instead of bytes are accepted.
- `quality*` groups without `what/quantity` are named from `how/task` (ODIM convention).
- `check_nyquist` understands dual-PRF sweeps (`lowprf` or legacy `rapic_UNFOLDING`)
  and skips the check when metadata is incomplete instead of raising `KeyError`.
- Unknown keyword arguments to `read_odim` raise `TypeError` instead of being ignored.

### Changed (breaking)
- The single `pyodim/pyodim.py` module is split into `reader`, `georef`, `metadata`,
  `decode` and `writer`. The public API is unchanged and re-exported from `pyodim`;
  `from pyodim.pyodim import ...` becomes e.g. `from pyodim.georef import ...`.
- `read_odim` is eager: it returns a list of `xarray.Dataset` (it used to return
  `dask.delayed` objects). `read_odim(..., lazy=True)` returns delayed sweeps.
- `read_sweep(source, sweep)` replaces `read_odim_slice_h5`: `source` is a path or an
  open `h5py.File`, `sweep` an index in elevation order or a `"datasetN"` key.
- `nslice` is replaced by `sweeps` (int or list of int); `read_odim` arguments are keyword-only.
- Removed: `read_write_odim`, `lazy_load`, `backend`, `compute`, `use_dask_arrays`,
  `field_chunks`, `read_odim_slice_h5`.
- `longitude`/`latitude` are no longer computed at read time. Call
  `pyodim.georeference(ds)` or `read_odim(..., georef=True)`. The pyproj aeqd inverse
  was ~70 % of the read time; the new pure-numpy WGS84 geodesic is 3x faster and
  exact to < 1 m, and `georeference(..., method="pyproj")` remains available.
- Dependencies are now `numpy`, `h5py`, `xarray` only. `dask` and `pyproj` are optional
  extras (`pip install pyodim[dask]`, `pyodim[pyproj]`, `pyodim[all]`).

### Performance
- Full 13-sweep test volume: ~3.2 s -> ~0.4 s, 349 MB -> 175 MB retained;
  `import pyodim` ~1.0 s / 130 MB -> ~0.5 s / 90 MB.
- One HDF5 open, one sweep sort and one root-metadata read per volume; fields are
  decoded through a float32 lookup table with no `numpy.ma` intermediate.

## [v0.6.2] 28 July 2026

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
- Next release (`0.6.2`): `read_write_odim` remains available with deprecation warning.
- Following release (`0.6.X`): documentation and examples use `read_odim` only.
- Next major release after `0.7`: `read_write_odim` planned for removal.
