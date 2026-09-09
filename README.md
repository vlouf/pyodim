# pyodim

`pyodim` is a Python library for reading ODIM H5 radar files, transforming them into xarray datasets with geographic coordinates. This library is designed for users needing direct access to ODIM H5 files, providing tools to read and process radar data.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

## Overview
The `pyodim` library provides essential functions for handling ODIM H5 radar data. It reads radar sweeps and converts them into xarray datasets, handling various metadata and radar coordinates transformations. The main function, `read_odim`, enables easy access to radar data in a format compatible with Python's data analysis ecosystem.

## Installation

`pyodim` is available on PyPI:
```bash
pip install pyodim
```

It requires only `numpy`, `h5py` and `xarray`. Optional extras: `pyodim[dask]` for `read_odim(lazy=True)`, `pyodim[pyproj]` for `georeference(method="pyproj")`.

## Usage

`read_odim` reads the sweeps of an ODIM H5 file into a list of `xarray.Dataset`
ordered by increasing elevation; `read_sweep` reads one sweep.

### Read a volume

```python
from pyodim import read_odim

sweeps = read_odim("radar_file.h5")          # list of xarray.Dataset, eager
lowest = read_odim("radar_file.h5", sweeps=0)[0]
some = read_odim("radar_file.h5", sweeps=[0, 3], include_fields=["DBZH", "VRADH"])
```

### Read one sweep

```python
import h5py
from pyodim import read_sweep

ds = read_sweep("radar_file.h5", 0)          # by position in elevation order
with h5py.File("radar_file.h5") as hfile:    # or from an open handle, by ODIM key
    ds = read_sweep(hfile, "dataset3")
```

### Lazy reading with dask (optional)

```python
import dask
from pyodim import read_odim

delayed_sweeps = read_odim("radar_file.h5", lazy=True)   # list of dask.delayed
sweeps = dask.compute(*delayed_sweeps)                    # read in parallel
```

Requires `pip install pyodim[dask]`.

### Keep the file handle open (edit workflows)

```python
from pyodim import read_odim

sweeps, hfile = read_odim("radar_file.h5", mode="r+", return_handle=True)
try:
    ds0 = sweeps[0]
    # ... update content through hfile as needed ...
finally:
    hfile.close()
```

### Geographic coordinates

Per-gate `longitude`/`latitude` are not computed at read time (they were the
most expensive part of a read and most workflows never use them). Add them when
you need them:

```python
from pyodim import read_odim, georeference

ds = read_odim("radar_file.h5", sweeps=0)[0]
ds = georeference(ds)            # pure-numpy WGS84 geodesic, exact to < 1 m
# or in one call:
ds = read_odim("radar_file.h5", sweeps=0, georef=True)[0]
```

Each sweep carries `x`, `y`, `z` (metres east/north of the radar and height
above mean sea level, 4/3-Earth refraction model), `range`, `azimuth`,
`elevation`, per-ray `time` and `prt`. Fields are `float32` with `NaN` for
`nodata` and `undetect` gates (`mask_undetect=False` keeps the decoded
`undetect` value); each field's `gain`, `offset`, `nodata`, `undetect` and
ODIM `id` are kept in its attributes.

### Parameters

`read_odim(odim_file, *, sweeps=None, lazy=False, mode="r", return_handle=False, **options)`

- `sweeps` (int or list of int, optional): sweep index/indices in elevation order; all if omitted.
- `lazy` (bool): return `dask.delayed` objects instead of datasets.
- `mode` (str): HDF5 mode, `"r"` or `"r+"`.
- `return_handle` (bool): return `(sweeps, hfile)` and leave the file open.

`read_sweep(source, sweep, *, mode="r", **options)`

- `source` (path or open `h5py.File`), `sweep` (int index or `"datasetN"` key).

Options accepted by both:

- `include_fields` / `exclude_fields` (list of str): fields to read / skip.
- `check_nyq` (bool): warn when the Nyquist velocity is inconsistent with the PRF.
- `max_field_elements` (int or None): guard against oversized fields (default 50,000,000).
- `mask_undetect` (bool): `NaN` for `undetect` gates (default `True`).
- `georef` (bool): add `longitude`/`latitude` (default `False`, see `georeference`).

Feel free to contribute to pyodim by submitting issues or pull requests.