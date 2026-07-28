# pyodim

`pyodim` is a Python library for reading ODIM H5 radar files, transforming them into xarray datasets with geographic coordinates. This library is designed for users needing direct access to ODIM H5 files, providing tools to read and process radar data.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Deprecations](#deprecations)

## Overview
The `pyodim` library provides essential functions for handling ODIM H5 radar data. It reads radar sweeps and converts them into xarray datasets, handling various metadata and radar coordinates transformations. The main function, `read_odim`, enables easy access to radar data in a format compatible with Python's data analysis ecosystem.

## Installation

`pyodim` is available on PyPI:
```bash
pip install pyodim
```

It requires the following packages: `h5py pyproj pandas numpy xarray dask`.

## Usage

The main entry point for pyodim is `read_odim`, which reads one or more sweeps
from an ODIM H5 file and returns xarray datasets.

### Read sweeps (default)

```python
from pyodim import read_odim

# Read all sweeps lazily (default backend='dask')
sweeps = read_odim("radar_file.h5")

# Compute the first sweep when needed
first = sweeps[0].compute()
print(first)
```

### Read one sweep eagerly

```python
from pyodim import read_odim

# Read one sweep immediately as an xarray.Dataset
sweeps = read_odim("radar_file.h5", nslice=0, backend="numpy")
print(sweeps[0])
```

### Keep the file handle open (edit workflows)

```python
from pyodim import read_odim

# Get datasets + open h5py handle
sweeps, hfile = read_odim(
    "radar_file.h5",
    backend="numpy",
    mode="r+",
    return_handle=True,
)

try:
    ds0 = sweeps[0]
    # ... update content through hfile as needed ...
finally:
    hfile.close()
```

`read_odim` key parameters:
- `odim_file` (str): Path to the ODIM H5 file.
- `nslice` (int, optional): Sweep index to read; if omitted, reads all sweeps.
- `backend` (str, optional): `"dask"` (lazy) or `"numpy"` (eager).
- `compute` (bool, optional): If `backend="dask"`, compute before returning.
- `mode` (str, optional): HDF5 mode (`"r"`, `"r+"`, etc.).
- `return_handle` (bool, optional): Return `(sweeps, hfile)` if `True`.
- `include_fields` (List[str], optional): Fields to include.
- `exclude_fields` (List[str], optional): Fields to exclude.
- `check_nyq` (bool, optional): Check Nyquist parameter consistency.
- `use_dask_arrays` (bool, optional): Dask-backed field arrays (requires `return_handle=True` and `backend="numpy"`).

## Deprecations

`read_write_odim` is deprecated in favor of:

```python
read_odim(..., return_handle=True, mode="r"|"r+")
```

See [CHANGELOG.md](CHANGELOG.md) for the deprecation timeline.

Feel free to contribute to pyodim by submitting issues or pull requests.