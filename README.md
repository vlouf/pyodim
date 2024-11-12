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

It requires the following packages: `h5py pyproj pandas numpy xarray dask`.

## Usage

The main entry point for pyodim is the read_odim function, which reads a sweep from an ODIM H5 file and outputs an xarray dataset.

### Example

```python
from pyodim import read_odim

# Read an ODIM H5 file
dataset = read_odim("radar_file.h5", nslice=0)
print(dataset)
```

`read_odim` takes the following parameters:
- `odim_file` (str): Path to the ODIM H5 file.
- `nslice` (int, optional): Sweep number to read (default is 0).
- `include_fields` (List, optional): Fields to read.
- `exclude_fields` (List, optional): Fields to exclude.
- `check_NI` (bool, optional): Check Nyquist parameter consistency (default is False).
- `read_write` (bool, optional): Open in read-write mode if True.

Feel free to contribute to pyodim by submitting issues or pull requests.