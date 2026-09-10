"""
Reading ODIM H5 sweeps into xarray Datasets: `read_sweep` and `read_odim`.
"""

import os
import warnings
from typing import IO, Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import xarray as xr

from .decode import decode_field
from .georef import coord_from_metadata, generate_timestamp, georeference, radar_coordinates_to_xyz
from .metadata import (
    _as_str,
    _clean_attrs,
    _sorted_sweep_keys,
    check_nyquist,
    field_metadata,
    get_dataset_metadata,
    get_root_metadata,
)

# `dask` (read_odim(lazy=True)) is optional and imported inside the function that needs it.


def read_sweep(
    source: Union[str, "os.PathLike[str]", IO[bytes], h5py.File],
    sweep: Union[int, str],
    *,
    mode: str = "r",
    include_fields: Optional[Sequence[str]] = None,
    exclude_fields: Optional[Sequence[str]] = None,
    check_nyq: bool = False,
    max_field_elements: Optional[int] = 50_000_000,
    mask_undetect: bool = True,
    georef: bool = False,
    _root_attrs: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    """
    Read one sweep of an ODIM HDF5 radar file into an xarray.Dataset.

    Parameters
    ----------
    source : str, path-like, binary file-like or h5py.File
        Path or binary file object (opened with `mode` and closed on return) or an
        already open HDF5 file handle (left open).
    sweep : int or str
        Sweep to read: an index into the sweeps ordered by increasing elevation
        angle then start time (0-based), or an ODIM group key such as `"dataset3"`.
    mode : str, optional
        HDF5 open mode used when `source` is a path (`"r"` or `"r+"`).
    include_fields : list of str, optional
        Radar fields (ODIM quantities) to read. All fields if empty.
    exclude_fields : list of str, optional
        Radar fields to skip.
    check_nyq : bool, optional
        If True, warn when the Nyquist velocity is inconsistent with the PRF.
    max_field_elements : int, optional
        Refuse to read a field with more elements than this (None disables the guard).
    mask_undetect : bool, optional
        If True (default), gates flagged `undetect` are NaN like `nodata` gates.
        If False they keep their decoded value (`gain * undetect + offset`).
    georef : bool, optional
        If True, add `longitude`/`latitude` variables (see `georeference`).

    Returns
    -------
    xr.Dataset
        - Radar fields (float32, NaN for missing data; `gain`, `offset`, `nodata`,
          `undetect` and the ODIM `id` kept in each field's attrs)
        - Coordinates: range, azimuth, elevation, time
        - Geometry: x, y, z (4/3 Earth model, z above mean sea level), prt
        - Metadata attributes (root and sweep-specific)
    """
    if isinstance(source, (h5py.File, h5py.Group)):
        return _build_sweep(
            source, sweep, _root_attrs, include_fields, exclude_fields, check_nyq, max_field_elements,
            mask_undetect, georef,
        )
    with h5py.File(source, mode) as hfile:
        return _build_sweep(
            hfile, sweep, _root_attrs, include_fields, exclude_fields, check_nyq, max_field_elements,
            mask_undetect, georef,
        )


def _build_sweep(
    hfile, sweep, root_attrs, include_fields, exclude_fields, check_nyq, max_field_elements, mask_undetect, georef
) -> xr.Dataset:
    """Body of `read_sweep` for an open file handle."""
    if include_fields is None:
        include_fields = []
    if exclude_fields is None:
        exclude_fields = []
    if not isinstance(include_fields, (list, tuple, set)):
        raise TypeError("Argument `include_fields` should be a sequence of field names")
    if not isinstance(exclude_fields, (list, tuple, set)):
        raise TypeError("Argument `exclude_fields` should be a sequence of field names")
    include_fields_set = set(include_fields)
    exclude_fields_set = set(exclude_fields)

    if isinstance(sweep, str):
        rootkey = sweep
        if not (rootkey.startswith("dataset") and rootkey in hfile):
            raise KeyError(f"No sweep group '{rootkey}' in file.")
    else:
        sweep_keys = _sorted_sweep_keys(hfile)
        if sweep < 0 or sweep >= len(sweep_keys):
            raise ValueError(f"sweep index {sweep} out of range (0-{len(sweep_keys) - 1})")
        rootkey = sweep_keys[sweep]

    # Retrieve dataset metadata and coordinates metadata.
    metadata, coordinates_metadata = get_dataset_metadata(hfile, rootkey)
    metadata["id"] = rootkey  # Remember sweep id
    prt = metadata.pop("prt", None)

    dataset_attrs = dict(get_root_metadata(hfile) if root_attrs is None else root_attrs)
    dataset_attrs.update(metadata)
    dataset_attrs = _clean_attrs(dataset_attrs)

    if check_nyq:
        try:
            check_nyquist(dataset_attrs)
        except ValueError as err:
            warnings.warn(str(err), UserWarning)

    nrays = coordinates_metadata["nrays"]
    nbins = coordinates_metadata["nbins"]
    expected_shape = (nrays, nbins)

    field_data = {}
    sweep_group = hfile[f"/{rootkey}"]
    for datakey in sweep_group.keys():
        if not (datakey.startswith("data") or datakey.startswith("quality")):
            continue

        field_group = sweep_group[datakey]
        what_attrs = dict(field_group["what"].attrs) if "what" in field_group else {}
        if "quantity" in what_attrs:
            name = _as_str(what_attrs["quantity"])
        elif datakey.startswith("quality") and "how" in field_group and "task" in field_group["how"].attrs:
            name = _as_str(field_group["how"].attrs["task"])  # ODIM quality fields are identified by how/task
        else:
            warnings.warn(f"No quantity attribute found for {rootkey}/{datakey}: skipped.", UserWarning)
            continue

        # Check if field should be read.
        if len(include_fields_set) > 0 and name not in include_fields_set:
            continue
        if name in exclude_fields_set:
            continue

        gain = float(what_attrs.get("gain", 1.0))
        offset = float(what_attrs.get("offset", 0.0))
        nodata = what_attrs.get("nodata")
        undetect = what_attrs.get("undetect")
        nodata = None if nodata is None else float(nodata)
        undetect = None if undetect is None else float(undetect)

        h5_data = field_group["data"]
        if max_field_elements is not None and h5_data.size > max_field_elements:
            raise ValueError(
                f"Field '{name}' has {h5_data.size} elements, above max_field_elements={max_field_elements}."
            )

        if h5_data.ndim != 2:
            raise ValueError(f"Field '{name}' has ndim={h5_data.ndim}; expected 2 dimensions (azimuth, range).")

        if h5_data.shape != expected_shape:
            raise ValueError(f"Field '{name}' shape {h5_data.shape} does not match expected {expected_shape}.")

        data_value = decode_field(h5_data[()], gain, offset, nodata, undetect, mask_undetect)

        if name in field_data:
            warnings.warn(
                f"Duplicate field '{name}' found in sweep. Using last occurrence. "
                "This indicates a potential issue with the ODIM file.",
                UserWarning,
            )

        attrs = field_metadata(name)
        attrs.update({"id": datakey, "gain": gain, "offset": offset})
        if nodata is not None:
            attrs["nodata"] = nodata
        if undetect is not None:
            attrs["undetect"] = undetect
        field_data[name] = (("azimuth", "range"), data_value, attrs)

    time = generate_timestamp(metadata["start_time"], metadata["end_time"], nrays, coordinates_metadata["a1gate"])
    r, azi, elev = coord_from_metadata(coordinates_metadata)
    x, y, z = radar_coordinates_to_xyz(r, azi, elev)
    z = z + np.float32(dataset_attrs["height"])

    field_data.update(
        {
            "x": (("azimuth", "range"), x, {"units": "m", "long_name": "Distance east of the radar"}),
            "y": (("azimuth", "range"), y, {"units": "m", "long_name": "Distance north of the radar"}),
            "z": (("azimuth", "range"), z, {"units": "m", "long_name": "Height above mean sea level"}),
        }
    )
    # Pulse repetition time per ray: from the (legacy dual-PRF) metadata when available,
    # otherwise constant 1/highprf, so that every sweep of a volume has the same variables.
    if prt is None and dataset_attrs.get("highprf"):
        prt = np.full(nrays, 1.0 / float(dataset_attrs["highprf"]))
    if prt is not None:
        prt = np.asarray(prt, dtype=np.float64)
        if prt.size == 1:
            prt = np.full(nrays, prt.ravel()[0])
        if prt.ndim == 1 and prt.size == nrays:
            field_data["prt"] = (("azimuth",), prt.astype(np.float32), {"units": "s", "long_name": "Pulse repetition time"})

    coords = {
        "range": (("range",), r, {"units": "m", "long_name": "Range from the radar to the gate centre"}),
        "azimuth": (("azimuth",), azi, {"units": "degrees", "long_name": "Azimuth of the ray centre"}),
        "elevation": (("elevation",), elev, {"units": "degrees", "long_name": "Elevation angle"}),
        "time": (("time",), time),
    }
    dataset = xr.Dataset(data_vars=field_data, coords=coords, attrs=dataset_attrs)

    if georef:
        dataset = georeference(dataset)

    return dataset


def read_odim(
    odim_file: Union[str, "os.PathLike[str]", IO[bytes]],
    *,
    sweeps: Union[None, int, Sequence[int]] = None,
    lazy: bool = False,
    mode: str = "r",
    return_handle: bool = False,
    **options,
) -> Union[List[xr.Dataset], Tuple[List[xr.Dataset], h5py.File]]:
    """
    Read the sweeps of an ODIM HDF5 radar file as a list of xarray.Dataset,
    ordered by increasing elevation angle.

    Parameters
    ----------
    odim_file : str, path-like or binary file-like
        Path to the ODIM HDF5 radar file, or an open binary file object such as
        ``io.BytesIO(zip_member_bytes)`` (anything ``h5py.File`` accepts; not with `lazy=True`).
    sweeps : int or sequence of int, optional
        Sweep index (or indices) to read, in elevation order. All sweeps if omitted.
    lazy : bool, optional
        If True, return `dask.delayed` objects (one per sweep, each opening the
        file on its own when computed) instead of datasets. Requires `dask`.
    mode : str, optional
        HDF5 open mode: `"r"` (default) or `"r+"` for in-place edits with `return_handle=True`.
    return_handle : bool, optional
        If True, return `(sweeps, hfile)` with the file left open; the caller
        must close it. Not available with `lazy=True`.
    **options
        Forwarded to `read_sweep`: `include_fields`, `exclude_fields`, `check_nyq`,
        `max_field_elements`, `mask_undetect`, `georef`. Unknown names raise `TypeError`.

    Returns
    -------
    list of xr.Dataset (or of dask.delayed when `lazy=True`)
    tuple (list, h5py.File) when `return_handle=True`
    """
    if lazy and return_handle:
        raise ValueError("return_handle=True is not available with lazy=True (each delayed sweep opens its own handle).")
    if lazy and mode != "r":
        raise ValueError("lazy=True requires mode='r' (HDF5 handles cannot be shared for writing).")

    hfile = h5py.File(odim_file, mode)
    try:
        sweep_keys = _sorted_sweep_keys(hfile)
        nsweep = len(sweep_keys)
        if sweeps is None:
            indices = list(range(nsweep))
        else:
            indices = [sweeps] if isinstance(sweeps, (int, np.integer)) else list(sweeps)
            for i in indices:
                if i < 0 or i >= nsweep:
                    raise ValueError(f"sweep index {i} out of range (0-{nsweep - 1})")

        if lazy:
            from dask import delayed  # optional dependency, imported on use only

            hfile.close()
            return [delayed(read_sweep)(odim_file, sweep_keys[i], **options) for i in indices]

        root_attrs = get_root_metadata(hfile)
        radar = [read_sweep(hfile, sweep_keys[i], _root_attrs=root_attrs, **options) for i in indices]
    except Exception:
        hfile.close()
        raise

    if return_handle:
        return radar, hfile
    hfile.close()
    return radar
