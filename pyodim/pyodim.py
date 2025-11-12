"""
Natively reading ODIM H5 radar files in Python.

@title: pyodim
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Bureau of Meteorology and Monash University.
@creation: 21/01/2020
@date: 30/05/2025

.. autosummary::
    :toctree: generated/

    buffer
    cartesian_to_geographic
    check_nyquist
    coord_from_metadata
    field_metadata
    generate_timestamp
    get_dataset_metadata
    get_root_metadata
    radar_coordinates_to_xyz
    read_odim_slice
    read_odim
"""

import warnings
import datetime
import traceback
from typing import Dict, List, Tuple

import dask
import h5py
import pyproj
import pandas as pd
import numpy as np
import xarray as xr


def buffer(func):
    """
    Decorator to catch and kill error message. Almost want to name the function
    dont_fail.
    """

    def wrapper(*args, **kwargs):
        try:
            rslt = func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            rslt = None
        return rslt

    return wrapper


def cartesian_to_geographic(x: np.ndarray, y: np.ndarray, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform cartesian coordinates to lat/lon using the Azimuth Equidistant
    projection.

    Parameters:
    ===========
    x: ndarray
        x-axis cartesian coordinates.
    y: ndarray
        y-axis cartesian coordinates. Same dimension as x
    lon0: float
        Radar site longitude.
    lat0: float
        Radar site latitude.

    Returns:
    lon: ndarray
        Longitude of each gate.
    lat: ndarray
        Latitude of each gate.
    """
    georef = pyproj.Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +ellps=WGS84")
    lon, lat = georef(x, y, inverse=True)
    lon = lon.astype(np.float32)
    lat = lat.astype(np.float32)
    return lon, lat


@buffer
def check_nyquist(dset: xr.Dataset) -> None:
    """
    This is a sanity check to ensure that the Nyquist velocity is consistent
    with the PRF and wavelength attributes in the dataset.

    Parameters:
    ===========
    dset: xarray.Dataset
        Dataset containing the attributes 'wavelength', 'highprf', and 'NI'.

    Raises:
    =======
    AssertionError: If the Nyquist velocity is not consistent with the PRF.
    """
    wavelength = dset.attrs["wavelength"]
    prf = dset.attrs["highprf"]
    nyquist = dset.attrs["NI"]
    ny_int = 1e-2 * prf * wavelength / 4

    assert np.abs(nyquist - ny_int) < 0.5, "Nyquist not consistent with PRF"


def coord_from_metadata(metadata: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the radar coordinates from the ODIM H5 metadata specification.

    Parameter:
    ==========
    metadata: dict()
        Metadata dictionnary containing the specific ODIM H5 keys: astart,
        nrays, nbins, rstart, rscale, elangle.

    Returns:
    ========
    r: ndarray<nbins>
        Sweep range
    azimuth: ndarray<nrays>
        Sweep azimuth
    elev: float
        Sweep elevation
    """
    da = 360 / metadata["nrays"]
    azimuth = np.linspace(metadata["astart"] + da / 2, 360 - da, metadata["nrays"], dtype=np.float32)

    # rstart is in KM !!! STUPID.
    rstart_center = 1e3 * metadata["rstart"] + metadata["rscale"] / 2
    r = np.arange(
        rstart_center, rstart_center + metadata["nbins"] * metadata["rscale"], metadata["rscale"], dtype=np.float32
    )

    elev = np.array([metadata["elangle"]], dtype=np.float32)
    return r, azimuth, elev


def field_metadata(quantity_name: str) -> Dict:
    """
    Populate metadata for common fields using Py-ART get_metadata() function.
    (Optionnal).

    Parameter:
    ==========
    quantity_name: str
        ODIM H5 quantity attribute name.

    Returns:
    ========
    attrs: dict()
        Metadata dictionnary.
    """
    metadata = {
        "TH": {"units": "dBZ", "standard_name": "equivalent_reflectivity_factor", "long_name": "Total power"},
        "TV": {"units": "dBZ", "standard_name": "equivalent_reflectivity_factor", "long_name": "Total power"},
        "DBZH": {"units": "dBZ", "standard_name": "equivalent_reflectivity_factor", "long_name": "Reflectivity"},
        "DBZH_CLEAN": {"units": "dBZ", "standard_name": "equivalent_reflectivity_factor", "long_name": "Reflectivity"},
        "DBZV": {"units": "dBZ", "standard_name": "equivalent_reflectivity_factor", "long_name": "Reflectivity"},
        "ZDR": {
            "units": "dB",
            "standard_name": "log_differential_reflectivity_hv",
            "long_name": "Differential reflectivity",
        },
        "RHOHV": {
            "units": "ratio",
            "standard_name": "cross_correlation_ratio_hv",
            "long_name": "Cross correlation ratio (RHOHV)",
            "valid_max": 1.0,
            "valid_min": 0.0,
        },
        "LDR": {
            "units": "dB",
            "standard_name": "log_linear_depolarization_ratio_hv",
            "long_name": "Linear depolarization ratio",
        },
        "PHIDP": {
            "units": "degrees",
            "standard_name": "differential_phase_hv",
            "long_name": "Differential phase (PhiDP)",
            "valid_max": 180.0,
            "valid_min": -180.0,
        },
        "KDP": {
            "units": "degrees/km",
            "standard_name": "specific_differential_phase_hv",
            "long_name": "Specific differential phase (KDP)",
        },
        "SQI": {
            "units": "ratio",
            "standard_name": "normalized_coherent_power",
            "long_name": "Normalized coherent power",
            "valid_max": 1.0,
            "valid_min": 0.0,
            "comment": "Also know as signal quality index (SQI)",
        },
        "SNR": {"units": "dB", "standard_name": "signal_to_noise_ratio", "long_name": "Signal to noise ratio"},
        "SNRH": {"units": "dB", "standard_name": "signal_to_noise_ratio", "long_name": "Signal to noise ratio"},
        "VRAD": {
            "units": "meters_per_second",
            "standard_name": "radial_velocity",
            "long_name": "Mean dopper velocity",
        },
        "VRADH": {
            "units": "meters_per_second",
            "standard_name": "radial_velocity",
            "long_name": "Mean dopper velocity",
        },
        "VRADDH": {
            "units": "meters_per_second",
            "standard_name": "corrected_radial_velocity",
            "long_name": "Corrected mean doppler velocity",
        },
        "VRADV": {
            "units": "meters_per_second",
            "standard_name": "radial_velocity",
            "long_name": "Mean dopper velocity",
        },
        "WRAD": {
            "units": "meters_per_second",
            "standard_name": "doppler_spectrum_width",
            "long_name": "Doppler spectrum width",
        },
    }

    try:
        attrs = metadata[quantity_name]
    except KeyError:
        return {}

    return attrs


def generate_timestamp(stime: str, etime: str, nrays: int, a1gate: int) -> np.ndarray:
    """
    Generate timestamp for each ray.

    Parameters:
    ===========
    stime: str
        Sweep starting time.
    etime:
        Sweep ending time.
    nrays: int
        Number of rays in sweep.
    a1gate: int
        Azimuth of the ray measured first by the radar.

    Returns:
    ========
    trange: Timestamp<nrays>
        Timestamp for each ray.
    """
    sdtime = datetime.datetime.strptime(stime, "%Y%m%d_%H%M%S")
    edtime = datetime.datetime.strptime(etime, "%Y%m%d_%H%M%S")
    trange = pd.date_range(sdtime, edtime, nrays)

    return np.roll(trange, a1gate)


def get_dataset_metadata(hfile, dataset: str = "dataset1") -> Tuple[Dict, Dict]:
    """
    Get the dataset metadata of the ODIM H5 file.

    Parameters:
    ===========
    hfile: h5py.File
        H5 file identifier.
    dataset: str
        Key of the dataset for which to extract the metadata

    Returns:
    ========
    metadata: dict
        General metadata of the dataset.
    coordinates_metadata: dict
        Coordinates-specific metadata.
    """
    metadata = dict()
    coordinates_metadata = dict()

    # NB: do not try/except KeyError for h5py attrs: it leaks [h5py issue 2350]

    # General metadata
    ds_how = hfile[f"/{dataset}/how"]
    for k in {"NI", "highprf", "product"} & ds_how.attrs.keys():
        metadata[k] = ds_how.attrs[k]

    sdate = hfile[f"/{dataset}/what"].attrs["startdate"].decode("utf-8")
    stime = hfile[f"/{dataset}/what"].attrs["starttime"].decode("utf-8")
    edate = hfile[f"/{dataset}/what"].attrs["enddate"].decode("utf-8")
    etime = hfile[f"/{dataset}/what"].attrs["endtime"].decode("utf-8")
    metadata["start_time"] = f"{sdate}_{stime}"
    metadata["end_time"] = f"{edate}_{etime}"

    # Coordinates:
    if "astart" in ds_how.attrs:
        coordinates_metadata["astart"] = ds_how.attrs["astart"]
    else:
        # Optional coordinates (!).
        coordinates_metadata["astart"] = 0
    coordinates_metadata["a1gate"] = hfile[f"/{dataset}/where"].attrs["a1gate"]
    coordinates_metadata["nrays"] = hfile[f"/{dataset}/where"].attrs["nrays"]

    coordinates_metadata["rstart"] = hfile[f"/{dataset}/where"].attrs["rstart"]
    coordinates_metadata["rscale"] = hfile[f"/{dataset}/where"].attrs["rscale"]
    coordinates_metadata["nbins"] = hfile[f"/{dataset}/where"].attrs["nbins"]

    coordinates_metadata["elangle"] = hfile[f"/{dataset}/where"].attrs["elangle"]

    return metadata, coordinates_metadata


def get_root_metadata(hfile) -> Dict:
    """
    Get the metadata at the root of the ODIM H5 file.

    Parameters:
    ===========
    hfile: h5py.File
        H5 file identifier.

    Returns:
    ========
    rootmetadata: dict
        Metadata at the root of the ODIM H5 file.
    """
    rootmetadata = {}

    # NB: do not try/except KeyError for h5py attrs: it leaks [h5py issue 2350]

    # Root
    rootmetadata["Conventions"] = hfile.attrs["Conventions"].decode("utf-8")

    # Where
    rootmetadata["latitude"] = hfile["/where"].attrs["lat"]
    rootmetadata["longitude"] = hfile["/where"].attrs["lon"]
    rootmetadata["height"] = hfile["/where"].attrs["height"]

    # What
    sdate = hfile["/what"].attrs["date"].decode("utf-8")
    stime = hfile["/what"].attrs["time"].decode("utf-8")
    rootmetadata["date"] = datetime.datetime.strptime(sdate + stime, "%Y%m%d%H%M%S").isoformat()
    for k in {"object", "source", "version"} & hfile["/what"].attrs.keys():
        rootmetadata[k] = hfile["/what"].attrs[k].decode("utf-8")

    # How
    for k in {"beamwH", "beamwV", "rpm", "wavelength"} & hfile["/how"].attrs.keys():
        rootmetadata[k] = hfile["/how"].attrs[k]

    if "copyright" in hfile["/how"].attrs:
        rootmetadata["copyright"] = hfile["/how"].attrs["copyright"].decode("utf-8")

    return rootmetadata


def radar_coordinates_to_xyz(
    r: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform radar coordinates to cartesian coordinates.

    Parameters:
    ===========
    r: ndarray<nbins>
        Sweep range.
    azimuth: ndarray<nrays>
        Sweep azimuth.
    elevation: float
        Sweep elevation.

    Returns:
    ========
    x, y, z: ndarray<nrays, nbins>
        XYZ cartesian coordinates.
    """
    # To proper spherical coordinates.
    theta = np.deg2rad(90 - elevation)
    phi = 450 - azimuth
    phi[phi >= 360] -= 360
    phi = np.deg2rad(phi)

    R, PHI = np.meshgrid(r, phi)
    R = R.astype(np.float32)
    PHI = PHI.astype(np.float32)

    x = R * np.sin(theta) * np.cos(PHI)
    y = R * np.sin(theta) * np.sin(PHI)
    z = R * np.cos(theta)

    return x, y, z


def read_odim_slice(
    odim_file: str,
    nslice: int = 0,
    include_fields: List = [],
    exclude_fields: List = [],
    check_NI: bool = False,
    read_write: bool = False,
) -> xr.Dataset:
    """
    Read into an xarray dataset one sweep of the ODIM file.

    Parameters:
    ===========
    odim_file: str
        ODIM H5 filename.
    nslice: int
        Slice number we want to extract (start indexing at 0).
    include_fields: list
        Specific fields to be exclusively read.
    exclude_fields: list
        Specific fields to be excluded from reading.
    check_NI: bool
        Check NI parameter in ODIM file and compare it to the PRF.
    read_write: open in read-write mode if True

    Returns:
    ========
    dataset: xarray.Dataset
        xarray dataset of one sweep of the ODIM file.
    """
    rw_mode = "r+" if read_write else "r"
    with h5py.File(odim_file, rw_mode) as hfile:
        return read_odim_slice_h5(
            hfile,
            nslice=nslice,
            include_fields=include_fields,
            exclude_fields=exclude_fields,
            check_NI=check_NI,
            read_write=read_write,
        )


def read_odim_slice_h5(
    hfile: h5py.File,
    nslice: int,
    include_fields: List = [],
    exclude_fields: List = [],
    check_NI: bool = False,
    read_write: bool = False,
) -> xr.Dataset:
    """
    Read a single sweep (slice) from an ODIM HDF5 radar file.

    Parameters
    ----------
    hfile : h5py.File
        Open HDF5 file handle pointing to an ODIM-compliant radar volume.
    nslice : int
        Index of the sweep to read (0-based). Must be less than or equal to the number of available sweeps.
    include_fields : list of str, optional
        List of radar fields (quantities) to include. If empty, all fields are read.
    exclude_fields : list of str, optional
        List of radar fields to exclude from reading.
    check_NI : bool, optional
        If True, validates Nyquist velocity consistency with PRF.
    read_write : bool, optional
        Indicates if the file is opened in read-write mode (affects internal handling).

    Returns
    -------
    xr.Dataset
        Radar sweep data including:
        - Radar fields (e.g., reflectivity, velocity)
        - Coordinates: range, azimuth, elevation, x, y, z, longitude, latitude
        - Time dimension
        - Metadata attributes (root and sweep-specific)
    """
    # if nslice == 0:
    #     raise ValueError('Slice numbering start at 1.')
    if type(include_fields) is not list:
        raise TypeError("Argument `include_fields` should be a list")

    # Number of sweep in dataset
    nsweep = len([k for k in hfile["/"].keys() if k.startswith("dataset")])
    assert nslice <= nsweep, f"Wrong slice number asked. Only {nsweep} available."

    # Order datasets by increasing elevation and time
    sweeps = dict()
    for key in hfile["/"].keys():
        if key.startswith("dataset"):
            sweeps[key] = (hfile[f"/{key}/where"].attrs["elangle"], hfile[f"/{key}/what"].attrs["starttime"])

    sorted_keys = sorted(sweeps, key=lambda k: sweeps[k])
    rootkey = sorted_keys[nslice]

    # Retrieve dataset metadata and coordinates metadata.
    metadata, coordinates_metadata = get_dataset_metadata(hfile, rootkey)
    # remember tilt id
    metadata["id"] = rootkey

    dataset = xr.Dataset()
    dataset.attrs = get_root_metadata(hfile)
    dataset.attrs.update(metadata)
    if check_NI:
        try:
            check_nyquist(dataset)
        except AssertionError:
            print("Nyquist not consistent with PRF")
            pass

    for datakey in hfile[f"/{rootkey}"].keys():
        if not (datakey.startswith("data") or datakey.startswith("quality")):
            continue

        gain = hfile[f"/{rootkey}/{datakey}/what"].attrs["gain"]
        nodata = hfile[f"/{rootkey}/{datakey}/what"].attrs["nodata"]
        offset = hfile[f"/{rootkey}/{datakey}/what"].attrs["offset"]
        hqtt = hfile[f"/{rootkey}/{datakey}/what"].attrs["quantity"]
        if isinstance(hqtt, bytes):
            name = hqtt.decode("utf-8")
        elif isinstance(hqtt, str):
            name = hqtt
        else:
            warnings.warn(f"Unknown type {type(hqtt)} for quantity attribute: {hqtt!r}.")
            continue

        # Check if field should be read.
        if len(include_fields) > 0:
            if name not in include_fields:
                continue
        if name in exclude_fields:
            continue

        data_value = hfile[f"/{rootkey}/{datakey}/data"][:].astype(np.float32)
        data_value = gain * np.ma.masked_equal(data_value, nodata) + offset
        dataset = dataset.merge({name: (("azimuth", "range"), data_value)})
        dataset[name].attrs = field_metadata(name)
        # remember dataset id
        dataset[name].attrs["id"] = datakey

    time = generate_timestamp(
        metadata["start_time"], metadata["end_time"], coordinates_metadata["nrays"], coordinates_metadata["a1gate"]
    )
    r, azi, elev = coord_from_metadata(coordinates_metadata)
    x, y, z = radar_coordinates_to_xyz(r, azi, elev)
    longitude, latitude = cartesian_to_geographic(x, y, dataset.attrs["longitude"], dataset.attrs["latitude"])

    dataset = dataset.merge(
        {
            "range": (("range"), r),
            "azimuth": (("azimuth"), azi),
            "elevation": (("elevation"), elev),
            "time": (("time"), time),
            "x": (("azimuth", "range"), x),
            "y": (("azimuth", "range"), y),
            "z": (("azimuth", "range"), z + dataset.attrs["height"]),
            "longitude": (("azimuth", "range"), longitude),
            "latitude": (("azimuth", "range"), latitude),
        }
    )

    return dataset


def read_write_odim(
    odim_file: str,
    lazy_load: bool = True,
    read_write: bool = False,
    **kwargs,
) -> Tuple[List[xr.Dataset], h5py.File]:
    """
    Read one or multiple sweeps from an ODIM HDF5 radar file, optionally lazily, and return both the data and the file handle.

    Parameters
    ----------
    odim_file : str
        Path to the ODIM HDF5 radar file.
    lazy_load : bool, optional
        If True, returns Dask-delayed objects for lazy evaluation. If False, computes immediately.
    read_write : bool, optional
        If True, opens the file in read-write mode ('r+'), otherwise read-only ('r').
    **kwargs
        Additional arguments forwarded to `read_odim_slice_h5`, such as:
        - nslice (int): Specific sweep index to read.
        - include_fields (list of str): Fields to include.
        - exclude_fields (list of str): Fields to exclude.

    Returns
    -------
    tuple
        (radar, hfile)
        radar : list of xr.Dataset or list of dask.delayed
            Radar sweeps ordered by elevation angle.
        hfile : h5py.File
            Open file handle for further inspection or modification.
    """
    rw_mode = "r+" if read_write else "r"
    hfile = h5py.File(odim_file, rw_mode)

    user_sweep = kwargs.get("nslice", None)
    nsweep = len([k for k in hfile["/"].keys() if k.startswith("dataset")])

    radar = []
    if user_sweep is not None:

        kwargs.pop("nslice", None)  # Prevent duplicate argument
        print(f"User asked for sweep #{user_sweep}")
        if user_sweep < 0 or user_sweep >= nsweep:
            raise ValueError(f"sweep index {user_sweep} out of range (0-{nsweep-1})")

        # Only process the requested sweep
        c = dask.delayed(read_odim_slice_h5)(hfile, user_sweep, **kwargs)
        radar.append(c)
    else:
        # Process all sweeps
        for sweep in range(nsweep):
            c = dask.delayed(read_odim_slice_h5)(hfile, sweep, **kwargs)
            radar.append(c)

    if not lazy_load:
        radar = [r.compute() for r in radar]

    return (radar, hfile)


def read_odim(
    odim_file: str,
    lazy_load: bool = True,
    **kwargs,
) -> List[xr.Dataset]:
    """
    Convenience wrapper to read radar sweeps from an ODIM HDF5 file and return them as a list of xarray.Dataset objects.

    Parameters
    ----------
    odim_file : str
        Path to the ODIM HDF5 radar file.
    lazy_load : bool, optional
        If True, returns Dask-delayed objects for lazy evaluation. If False, directly load all the data in memory and returns the list of xarray.
    **kwargs
        Passed to `read_write_odim` (e.g., nslice, include_fields, exclude_fields).

    Returns
    -------
    list of xr.Dataset
        Radar sweeps ordered by elevation angle, each as an xarray.Dataset.
    """
    (radar, _) = read_write_odim(odim_file, lazy_load=lazy_load, **kwargs)
    return radar


def odim_str_type_id(text_bytes):
    """Generate ODIM-conformant h5py string type ID for `text_bytes`."""
    # h5py default string type is STRPAD STR_NULLPAD
    # ODIM spec string type is STRPAD STR_NULLTERM
    type_id = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
    type_id.set_strpad(h5py.h5t.STR_NULLTERM)
    type_id.set_size(len(text_bytes) + 1)
    return type_id


def write_odim_str_attrib(group, attrib_name: str, text: str) -> None:
    """
    Write ODIM-conformant h5py string attribute.
    If the attribute already exists, it will be overwritten.

    Parameters:
    ===========
    group:
        h5py group to which the attribute will be added.
    attrib_name:
        name of the attribute to be added.
    text:
        text to be written as the attribute value.
    """
    if attrib_name in group.attrs:
        del group.attrs[attrib_name]

    group_id = group.id
    text_bytes = text.encode("utf-8")
    type_id = odim_str_type_id(text_bytes)
    space = h5py.h5s.create(h5py.h5s.SCALAR)
    att_id = h5py.h5a.create(group_id, attrib_name.encode("utf-8"), type_id, space)
    text_array = np.array(text_bytes)
    att_id.write(text_array)

    return None


def copy_h5_data(h5_tilt, orig_id: str) -> str:
    """Add a data array to `h5_tilt` by copying data with `orig_id`.
    This function is used to duplicate an existing data array in the HDF5 file
    and return the new data ID.
    The new data ID is generated based on the current number of data arrays
    in the HDF5 file, ensuring that it is unique.

    Parameters:
    ===========
    h5_tilt: h5py.File
        HDF5 Dataset tilt where the data will be copied.
    orig_id: str
        The ID of the original data array to be copied.
    """
    # current data_count
    data_count = len([k for k in h5_tilt.keys() if k.startswith("data")])

    # use data_count for data_id
    data_id = f"data{data_count + 1}"

    # duplicate original
    h5_tilt.copy(orig_id, data_id)

    # return id
    return data_id
