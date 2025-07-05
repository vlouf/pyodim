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
import logging
from typing import Dict, List, Tuple, Optional

import dask
import h5py
import pyproj
import pandas as pd
import numpy as np
import xarray as xr

from .exceptions import (
    PyOdimError,
    MetadataError,
    CoordinateError,
    NyquistConsistencyError,
    H5FileError,
)
from .validation import (
    validate_file_path,
    validate_odim_file_format,
    validate_slice_number,
    validate_field_lists,
    validate_boolean_parameter,
    validate_geographic_coordinates,
)

# Set up logging
logger = logging.getLogger(__name__)


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


def cartesian_to_geographic(
    x: np.ndarray, y: np.ndarray, lon0: float, lat0: float
) -> Tuple[np.ndarray, np.ndarray]:
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
    ========
    lon: ndarray
        Longitude of each gate.
    lat: ndarray
        Latitude of each gate.

    Raises:
    =======
    CoordinateError: If coordinate transformation fails.
    TypeError: If input arrays are not numpy arrays.
    ValueError: If input coordinates are invalid.
    """
    # Validate inputs
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("x and y must be numpy arrays")

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")

    # Validate geographic coordinates
    lon0, lat0 = validate_geographic_coordinates(lon0, lat0)

    try:
        georef = pyproj.Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +ellps=WGS84")
        lon, lat = georef(x, y, inverse=True)
        lon = lon.astype(np.float32)
        lat = lat.astype(np.float32)
        return lon, lat
    except Exception as e:
        raise CoordinateError(f"Failed to transform coordinates: {e}")


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
    NyquistConsistencyError: If the Nyquist velocity is not consistent with the PRF.
    MetadataError: If required attributes are missing.
    """
    required_attrs = ["wavelength", "highprf", "NI"]
    missing_attrs = [attr for attr in required_attrs if attr not in dset.attrs]

    if missing_attrs:
        raise MetadataError(f"Missing required attributes for Nyquist check: {missing_attrs}")

    try:
        wavelength = float(dset.attrs["wavelength"])
        prf = float(dset.attrs["highprf"])
        nyquist = float(dset.attrs["NI"])

        ny_int = 1e-2 * prf * wavelength / 4

        if np.abs(nyquist - ny_int) >= 0.5:
            raise NyquistConsistencyError(
                f"Nyquist velocity ({nyquist}) not consistent with PRF. "
                f"Expected: {ny_int:.2f}, got: {nyquist:.2f}"
            )

    except (ValueError, TypeError) as e:
        raise MetadataError(f"Invalid attribute values for Nyquist check: {e}")


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
    azimuth = np.linspace(
        metadata["astart"] + da / 2, 360 - da, metadata["nrays"], dtype=np.float32
    )

    # rstart is in KM !!! STUPID.
    rstart_center = 1e3 * metadata["rstart"] + metadata["rscale"] / 2
    r = np.arange(
        rstart_center,
        rstart_center + metadata["nbins"] * metadata["rscale"],
        metadata["rscale"],
        dtype=np.float32,
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
        "TH": {
            "units": "dBZ",
            "standard_name": "equivalent_reflectivity_factor",
            "long_name": "Total power",
        },
        "TV": {
            "units": "dBZ",
            "standard_name": "equivalent_reflectivity_factor",
            "long_name": "Total power",
        },
        "DBZH": {
            "units": "dBZ",
            "standard_name": "equivalent_reflectivity_factor",
            "long_name": "Reflectivity",
        },
        "DBZH_CLEAN": {
            "units": "dBZ",
            "standard_name": "equivalent_reflectivity_factor",
            "long_name": "Reflectivity",
        },
        "DBZV": {
            "units": "dBZ",
            "standard_name": "equivalent_reflectivity_factor",
            "long_name": "Reflectivity",
        },
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
        "SNR": {
            "units": "dB",
            "standard_name": "signal_to_noise_ratio",
            "long_name": "Signal to noise ratio",
        },
        "SNRH": {
            "units": "dB",
            "standard_name": "signal_to_noise_ratio",
            "long_name": "Signal to noise ratio",
        },
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
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
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
    include_fields: list, optional
        Specific fields to be exclusively read.
    exclude_fields: list, optional
        Specific fields to be excluded from reading.
    check_NI: bool
        Check NI parameter in ODIM file and compare it to the PRF.
    read_write: bool
        Open in read-write mode if True.

    Returns:
    ========
    dataset: xarray.Dataset
        xarray dataset of one sweep of the ODIM file.

    Raises:
    =======
    FileNotFoundError: If the ODIM file doesn't exist.
    InvalidFileFormatError: If the file is not a valid ODIM H5 format.
    InvalidSliceError: If the slice number is invalid.
    H5FileError: If there's an error accessing the H5 file.
    """
    # Validate inputs
    odim_file = validate_file_path(odim_file)
    validate_odim_file_format(odim_file)
    include_fields, exclude_fields = validate_field_lists(include_fields, exclude_fields)
    check_NI = validate_boolean_parameter(check_NI, "check_NI")
    read_write = validate_boolean_parameter(read_write, "read_write")

    rw_mode = "r+" if read_write else "r"

    try:
        with h5py.File(odim_file, rw_mode) as hfile:
            # Validate slice number
            nsweep = len([k for k in hfile["/"].keys() if k.startswith("dataset")])
            nslice = validate_slice_number(nslice, nsweep)

            return read_odim_slice_h5(
                hfile,
                nslice=nslice,
                include_fields=include_fields,
                exclude_fields=exclude_fields,
                check_NI=check_NI,
                read_write=read_write,
            )
    except (OSError, IOError) as e:
        raise H5FileError(f"Cannot open H5 file {odim_file}: {e}")
    except Exception as e:
        if isinstance(e, PyOdimError):
            raise
        raise H5FileError(f"Unexpected error reading ODIM file: {e}")


def read_odim_slice_h5(
    hfile: h5py.File,
    nslice: int = 0,
    include_fields: List = [],
    exclude_fields: List = [],
    check_NI: bool = False,
    read_write: bool = False,
) -> xr.Dataset:
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
            sweeps[key] = (
                hfile[f"/{key}/where"].attrs["elangle"],
                hfile[f"/{key}/what"].attrs["starttime"],
            )

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
        metadata["start_time"],
        metadata["end_time"],
        coordinates_metadata["nrays"],
        coordinates_metadata["a1gate"],
    )
    r, azi, elev = coord_from_metadata(coordinates_metadata)
    x, y, z = radar_coordinates_to_xyz(r, azi, elev)
    longitude, latitude = cartesian_to_geographic(
        x, y, dataset.attrs["longitude"], dataset.attrs["latitude"]
    )

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
    """Read an ODIM H5 file and return h5py handle.

    @param read_write: open in read-write mode if True.
    @see read_odim().
    """
    rw_mode = "r+" if read_write else "r"
    hfile = h5py.File(odim_file, rw_mode)

    nsweep = len([k for k in hfile["/"].keys() if k.startswith("dataset")])

    radar = []
    for sweep in range(0, nsweep):
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
    Read an ODIM H5 file.

    Parameters:
    ===========
    odim_file: str
        ODIM H5 filename.
    lazy_load: bool
        Lazily load the data if true, read and load in memory the entire dataset
        if false.
    include_fields: list, optional
        Specific fields to be exclusively read.
    exclude_fields: list, optional
        Specific fields to be excluded from reading.
    check_NI: bool, optional
        Check NI parameter in ODIM file and compare it to the PRF.
    read_write: bool, optional
        Open in read-write mode if True.

    Returns:
    ========
    radar: list
        List of xarray datasets, each item in a the list is one sweep of the
        radar data (ordered from lowest elevation scan to highest).

    Raises:
    =======
    FileNotFoundError: If the ODIM file doesn't exist.
    InvalidFileFormatError: If the file is not a valid ODIM H5 format.
    H5FileError: If there's an error accessing the H5 file.

    Examples:
    =========
    >>> # Read all sweeps from an ODIM file
    >>> datasets = read_odim("radar_file.h5")
    >>>
    >>> # Read only reflectivity data
    >>> datasets = read_odim("radar_file.h5", include_fields=["DBZH"])
    >>>
    >>> # Load data immediately (not lazy)
    >>> datasets = read_odim("radar_file.h5", lazy_load=False)
    """
    # Validate inputs
    odim_file = validate_file_path(odim_file)
    validate_odim_file_format(odim_file)
    lazy_load = validate_boolean_parameter(lazy_load, "lazy_load")

    try:
        (radar, _) = read_write_odim(odim_file, lazy_load=lazy_load, **kwargs)
        return radar
    except Exception as e:
        if isinstance(e, PyOdimError):
            raise
        logger.error(f"Unexpected error reading ODIM file {odim_file}: {e}")
        raise H5FileError(f"Failed to read ODIM file: {e}")


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
