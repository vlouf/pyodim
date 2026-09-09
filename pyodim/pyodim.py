"""
Natively reading ODIM H5 radar files in Python.

@title: pyodim
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Bureau of Meteorology and Monash University.
@creation: 21/01/2020
@date: 09/09/2026

.. autosummary::
    :toctree: generated/

    antenna_to_ground
    cartesian_to_geographic
    check_nyquist
    coord_from_metadata
    decode_field
    field_metadata
    generate_timestamp
    geodesic_forward
    georeference
    get_dataset_metadata
    get_root_metadata
    radar_coordinates_to_xyz
    read_odim
    read_sweep

Since 0.7, `read_odim` is eager (`lazy=True` for dask-delayed sweeps),
`read_sweep` reads one sweep, and `longitude`/`latitude` are not computed at
read time: call `georeference(dataset)` (or `read_odim(..., georef=True)`).
"""

import datetime
import os
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import xarray as xr

# Only numpy, h5py and xarray are imported at module level. `dask` (read_odim(lazy=True))
# and `pyproj` (georeference(method="pyproj")) are optional and imported inside the
# functions that need them.

EARTH_RADIUS = 6371000.0  # mean Earth radius (m)
REFRACTION_KE = 4.0 / 3.0  # effective Earth radius factor (Doviak & Zrnić)
WGS84_A = 6378137.0  # WGS84 semi-major axis (m)
WGS84_F = 1.0 / 298.257223563  # WGS84 flattening

FIELD_METADATA: Dict[str, Dict[str, Any]] = {
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
    "VRAD": {"units": "meters_per_second", "standard_name": "radial_velocity", "long_name": "Mean dopper velocity"},
    "VRADH": {"units": "meters_per_second", "standard_name": "radial_velocity", "long_name": "Mean dopper velocity"},
    "VRADDH": {
        "units": "meters_per_second",
        "standard_name": "corrected_radial_velocity",
        "long_name": "Corrected mean doppler velocity",
    },
    "VRADV": {"units": "meters_per_second", "standard_name": "radial_velocity", "long_name": "Mean dopper velocity"},
    "WRAD": {
        "units": "meters_per_second",
        "standard_name": "doppler_spectrum_width",
        "long_name": "Doppler spectrum width",
    },
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _as_str(value: Any) -> str:
    """Return an HDF5 string attribute as `str` whether it was stored as bytes or str."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "U"):
        return _as_str(value.item()) if value.size == 1 else value.astype(str).tolist()
    return str(value)


def _as_python(value: Any) -> Any:
    """Convert an HDF5 attribute value into a plain, netCDF-serialisable Python object."""
    if isinstance(value, (bytes, np.bytes_)):
        return _as_str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "U"):
            return _as_str(value)
        return value.item() if value.size == 1 else value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clean_attrs(attrs: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop `None` values and convert bytes/numpy scalars so that `Dataset.to_netcdf()` works."""
    return {k: _as_python(v) for k, v in attrs.items() if v is not None}


def _sorted_sweep_keys(hfile: h5py.File) -> List[str]:
    """Return the `datasetN` keys of the file ordered by (elevation angle, start time)."""
    sweeps = {}
    for key in hfile["/"].keys():
        if key.startswith("dataset"):
            grp = hfile[key]
            sweeps[key] = (float(grp["where"].attrs["elangle"]), _as_str(grp["what"].attrs["starttime"]))
    return sorted(sweeps, key=lambda k: sweeps[k])


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def prt_from_rapic_metadata(metadata: Mapping[str, Any], nrays: int) -> Optional[np.ndarray]:
    """
    Generate PRT value for each ray using the legacy rapic metadata

    Parameters:
    ===========
    metadata: dict
        sweep metadata
    nrays: int
        number of rays in the sweep

    Returns:
    prt: ndarray
        PRT of each gate
    """
    prf_ratio_str = _as_str(metadata["rapic_UNFOLDING"])
    high_prf_loc_str = _as_str(metadata["rapic_HIPRF"])
    # abort if metadata is incomplete
    if prf_ratio_str == "None" or high_prf_loc_str == "None":
        return None
    # calculate prt ratio, high prt and low prt
    ratio_lhs = float(prf_ratio_str[0])
    ratio_rhs = float(prf_ratio_str[2])
    prt_ratio = ratio_rhs / ratio_lhs
    prt_high = 1 / metadata["highprf"]
    prt_low = prt_high * prt_ratio
    # initialise prt array with low prt values
    prt = np.zeros(nrays) + prt_low
    # insert high values
    if high_prf_loc_str == "EVENS":
        prt[1::2] = prt_high
    elif high_prf_loc_str == "ODDS":
        prt[::2] = prt_high

    return prt


def field_metadata(quantity_name: str) -> Dict:
    """
    Populate metadata for common fields using CF-style names (Optionnal).

    Parameter:
    ==========
    quantity_name: str
        ODIM H5 quantity attribute name.

    Returns:
    ========
    attrs: dict()
        Metadata dictionnary (a copy; empty if the quantity is unknown).
    """
    return dict(FIELD_METADATA.get(quantity_name, {}))


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
    metadata: Dict[str, Any] = dict()
    coordinates_metadata: Dict[str, Any] = dict()

    # NB: do not try/except KeyError for h5py attrs: it leaks [h5py issue 2350]
    grp = hfile[f"/{dataset}"]
    how_attrs = dict(grp["how"].attrs) if "how" in grp else {}
    what_attrs = dict(grp["what"].attrs)
    where_attrs = dict(grp["where"].attrs)

    # General metadata
    metadata["prt"] = None  # initialize prt key
    for k in ("NI", "highprf", "lowprf", "product", "prt", "rapic_UNFOLDING", "rapic_HIPRF"):
        if k in how_attrs:
            metadata[k] = how_attrs[k]

    metadata["start_time"] = f"{_as_str(what_attrs['startdate'])}_{_as_str(what_attrs['starttime'])}"
    metadata["end_time"] = f"{_as_str(what_attrs['enddate'])}_{_as_str(what_attrs['endtime'])}"

    # Coordinates:
    coordinates_metadata["astart"] = float(how_attrs.get("astart", 0.0))  # Optional coordinates (!).
    coordinates_metadata["a1gate"] = int(where_attrs["a1gate"])
    nrays = int(where_attrs["nrays"])
    coordinates_metadata["nrays"] = nrays

    rstart = float(where_attrs["rstart"])
    if rstart < 10:  # convert to meters if in km (legacy ODIM files) - units are unreliable in legacy ODIM
        rstart *= 1e3
    coordinates_metadata["rstart"] = rstart
    coordinates_metadata["rscale"] = float(where_attrs["rscale"])
    coordinates_metadata["nbins"] = int(where_attrs["nbins"])
    coordinates_metadata["elangle"] = float(where_attrs["elangle"])

    # Per-ray azimuths (ODIM >= 2.2, optional): used instead of the regular grid when present.
    if "startazA" in how_attrs and "stopazA" in how_attrs:
        startaz = np.asarray(how_attrs["startazA"], dtype=np.float64).ravel()
        stopaz = np.asarray(how_attrs["stopazA"], dtype=np.float64).ravel()
        if startaz.size == nrays and stopaz.size == nrays:
            coordinates_metadata["startazA"] = startaz
            coordinates_metadata["stopazA"] = stopaz

    # generate prt array from rapic metadata (support legacy dual prf metadata)
    if all(k in metadata for k in ("rapic_HIPRF", "rapic_UNFOLDING", "highprf")):
        try:
            metadata["prt"] = prt_from_rapic_metadata(metadata, nrays)
        except Exception as e:
            warnings.warn(
                f"Failed to build PRT array for {dataset} from legacy metadata due to error: {e}", UserWarning
            )

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
    rootmetadata: Dict[str, Any] = {}

    # NB: do not try/except KeyError for h5py attrs: it leaks [h5py issue 2350]
    what_attrs = dict(hfile["/what"].attrs)
    where_attrs = dict(hfile["/where"].attrs)
    how_attrs = dict(hfile["/how"].attrs) if "how" in hfile else {}

    # Root
    rootmetadata["Conventions"] = _as_str(hfile.attrs["Conventions"])

    # Where
    rootmetadata["latitude"] = float(where_attrs["lat"])
    rootmetadata["longitude"] = float(where_attrs["lon"])
    rootmetadata["height"] = float(where_attrs["height"])

    # What
    sdate = _as_str(what_attrs["date"])
    stime = _as_str(what_attrs["time"])
    rootmetadata["date"] = datetime.datetime.strptime(sdate + stime, "%Y%m%d%H%M%S").isoformat()
    for k in ("object", "source", "version"):
        if k in what_attrs:
            rootmetadata[k] = _as_str(what_attrs[k])

    # How
    for k in ("beamwH", "beamwV", "rpm", "wavelength"):
        if k in how_attrs:
            rootmetadata[k] = how_attrs[k]

    if "copyright" in how_attrs:
        rootmetadata["copyright"] = _as_str(how_attrs["copyright"])

    return rootmetadata


def check_nyquist(dset: Union[xr.Dataset, Mapping[str, Any]]) -> None:
    """
    This is a sanity check to ensure that the Nyquist velocity is consistent
    with the PRF and wavelength attributes in the dataset.

    Dual-PRF sweeps are handled: when `lowprf` (or the legacy
    `rapic_UNFOLDING` ratio "N:M") is present, the extended Nyquist velocity
    `lambda/4 * highprf*lowprf/(highprf-lowprf)` is used. The check is skipped
    when `wavelength`, `highprf` or `NI` is missing.

    Parameters:
    ===========
    dset: xarray.Dataset or dict
        Dataset (or attrs mapping) containing 'wavelength' (cm), 'highprf' (Hz), 'NI' (m/s).

    Raises:
    =======
    ValueError: If the Nyquist velocity is not consistent with the PRF.
    """
    attrs = dset.attrs if isinstance(dset, xr.Dataset) else dset
    if any(attrs.get(k) is None for k in ("wavelength", "highprf", "NI")):
        return None
    wavelength = float(attrs["wavelength"])
    prf_high = float(attrs["highprf"])
    nyquist = float(attrs["NI"])

    prf_low = attrs.get("lowprf")
    if prf_low is None and attrs.get("rapic_UNFOLDING") is not None:
        ratio = _as_str(attrs["rapic_UNFOLDING"])
        if ratio != "None" and ":" in ratio:
            lhs, rhs = (float(v) for v in ratio.split(":")[:2])
            prf_low = prf_high * min(lhs, rhs) / max(lhs, rhs)

    if prf_low is not None and float(prf_low) > 0 and float(prf_low) != prf_high:
        prf_low = float(prf_low)
        prf_eff = prf_high * prf_low / abs(prf_high - prf_low)  # dual-PRF extended Nyquist
    else:
        prf_eff = prf_high
    ny_int = 1e-2 * prf_eff * wavelength / 4

    if np.abs(nyquist - ny_int) >= 0.5:
        raise ValueError("Nyquist not consistent with PRF")


# --------------------------------------------------------------------------- #
# Coordinates and geometry
# --------------------------------------------------------------------------- #
def coord_from_metadata(metadata: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the radar coordinates from the ODIM H5 metadata specification.

    Parameter:
    ==========
    metadata: dict()
        Metadata dictionnary containing the specific ODIM H5 keys: astart,
        nrays, nbins, rstart, rscale, elangle (and optionally the per-ray
        startazA/stopazA arrays).

    Returns:
    ========
    r: ndarray<nbins>
        Sweep range (gate centres, m).
    azimuth: ndarray<nrays>
        Sweep azimuth (ray centres, deg).
    elev: float
        Sweep elevation
    """
    nrays = int(metadata["nrays"])
    startaz = metadata.get("startazA")
    stopaz = metadata.get("stopazA")
    if startaz is not None and stopaz is not None:
        # Ray centre = start + half of the (wrapped) angular extent.
        azimuth = (startaz + ((stopaz - startaz) % 360.0) / 2.0) % 360.0
        azimuth = azimuth.astype(np.float32)
    else:
        da = 360.0 / nrays
        azimuth = (metadata["astart"] + da / 2.0 + da * np.arange(nrays)).astype(np.float32)

    rstart_center = metadata["rstart"] + metadata["rscale"] / 2.0
    r = (rstart_center + metadata["rscale"] * np.arange(int(metadata["nbins"]))).astype(np.float32)

    elev = np.array([metadata["elangle"]], dtype=np.float32)
    return r, azimuth, elev


def antenna_to_ground(
    r: np.ndarray, elevation: np.ndarray, ke: float = REFRACTION_KE, earth_radius: float = EARTH_RADIUS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ground distance and height above the radar of a gate, using the
    effective-Earth-radius (4/3) refraction model (Doviak & Zrnić, eq. 2.28).

    Parameters:
    ===========
    r: ndarray
        Slant range (m).
    elevation: ndarray
        Elevation angle (deg). Broadcast against `r`.
    ke: float
        Effective Earth radius factor (default 4/3).
    earth_radius: float
        Earth radius (m).

    Returns:
    ========
    s: ndarray
        Great-circle ground distance from the radar (m).
    z: ndarray
        Height above the radar (m).
    """
    re = ke * earth_radius
    r = np.asarray(r, dtype=np.float64)
    el = np.deg2rad(np.asarray(elevation, dtype=np.float64))
    z = np.sqrt(r**2 + re**2 + 2.0 * r * re * np.sin(el)) - re
    s = re * np.arcsin(r * np.cos(el) / (re + z))
    return s, z


def radar_coordinates_to_xyz(
    r: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform radar coordinates to cartesian coordinates (4/3 Earth model).

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
        XYZ cartesian coordinates (m): x east, y north, z above the radar.
    """
    r = np.asarray(r, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)
    az = np.deg2rad(np.asarray(azimuth, dtype=np.float64))[:, None]

    if elevation.ndim == 1 and elevation.size == len(np.atleast_1d(azimuth)) and elevation.size > 1:
        s, z = antenna_to_ground(r[None, :], elevation[:, None])  # per-ray elevation
    else:
        s, z = antenna_to_ground(r[None, :], np.atleast_1d(elevation).ravel()[0])

    x = s * np.sin(az)
    y = s * np.cos(az)
    z = np.broadcast_to(z, x.shape)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def geodesic_forward(
    lon0: float,
    lat0: float,
    azimuth: np.ndarray,
    distance: np.ndarray,
    a: float = WGS84_A,
    f: float = WGS84_F,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct geodesic problem on the ellipsoid (Vincenty, vectorised): the
    longitude/latitude of the points at a given azimuth and distance from
    (lon0, lat0). `azimuth` and `distance` are broadcast against each other,
    so passing `azimuth[:, None]` (per ray) and `distance[None, :]` (per
    gate) computes the per-ray terms only once.

    Parameters:
    ===========
    lon0, lat0: float
        Origin (deg).
    azimuth: ndarray
        Forward azimuth from the origin (deg, clockwise from North).
    distance: ndarray
        Geodesic distance from the origin (m).
    a, f: float
        Ellipsoid semi-major axis (m) and flattening (default WGS84).

    Returns:
    ========
    lon, lat: ndarray<float64>
        Longitude and latitude (deg) of the end points.
    """
    b = a * (1.0 - f)
    az = np.deg2rad(np.asarray(azimuth, dtype=np.float64))
    s = np.asarray(distance, dtype=np.float64)

    phi1 = np.deg2rad(lat0)
    u1 = np.arctan((1.0 - f) * np.tan(phi1))
    sin_u1, cos_u1 = np.sin(u1), np.cos(u1)
    sin_a1, cos_a1 = np.sin(az), np.cos(az)

    sigma1 = np.arctan2(np.tan(u1), cos_a1)
    sin_alpha = cos_u1 * sin_a1
    cos2_alpha = 1.0 - sin_alpha**2
    u2 = cos2_alpha * (a**2 - b**2) / b**2
    big_a = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
    big_b = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
    big_c = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha))

    sigma0 = s / (b * big_a)
    sigma = sigma0
    for _ in range(max_iter):
        cos_2sm = np.cos(2.0 * sigma1 + sigma)
        sin_s, cos_s = np.sin(sigma), np.cos(sigma)
        delta = big_b * sin_s * (
            cos_2sm
            + big_b / 4.0 * (cos_s * (-1.0 + 2.0 * cos_2sm**2)
                             - big_b / 6.0 * cos_2sm * (-3.0 + 4.0 * sin_s**2) * (-3.0 + 4.0 * cos_2sm**2))
        )
        new_sigma = sigma0 + delta
        converged = np.all(np.abs(new_sigma - sigma) < tol)
        sigma = new_sigma
        if converged:
            break

    cos_2sm = np.cos(2.0 * sigma1 + sigma)
    sin_s, cos_s = np.sin(sigma), np.cos(sigma)
    lat = np.arctan2(
        sin_u1 * cos_s + cos_u1 * sin_s * cos_a1,
        (1.0 - f) * np.hypot(sin_alpha, sin_u1 * sin_s - cos_u1 * cos_s * cos_a1),
    )
    lam = np.arctan2(sin_s * sin_a1, cos_u1 * cos_s - sin_u1 * sin_s * cos_a1)
    big_l = lam - (1.0 - big_c) * f * sin_alpha * (
        sigma + big_c * sin_s * (cos_2sm + big_c * cos_s * (-1.0 + 2.0 * cos_2sm**2))
    )
    lon = np.rad2deg(np.deg2rad(lon0) + big_l)
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, np.rad2deg(lat)


def cartesian_to_geographic(x: np.ndarray, y: np.ndarray, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform cartesian coordinates to lat/lon using the Azimuth Equidistant
    projection on the WGS84 ellipsoid (pure numpy, no pyproj needed).

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
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    azimuth = np.rad2deg(np.arctan2(x, y))
    distance = np.hypot(x, y)
    lon, lat = geodesic_forward(lon0, lat0, azimuth, distance)
    return lon.astype(np.float32), lat.astype(np.float32)


def georeference(dataset: xr.Dataset, method: str = "numpy") -> xr.Dataset:
    """
    Add `longitude` and `latitude` (azimuth, range) to a sweep read by pyodim.

    Parameters:
    ===========
    dataset: xr.Dataset
        Sweep with `range`, `azimuth`, `elevation` coordinates and the
        `longitude`/`latitude` (radar site) attributes.
    method: {"numpy", "pyproj"}
        "numpy" (default): WGS84 geodesic in pure numpy, exact to < 1 m.
        "pyproj": azimuthal equidistant inverse projection with pyproj
        (requires the optional `pyproj` package).

    Returns:
    ========
    dataset: xr.Dataset
        A new dataset with `longitude` and `latitude` variables (float32, deg).
    """
    lon0 = float(dataset.attrs["longitude"])
    lat0 = float(dataset.attrs["latitude"])

    if method == "numpy":
        r = dataset["range"].values
        azimuth = dataset["azimuth"].values
        elevation = np.atleast_1d(dataset["elevation"].values).ravel()[0]
        s, _ = antenna_to_ground(r, elevation)
        lon, lat = geodesic_forward(lon0, lat0, azimuth[:, None], s[None, :])
        lon = lon.astype(np.float32)
        lat = lat.astype(np.float32)
    elif method == "pyproj":
        try:
            import pyproj
        except ImportError as err:
            raise ImportError("georeference(method='pyproj') requires pyproj: pip install pyproj") from err
        georef = pyproj.Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +ellps=WGS84")
        lon, lat = georef(dataset["x"].values.astype(np.float64), dataset["y"].values.astype(np.float64), inverse=True)
        lon = lon.astype(np.float32)
        lat = lat.astype(np.float32)
    else:
        raise ValueError("Invalid method. Expected one of: 'numpy', 'pyproj'.")

    return dataset.assign(
        longitude=(("azimuth", "range"), lon, {"units": "degrees_east", "standard_name": "longitude"}),
        latitude=(("azimuth", "range"), lat, {"units": "degrees_north", "standard_name": "latitude"}),
    )


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
    trange: datetime64[ns]<nrays>
        Timestamp for each ray.
    """
    sdtime = np.datetime64(datetime.datetime.strptime(stime, "%Y%m%d_%H%M%S"), "ns")
    edtime = np.datetime64(datetime.datetime.strptime(etime, "%Y%m%d_%H%M%S"), "ns")
    span_ns = (edtime - sdtime).astype(np.int64)
    if nrays > 1:
        offsets = np.round(span_ns * np.arange(nrays) / (nrays - 1)).astype(np.int64)
    else:
        offsets = np.zeros(nrays, dtype=np.int64)
    trange = sdtime + offsets.astype("timedelta64[ns]")

    return np.roll(trange, a1gate)


# --------------------------------------------------------------------------- #
# Field decoding
# --------------------------------------------------------------------------- #
def decode_field(
    raw: np.ndarray,
    gain: float,
    offset: float,
    nodata: Optional[float],
    undetect: Optional[float] = None,
    mask_undetect: bool = True,
) -> np.ndarray:
    """
    Decode an ODIM integer field into physical values: `gain * raw + offset`
    as float32, with `nodata` (and `undetect`, if `mask_undetect`) set to NaN.
    8- and 16-bit unsigned data are decoded through a lookup table.

    Parameters:
    ===========
    raw: ndarray
        Stored (encoded) data.
    gain, offset: float
        ODIM scaling attributes.
    nodata: float or None
        Encoded value for missing data.
    undetect: float or None
        Encoded value for "no echo detected".
    mask_undetect: bool
        Whether `undetect` gates become NaN (default) or keep their decoded value.

    Returns:
    ========
    data: ndarray<float32>
    """
    gain32 = np.float32(gain)
    offset32 = np.float32(offset)
    specials = [nodata] if nodata is not None else []
    if mask_undetect and undetect is not None:
        specials.append(undetect)

    if raw.dtype.kind == "u" and raw.dtype.itemsize <= 2:
        table = gain32 * np.arange(np.iinfo(raw.dtype).max + 1, dtype=np.float32) + offset32
        remaining = []
        for value in specials:
            fvalue = float(value)
            if fvalue.is_integer() and 0 <= fvalue < table.size:
                table[int(fvalue)] = np.nan
            else:
                remaining.append(value)
        data = np.take(table, raw)
        for value in remaining:  # non-integer special values: cannot go through the table
            data[raw == value] = np.nan
        return data

    data = raw.astype(np.float32)
    data *= gain32
    data += offset32
    for value in specials:
        data[raw == value] = np.nan
    return data


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_sweep(
    source: Union[str, "os.PathLike[str]", h5py.File],
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
    source : str, path-like or h5py.File
        Path to the file (opened with `mode` and closed on return) or an
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
    odim_file: Union[str, "os.PathLike[str]"],
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
    odim_file : str or path-like
        Path to the ODIM HDF5 radar file.
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


# --------------------------------------------------------------------------- #
# HDF5 write helpers
# --------------------------------------------------------------------------- #
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
    if orig_id not in h5_tilt:
        raise KeyError(f"Cannot copy missing field id '{orig_id}'.")

    existing_indices = [int(k[4:]) for k in h5_tilt.keys() if k.startswith("data") and k[4:].isdigit()]
    next_index = (max(existing_indices) + 1) if existing_indices else 1
    data_id = f"data{next_index}"
    while data_id in h5_tilt:
        next_index += 1
        data_id = f"data{next_index}"

    # duplicate original
    h5_tilt.copy(orig_id, data_id)

    return data_id


def odim_str_type_id(text_bytes: bytes) -> h5py.h5t.TypeID:
    """
    Generate ODIM-conformant HDF5 string type ID with null-termination.

    Parameters
    ----------
    text_bytes : bytes
        Byte string for which to create the type ID.

    Returns
    -------
    h5py.h5t.TypeID
        String type ID sized for text_bytes with STR_NULLTERM padding.
    """
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
