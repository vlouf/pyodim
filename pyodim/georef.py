"""
Radar geometry: sweep coordinates from ODIM metadata, antenna-to-cartesian
transform (4/3 Earth model), WGS84 geodesic for per-gate longitude/latitude,
and per-ray timestamps.
"""

import datetime
from typing import Dict, Tuple

import numpy as np
import xarray as xr


EARTH_RADIUS = 6371000.0  # mean Earth radius (m)
REFRACTION_KE = 4.0 / 3.0  # effective Earth radius factor (Doviak & Zrnić)
WGS84_A = 6378137.0  # WGS84 semi-major axis (m)
WGS84_F = 1.0 / 298.257223563  # WGS84 flattening


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
