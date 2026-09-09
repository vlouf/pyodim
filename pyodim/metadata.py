"""
ODIM H5 metadata: attribute conversion at the h5py boundary, sweep discovery,
the field metadata table and the root/dataset metadata readers.

Only numpy and h5py are imported here so that this module can be shared by the
reader and the writer.
"""

import datetime
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple

import h5py
import numpy as np


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


def check_nyquist(dset: Any) -> None:
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
    attrs = getattr(dset, "attrs", dset)  # xarray.Dataset or a plain mapping
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
