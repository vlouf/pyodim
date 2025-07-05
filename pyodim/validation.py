"""
Input validation utilities for pyodim library.

This module provides functions to validate inputs and ODIM H5 file structure
before processing.
"""

import os
import h5py
import numpy as np
from typing import List, Optional, Union

from .exceptions import (
    FileNotFoundError,
    InvalidFileFormatError,
    InvalidSliceError,
    MetadataError,
    DataValidationError,
    H5FileError,
)


def validate_file_path(file_path: str) -> str:
    """
    Validate that the file path exists and is readable.

    Parameters:
    ===========
    file_path: str
        Path to the ODIM H5 file.

    Returns:
    ========
    file_path: str
        The validated file path.

    Raises:
    =======
    FileNotFoundError: If the file doesn't exist or is not readable.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"File path must be a string, got {type(file_path)}")

    if not file_path.strip():
        raise ValueError("File path cannot be empty")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Path is not a file: {file_path}")

    if not os.access(file_path, os.R_OK):
        raise FileNotFoundError(f"File is not readable: {file_path}")

    return file_path


def validate_odim_file_format(file_path: str) -> None:
    """
    Validate that the file is a valid ODIM H5 format.

    Parameters:
    ===========
    file_path: str
        Path to the ODIM H5 file.

    Raises:
    =======
    InvalidFileFormatError: If the file is not a valid ODIM H5 format.
    H5FileError: If there's an error accessing the H5 file.
    """
    try:
        with h5py.File(file_path, "r") as hfile:
            # Check for required root attributes
            if "Conventions" not in hfile.attrs:
                raise InvalidFileFormatError("Missing 'Conventions' attribute in root")

            conventions = hfile.attrs["Conventions"]
            if isinstance(conventions, bytes):
                conventions = conventions.decode("utf-8")

            if "ODIM" not in conventions:
                raise InvalidFileFormatError(f"Not an ODIM file. Conventions: {conventions}")

            # Check for required root groups
            required_groups = ["/what", "/where"]
            for group in required_groups:
                if group not in hfile:
                    raise InvalidFileFormatError(f"Missing required group: {group}")

            # Check for at least one dataset
            datasets = [k for k in hfile.keys() if k.startswith("dataset")]
            if not datasets:
                raise InvalidFileFormatError("No datasets found in ODIM file")

    except (OSError, IOError) as e:
        raise H5FileError(f"Cannot open H5 file: {e}")
    except Exception as e:
        if isinstance(e, (InvalidFileFormatError, H5FileError)):
            raise
        raise H5FileError(f"Error validating ODIM file: {e}")


def validate_slice_number(nslice: int, max_slices: int) -> int:
    """
    Validate the slice number.

    Parameters:
    ===========
    nslice: int
        Slice number to validate.
    max_slices: int
        Maximum number of available slices.

    Returns:
    ========
    nslice: int
        The validated slice number.

    Raises:
    =======
    InvalidSliceError: If the slice number is invalid.
    """
    if not isinstance(nslice, int):
        raise TypeError(f"Slice number must be an integer, got {type(nslice)}")

    if nslice < 0:
        raise InvalidSliceError(f"Slice number must be non-negative, got {nslice}")

    if nslice >= max_slices:
        raise InvalidSliceError(
            f"Slice number {nslice} exceeds available slices (0-{max_slices-1})"
        )

    return nslice


def validate_field_lists(
    include_fields: Optional[List[str]], exclude_fields: Optional[List[str]]
) -> tuple:
    """
    Validate include and exclude field lists.

    Parameters:
    ===========
    include_fields: list or None
        List of fields to include.
    exclude_fields: list or None
        List of fields to exclude.

    Returns:
    ========
    tuple: (validated_include_fields, validated_exclude_fields)

    Raises:
    =======
    TypeError: If field lists are not lists or None.
    ValueError: If both include and exclude lists are provided and overlap.
    """
    # Handle None values
    if include_fields is None:
        include_fields = []
    if exclude_fields is None:
        exclude_fields = []

    # Type validation
    if not isinstance(include_fields, list):
        raise TypeError(f"include_fields must be a list or None, got {type(include_fields)}")

    if not isinstance(exclude_fields, list):
        raise TypeError(f"exclude_fields must be a list or None, got {type(exclude_fields)}")

    # Validate list contents
    for field in include_fields:
        if not isinstance(field, str):
            raise TypeError(f"All fields in include_fields must be strings, got {type(field)}")

    for field in exclude_fields:
        if not isinstance(field, str):
            raise TypeError(f"All fields in exclude_fields must be strings, got {type(field)}")

    # Check for overlap
    if include_fields and exclude_fields:
        overlap = set(include_fields) & set(exclude_fields)
        if overlap:
            raise ValueError(f"Fields cannot be both included and excluded: {overlap}")

    return include_fields, exclude_fields


def validate_boolean_parameter(
    param: Union[bool, None], param_name: str, default: bool = False
) -> bool:
    """
    Validate a boolean parameter.

    Parameters:
    ===========
    param: bool or None
        The parameter to validate.
    param_name: str
        Name of the parameter for error messages.
    default: bool
        Default value if param is None.

    Returns:
    ========
    bool: The validated boolean value.

    Raises:
    =======
    TypeError: If the parameter is not a boolean or None.
    """
    if param is None:
        return default

    if not isinstance(param, bool):
        raise TypeError(f"{param_name} must be a boolean or None, got {type(param)}")

    return param


def validate_dataset_metadata(metadata: dict, coordinates_metadata: dict) -> None:
    """
    Validate that required metadata is present and valid.

    Parameters:
    ===========
    metadata: dict
        General metadata dictionary.
    coordinates_metadata: dict
        Coordinates metadata dictionary.

    Raises:
    =======
    MetadataError: If required metadata is missing or invalid.
    """
    # Required coordinate metadata
    required_coord_keys = ["nrays", "nbins", "rstart", "rscale", "elangle"]
    for key in required_coord_keys:
        if key not in coordinates_metadata:
            raise MetadataError(f"Missing required coordinate metadata: {key}")

    # Validate coordinate values
    if coordinates_metadata["nrays"] <= 0:
        raise MetadataError(f"Invalid nrays: {coordinates_metadata['nrays']}")

    if coordinates_metadata["nbins"] <= 0:
        raise MetadataError(f"Invalid nbins: {coordinates_metadata['nbins']}")

    if coordinates_metadata["rscale"] <= 0:
        raise MetadataError(f"Invalid rscale: {coordinates_metadata['rscale']}")

    # Validate elevation angle
    elangle = coordinates_metadata["elangle"]
    if not (-90 <= elangle <= 90):
        raise MetadataError(
            f"Invalid elevation angle: {elangle} (must be between -90 and 90 degrees)"
        )


def validate_data_array(data: np.ndarray, field_name: str) -> None:
    """
    Validate a data array.

    Parameters:
    ===========
    data: np.ndarray
        The data array to validate.
    field_name: str
        Name of the field for error messages.

    Raises:
    =======
    DataValidationError: If the data array is invalid.
    """
    if not isinstance(data, np.ndarray):
        raise DataValidationError(
            f"Data for field '{field_name}' must be a numpy array, got {type(data)}"
        )

    if data.size == 0:
        raise DataValidationError(f"Data array for field '{field_name}' is empty")

    if data.ndim != 2:
        raise DataValidationError(
            f"Data array for field '{field_name}' must be 2D, got {data.ndim}D"
        )

    if not np.isfinite(data).any():
        raise DataValidationError(f"Data array for field '{field_name}' contains no finite values")


def validate_geographic_coordinates(longitude: float, latitude: float) -> tuple:
    """
    Validate geographic coordinates.

    Parameters:
    ===========
    longitude: float
        Longitude in degrees.
    latitude: float
        Latitude in degrees.

    Returns:
    ========
    tuple: (validated_longitude, validated_latitude)

    Raises:
    =======
    ValueError: If coordinates are invalid.
    """
    if not isinstance(longitude, (int, float)):
        raise TypeError(f"Longitude must be a number, got {type(longitude)}")

    if not isinstance(latitude, (int, float)):
        raise TypeError(f"Latitude must be a number, got {type(latitude)}")

    if not (-180 <= longitude <= 180):
        raise ValueError(f"Invalid longitude: {longitude} (must be between -180 and 180 degrees)")

    if not (-90 <= latitude <= 90):
        raise ValueError(f"Invalid latitude: {latitude} (must be between -90 and 90 degrees)")

    return float(longitude), float(latitude)
