"""
Custom exceptions for pyodim library.

This module defines specific exception classes for different types of errors
that can occur when working with ODIM H5 radar files.
"""


class PyOdimError(Exception):
    """Base exception class for all pyodim-related errors."""

    pass


class FileNotFoundError(PyOdimError):
    """Raised when an ODIM H5 file cannot be found."""

    pass


class InvalidFileFormatError(PyOdimError):
    """Raised when a file is not a valid ODIM H5 format."""

    pass


class InvalidSliceError(PyOdimError):
    """Raised when an invalid slice number is requested."""

    pass


class MetadataError(PyOdimError):
    """Raised when required metadata is missing or invalid."""

    pass


class CoordinateError(PyOdimError):
    """Raised when coordinate transformation fails."""

    pass


class DataValidationError(PyOdimError):
    """Raised when data validation fails."""

    pass


class NyquistConsistencyError(PyOdimError):
    """Raised when Nyquist velocity is not consistent with PRF."""

    pass


class FieldNotFoundError(PyOdimError):
    """Raised when a requested field is not found in the dataset."""

    pass


class H5FileError(PyOdimError):
    """Raised when there's an error accessing the H5 file."""

    pass
