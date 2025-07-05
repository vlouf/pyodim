# primary read routines
from .pyodim import read_odim
from .pyodim import read_write_odim
from .pyodim import read_odim_slice

# helper routines
from .pyodim import copy_h5_data
from .pyodim import write_odim_str_attrib

# exceptions
from .exceptions import (
    PyOdimError,
    FileNotFoundError,
    InvalidFileFormatError,
    InvalidSliceError,
    MetadataError,
    CoordinateError,
    DataValidationError,
    NyquistConsistencyError,
    FieldNotFoundError,
    H5FileError,
)

# validation utilities
from .validation import (
    validate_file_path,
    validate_odim_file_format,
    validate_slice_number,
    validate_field_lists,
    validate_boolean_parameter,
    validate_dataset_metadata,
    validate_data_array,
    validate_geographic_coordinates,
)

__all__ = [
    # Primary functions
    "read_odim",
    "read_write_odim",
    "read_odim_slice",
    # Helper functions
    "copy_h5_data",
    "write_odim_str_attrib",
    # Exceptions
    "PyOdimError",
    "FileNotFoundError",
    "InvalidFileFormatError",
    "InvalidSliceError",
    "MetadataError",
    "CoordinateError",
    "DataValidationError",
    "NyquistConsistencyError",
    "FieldNotFoundError",
    "H5FileError",
    # Validation functions
    "validate_file_path",
    "validate_odim_file_format",
    "validate_slice_number",
    "validate_field_lists",
    "validate_boolean_parameter",
    "validate_dataset_metadata",
    "validate_data_array",
    "validate_geographic_coordinates",
]
