"""
Test cases for pyodim validation functions.
"""

import os
import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock

from pyodim.validation import (
    validate_file_path,
    validate_odim_file_format,
    validate_slice_number,
    validate_field_lists,
    validate_boolean_parameter,
    validate_dataset_metadata,
    validate_data_array,
    validate_geographic_coordinates,
)
from pyodim.exceptions import (
    FileNotFoundError,
    InvalidFileFormatError,
    InvalidSliceError,
    MetadataError,
    DataValidationError,
)


class TestValidateFilePath:
    """Test cases for validate_file_path function."""

    def test_valid_file_path(self):
        """Test validation with a valid file path."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            result = validate_file_path(temp_path)
            assert result == temp_path
        finally:
            os.unlink(temp_path)

    def test_nonexistent_file(self):
        """Test validation with non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            validate_file_path("/nonexistent/path/file.h5")

    def test_empty_path(self):
        """Test validation with empty path."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            validate_file_path("")

        with pytest.raises(ValueError, match="File path cannot be empty"):
            validate_file_path("   ")

    def test_non_string_path(self):
        """Test validation with non-string path."""
        with pytest.raises(TypeError, match="File path must be a string"):
            validate_file_path(123)

        with pytest.raises(TypeError, match="File path must be a string"):
            validate_file_path(None)

    def test_directory_instead_of_file(self):
        """Test validation when path is a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError, match="Path is not a file"):
                validate_file_path(temp_dir)

    @patch("os.access")
    def test_unreadable_file(self, mock_access):
        """Test validation with unreadable file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            # Mock os.access to return False for read permission
            mock_access.return_value = False

            with pytest.raises(FileNotFoundError, match="File is not readable"):
                validate_file_path(temp_path)
        finally:
            os.unlink(temp_path)


class TestValidateOdimFileFormat:
    """Test cases for validate_odim_file_format function."""

    @patch("h5py.File")
    def test_valid_odim_file(self, mock_h5py_file):
        """Test validation with valid ODIM file."""
        # Mock HDF5 file structure
        mock_file = MagicMock()
        mock_file.attrs = {"Conventions": b"ODIM_H5/V2_2"}
        mock_file.__contains__ = lambda self, key: key in ["/what", "/where"]
        mock_file.keys.return_value = ["what", "where", "dataset1", "dataset2"]
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        # Should not raise an exception
        validate_odim_file_format("test.h5")

    @patch("h5py.File")
    def test_missing_conventions(self, mock_h5py_file):
        """Test validation with missing Conventions attribute."""
        mock_file = MagicMock()
        mock_file.attrs = {}
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        with pytest.raises(InvalidFileFormatError, match="Missing 'Conventions' attribute"):
            validate_odim_file_format("test.h5")

    @patch("h5py.File")
    def test_non_odim_conventions(self, mock_h5py_file):
        """Test validation with non-ODIM conventions."""
        mock_file = MagicMock()
        mock_file.attrs = {"Conventions": b"CF-1.6"}
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        with pytest.raises(InvalidFileFormatError, match="Not an ODIM file"):
            validate_odim_file_format("test.h5")

    @patch("h5py.File")
    def test_missing_required_groups(self, mock_h5py_file):
        """Test validation with missing required groups."""
        mock_file = MagicMock()
        mock_file.attrs = {"Conventions": b"ODIM_H5/V2_2"}
        mock_file.__contains__ = lambda self, key: key == "/what"  # Missing /where
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        with pytest.raises(InvalidFileFormatError, match="Missing required group: /where"):
            validate_odim_file_format("test.h5")

    @patch("h5py.File")
    def test_no_datasets(self, mock_h5py_file):
        """Test validation with no datasets."""
        mock_file = MagicMock()
        mock_file.attrs = {"Conventions": b"ODIM_H5/V2_2"}
        mock_file.__contains__ = lambda self, key: key in ["/what", "/where"]
        mock_file.keys.return_value = ["what", "where", "how"]  # No datasets
        mock_h5py_file.return_value.__enter__.return_value = mock_file

        with pytest.raises(InvalidFileFormatError, match="No datasets found"):
            validate_odim_file_format("test.h5")


class TestValidateSliceNumber:
    """Test cases for validate_slice_number function."""

    def test_valid_slice_numbers(self):
        """Test validation with valid slice numbers."""
        assert validate_slice_number(0, 5) == 0
        assert validate_slice_number(4, 5) == 4
        assert validate_slice_number(0, 1) == 0

    def test_invalid_slice_numbers(self):
        """Test validation with invalid slice numbers."""
        with pytest.raises(InvalidSliceError, match="exceeds available slices"):
            validate_slice_number(5, 5)

        with pytest.raises(InvalidSliceError, match="exceeds available slices"):
            validate_slice_number(10, 5)

    def test_negative_slice_number(self):
        """Test validation with negative slice number."""
        with pytest.raises(InvalidSliceError, match="must be non-negative"):
            validate_slice_number(-1, 5)

        with pytest.raises(InvalidSliceError, match="must be non-negative"):
            validate_slice_number(-10, 5)

    def test_non_integer_slice_number(self):
        """Test validation with non-integer slice number."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_slice_number(1.5, 5)

        with pytest.raises(TypeError, match="must be an integer"):
            validate_slice_number("1", 5)


class TestValidateFieldLists:
    """Test cases for validate_field_lists function."""

    def test_valid_field_lists(self):
        """Test validation with valid field lists."""
        include, exclude = validate_field_lists(["TH", "VRAD"], ["CLASS"])
        assert include == ["TH", "VRAD"]
        assert exclude == ["CLASS"]

    def test_none_field_lists(self):
        """Test validation with None field lists."""
        include, exclude = validate_field_lists(None, None)
        assert include == []
        assert exclude == []

        include, exclude = validate_field_lists(["TH"], None)
        assert include == ["TH"]
        assert exclude == []

    def test_empty_field_lists(self):
        """Test validation with empty field lists."""
        include, exclude = validate_field_lists([], [])
        assert include == []
        assert exclude == []

    def test_overlapping_field_lists(self):
        """Test validation with overlapping field lists."""
        with pytest.raises(ValueError, match="cannot be both included and excluded"):
            validate_field_lists(["TH", "VRAD"], ["TH", "CLASS"])

    def test_non_list_field_lists(self):
        """Test validation with non-list field lists."""
        with pytest.raises(TypeError, match="must be a list or None"):
            validate_field_lists("TH", [])

        with pytest.raises(TypeError, match="must be a list or None"):
            validate_field_lists([], "CLASS")

    def test_non_string_fields(self):
        """Test validation with non-string fields."""
        with pytest.raises(TypeError, match="must be strings"):
            validate_field_lists(["TH", 123], [])

        with pytest.raises(TypeError, match="must be strings"):
            validate_field_lists([], ["CLASS", None])


class TestValidateBooleanParameter:
    """Test cases for validate_boolean_parameter function."""

    def test_valid_boolean_values(self):
        """Test validation with valid boolean values."""
        assert validate_boolean_parameter(True, "test") == True
        assert validate_boolean_parameter(False, "test") == False

    def test_none_with_default(self):
        """Test validation with None and default value."""
        assert validate_boolean_parameter(None, "test", default=True) == True
        assert validate_boolean_parameter(None, "test", default=False) == False
        assert validate_boolean_parameter(None, "test") == False  # Default default

    def test_non_boolean_values(self):
        """Test validation with non-boolean values."""
        with pytest.raises(TypeError, match="must be a boolean or None"):
            validate_boolean_parameter("true", "test")

        with pytest.raises(TypeError, match="must be a boolean or None"):
            validate_boolean_parameter(1, "test")

        with pytest.raises(TypeError, match="must be a boolean or None"):
            validate_boolean_parameter([], "test")


class TestValidateDatasetMetadata:
    """Test cases for validate_dataset_metadata function."""

    def test_valid_metadata(self):
        """Test validation with valid metadata."""
        metadata = {}
        coordinates_metadata = {
            "nrays": 360,
            "nbins": 1000,
            "rstart": 0.0,
            "rscale": 250.0,
            "elangle": 0.5,
        }

        # Should not raise an exception
        validate_dataset_metadata(metadata, coordinates_metadata)

    def test_missing_required_keys(self):
        """Test validation with missing required keys."""
        metadata = {}
        coordinates_metadata = {
            "nrays": 360,
            "nbins": 1000,
            # Missing 'rstart', 'rscale', 'elangle'
        }

        with pytest.raises(MetadataError, match="Missing required coordinate metadata"):
            validate_dataset_metadata(metadata, coordinates_metadata)

    def test_invalid_nrays(self):
        """Test validation with invalid nrays."""
        metadata = {}
        coordinates_metadata = {
            "nrays": 0,  # Invalid
            "nbins": 1000,
            "rstart": 0.0,
            "rscale": 250.0,
            "elangle": 0.5,
        }

        with pytest.raises(MetadataError, match="Invalid nrays"):
            validate_dataset_metadata(metadata, coordinates_metadata)

    def test_invalid_elevation_angle(self):
        """Test validation with invalid elevation angle."""
        metadata = {}
        coordinates_metadata = {
            "nrays": 360,
            "nbins": 1000,
            "rstart": 0.0,
            "rscale": 250.0,
            "elangle": 95.0,  # Invalid (> 90)
        }

        with pytest.raises(MetadataError, match="Invalid elevation angle"):
            validate_dataset_metadata(metadata, coordinates_metadata)


class TestValidateDataArray:
    """Test cases for validate_data_array function."""

    def test_valid_data_array(self):
        """Test validation with valid data array."""
        data = np.random.rand(360, 1000)

        # Should not raise an exception
        validate_data_array(data, "test_field")

    def test_non_numpy_array(self):
        """Test validation with non-numpy array."""
        with pytest.raises(DataValidationError, match="must be a numpy array"):
            validate_data_array([[1, 2], [3, 4]], "test_field")

    def test_empty_array(self):
        """Test validation with empty array."""
        with pytest.raises(DataValidationError, match="is empty"):
            validate_data_array(np.array([]), "test_field")

    def test_wrong_dimensions(self):
        """Test validation with wrong dimensions."""
        # 1D array
        with pytest.raises(DataValidationError, match="must be 2D"):
            validate_data_array(np.array([1, 2, 3]), "test_field")

        # 3D array
        with pytest.raises(DataValidationError, match="must be 2D"):
            validate_data_array(np.random.rand(10, 10, 10), "test_field")

    def test_all_non_finite_values(self):
        """Test validation with all non-finite values."""
        data = np.full((10, 10), np.nan)

        with pytest.raises(DataValidationError, match="contains no finite values"):
            validate_data_array(data, "test_field")


class TestValidateGeographicCoordinates:
    """Test cases for validate_geographic_coordinates function."""

    def test_valid_coordinates(self):
        """Test validation with valid coordinates."""
        lon, lat = validate_geographic_coordinates(-122.5, 37.8)
        assert lon == -122.5
        assert lat == 37.8

        # Test edge cases
        lon, lat = validate_geographic_coordinates(-180, -90)
        assert lon == -180.0
        assert lat == -90.0

        lon, lat = validate_geographic_coordinates(180, 90)
        assert lon == 180.0
        assert lat == 90.0

    def test_invalid_longitude(self):
        """Test validation with invalid longitude."""
        with pytest.raises(ValueError, match="Invalid longitude"):
            validate_geographic_coordinates(181, 0)

        with pytest.raises(ValueError, match="Invalid longitude"):
            validate_geographic_coordinates(-181, 0)

    def test_invalid_latitude(self):
        """Test validation with invalid latitude."""
        with pytest.raises(ValueError, match="Invalid latitude"):
            validate_geographic_coordinates(0, 91)

        with pytest.raises(ValueError, match="Invalid latitude"):
            validate_geographic_coordinates(0, -91)

    def test_non_numeric_coordinates(self):
        """Test validation with non-numeric coordinates."""
        with pytest.raises(TypeError, match="Longitude must be a number"):
            validate_geographic_coordinates("0", 0)

        with pytest.raises(TypeError, match="Latitude must be a number"):
            validate_geographic_coordinates(0, "0")

    def test_integer_coordinates(self):
        """Test validation with integer coordinates."""
        lon, lat = validate_geographic_coordinates(-122, 37)
        assert lon == -122.0
        assert lat == 37.0
        assert isinstance(lon, float)
        assert isinstance(lat, float)
