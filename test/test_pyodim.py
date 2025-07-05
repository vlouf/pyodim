# tests/test_pyodim.py

import os
import pytest
import numpy as np
import tempfile
import h5py
from unittest.mock import patch, MagicMock

from pyodim import (
    read_odim,
    read_odim_slice,
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
from pyodim.pyodim import cartesian_to_geographic, check_nyquist
import xarray as xr

# Define the path to the ODIM H5 file
ODIM_FILE_PATH = "test/8_20241112_005000.pvol.h5"


@pytest.fixture
def sample_odim_file():
    """
    Fixture to check the presence of the sample ODIM file.
    """
    if not os.path.exists(ODIM_FILE_PATH):
        pytest.fail(f"Test file '{ODIM_FILE_PATH}' does not exist.")
    return ODIM_FILE_PATH


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_dataset():
    """Create a mock xarray dataset for testing."""
    dataset = xr.Dataset()
    dataset.attrs = {"wavelength": 10.0, "highprf": 1000.0, "NI": 25.0}
    return dataset


class TestReadOdim:
    """Test cases for the main read_odim function."""

    def test_read_odim_basic(self, sample_odim_file):
        """
        Test the read_odim function to ensure it correctly reads an ODIM H5 file and
        returns an xarray Dataset with expected properties.
        """
        # Call read_odim on the test file and get the dataset
        rsets = read_odim(sample_odim_file)
        assert len(rsets) > 0, "No sweeps in radar datasets found"

        dataset = rsets[0].compute()
        # Assert that the output is an xarray Dataset
        assert isinstance(dataset, xr.Dataset), "Output is not an xarray Dataset."

        # Check that the dataset is not empty
        assert len(dataset.data_vars) > 0, "Dataset has no data variables."

        # Check for specific metadata or data variables
        assert "latitude" in dataset.data_vars, "Latitude coordinate is missing."
        assert "longitude" in dataset.data_vars, "Longitude coordinate is missing."

        # Optional: Check for specific radar fields
        assert "TH" in dataset.data_vars, "Expected data variable 'TH' is missing."
        assert "CLASS" in dataset.data_vars, "Expected data variable 'CLASS' is missing"

        # Check a basic property of the data
        assert dataset["TH"].shape[0] > 0, "TH data is empty."

    def test_read_odim_with_field_filtering(self, sample_odim_file):
        """Test reading ODIM file with field filtering."""
        # Test include_fields
        rsets = read_odim(sample_odim_file, include_fields=["TH"])
        dataset = rsets[0].compute()
        assert "TH" in dataset.data_vars
        # Should not have other fields (except coordinates)
        data_fields = [
            var for var in dataset.data_vars if var not in ["latitude", "longitude", "x", "y", "z"]
        ]
        assert len(data_fields) == 1

        # Test exclude_fields
        rsets = read_odim(sample_odim_file, exclude_fields=["CLASS"])
        dataset = rsets[0].compute()
        assert "CLASS" not in dataset.data_vars
        assert "DBZH" in dataset.data_vars

    def test_read_odim_lazy_vs_immediate(self, sample_odim_file):
        """Test lazy vs immediate loading."""
        # Lazy loading (default)
        rsets_lazy = read_odim(sample_odim_file, lazy_load=True)

        # Immediate loading
        rsets_immediate = read_odim(sample_odim_file, lazy_load=False)

        # Both should have the same number of sweeps
        assert len(rsets_lazy) == len(rsets_immediate)

        # Lazy datasets should be dask delayed objects
        assert hasattr(rsets_lazy[0], "compute")

        # Immediate datasets should be xarray datasets
        assert isinstance(rsets_immediate[0], xr.Dataset)


class TestReadOdimSlice:
    """Test cases for the read_odim_slice function."""

    def test_read_odim_slice_basic(self, sample_odim_file):
        """Test reading a single slice."""
        dataset = read_odim_slice(sample_odim_file, nslice=0)
        assert isinstance(dataset, xr.Dataset)
        assert len(dataset.data_vars) > 0

    def test_read_odim_slice_invalid_slice(self, sample_odim_file):
        """Test reading with invalid slice number."""
        with pytest.raises(InvalidSliceError):
            read_odim_slice(sample_odim_file, nslice=999)

    def test_read_odim_slice_negative_slice(self, sample_odim_file):
        """Test reading with negative slice number."""
        with pytest.raises(InvalidSliceError):
            read_odim_slice(sample_odim_file, nslice=-1)


class TestValidation:
    """Test cases for validation functions."""

    def test_validate_file_path_valid(self, sample_odim_file):
        """Test validation of valid file path."""
        result = validate_file_path(sample_odim_file)
        assert result == sample_odim_file

    def test_validate_file_path_nonexistent(self):
        """Test validation of non-existent file."""
        with pytest.raises(FileNotFoundError):
            validate_file_path("nonexistent_file.h5")

    def test_validate_file_path_empty(self):
        """Test validation of empty file path."""
        with pytest.raises(ValueError):
            validate_file_path("")

    def test_validate_file_path_wrong_type(self):
        """Test validation of wrong type for file path."""
        with pytest.raises(TypeError):
            validate_file_path(123)

    def test_validate_slice_number_valid(self):
        """Test validation of valid slice number."""
        result = validate_slice_number(0, 5)
        assert result == 0

        result = validate_slice_number(4, 5)
        assert result == 4

    def test_validate_slice_number_invalid(self):
        """Test validation of invalid slice numbers."""
        with pytest.raises(InvalidSliceError):
            validate_slice_number(5, 5)  # Equal to max

        with pytest.raises(InvalidSliceError):
            validate_slice_number(-1, 5)  # Negative

        with pytest.raises(TypeError):
            validate_slice_number("0", 5)  # Wrong type

    def test_validate_field_lists_valid(self):
        """Test validation of valid field lists."""
        include, exclude = validate_field_lists(["TH", "VRAD"], ["CLASS"])
        assert include == ["TH", "VRAD"]
        assert exclude == ["CLASS"]

        # Test None values
        include, exclude = validate_field_lists(None, None)
        assert include == []
        assert exclude == []

    def test_validate_field_lists_overlap(self):
        """Test validation of overlapping field lists."""
        with pytest.raises(ValueError):
            validate_field_lists(["TH", "VRAD"], ["TH", "CLASS"])

    def test_validate_field_lists_wrong_type(self):
        """Test validation of wrong type field lists."""
        with pytest.raises(TypeError):
            validate_field_lists("TH", [])

        with pytest.raises(TypeError):
            validate_field_lists(["TH", 123], [])

    def test_validate_boolean_parameter(self):
        """Test validation of boolean parameters."""
        assert validate_boolean_parameter(True, "test") == True
        assert validate_boolean_parameter(False, "test") == False
        assert validate_boolean_parameter(None, "test", default=True) == True

        with pytest.raises(TypeError):
            validate_boolean_parameter("true", "test")

    def test_validate_geographic_coordinates_valid(self):
        """Test validation of valid geographic coordinates."""
        lon, lat = validate_geographic_coordinates(-122.5, 37.8)
        assert lon == -122.5
        assert lat == 37.8

    def test_validate_geographic_coordinates_invalid(self):
        """Test validation of invalid geographic coordinates."""
        with pytest.raises(ValueError):
            validate_geographic_coordinates(181, 0)  # Invalid longitude

        with pytest.raises(ValueError):
            validate_geographic_coordinates(0, 91)  # Invalid latitude

        with pytest.raises(TypeError):
            validate_geographic_coordinates("0", 0)  # Wrong type


class TestCoordinateTransformation:
    """Test cases for coordinate transformation functions."""

    def test_cartesian_to_geographic_basic(self):
        """Test basic coordinate transformation."""
        x = np.array([1000.0, 2000.0])
        y = np.array([1000.0, 2000.0])
        lon0, lat0 = -122.5, 37.8

        lon, lat = cartesian_to_geographic(x, y, lon0, lat0)

        assert isinstance(lon, np.ndarray)
        assert isinstance(lat, np.ndarray)
        assert lon.shape == x.shape
        assert lat.shape == y.shape
        assert lon.dtype == np.float32
        assert lat.dtype == np.float32

    def test_cartesian_to_geographic_invalid_inputs(self):
        """Test coordinate transformation with invalid inputs."""
        x = np.array([1000.0, 2000.0])
        y = np.array([1000.0])  # Different shape

        with pytest.raises(ValueError):
            cartesian_to_geographic(x, y, -122.5, 37.8)

        with pytest.raises(TypeError):
            cartesian_to_geographic([1000.0], [1000.0], -122.5, 37.8)

        with pytest.raises(ValueError):
            cartesian_to_geographic(x, x, 181, 37.8)  # Invalid longitude


class TestNyquistCheck:
    """Test cases for Nyquist velocity checking."""

    def test_check_nyquist_consistent(self, mock_dataset):
        """Test Nyquist check with consistent values."""
        # Should not raise an exception
        check_nyquist(mock_dataset)

    def test_check_nyquist_inconsistent(self, mock_dataset):
        """Test Nyquist check with inconsistent values."""
        mock_dataset.attrs["NI"] = 50.0  # Inconsistent value

        with pytest.raises(NyquistConsistencyError):
            check_nyquist(mock_dataset)

    def test_check_nyquist_missing_attributes(self):
        """Test Nyquist check with missing attributes."""
        dataset = xr.Dataset()
        dataset.attrs = {"wavelength": 10.0}  # Missing other attributes

        with pytest.raises(MetadataError):
            check_nyquist(dataset)


class TestErrorHandling:
    """Test cases for error handling."""

    def test_file_not_found_error(self):
        """Test FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            read_odim("nonexistent_file.h5")

    def test_invalid_file_format_error(self, temp_file):
        """Test InvalidFileFormatError is raised for invalid files."""
        # Create a non-HDF5 file
        with open(temp_file, "w") as f:
            f.write("This is not an HDF5 file")

        with pytest.raises((InvalidFileFormatError, H5FileError)):
            read_odim(temp_file)

    def test_data_validation_error(self):
        """Test DataValidationError for invalid data arrays."""
        with pytest.raises(DataValidationError):
            validate_data_array([], "test_field")  # Empty list instead of array

        with pytest.raises(DataValidationError):
            validate_data_array(np.array([]), "test_field")  # Empty array

        with pytest.raises(DataValidationError):
            validate_data_array(np.array([1, 2, 3]), "test_field")  # 1D array instead of 2D


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self, sample_odim_file):
        """Test a complete workflow from file reading to data access."""
        # Read the file
        datasets = read_odim(sample_odim_file, lazy_load=False)

        # Check we got datasets
        assert len(datasets) > 0

        # Get first dataset
        dataset = datasets[0]

        # Check basic properties
        assert isinstance(dataset, xr.Dataset)
        assert len(dataset.data_vars) > 0

        # Check coordinates exist
        assert "latitude" in dataset.data_vars
        assert "longitude" in dataset.data_vars
        assert "range" in dataset.coords
        assert "azimuth" in dataset.coords

        # Check data fields exist
        data_fields = [
            var for var in dataset.data_vars if var not in ["latitude", "longitude", "x", "y", "z"]
        ]
        assert len(data_fields) > 0

        # Check data has reasonable values
        for field in data_fields:
            data = dataset[field].values
            assert not np.all(np.isnan(data)), f"Field {field} is all NaN"
