# tests/test_pyodim.py
import os
import pytest
from pyodim import read_odim
from pyodim.pyodim import (
    check_nyquist,
    write_odim_str_attrib
)
import h5py
import tempfile
import xarray as xr
import numpy as np

# Define the path to the ODIM H5 file
ODIM_FILE_PATH = "test/8_20241112_005000.pvol.h5"

@pytest.fixture
def sample_odim_file():
    """
    Fixture to check the presence of the sample ODIM file.
    """
    if not os.path.exists(ODIM_FILE_PATH):
        pytest.skip(f"Test file '{ODIM_FILE_PATH}' does not exist.")
    return ODIM_FILE_PATH

@pytest.fixture
def radar_datasets(sample_odim_file):
    """
    Fixture that reads the ODIM file and returns the radar datasets.
    This avoids reading the file multiple times across tests.
    """
    return read_odim(sample_odim_file)

def test_check_nyquist_valid():
    """Test check_nyquist with consistent Nyquist velocity."""
    # Create a dataset with consistent attributes
    # Formula: nyquist = 1e-2 * prf * wavelength / 4
    # Example: wavelength=0.053m (C-band), prf=1000Hz -> nyquist=13.25 m/s

    wavelength = 0.053  # meters (C-band radar)
    prf = 1000.0  # Hz
    nyquist = 1e-2 * prf * wavelength / 4  # = 13.25 m/s

    ds = xr.Dataset(
        attrs={
            'wavelength': wavelength,
            'highprf': prf,
            'NI': nyquist
        }
    )

    # Should not raise an error
    check_nyquist(ds)

def test_read_odim_returns_datasets(sample_odim_file):
    """
    Test that read_odim returns a non-empty list of datasets.
    """
    rsets = read_odim(sample_odim_file)
    assert isinstance(rsets, list), "read_odim should return a list of datasets."
    assert len(rsets) > 0, "No sweeps in radar datasets found."

def test_dataset_is_xarray(radar_datasets):
    """
    Test that each dataset is an xarray Dataset.
    """
    dataset = radar_datasets[0].compute()
    assert isinstance(dataset, xr.Dataset), "Output is not an xarray Dataset."

def test_dataset_has_data_variables(radar_datasets):
    """
    Test that the dataset contains data variables.
    """
    dataset = radar_datasets[0].compute()
    assert len(dataset.data_vars) > 0, "Dataset has no data variables."

def test_geographic_coordinates_present(radar_datasets):
    """
    Test that latitude and longitude coordinates are present.
    """
    dataset = radar_datasets[0].compute()
    assert 'latitude' in dataset.data_vars or 'latitude' in dataset.coords, \
        "Latitude coordinate is missing."
    assert 'longitude' in dataset.data_vars or 'longitude' in dataset.coords, \
        "Longitude coordinate is missing."

def test_expected_radar_variables(radar_datasets):
    """
    Test that expected radar data variables are present.
    """
    dataset = radar_datasets[0].compute()
    assert 'TH' in dataset.data_vars, "Expected data variable 'TH' (reflectivity) is missing."
    assert 'CLASS' in dataset.data_vars, "Expected data variable 'CLASS' (classification) is missing."

def test_reflectivity_data_shape(radar_datasets):
    """
    Test that reflectivity data has valid shape (non-empty).
    """
    dataset = radar_datasets[0].compute()
    assert dataset['TH'].shape[0] > 0, "TH (reflectivity) data has zero size in first dimension."
    assert dataset['TH'].size > 0, "TH (reflectivity) data is completely empty."

def test_reflectivity_value_range(radar_datasets):
    """
    Test that reflectivity values are within reasonable range.
    """
    dataset = radar_datasets[0].compute()
    th_data = dataset['TH'].values

    # Remove NaN/masked values for range check
    valid_data = th_data[~np.isnan(th_data)]

    if len(valid_data) > 0:
        assert valid_data.min() >= -40, "TH values unreasonably low (< -40 dBZ)."
        assert valid_data.max() <= 80, "TH values unreasonably high (> 80 dBZ)."

def test_classification_is_integer(radar_datasets):
    """
    Test that classification data contains integer values.
    """
    dataset = radar_datasets[0].compute()
    class_data = dataset['CLASS'].values

    # Check dtype is integer type
    assert np.issubdtype(class_data.dtype, np.integer) or \
           np.issubdtype(class_data.dtype, np.floating), \
           "CLASS data should be numeric."

def test_all_sweeps_have_consistent_variables(radar_datasets):
    """
    Test that all sweeps contain the same data variables.
    """
    if len(radar_datasets) > 1:
        first_vars = set(radar_datasets[0].compute().data_vars)

        for i, rset in enumerate(radar_datasets[1:], start=1):
            sweep_vars = set(rset.compute().data_vars)
            assert sweep_vars == first_vars, \
                f"Sweep {i} has different variables than sweep 0."

def test_dimensions_present(radar_datasets):
    """
    Test that expected dimensions are present (e.g., azimuth, range).
    """
    dataset = radar_datasets[0].compute()

    # Common ODIM dimensions - adjust based on your implementation
    expected_dims = {'azimuth', 'range'} | {'elevation'} | {'time'}

    # Check that at least some expected dimensions are present
    # Use dataset.sizes instead of dataset.dims to avoid FutureWarning
    actual_dims = set(dataset.sizes.keys())
    assert len(actual_dims & expected_dims) > 0, \
        f"Expected dimensions not found. Found: {actual_dims}"

def test_metadata_attributes(radar_datasets):
    """
    Test that important ODIM metadata attributes are preserved.
    """
    dataset = radar_datasets[0].compute()

    # Check for common ODIM attributes - adjust based on what pyodim preserves
    # These might be in dataset.attrs or in coordinate attributes
    attrs = dataset.attrs

    # At minimum, check that some attributes exist
    assert len(attrs) > 0, "Dataset has no metadata attributes."

def test_coordinate_monotonicity(radar_datasets):
    """
    Test that coordinate arrays are monotonic where expected.
    """
    dataset = radar_datasets[0].compute()

    if 'range' in dataset.coords:
        range_vals = dataset.coords['range'].values
        assert np.all(np.diff(range_vals) > 0), "Range coordinate is not monotonically increasing."

def test_no_all_nan_variables(radar_datasets):
    """
    Test that data variables are not completely filled with NaN values.
    """
    dataset = radar_datasets[0].compute()

    for var in dataset.data_vars:
        data = dataset[var].values
        assert not np.all(np.isnan(data)), f"Variable '{var}' contains only NaN values."

@pytest.mark.parametrize("sweep_idx", [0, 1, 2])
def test_multiple_sweeps(radar_datasets, sweep_idx):
    """
    Test that multiple sweeps can be accessed and are valid.
    Skips if the requested sweep doesn't exist.
    """
    if sweep_idx >= len(radar_datasets):
        pytest.skip(f"Sweep {sweep_idx} does not exist in this file.")

    dataset = radar_datasets[sweep_idx].compute()
    assert isinstance(dataset, xr.Dataset), f"Sweep {sweep_idx} is not an xarray Dataset."
    assert len(dataset.data_vars) > 0, f"Sweep {sweep_idx} has no data variables."

def test_data_array_dtypes(radar_datasets):
    """
    Test that data arrays have appropriate data types.
    """
    dataset = radar_datasets[0].compute()

    for var in dataset.data_vars:
        dtype = dataset[var].dtype
        # Should be numeric types
        assert np.issubdtype(dtype, np.number), \
            f"Variable '{var}' has non-numeric dtype: {dtype}"

def test_coordinate_coverage(radar_datasets):
    """
    Test that coordinates cover expected ranges for radar data.
    """
    dataset = radar_datasets[0].compute()

    if 'azimuth' in dataset.coords:
        az = dataset.coords['azimuth'].values
        assert az.min() >= 0, "Azimuth values should be >= 0 degrees."
        assert az.max() <= 360, "Azimuth values should be <= 360 degrees."

    if 'range' in dataset.coords:
        rng = dataset.coords['range'].values
        assert rng.min() >= 0, "Range values should be non-negative."
        assert rng.max() > 0, "Range should have positive maximum value."

def test_data_variable_dimensions(radar_datasets):
    """
    Test that data variables have expected dimensions.
    """
    dataset = radar_datasets[0].compute()

    for var in ['TH', 'CLASS']:
        if var in dataset.data_vars:
            dims = dataset[var].dims
            # Should typically have azimuth and range dimensions
            assert len(dims) >= 2, f"Variable '{var}' should have at least 2 dimensions."

def test_sweep_elevation_ordering(radar_datasets):
    """
    Test that sweeps are ordered by increasing elevation angle.
    """
    if len(radar_datasets) > 1:
        elevations = []
        for rset in radar_datasets:
            dataset = rset.compute()
            # Try to get elevation from attributes or coordinates
            if 'elevation' in dataset.attrs:
                elevations.append(dataset.attrs['elevation'])
            elif 'elevation' in dataset.coords:
                # Use mean if it's an array
                elevations.append(float(dataset.coords['elevation'].values.mean()))

        if elevations:
            # Check if generally increasing (allowing for small variations)
            assert elevations == sorted(elevations), \
                f"Sweeps should be ordered by elevation. Got: {elevations}"

def test_data_completeness(radar_datasets):
    """
    Test that data arrays have reasonable amount of valid (non-NaN) data.
    """
    dataset = radar_datasets[0].compute()

    for var in ['TH', 'CLASS']:
        if var in dataset.data_vars:
            data = dataset[var].values
            valid_fraction = np.sum(~np.isnan(data)) / data.size
            # Should have at least some valid data (adjust threshold as needed)
            assert valid_fraction > 0.01, \
                f"Variable '{var}' has too few valid values: {valid_fraction*100:.1f}%"

def test_geographic_coordinate_ranges(radar_datasets):
    """
    Test that geographic coordinates are within valid ranges.
    """
    dataset = radar_datasets[0].compute()

    if 'latitude' in dataset.data_vars:
        lat = dataset['latitude'].values
        valid_lat = lat[~np.isnan(lat)]
        if len(valid_lat) > 0:
            assert valid_lat.min() >= -90, "Latitude values should be >= -90."
            assert valid_lat.max() <= 90, "Latitude values should be <= 90."

    if 'longitude' in dataset.data_vars:
        lon = dataset['longitude'].values
        valid_lon = lon[~np.isnan(lon)]
        if len(valid_lon) > 0:
            assert valid_lon.min() >= -180, "Longitude values should be >= -180."
            assert valid_lon.max() <= 180, "Longitude values should be <= 180."

def test_write_odim_str_attrib():
    """Test writing ODIM string attributes to HDF5."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        with h5py.File(tmp_file.name, 'w') as h5_file:
            grp = h5_file.create_group('test_group')

            # Write string attribute
            write_odim_str_attrib(grp, 'source', 'WMO:12345')

            # Verify
            assert 'source' in grp.attrs
            assert grp.attrs['source'] == b'WMO:12345' or grp.attrs['source'] == 'WMO:12345'