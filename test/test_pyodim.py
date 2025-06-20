# tests/test_pyodim.py

import os
import pytest
from pyodim import read_odim
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


def test_read_odim(sample_odim_file):
    """
    Test the read_odim function to ensure it correctly reads an ODIM H5 file and
    returns an xarray Dataset with expected properties.
    """
    # Call read_odim on the test file and get the dataset
    rsets = read_odim(sample_odim_file)
    assert len(rsets) > 1, "No sweeps in radar datasets found"
    
    dataset = rsets[0].compute()
    # Assert that the output is an xarray Dataset
    assert isinstance(dataset, xr.Dataset), "Output is not an xarray Dataset."

    # Check that the dataset is not empty
    assert len(dataset.data_vars) > 0, "Dataset has no data variables."

    # Check for specific metadata or data variables (example: 'latitude' or 'longitude' might be expected)
    assert 'latitude' in dataset.data_vars, "Latitude coordinate is missing."
    assert 'longitude' in dataset.data_vars, "Longitude coordinate is missing."

    # Optional: Check for specific radar fields (e.g., reflectivity, if expected)
    assert 'TH' in dataset.data_vars, "Expected data variable 'reflectivity' is missing."
    assert 'CLASS' in dataset.data_vars, "Expected data variable 'classification' is missing"

    # Check a basic property of the data (example: shape or value range)
    assert dataset['TH'].shape[0] > 0, "Reflectivity data is empty."

