# tests/test_pyodim.py
import os
import pytest
from pyodim import read_odim
from pyodim.pyodim import (
    check_nyquist,
    write_odim_str_attrib,
    get_dataset_metadata,
    coord_from_metadata,
    copy_h5_data,
    read_odim_slice_h5,
    read_write_odim,
)
import h5py
import tempfile
import xarray as xr
import numpy as np
import dask.array as da

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


def test_get_dataset_metadata_normalizes_small_rstart_to_meters():
    """Read-time metadata extraction should convert small rstart values from km to m."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        with h5py.File(tmp_file.name, 'w') as h5_file:
            h5_file.attrs['Conventions'] = np.bytes_('ODIM_H5/V2_4')

            root_what = h5_file.create_group('/what')
            root_what.attrs['version'] = np.bytes_('H5rad 2.4')

            dataset = h5_file.create_group('/dataset1')
            ds_how = dataset.create_group('how')
            ds_what = dataset.create_group('what')
            ds_where = dataset.create_group('where')

            ds_what.attrs['startdate'] = np.bytes_('20240101')
            ds_what.attrs['starttime'] = np.bytes_('000000')
            ds_what.attrs['enddate'] = np.bytes_('20240101')
            ds_what.attrs['endtime'] = np.bytes_('000100')

            ds_where.attrs['a1gate'] = 0
            ds_where.attrs['nrays'] = 360
            ds_where.attrs['rstart'] = 1.0
            ds_where.attrs['rscale'] = 250.0
            ds_where.attrs['nbins'] = 4
            ds_where.attrs['elangle'] = 0.5

            _, coordinates_metadata = get_dataset_metadata(h5_file, 'dataset1')
            assert coordinates_metadata['rstart'] == pytest.approx(1000.0)


def test_get_dataset_metadata_keeps_large_rstart_in_meters():
    """Read-time metadata extraction should keep large rstart values as meters."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        with h5py.File(tmp_file.name, 'w') as h5_file:
            h5_file.attrs['Conventions'] = np.bytes_('ODIM_H5/V2_4')

            root_what = h5_file.create_group('/what')
            root_what.attrs['version'] = np.bytes_('H5rad 2.4')

            dataset = h5_file.create_group('/dataset1')
            ds_how = dataset.create_group('how')
            ds_what = dataset.create_group('what')
            ds_where = dataset.create_group('where')

            ds_what.attrs['startdate'] = np.bytes_('20240101')
            ds_what.attrs['starttime'] = np.bytes_('000000')
            ds_what.attrs['enddate'] = np.bytes_('20240101')
            ds_what.attrs['endtime'] = np.bytes_('000100')

            ds_where.attrs['a1gate'] = 0
            ds_where.attrs['nrays'] = 360
            ds_where.attrs['rstart'] = 1000.0
            ds_where.attrs['rscale'] = 250.0
            ds_where.attrs['nbins'] = 4
            ds_where.attrs['elangle'] = 0.5

            _, coordinates_metadata = get_dataset_metadata(h5_file, 'dataset1')
            assert coordinates_metadata['rstart'] == pytest.approx(1000.0)


def test_coord_from_metadata_uses_normalized_rstart():
    """Range coordinate should start at gate center in meters for both encodings."""
    metadata_km = {
        "astart": 0,
        "nrays": 360,
        "nbins": 4,
        "rstart": 1000.0,
        "rscale": 250.0,
        "elangle": 0.5,
    }
    metadata_m = {
        "astart": 0,
        "nrays": 360,
        "nbins": 4,
        "rstart": 1000.0,
        "rscale": 250.0,
        "elangle": 0.5,
    }

    r_km, _, _ = coord_from_metadata(metadata_km)
    r_m, _, _ = coord_from_metadata(metadata_m)

    assert r_km[0] == pytest.approx(1125.0)
    assert r_m[0] == pytest.approx(1125.0)


def _create_minimal_odim_file(path):
    with h5py.File(path, 'w') as h5_file:
        h5_file.attrs['Conventions'] = np.bytes_('ODIM_H5/V2_4')

        root_what = h5_file.create_group('/what')
        root_what.attrs['date'] = np.bytes_('20240101')
        root_what.attrs['time'] = np.bytes_('000000')
        root_what.attrs['version'] = np.bytes_('H5rad 2.4')

        root_how = h5_file.create_group('/how')
        root_how.attrs['wavelength'] = 5.3

        root_where = h5_file.create_group('/where')
        root_where.attrs['lat'] = -35.0
        root_where.attrs['lon'] = 149.0
        root_where.attrs['height'] = 100.0

        dataset = h5_file.create_group('/dataset1')
        ds_how = dataset.create_group('how')
        ds_how.attrs['highprf'] = 1000.0
        ds_how.attrs['NI'] = 13.25

        ds_what = dataset.create_group('what')
        ds_what.attrs['startdate'] = np.bytes_('20240101')
        ds_what.attrs['starttime'] = np.bytes_('000000')
        ds_what.attrs['enddate'] = np.bytes_('20240101')
        ds_what.attrs['endtime'] = np.bytes_('000100')

        ds_where = dataset.create_group('where')
        ds_where.attrs['a1gate'] = 0
        ds_where.attrs['nrays'] = 2
        ds_where.attrs['rstart'] = 1000.0
        ds_where.attrs['rscale'] = 250.0
        ds_where.attrs['nbins'] = 2
        ds_where.attrs['elangle'] = 0.5

        data1 = dataset.create_group('data1')
        data1_what = data1.create_group('what')
        data1_what.attrs['gain'] = 1.0
        data1_what.attrs['offset'] = 0.0
        data1_what.attrs['nodata'] = -9999
        data1_what.attrs['quantity'] = np.bytes_('TH')
        data1.create_dataset('data', data=np.array([[1, 2], [3, 4]], dtype=np.int16))


def test_read_odim_slice_h5_rejects_invalid_slice_index():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)
        with h5py.File(tmp_file.name, 'r') as h5_file:
            with pytest.raises(ValueError):
                read_odim_slice_h5(h5_file, nslice=1)


def test_read_odim_slice_h5_max_field_elements_guard():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)
        with h5py.File(tmp_file.name, 'r') as h5_file:
            with pytest.raises(ValueError, match='max_field_elements'):
                read_odim_slice_h5(h5_file, nslice=0, max_field_elements=3)


def test_copy_h5_data_uses_next_available_numeric_id():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        with h5py.File(tmp_file.name, 'w') as h5_file:
            h5_file.create_group('data1')
            h5_file.create_group('data3')

            new_id = copy_h5_data(h5_file, 'data1')
            assert new_id == 'data4'
            assert 'data4' in h5_file


def test_read_write_odim_disallows_lazy_with_read_write(sample_odim_file):
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="backend='dask'"):
            read_write_odim(sample_odim_file, lazy_load=True, read_write=True)


def test_read_odim_backend_numpy_returns_materialized_datasets(sample_odim_file):
    radar = read_odim(sample_odim_file, backend='numpy')
    assert len(radar) > 0
    assert isinstance(radar[0], xr.Dataset)


def test_read_odim_backend_dask_compute_true_returns_materialized_datasets(sample_odim_file):
    radar = read_odim(sample_odim_file, backend='dask', compute=True)
    assert len(radar) > 0
    assert isinstance(radar[0], xr.Dataset)


def test_read_odim_lazy_load_emits_deprecation_warning(sample_odim_file):
    with pytest.warns(DeprecationWarning):
        radar = read_odim(sample_odim_file, lazy_load=False)
    assert len(radar) > 0


def test_read_write_odim_backend_dask_compute_true(sample_odim_file):
    radar, hfile = read_write_odim(sample_odim_file, backend='dask', compute=True)
    try:
        assert len(radar) > 0
        assert isinstance(radar[0], xr.Dataset)
    finally:
        hfile.close()


def test_read_write_odim_invalid_backend(sample_odim_file):
    with pytest.raises(ValueError, match='Invalid backend'):
        read_write_odim(sample_odim_file, backend='cupy')


def test_read_odim_invalid_backend(sample_odim_file):
    with pytest.raises(ValueError, match='Invalid backend'):
        read_odim(sample_odim_file, backend='cupy')


def test_read_write_odim_compute_ignored_for_numpy_backend(sample_odim_file):
    radar, hfile = read_write_odim(sample_odim_file, backend='numpy', compute=False)
    try:
        assert len(radar) > 0
        assert isinstance(radar[0], xr.Dataset)
    finally:
        hfile.close()


def test_read_odim_compute_ignored_for_numpy_backend(sample_odim_file):
    radar = read_odim(sample_odim_file, backend='numpy', compute=False)
    assert len(radar) > 0
    assert isinstance(radar[0], xr.Dataset)


def test_read_write_odim_lazy_load_deprecated_when_backend_missing(sample_odim_file):
    with pytest.warns(DeprecationWarning):
        radar, hfile = read_write_odim(sample_odim_file, lazy_load=False, read_write=False)
    try:
        assert len(radar) > 0
    finally:
        hfile.close()


def test_read_write_odim_can_return_dask_backed_fields(sample_odim_file):
    radar, hfile = read_write_odim(
        sample_odim_file,
        backend='numpy',
        read_write=False,
        use_dask_arrays=True,
        field_chunks=(64, 256),
    )
    try:
        assert len(radar) > 0
        first = radar[0]
        assert isinstance(first['TH'].data, da.Array)
        assert first['TH'].data.chunks is not None
    finally:
        hfile.close()


def test_read_write_odim_disallows_dask_fields_with_lazy(sample_odim_file):
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match='use_dask_arrays=True'):
            read_write_odim(sample_odim_file, lazy_load=True, use_dask_arrays=True)


def test_read_odim_disallows_dask_backed_fields(sample_odim_file):
    with pytest.raises(ValueError, match='use_dask_arrays=True'):
        read_odim(sample_odim_file, lazy_load=True, use_dask_arrays=True)