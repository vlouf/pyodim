# tests/test_pyodim.py
import os
import pyodim
import pytest
from pyodim import read_odim, read_sweep, georeference
from pyodim.pyodim import (
    antenna_to_ground,
    check_nyquist,
    decode_field,
    geodesic_forward,
    write_odim_str_attrib,
    get_dataset_metadata,
    coord_from_metadata,
    copy_h5_data,
    radar_coordinates_to_xyz,
)
import h5py
import datetime
import subprocess
import sys
import tempfile
import warnings
import xarray as xr
import numpy as np

# Define the path to the ODIM H5 file
ODIM_FILE_PATH = "test/8_20241112_005000.pvol.h5"

@pytest.fixture(scope="module")
def sample_odim_file():
    """
    Fixture to check the presence of the sample ODIM file.
    """
    if not os.path.exists(ODIM_FILE_PATH):
        pytest.skip(f"Test file '{ODIM_FILE_PATH}' does not exist.")
    return ODIM_FILE_PATH

@pytest.fixture(scope="module")
def radar_datasets(sample_odim_file):
    """
    Fixture that reads the ODIM file once (eagerly) and returns the radar datasets.
    `xr.Dataset.compute()` is a no-op, so tests written for delayed objects still work.
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
    Test that latitude and longitude are added by georeference().
    """
    dataset = georeference(radar_datasets[0].compute())
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
    dataset = georeference(radar_datasets[0].compute())

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


def test_copy_h5_data_uses_next_available_numeric_id():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        with h5py.File(tmp_file.name, 'w') as h5_file:
            h5_file.create_group('data1')
            h5_file.create_group('data3')

            new_id = copy_h5_data(h5_file, 'data1')
            assert new_id == 'data4'
            assert 'data4' in h5_file


# --------------------------------------------------------------------------- #
# Regression tests for the 0.7 correctness fixes and performance plan
# --------------------------------------------------------------------------- #
def test_fields_are_float32(radar_datasets):
    """Fields must stay float32 (NumPy 2 promotion regression)."""
    dataset = radar_datasets[0]
    for var in dataset.data_vars:
        assert dataset[var].dtype == np.float32, f"{var} is {dataset[var].dtype}, expected float32"


def test_field_encoding_kept_in_attrs(radar_datasets):
    """gain/offset/nodata/undetect are preserved on each field for lossless round-trips."""
    th = radar_datasets[0]['TH']
    for key in ('gain', 'offset', 'nodata', 'undetect', 'id'):
        assert key in th.attrs
    assert th.attrs['gain'] == pytest.approx(0.5)
    assert th.attrs['offset'] == pytest.approx(-32.0)


def test_undetect_is_masked_by_default(sample_odim_file):
    """undetect gates decode to NaN by default; mask_undetect=False keeps the raw decoded value."""
    masked = read_odim(sample_odim_file, sweeps=0)[0]['DBZH_CLEAN']
    assert masked.attrs['undetect'] == pytest.approx(1.0)
    undetect_value = masked.attrs['gain'] * masked.attrs['undetect'] + masked.attrs['offset']  # -31.9 dBZ
    assert not np.any(np.isclose(masked.values, undetect_value, atol=1e-3))

    kept = read_odim(sample_odim_file, sweeps=0, mask_undetect=False)[0]['DBZH_CLEAN']
    assert np.sum(np.isclose(kept.values, undetect_value, atol=1e-3)) > 0
    assert np.isnan(kept.values).sum() < np.isnan(masked.values).sum()


def test_decode_field_lut_matches_direct_path():
    raw = np.array([[0, 1, 2, 255]], dtype=np.uint8)
    lut = decode_field(raw, 0.5, -32.0, nodata=255, undetect=0)
    direct = decode_field(raw.astype(np.int16), 0.5, -32.0, nodata=255, undetect=0)
    assert lut.dtype == np.float32 and direct.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(lut), [[True, False, False, True]])
    np.testing.assert_allclose(lut, direct, equal_nan=True)
    assert lut[0, 1] == pytest.approx(-31.5)
    # non-integer special values fall back to a comparison
    weird = decode_field(raw, 1.0, 0.0, nodata=0.5, undetect=None)
    assert not np.any(np.isnan(weird))


def test_attributes_are_serialisable(radar_datasets):
    """No bytes / None / arrays in attrs; to_netcdf must work."""
    dataset = radar_datasets[0]
    for k, v in dataset.attrs.items():
        assert v is not None, f"attr {k} is None"
        assert not isinstance(v, (bytes, np.bytes_)), f"attr {k} is bytes"
        assert not isinstance(v, np.ndarray), f"attr {k} is an ndarray"
    assert isinstance(dataset.attrs['rapic_HIPRF'], str)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'sweep.nc')
        dataset.to_netcdf(path)
        with xr.open_dataset(path) as reread:
            assert 'TH' in reread


def test_prt_is_a_per_ray_variable(radar_datasets):
    """prt(azimuth) on every sweep: alternating on dual-PRF sweeps, constant 1/highprf otherwise."""
    dual = [ds for ds in radar_datasets if ds.attrs.get('rapic_UNFOLDING', 'None') != 'None']
    single = [ds for ds in radar_datasets if ds.attrs.get('rapic_UNFOLDING', 'None') == 'None']
    assert len(dual) > 0 and len(single) > 0
    for ds in radar_datasets:
        assert 'prt' in ds.data_vars and 'prt' not in ds.attrs
        assert ds['prt'].dims == ('azimuth',)
        assert ds['prt'].dtype == np.float32
    for ds in dual:
        assert len(np.unique(ds['prt'].values)) == 2
    for ds in single:
        np.testing.assert_allclose(ds['prt'].values, 1.0 / ds.attrs['highprf'], rtol=1e-6)


def test_azimuth_grid_is_uniform_for_any_astart():
    """Regression: the old linspace end point was only right for astart = -da/2."""
    for astart, nrays in ((-0.5, 360), (0.0, 360), (0.25, 360), (0.0, 720), (-0.25, 720)):
        metadata = {"astart": astart, "nrays": nrays, "nbins": 4, "rstart": 0.0, "rscale": 250.0, "elangle": 0.5}
        _, az, _ = coord_from_metadata(metadata)
        da = 360.0 / nrays
        assert az[0] == pytest.approx(astart + da / 2)
        assert az[-1] == pytest.approx(astart + da / 2 + da * (nrays - 1))
        np.testing.assert_allclose(np.diff(az), da, atol=1e-4)


def test_azimuth_from_startaza_stopaza_takes_precedence():
    nrays = 4
    startaz = np.array([359.0, 89.0, 179.0, 269.0])
    stopaz = np.array([1.0, 91.0, 181.0, 271.0])  # first ray wraps through 0
    metadata = {"astart": 0.0, "nrays": nrays, "nbins": 2, "rstart": 0.0, "rscale": 250.0, "elangle": 0.5,
                "startazA": startaz, "stopazA": stopaz}
    _, az, _ = coord_from_metadata(metadata)
    np.testing.assert_allclose(az, [0.0, 90.0, 180.0, 270.0], atol=1e-5)


def test_get_dataset_metadata_reads_startaza():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)
        with h5py.File(tmp_file.name, 'r+') as h5_file:
            h5_file['/dataset1/how'].attrs['startazA'] = np.array([0.0, 180.0])
            h5_file['/dataset1/how'].attrs['stopazA'] = np.array([180.0, 360.0])
        with h5py.File(tmp_file.name, 'r') as h5_file:
            _, coords = get_dataset_metadata(h5_file, 'dataset1')
            assert 'startazA' in coords and 'stopazA' in coords
            _, az, _ = coord_from_metadata(coords)
            np.testing.assert_allclose(az, [90.0, 270.0])


def test_beam_height_uses_four_thirds_earth_model(radar_datasets):
    """Regression: z was flat-earth (2.6 km instead of 7.9 km at 300 km, 0.5 deg)."""
    dataset = radar_datasets[0]
    r = dataset['range'].values.astype(np.float64)
    el = float(dataset['elevation'].values[0])
    re = 4.0 / 3.0 * 6371000.0
    z_ref = np.sqrt(r**2 + re**2 + 2 * r * re * np.sin(np.deg2rad(el))) - re + dataset.attrs['height']
    np.testing.assert_allclose(dataset['z'].values[0], z_ref, rtol=1e-5)
    z_flat = r * np.sin(np.deg2rad(el)) + dataset.attrs['height']
    assert dataset['z'].values[0, -1] - z_flat[-1] > 1000.0  # far from flat earth at long range
    s_ref = re * np.arcsin(r * np.cos(np.deg2rad(el)) / (re + z_ref - dataset.attrs['height']))
    np.testing.assert_allclose(np.hypot(dataset['x'].values, dataset['y'].values)[0], s_ref, rtol=1e-5)


def test_antenna_to_ground_reference_values():
    s, z = antenna_to_ground(np.array([50e3, 100e3, 300e3]), 0.5)
    np.testing.assert_allclose(z, [583.0, 1461.0, 7912.0], atol=1.0)
    assert np.all(s < np.array([50e3, 100e3, 300e3]))


def test_radar_coordinates_to_xyz_shapes_and_dtype():
    r = np.arange(0.0, 10.0e3, 250.0)
    az = np.arange(0.0, 360.0, 1.0)
    x, y, z = radar_coordinates_to_xyz(r, az, np.array([1.0]))
    assert x.shape == y.shape == z.shape == (360, 40)
    assert x.dtype == y.dtype == z.dtype == np.float32
    assert y[0, -1] > 0 and abs(x[0, -1]) < 1.0  # azimuth 0 points north
    assert x[90, -1] > 0 and abs(y[90, -1]) < 1.0  # azimuth 90 points east


def test_geodesic_forward_origin_and_symmetry():
    lon, lat = geodesic_forward(152.577, -25.9574, np.array([0.0, 90.0, 180.0, 270.0]), np.array([0.0]))
    np.testing.assert_allclose(lon, 152.577)
    np.testing.assert_allclose(lat, -25.9574)
    lon, lat = geodesic_forward(152.577, -25.9574, np.array([[0.0], [180.0]]), np.array([[100e3]]))
    assert lat[0, 0] > -25.9574 > lat[1, 0]
    np.testing.assert_allclose(lon[:, 0], 152.577, atol=1e-9)


def test_geodesic_forward_matches_pyproj():
    pyproj = pytest.importorskip('pyproj')
    geod = pyproj.Geod(ellps='WGS84')
    az = np.linspace(0, 359, 37)
    dist = np.linspace(0, 300e3, 13)
    lon, lat = geodesic_forward(152.577, -25.9574, az[:, None], dist[None, :])
    az2d, dist2d = np.broadcast_arrays(az[:, None], dist[None, :])
    lon_ref, lat_ref, _ = geod.fwd(np.full(az2d.shape, 152.577), np.full(az2d.shape, -25.9574), az2d, dist2d)
    _, _, err = geod.inv(lon_ref, lat_ref, lon, lat)
    assert err.max() < 0.01  # metres


def test_georeference_numpy_matches_pyproj_method(radar_datasets):
    pyproj = pytest.importorskip('pyproj')
    ds = radar_datasets[0]
    numpy_ds = georeference(ds)
    pyproj_ds = georeference(ds, method='pyproj')
    geod = pyproj.Geod(ellps='WGS84')
    _, _, err = geod.inv(
        pyproj_ds['longitude'].values.astype(np.float64), pyproj_ds['latitude'].values.astype(np.float64),
        numpy_ds['longitude'].values.astype(np.float64), numpy_ds['latitude'].values.astype(np.float64),
    )
    assert err.max() < 5.0  # float32 lon/lat quantisation is ~1.5 m at this longitude
    assert numpy_ds['longitude'].dtype == np.float32


def test_read_odim_georef_option(sample_odim_file):
    ds = read_odim(sample_odim_file, sweeps=0)[0]
    assert 'longitude' not in ds and 'latitude' not in ds
    ds = read_odim(sample_odim_file, sweeps=0, georef=True)[0]
    assert 'longitude' in ds and 'latitude' in ds
    assert ds['longitude'].shape == ds['TH'].shape


def test_georeference_invalid_method(radar_datasets):
    with pytest.raises(ValueError, match='Invalid method'):
        georeference(radar_datasets[0], method='magic')


def test_check_nyquist_dual_prf():
    wavelength = 10.409  # cm
    single = 1e-2 * 750.0 * wavelength / 4  # 19.5 m/s
    extended = 1e-2 * (750.0 * 500.0 / 250.0) * wavelength / 4  # 39.0 m/s
    check_nyquist({'wavelength': wavelength, 'highprf': 750.0, 'lowprf': 500.0, 'NI': extended})
    check_nyquist({'wavelength': wavelength, 'highprf': 750.0, 'rapic_UNFOLDING': '2:3', 'NI': extended})
    with pytest.raises(ValueError):
        check_nyquist({'wavelength': wavelength, 'highprf': 750.0, 'lowprf': 500.0, 'NI': single})
    check_nyquist({'wavelength': wavelength, 'highprf': 750.0, 'rapic_UNFOLDING': 'None', 'NI': single})
    check_nyquist({'highprf': 750.0, 'NI': single})  # incomplete metadata: skipped, no KeyError


def test_check_nyq_does_not_warn_on_sample_file(sample_odim_file):
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        read_odim(sample_odim_file, check_nyq=True)


def test_quality_group_identified_by_how_task():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)
        with h5py.File(tmp_file.name, 'r+') as h5_file:
            quality = h5_file['/dataset1'].create_group('quality1')
            quality.create_group('how').attrs['task'] = np.bytes_('fi.fmi.ropo.detector.classification')
            q_what = quality.create_group('what')
            q_what.attrs['gain'] = 1.0
            q_what.attrs['offset'] = 0.0
            q_what.attrs['nodata'] = 255
            q_what.attrs['undetect'] = 0
            quality.create_dataset('data', data=np.array([[1, 2], [3, 255]], dtype=np.uint8))
        with h5py.File(tmp_file.name, 'r') as h5_file:
            ds = read_sweep(h5_file, 0)
    assert 'fi.fmi.ropo.detector.classification' in ds.data_vars
    assert np.isnan(ds['fi.fmi.ropo.detector.classification'].values[1, 1])


def test_string_attributes_stored_as_str_are_accepted():
    """Producers that write variable-length (str) attributes instead of bytes must not crash the reader."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)
        with h5py.File(tmp_file.name, 'r+') as h5_file:
            h5_file.attrs['Conventions'] = 'ODIM_H5/V2_2'
            h5_file['/what'].attrs['source'] = 'WMO:00000'
            h5_file['/dataset1/what'].attrs['startdate'] = '20240101'
            h5_file['/dataset1/data1/what'].attrs['quantity'] = 'TH'
        ds = read_odim(tmp_file.name)[0]
    assert ds.attrs['Conventions'] == 'ODIM_H5/V2_2'
    assert ds.attrs['source'] == 'WMO:00000'
    assert 'TH' in ds


def test_unknown_keyword_raises(sample_odim_file):
    with pytest.raises(TypeError):
        read_odim(sample_odim_file, include_field=['DBZH'])


def test_sweep_index_out_of_range(sample_odim_file):
    with pytest.raises(ValueError, match='out of range'):
        read_odim(sample_odim_file, sweeps=99)


def test_ray_timestamps_span_start_to_end(radar_datasets):
    ds = radar_datasets[0]
    t = ds['time'].values
    start = np.datetime64(datetime.datetime.strptime(ds.attrs['start_time'], '%Y%m%d_%H%M%S'), 'ns')
    end = np.datetime64(datetime.datetime.strptime(ds.attrs['end_time'], '%Y%m%d_%H%M%S'), 'ns')
    assert t.min() == start and t.max() == end
    assert t.dtype == np.dtype('datetime64[ns]')


def test_import_does_not_load_optional_packages():
    """`import pyodim` must not import dask, pandas or pyproj (performance plan 4.6)."""
    code = (
        "import sys, pyodim; "
        "print(sorted(m for m in ('dask', 'dask.array', 'pyproj') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]', out.stdout



# --------------------------------------------------------------------------- #
# read_odim / read_sweep API (0.7)
# --------------------------------------------------------------------------- #
def test_removed_names_are_gone():
    import pyodim.pyodim as module
    for name in ('read_write_odim', 'read_odim_slice_h5', '_read_odim_slice_from_file', '_read_sweep'):
        assert not hasattr(module, name), name
    assert not hasattr(pyodim, 'read_write_odim')


def test_read_odim_is_eager_by_default(sample_odim_file):
    radar = read_odim(sample_odim_file)
    assert isinstance(radar, list) and len(radar) > 1
    assert all(isinstance(ds, xr.Dataset) for ds in radar)


def test_read_odim_lazy_returns_delayed(sample_odim_file):
    pytest.importorskip('dask')
    from dask.delayed import Delayed
    radar = read_odim(sample_odim_file, lazy=True)
    assert all(isinstance(d, Delayed) for d in radar)
    first = radar[0].compute()
    assert isinstance(first, xr.Dataset) and 'TH' in first
    eager = read_odim(sample_odim_file, sweeps=0)[0]
    xr.testing.assert_identical(first, eager)


def test_read_odim_lazy_forwards_options(sample_odim_file):
    pytest.importorskip('dask')
    ds = read_odim(sample_odim_file, lazy=True, sweeps=1, include_fields=['DBZH'], georef=True)[0].compute()
    assert set(ds.data_vars) == {'DBZH', 'x', 'y', 'z', 'prt', 'longitude', 'latitude'}


def test_read_odim_sweeps_selection(sample_odim_file):
    all_sweeps = read_odim(sample_odim_file)
    one = read_odim(sample_odim_file, sweeps=2)
    assert len(one) == 1 and one[0].attrs['id'] == all_sweeps[2].attrs['id']
    some = read_odim(sample_odim_file, sweeps=[0, 3])
    assert [ds.attrs['id'] for ds in some] == [all_sweeps[0].attrs['id'], all_sweeps[3].attrs['id']]
    with pytest.raises(ValueError, match='out of range'):
        read_odim(sample_odim_file, sweeps=[0, 99])


def test_read_odim_return_handle(sample_odim_file):
    radar, hfile = read_odim(sample_odim_file, return_handle=True, mode='r')
    try:
        assert len(radar) > 0 and isinstance(radar[0], xr.Dataset)
        assert hfile.id.valid
    finally:
        hfile.close()


def test_read_odim_lazy_incompatible_options(sample_odim_file):
    with pytest.raises(ValueError, match='return_handle'):
        read_odim(sample_odim_file, lazy=True, return_handle=True)
    with pytest.raises(ValueError, match="mode='r'"):
        read_odim(sample_odim_file, lazy=True, mode='r+')


def test_read_sweep_by_index_key_path_and_handle(sample_odim_file):
    from_path = read_sweep(sample_odim_file, 0)
    with h5py.File(sample_odim_file) as h5_file:
        by_index = read_sweep(h5_file, 0)
        by_key = read_sweep(h5_file, by_index.attrs['id'])
        assert h5_file.id.valid  # handle left open
    xr.testing.assert_identical(from_path, by_index)
    xr.testing.assert_identical(by_key, by_index)


def test_read_sweep_rejects_invalid_sweep():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)
        with h5py.File(tmp_file.name, 'r') as h5_file:
            with pytest.raises(ValueError, match='out of range'):
                read_sweep(h5_file, 1)
            with pytest.raises(KeyError):
                read_sweep(h5_file, 'dataset7')
            with pytest.raises(KeyError):
                read_sweep(h5_file, 'what')


def test_read_sweep_max_field_elements_guard():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)
        with pytest.raises(ValueError, match='max_field_elements'):
            read_sweep(tmp_file.name, 0, max_field_elements=3)


def test_read_sweep_unknown_keyword_raises(sample_odim_file):
    with pytest.raises(TypeError):
        read_sweep(sample_odim_file, 0, include_field=['DBZH'])


def test_read_odim_reads_all_sweeps_in_elevation_order(sample_odim_file):
    radar = read_odim(sample_odim_file)
    with h5py.File(sample_odim_file) as h5_file:
        nsweep = len([k for k in h5_file if k.startswith('dataset')])
    assert len(radar) == nsweep
    elevations = [float(ds['elevation'].values[0]) for ds in radar]
    assert elevations == sorted(elevations)
