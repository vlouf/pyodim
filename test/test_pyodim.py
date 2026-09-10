# tests/test_pyodim.py
import os
import pyodim
import pytest
from pyodim import read_odim, read_sweep, georeference
from pyodim.metadata import check_nyquist, get_dataset_metadata
from pyodim.georef import antenna_to_ground, coord_from_metadata, geodesic_forward, radar_coordinates_to_xyz
from pyodim.decode import decode_field
from pyodim.writer import copy_h5_data, write_odim_str_attrib
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

# --------------------------------------------------------------------------- #
# Reading the sample file: independent oracle (plain h5py + numpy) and frozen values
# --------------------------------------------------------------------------- #
def _reference_decode(group):
    """Decode one ODIM data group the long way: raw * gain + offset, nodata/undetect -> NaN."""
    what = dict(group['what'].attrs)
    raw = group['data'][()]
    values = float(what['gain']) * raw.astype(np.float64) + float(what['offset'])
    missing = (raw == what['nodata']) | (raw == what['undetect'])
    return np.where(missing, np.nan, values), what


def test_read_matches_independent_h5py_decode(sample_odim_file, radar_datasets):
    """The basic test: every field of every sweep equals a hand-decoded read of the HDF5 file."""
    with h5py.File(sample_odim_file) as h5_file:
        keys = [k for k in h5_file if k.startswith('dataset')]
        expected_order = sorted(
            keys, key=lambda k: (float(h5_file[k]['where'].attrs['elangle']), h5_file[k]['what'].attrs['starttime'])
        )
        assert [ds.attrs['id'] for ds in radar_datasets] == expected_order

        for ds in radar_datasets:
            group = h5_file[ds.attrs['id']]
            where = dict(group['where'].attrs)
            nrays, nbins = int(where['nrays']), int(where['nbins'])
            fields_checked = 0
            for key in group:
                if not key.startswith(('data', 'quality')):
                    continue
                expected, what = _reference_decode(group[key])
                name = what['quantity'].decode()
                got = ds[name]
                assert got.dims == ('azimuth', 'range')
                assert got.shape == (nrays, nbins) == expected.shape
                assert got.dtype == np.float32
                np.testing.assert_array_equal(np.isnan(got.values), np.isnan(expected), err_msg=name)
                # atol: float32 quantisation of values up to ~300 (uint16 fields), far below the 0.1 finest gain
                np.testing.assert_allclose(got.values, expected, rtol=1e-6, atol=1e-4, equal_nan=True, err_msg=name)
                assert got.attrs['id'] == key
                fields_checked += 1
            assert fields_checked == 8

            # coordinates straight from the ODIM attributes
            rstart_m = float(where['rstart']) * (1e3 if where['rstart'] < 10 else 1.0)
            np.testing.assert_allclose(ds['range'].values, rstart_m + where['rscale'] / 2 + where['rscale'] * np.arange(nbins))
            astart = float(group['how'].attrs['astart'])
            np.testing.assert_allclose(ds['azimuth'].values, astart + 0.5 + np.arange(nrays), atol=1e-5)
            assert ds['elevation'].values.tolist() == [pytest.approx(float(where['elangle']), abs=1e-6)]
            assert ds.attrs['NI'] == float(group['how'].attrs['NI'])
            assert ds.attrs['highprf'] == float(group['how'].attrs['highprf'])

        assert radar_datasets[0].attrs['latitude'] == float(h5_file['where'].attrs['lat'])
        assert radar_datasets[0].attrs['longitude'] == float(h5_file['where'].attrs['lon'])
        assert radar_datasets[0].attrs['height'] == float(h5_file['where'].attrs['height'])
        assert radar_datasets[0].attrs['source'] == h5_file['what'].attrs['source'].decode()


def test_sample_file_frozen_values(radar_datasets):
    """Values computed once with plain h5py (not pyodim) and frozen here; a change means the reader changed."""
    assert len(radar_datasets) == 13
    elevations = [float(ds['elevation'].values[0]) for ds in radar_datasets]
    np.testing.assert_allclose(elevations, [0.5, 0.8, 1.4, 2.4, 3.5, 4.7, 6.0, 7.8, 10.0, 13.0, 17.0, 23.0, 32.0], atol=1e-6)
    assert [ds.attrs['id'] for ds in radar_datasets][:3] == ['dataset13', 'dataset12', 'dataset11']

    ds = radar_datasets[0]
    assert ds.attrs['source'] == 'RAD:AU08,PLC:Kanign,CTY:500,STN:40625'
    assert (ds.attrs['latitude'], ds.attrs['longitude'], ds.attrs['height']) == (-25.9574, 152.577, 375.0)
    assert ds.attrs['start_time'] == '20241112_005421' and ds.attrs['end_time'] == '20241112_005451'
    assert set(ds.data_vars) == {'DBZH', 'VRADH', 'WRADH', 'TH', 'QCFLAGS', 'DBZH_CLEAN', 'VRADDH', 'CLASS',
                                 'x', 'y', 'z', 'prt'}
    assert dict(ds.sizes) == {'azimuth': 360, 'range': 1196, 'elevation': 1, 'time': 360}

    dbzh = ds['DBZH'].values
    assert dbzh[0, :6].tolist() == [25.0, 23.5, 19.0, 3.0, 5.0, 23.5]
    assert np.all(np.isnan(dbzh[200, 100:104]))
    assert int(np.isfinite(dbzh).sum()) == 68593
    assert float(np.nanmean(dbzh)) == pytest.approx(14.3925, abs=1e-3)
    assert float(np.nanmax(dbzh)) == 65.5 and float(np.nanmin(dbzh)) == -30.0
    assert ds['range'].values[0] == 1125.0 and ds['range'].values[-1] == 1125.0 + 250.0 * 1195
    assert ds['azimuth'].values[0] == 0.0 and ds['azimuth'].values[-1] == 359.0


def test_every_sweep_has_the_same_layout(radar_datasets):
    first = radar_datasets[0]
    for ds in radar_datasets:
        assert set(ds.data_vars) == set(first.data_vars)
        assert set(ds.coords) == {'range', 'azimuth', 'elevation', 'time'}
        for name in ds.data_vars:
            expected_dims = ('azimuth',) if name == 'prt' else ('azimuth', 'range')
            assert ds[name].dims == expected_dims, name
            assert ds[name].dtype == np.float32, name
        assert ds.sizes['time'] == ds.sizes['azimuth']
        assert ds.sizes['elevation'] == 1


def test_physical_plausibility(radar_datasets):
    """Reflectivity in a sane dBZ range with a meaningful fraction of echoes; CLASS is integer-valued."""
    for ds in radar_datasets:
        for name in ('TH', 'DBZH'):
            values = ds[name].values
            valid = values[np.isfinite(values)]
            assert 0.01 < valid.size / values.size < 0.9, name
            assert valid.min() >= -40 and valid.max() <= 80, name
        cls = ds['CLASS'].values
        cls_valid = cls[np.isfinite(cls)]
        assert cls_valid.size > 0
        np.testing.assert_array_equal(cls_valid, np.round(cls_valid))
        assert np.all(np.diff(ds['range'].values) == 250.0)
        assert ds['azimuth'].values.min() >= 0 and ds['azimuth'].values.max() < 360


def test_ray_times_are_consistent_with_a1gate(sample_odim_file, radar_datasets):
    with h5py.File(sample_odim_file) as h5_file:
        for ds in radar_datasets:
            a1gate = int(h5_file[ds.attrs['id']]['where'].attrs['a1gate'])
            t = ds['time'].values
            unrolled = np.roll(t, -a1gate)  # acquisition order
            assert np.all(np.diff(unrolled).astype(np.int64) > 0)
            assert np.argmin(t) == a1gate
            start = np.datetime64(datetime.datetime.strptime(ds.attrs['start_time'], '%Y%m%d_%H%M%S'), 'ns')
            end = np.datetime64(datetime.datetime.strptime(ds.attrs['end_time'], '%Y%m%d_%H%M%S'), 'ns')
            assert unrolled[0] == start and unrolled[-1] == end


def test_georeference_is_anchored_to_the_site(radar_datasets):
    ds = georeference(radar_datasets[0])
    lat0, lon0 = ds.attrs['latitude'], ds.attrs['longitude']
    assert ds['latitude'].dims == ds['longitude'].dims == ('azimuth', 'range')
    # first gate (1.125 km) is within ~0.02 deg of the site, in every direction
    assert np.abs(ds['latitude'].values[:, 0] - lat0).max() < 0.02
    assert np.abs(ds['longitude'].values[:, 0] - lon0).max() < 0.02
    # due north at 300 km: ~2.7 deg of latitude, same longitude; due east: longitude grows
    north = int(np.argmin(np.abs(ds['azimuth'].values - 0.0)))
    east = int(np.argmin(np.abs(ds['azimuth'].values - 90.0)))
    assert ds['latitude'].values[north, -1] - lat0 == pytest.approx(300.0 / 111.0, abs=0.1)
    assert ds['longitude'].values[north, -1] == pytest.approx(lon0, abs=1e-3)
    assert ds['longitude'].values[east, -1] > lon0 + 2.5
    assert np.all(np.abs(ds['latitude'].values) <= 90) and np.all(np.abs(ds['longitude'].values) <= 180)


# --------------------------------------------------------------------------- #
# Unit tests of the helpers
# --------------------------------------------------------------------------- #
def test_check_nyquist_single_prf():
    """ODIM stores wavelength in cm: 5.3 cm at 1000 Hz gives a 13.25 m/s Nyquist velocity."""
    check_nyquist({'wavelength': 5.3, 'highprf': 1000.0, 'NI': 13.25})
    check_nyquist(xr.Dataset(attrs={'wavelength': 5.3, 'highprf': 1000.0, 'NI': 13.25}))
    with pytest.raises(ValueError, match='Nyquist'):
        check_nyquist({'wavelength': 5.3, 'highprf': 1000.0, 'NI': 26.5})


def test_write_odim_str_attrib_is_null_terminated_fixed_length():
    """ODIM requires fixed-length, null-terminated strings (not h5py's default null-padded ones)."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        with h5py.File(tmp_file.name, 'w') as h5_file:
            grp = h5_file.create_group('test_group')
            write_odim_str_attrib(grp, 'source', 'WMO:12345')
            write_odim_str_attrib(grp, 'source', 'WMO:54321')  # overwrite
            assert grp.attrs['source'] == b'WMO:54321'
            type_id = h5py.h5a.open(grp.id, b'source').get_type()
            assert type_id.get_class() == h5py.h5t.STRING
            assert type_id.get_strpad() == h5py.h5t.STR_NULLTERM
            assert type_id.get_size() == len(b'WMO:54321') + 1
            assert not type_id.is_variable_str()


def test_get_dataset_metadata_normalizes_small_rstart_to_meters():
    """Read-time metadata extraction should convert small rstart values from km to m (BOM legacy files)."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)
        with h5py.File(tmp_file.name, 'r+') as h5_file:
            h5_file['/dataset1/where'].attrs['rstart'] = 1.0
        with h5py.File(tmp_file.name, 'r') as h5_file:
            _, coordinates_metadata = get_dataset_metadata(h5_file, 'dataset1')
            assert coordinates_metadata['rstart'] == pytest.approx(1000.0)


def test_get_dataset_metadata_keeps_large_rstart_in_meters():
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        _create_minimal_odim_file(tmp_file.name)  # rstart = 1000.0 m
        with h5py.File(tmp_file.name, 'r') as h5_file:
            _, coordinates_metadata = get_dataset_metadata(h5_file, 'dataset1')
            assert coordinates_metadata['rstart'] == pytest.approx(1000.0)


def test_coord_from_metadata_range_is_gate_centre():
    metadata = {"astart": 0, "nrays": 360, "nbins": 4, "rstart": 1000.0, "rscale": 250.0, "elangle": 0.5}
    r, az, elev = coord_from_metadata(metadata)
    np.testing.assert_allclose(r, [1125.0, 1375.0, 1625.0, 1875.0])
    assert r.dtype == np.float32 and az.shape == (360,) and elev.tolist() == [0.5]


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
    import pyodim.reader as module
    for name in ('read_write_odim', 'read_odim_slice_h5', '_read_odim_slice_from_file', '_read_sweep'):
        assert not hasattr(module, name), name
        assert not hasattr(pyodim, name), name
    assert not hasattr(pyodim, 'pyodim')  # the monolithic module is gone


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


def test_read_from_in_memory_buffer(sample_odim_file, radar_datasets):
    """h5py accepts file-like objects, so zip members can be read without touching disk."""
    import io
    import zipfile
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_file = os.path.join(tmpdir, 'archive.zip')
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(sample_odim_file, 'volume.h5')
        with zipfile.ZipFile(zip_file) as zf:
            payload = zf.read('volume.h5')
    from_buffer = read_odim(io.BytesIO(payload), sweeps=[0, 5])
    xr.testing.assert_identical(from_buffer[0], radar_datasets[0])
    xr.testing.assert_identical(from_buffer[1], radar_datasets[5])
    xr.testing.assert_identical(read_sweep(io.BytesIO(payload), 3), radar_datasets[3])
