#!/usr/bin/env python3
"""
Test suite for write_odim functionality.
"""

import os
import tempfile
import numpy as np
import xarray as xr
import h5py
import pytest

from pyodim import write_odim, read_odim


def create_test_dataset(elangle=0.5, nrays=360, nbins=500):
    """Create a test xarray dataset with required ODIM structure."""
    azimuth = np.linspace(0, 359, nrays)
    range_bins = np.arange(nbins) * 250.0 + 125.0  # 250m resolution, start at 125m
    
    th_data = np.random.uniform(0, 60, (nrays, nbins))  # Reflectivity
    vrad_data = np.random.uniform(-20, 20, (nrays, nbins))  # Velocity
    
    ds = xr.Dataset(
        {
            'TH': (['azimuth', 'range'], th_data, {
                'gain': 0.5,
                'offset': -32.0,
                'nodata': 255,
                'quantity': 'TH'
            }),
            'VRAD': (['azimuth', 'range'], vrad_data, {
                'gain': 0.5,
                'offset': -63.5,
                'nodata': 255,
                'quantity': 'VRAD'
            })
        },
        coords={
            'azimuth': azimuth,
            'range': range_bins
        },
        attrs={
            'latitude': -37.855,
            'longitude': 144.755,
            'height': 50.0,
            'wavelength': 0.053,
            'beamwH': 1.0,
            'beamwV': 1.0,
            'start_time': '20241112_005000',
            'end_time': '20241112_005500',
            'elangle': elangle,
            'NI': 25.0,
            'highprf': 1200.0,
            'astart': 0.0,
            'nrays': nrays,
            'nbins': nbins,
            'rstart': 125.0,
            'rscale': 250.0,
            'a1gate': 0
        }
    )
    
    return ds


def test_write_odim_basic():
    """Test basic write_odim functionality."""
    datasets = [
        create_test_dataset(elangle=0.5),
        create_test_dataset(elangle=1.5),
        create_test_dataset(elangle=2.5)
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        output_file = tmp.name
    
    try:
        write_odim(datasets, output_file)
        
        assert os.path.exists(output_file)
        
        with h5py.File(output_file, 'r') as f:
            assert 'where' in f
            assert 'what' in f
            assert 'how' in f
            assert 'dataset1' in f
            assert 'dataset2' in f
            assert 'dataset3' in f
            
            assert f.attrs['Conventions'] == b'ODIM_H5/V2_4'
            
            assert f['where'].attrs['lat'] == -37.855
            assert f['where'].attrs['lon'] == 144.755
            assert f['where'].attrs['height'] == 50.0
            
            ds1 = f['dataset1']
            assert 'where' in ds1
            assert 'what' in ds1
            assert 'how' in ds1
            assert 'data1' in ds1
            assert 'data2' in ds1
            
            data1 = ds1['data1']
            assert 'what' in data1
            assert 'data' in data1
            assert data1['data'].shape == (360, 500)
            assert data1['data'].dtype == np.uint8
            
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_odim_validation():
    """Test dataset validation."""
    invalid_ds = xr.Dataset({'temp': (['x'], [1, 2, 3])})
    
    with pytest.raises(ValueError, match="Missing required dimensions"):
        write_odim([invalid_ds], 'test.h5')
    
    ds = create_test_dataset()
    del ds.attrs['latitude']
    
    with pytest.raises(ValueError, match="Missing required attributes"):
        write_odim([ds], 'test.h5')


def test_write_odim_roundtrip():
    """Test writing and reading back ODIM file."""
    original_datasets = [create_test_dataset(elangle=1.0)]
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        output_file = tmp.name
    
    try:
        write_odim(original_datasets, output_file)
        
        read_datasets = read_odim(output_file, lazy_load=False)
        
        assert len(read_datasets) == 1
        ds = read_datasets[0]
        
        assert 'azimuth' in ds.dims
        assert 'range' in ds.dims
        assert ds.sizes['azimuth'] == 360
        assert ds.sizes['range'] == 500
        
        assert 'TH' in ds.data_vars
        assert 'VRAD' in ds.data_vars
        
        assert abs(ds.attrs['latitude'] - (-37.855)) < 0.001
        
        assert 'elevation' in ds.coords
        assert abs(ds.coords['elevation'].values[0] - 1.0) < 0.001
        
    finally:
        if os.path.exists(output_file):
            os.unlink(output_file)


def test_write_odim_empty_list():
    """Test error handling for empty dataset list."""
    with pytest.raises(ValueError, match="datasets list cannot be empty"):
        write_odim([], 'test.h5')


def test_write_odim_invalid_type():
    """Test error handling for invalid input types."""
    with pytest.raises(TypeError, match="datasets must be a list"):
        write_odim("not a list", 'test.h5')
    
    with pytest.raises(TypeError, match="All items in datasets must be xarray.Dataset"):
        write_odim([1, 2, 3], 'test.h5')
