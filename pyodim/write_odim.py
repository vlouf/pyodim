"""
Write ODIM H5 radar files from xarray datasets.

@title: write_odim
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Bureau of Meteorology and Monash University.
@creation: 05/07/2025

.. autosummary::
    :toctree: generated/

    validate_dataset
    write_odim
"""
import datetime
from typing import Dict, List

import h5py
import numpy as np
import xarray as xr

from .pyodim import write_odim_str_attrib


def validate_dataset(dataset: xr.Dataset) -> Dict[str, str]:
    """
    Validate that an xarray dataset has the required structure for ODIM writing.

    Parameters:
    ===========
    dataset: xarray.Dataset
        Dataset to validate for ODIM compliance.

    Returns:
    ========
    errors: dict
        Dictionary of validation errors, empty if valid.
    """
    errors = {}

    required_dims = {'azimuth', 'range'}
    if not required_dims.issubset(set(dataset.dims)):
        missing_dims = required_dims - set(dataset.dims)
        errors['dimensions'] = f"Missing required dimensions: {missing_dims}"

    required_attrs = {
        'latitude', 'longitude', 'height', 'wavelength', 'beamwH', 'beamwV',
        'start_time', 'end_time', 'elangle', 'NI', 'highprf',
        'astart', 'nrays', 'nbins', 'rstart', 'rscale', 'a1gate'
    }
    missing_attrs = required_attrs - set(dataset.attrs.keys())
    if missing_attrs:
        errors['attributes'] = f"Missing required attributes: {missing_attrs}"

    for var_name, var in dataset.data_vars.items():
        if var.dims == ('azimuth', 'range'):
            required_var_attrs = {'gain', 'offset', 'nodata', 'quantity'}
            missing_var_attrs = required_var_attrs - set(var.attrs.keys())
            if missing_var_attrs:
                msg = f"Variable {var_name} missing attributes: {missing_var_attrs}"
                errors[f'{var_name}_attrs'] = msg

    if 'nrays' in dataset.attrs and 'azimuth' in dataset.sizes:
        if dataset.attrs['nrays'] != dataset.sizes['azimuth']:
            nrays_val = dataset.attrs['nrays']
            azimuth_dim = dataset.sizes['azimuth']
            msg = (f"nrays attribute ({nrays_val}) doesn't match "
                   f"azimuth dimension ({azimuth_dim})")
            errors['nrays_consistency'] = msg

    if 'nbins' in dataset.attrs and 'range' in dataset.sizes:
        if dataset.attrs['nbins'] != dataset.sizes['range']:
            nbins_val = dataset.attrs['nbins']
            range_dim = dataset.sizes['range']
            msg = (f"nbins attribute ({nbins_val}) doesn't match "
                   f"range dimension ({range_dim})")
            errors['nbins_consistency'] = msg

    return errors


def _create_root_groups(hfile: h5py.File, dataset: xr.Dataset) -> None:
    """
    Create root-level groups (/where, /what, /how) with metadata.

    Parameters:
    ===========
    hfile: h5py.File
        Open HDF5 file handle.
    dataset: xarray.Dataset
        Dataset containing root metadata.
    """
    write_odim_str_attrib(hfile, 'Conventions', 'ODIM_H5/V2_4')

    where_group = hfile.create_group('where')
    where_group.attrs['lat'] = dataset.attrs['latitude']
    where_group.attrs['lon'] = dataset.attrs['longitude']
    where_group.attrs['height'] = dataset.attrs['height']

    what_group = hfile.create_group('what')

    start_time = dataset.attrs['start_time']
    if isinstance(start_time, str) and '_' in start_time:
        date_str, time_str = start_time.split('_')
        write_odim_str_attrib(what_group, 'date', date_str)
        write_odim_str_attrib(what_group, 'time', time_str)
    else:
        now = datetime.datetime.utcnow()
        write_odim_str_attrib(what_group, 'date', now.strftime('%Y%m%d'))
        write_odim_str_attrib(what_group, 'time', now.strftime('%H%M%S'))

    write_odim_str_attrib(what_group, 'object', dataset.attrs.get('object', 'PVOL'))
    write_odim_str_attrib(what_group, 'source', dataset.attrs.get('source', 'pyodim'))
    write_odim_str_attrib(what_group, 'version', dataset.attrs.get('version', 'pyodim'))

    how_group = hfile.create_group('how')
    how_group.attrs['beamwH'] = dataset.attrs['beamwH']
    how_group.attrs['beamwV'] = dataset.attrs['beamwV']
    how_group.attrs['wavelength'] = dataset.attrs['wavelength']

    for attr in ['copyright', 'rpm']:
        if attr in dataset.attrs:
            if isinstance(dataset.attrs[attr], str):
                write_odim_str_attrib(how_group, attr, dataset.attrs[attr])
            else:
                how_group.attrs[attr] = dataset.attrs[attr]


def _create_dataset_group(hfile: h5py.File, dataset: xr.Dataset, dataset_num: int) -> h5py.Group:
    """
    Create a dataset group (/datasetN) with sweep metadata and data.

    Parameters:
    ===========
    hfile: h5py.File
        Open HDF5 file handle.
    dataset: xarray.Dataset
        Dataset containing sweep data and metadata.
    dataset_num: int
        Dataset number (1-indexed).

    Returns:
    ========
    dataset_group: h5py.Group
        Created dataset group.
    """
    dataset_group = hfile.create_group(f'dataset{dataset_num}')

    where_group = dataset_group.create_group('where')
    where_group.attrs['elangle'] = dataset.attrs['elangle']
    where_group.attrs['nrays'] = dataset.attrs['nrays']
    where_group.attrs['nbins'] = dataset.attrs['nbins']
    where_group.attrs['rstart'] = dataset.attrs['rstart']
    where_group.attrs['rscale'] = dataset.attrs['rscale']
    where_group.attrs['a1gate'] = dataset.attrs['a1gate']

    what_group = dataset_group.create_group('what')

    start_time = dataset.attrs['start_time']
    end_time = dataset.attrs['end_time']

    if isinstance(start_time, str) and '_' in start_time:
        start_date, start_time_str = start_time.split('_')
        write_odim_str_attrib(what_group, 'startdate', start_date)
        write_odim_str_attrib(what_group, 'starttime', start_time_str)

    if isinstance(end_time, str) and '_' in end_time:
        end_date, end_time_str = end_time.split('_')
        write_odim_str_attrib(what_group, 'enddate', end_date)
        write_odim_str_attrib(what_group, 'endtime', end_time_str)

    write_odim_str_attrib(what_group, 'product', 'SCAN')

    how_group = dataset_group.create_group('how')
    how_group.attrs['NI'] = dataset.attrs['NI']
    how_group.attrs['highprf'] = dataset.attrs['highprf']
    how_group.attrs['astart'] = dataset.attrs['astart']

    for attr in ['rpm', 'pulsewidth', 'peakpwrH']:
        if attr in dataset.attrs:
            how_group.attrs[attr] = dataset.attrs[attr]

    return dataset_group


def _write_data_field(dataset_group: h5py.Group, var_name: str,
                      var_data: xr.DataArray, data_num: int) -> None:
    """
    Write a data field to the dataset group.

    Parameters:
    ===========
    dataset_group: h5py.Group
        Dataset group to write data to.
    var_name: str
        Variable name.
    var_data: xr.DataArray
        Variable data and attributes.
    data_num: int
        Data number (1-indexed).
    """
    data_group = dataset_group.create_group(f'data{data_num}')

    what_group = data_group.create_group('what')
    what_group.attrs['gain'] = var_data.attrs['gain']
    what_group.attrs['offset'] = var_data.attrs['offset']
    what_group.attrs['nodata'] = var_data.attrs['nodata']

    if 'undetect' in var_data.attrs:
        what_group.attrs['undetect'] = var_data.attrs['undetect']

    quantity = var_data.attrs.get('quantity', var_name)
    write_odim_str_attrib(what_group, 'quantity', quantity)

    gain = var_data.attrs['gain']
    offset = var_data.attrs['offset']
    nodata = var_data.attrs['nodata']

    data_array = var_data.values

    if hasattr(data_array, 'mask'):
        scaled_data = np.where(data_array.mask, nodata, (data_array.data - offset) / gain)
    else:
        scaled_data = np.where(np.isnan(data_array), nodata, (data_array - offset) / gain)

    scaled_data = np.clip(scaled_data, 0, 255).astype(np.uint8)

    data_group.create_dataset('data', data=scaled_data, compression='gzip', compression_opts=6)


def write_odim(datasets: List[xr.Dataset], output_file: str, validate: bool = True) -> None:
    """
    Write a list of xarray datasets to an ODIM H5 file.

    Parameters:
    ===========
    datasets: List[xr.Dataset]
        List of radar sweep datasets. Each dataset should have:
        - Dimensions: 'azimuth', 'range'
        - Required global attributes: latitude, longitude, height, wavelength,
          beamwH, beamwV, start_time, end_time, elangle, NI, highprf,
          astart, nrays, nbins, rstart, rscale, a1gate
        - Data variables with attributes: gain, offset, nodata, quantity
    output_file: str
        Output ODIM H5 filename.
    validate: bool
        Validate dataset structure before writing (default: True).

    Raises:
    =======
    ValueError: If validation fails or datasets are invalid.
    TypeError: If input types are incorrect.
    """
    if not isinstance(datasets, list):
        raise TypeError("datasets must be a list of xarray.Dataset objects")

    if len(datasets) == 0:
        raise ValueError("datasets list cannot be empty")

    if not all(isinstance(ds, xr.Dataset) for ds in datasets):
        raise TypeError("All items in datasets must be xarray.Dataset objects")

    if validate:
        for i, dataset in enumerate(datasets):
            errors = validate_dataset(dataset)
            if errors:
                error_msg = f"Dataset {i+1} validation failed:\n"
                for key, msg in errors.items():
                    error_msg += f"  {key}: {msg}\n"
                raise ValueError(error_msg)

    sorted_datasets = sorted(datasets, key=lambda ds: ds.attrs['elangle'])

    with h5py.File(output_file, 'w') as hfile:
        root_dataset = sorted_datasets[0]
        _create_root_groups(hfile, root_dataset)

        for dataset_num, dataset in enumerate(sorted_datasets, 1):
            dataset_group = _create_dataset_group(hfile, dataset, dataset_num)

            data_num = 1
            for var_name, var_data in dataset.data_vars.items():
                if var_data.dims == ('azimuth', 'range'):
                    _write_data_field(dataset_group, var_name, var_data, data_num)
                    data_num += 1
