"""
HDF5 write helpers for ODIM files (ODIM-conformant string attributes, dataset
duplication). `write_odim` / `empty_sweep` will be added here.
"""

import h5py
import numpy as np


def copy_h5_data(h5_tilt, orig_id: str) -> str:
    """Add a data array to `h5_tilt` by copying data with `orig_id`.
    This function is used to duplicate an existing data array in the HDF5 file
    and return the new data ID.
    The new data ID is generated based on the current number of data arrays
    in the HDF5 file, ensuring that it is unique.

    Parameters:
    ===========
    h5_tilt: h5py.File
        HDF5 Dataset tilt where the data will be copied.
    orig_id: str
        The ID of the original data array to be copied.
    """
    if orig_id not in h5_tilt:
        raise KeyError(f"Cannot copy missing field id '{orig_id}'.")

    existing_indices = [int(k[4:]) for k in h5_tilt.keys() if k.startswith("data") and k[4:].isdigit()]
    next_index = (max(existing_indices) + 1) if existing_indices else 1
    data_id = f"data{next_index}"
    while data_id in h5_tilt:
        next_index += 1
        data_id = f"data{next_index}"

    # duplicate original
    h5_tilt.copy(orig_id, data_id)

    return data_id


def odim_str_type_id(text_bytes: bytes) -> h5py.h5t.TypeID:
    """
    Generate ODIM-conformant HDF5 string type ID with null-termination.

    Parameters
    ----------
    text_bytes : bytes
        Byte string for which to create the type ID.

    Returns
    -------
    h5py.h5t.TypeID
        String type ID sized for text_bytes with STR_NULLTERM padding.
    """
    # h5py default string type is STRPAD STR_NULLPAD
    # ODIM spec string type is STRPAD STR_NULLTERM
    type_id = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
    type_id.set_strpad(h5py.h5t.STR_NULLTERM)
    type_id.set_size(len(text_bytes) + 1)
    return type_id


def write_odim_str_attrib(group, attrib_name: str, text: str) -> None:
    """
    Write ODIM-conformant h5py string attribute.
    If the attribute already exists, it will be overwritten.

    Parameters:
    ===========
    group:
        h5py group to which the attribute will be added.
    attrib_name:
        name of the attribute to be added.
    text:
        text to be written as the attribute value.
    """
    if attrib_name in group.attrs:
        del group.attrs[attrib_name]

    group_id = group.id
    text_bytes = text.encode("utf-8")
    type_id = odim_str_type_id(text_bytes)
    space = h5py.h5s.create(h5py.h5s.SCALAR)
    att_id = h5py.h5a.create(group_id, attrib_name.encode("utf-8"), type_id, space)
    text_array = np.array(text_bytes)
    att_id.write(text_array)

    return None
