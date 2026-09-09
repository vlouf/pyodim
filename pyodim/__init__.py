"""
Natively reading ODIM H5 radar files in Python.

@title: pyodim
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Bureau of Meteorology and Monash University.
@creation: 21/01/2020

Modules
-------
reader    read_odim, read_sweep
georef    georeference and the radar geometry functions
metadata  ODIM attribute readers and the field metadata table
decode    integer <-> physical value conversion
writer    HDF5 write helpers
"""

# primary read routines
from .reader import read_odim
from .reader import read_sweep
from .georef import georeference

# helper routines
from .writer import copy_h5_data
from .writer import write_odim_str_attrib

__all__ = ["read_odim", "read_sweep", "georeference", "copy_h5_data", "write_odim_str_attrib"]
