"""
Conversion between ODIM integer storage and physical float32 values.
(`decode_field` today; its inverse will live here when the writer is added.)
"""

from typing import Optional

import numpy as np


def decode_field(
    raw: np.ndarray,
    gain: float,
    offset: float,
    nodata: Optional[float],
    undetect: Optional[float] = None,
    mask_undetect: bool = True,
) -> np.ndarray:
    """
    Decode an ODIM integer field into physical values: `gain * raw + offset`
    as float32, with `nodata` (and `undetect`, if `mask_undetect`) set to NaN.
    8- and 16-bit unsigned data are decoded through a lookup table.

    Parameters:
    ===========
    raw: ndarray
        Stored (encoded) data.
    gain, offset: float
        ODIM scaling attributes.
    nodata: float or None
        Encoded value for missing data.
    undetect: float or None
        Encoded value for "no echo detected".
    mask_undetect: bool
        Whether `undetect` gates become NaN (default) or keep their decoded value.

    Returns:
    ========
    data: ndarray<float32>
    """
    gain32 = np.float32(gain)
    offset32 = np.float32(offset)
    specials = [nodata] if nodata is not None else []
    if mask_undetect and undetect is not None:
        specials.append(undetect)

    if raw.dtype.kind == "u" and raw.dtype.itemsize <= 2:
        table = gain32 * np.arange(np.iinfo(raw.dtype).max + 1, dtype=np.float32) + offset32
        remaining = []
        for value in specials:
            fvalue = float(value)
            if fvalue.is_integer() and 0 <= fvalue < table.size:
                table[int(fvalue)] = np.nan
            else:
                remaining.append(value)
        data = np.take(table, raw)
        for value in remaining:  # non-integer special values: cannot go through the table
            data[raw == value] = np.nan
        return data

    data = raw.astype(np.float32)
    data *= gain32
    data += offset32
    for value in specials:
        data[raw == value] = np.nan
    return data
