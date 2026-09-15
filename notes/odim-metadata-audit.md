# ODIM metadata audit: what pyodim reads, what it drops

Context: pyodim 0.7.0, audited 2026-09-15 against `test/8_20241112_005000.pvol.h5`
(BoM Kanigan PVOL, 13 sweeps, 360 x 1196, 7 data + 1 quality group per sweep).

The question behind this audit: **if we want `write_odim(sweeps, filename)` to produce a
file as complete as the one we read, what must the reader stop throwing away?**

## 1. Scale of the loss

The sample file carries 1450 HDF5 attributes (26 at root, 1424 across sweeps and their
data groups), spread over ~74 distinct names. `read_odim` currently exposes **20 names**
in `Dataset.attrs` plus 5 per field (`gain`, `offset`, `nodata`, `undetect`, `id`).

Everything else is dropped by the two whitelists in `metadata.py`:

- `get_root_metadata` keeps `Conventions`, `lat`/`lon`/`height`, `date`+`time`, `object`,
  `source`, `version`, and only `beamwH`, `beamwV`, `rpm`, `wavelength`, `copyright` from `/how`.
- `get_dataset_metadata` keeps `startdate`/`starttime`/`enddate`/`endtime` and only
  `NI`, `highprf`, `lowprf`, `product`, `prt`, `rapic_UNFOLDING`, `rapic_HIPRF` from `dataset/how`.
- Per-field `how` groups are never opened at all.

## 2. The three categories

### 2a. Geometry attributes: consumed, but fully recoverable

`nrays`, `nbins`, `rstart`, `rscale`, `elangle`, `astart`, `a1gate` feed
`coord_from_metadata` and are never stored in `attrs`. All are invertible from the
coordinates, verified against the sample file:

| ODIM attribute | recovered from the Dataset | checked |
| --- | --- | --- |
| `where/nrays`, `where/nbins` | `ds.sizes["azimuth"]`, `ds.sizes["range"]` | yes |
| `where/rscale` | `diff(ds.range)[0]` | yes |
| `where/rstart` | `ds.range[0] - rscale/2` | yes |
| `where/elangle` | `ds.elevation[0]` | yes |
| `how/astart` | `ds.azimuth[0] - 180/nrays` | yes (-0.5 exact) |
| `where/a1gate` | `argmin(ds.time)` - `generate_timestamp` ends in `np.roll(t, a1gate)` | yes (95 == 95) |

**No reader change needed for these.** Two caveats:

- `astart` inversion assumes a regular azimuth grid. When a file supplies per-ray
  `startazA`/`stopazA`, `coord_from_metadata` keeps only the ray centres, so the ray
  extents and the regular-grid assumption are both gone. The sample file has no
  `startazA`, so this path is currently untested either way.
- `startazT`/`stopazT` (per-ray times) are never read. `time` is synthesised as a linear
  ramp between `starttime` and `endtime`, so a written file can only ever carry that
  approximation - even when the source had exact per-ray times.

### 2b. `/how` metadata: genuinely lost, and the reason for step 1 of the plan

Dropped from the sample file:

- **root `/how`** (12 of 17): `system`, `sw_version`, `nsampleH`, `monitoring_az_error`,
  `monitoring_el_error`, `monitoring_calibration`, `rapic_ANTDIAM`, `rapic_AZCORR`,
  `rapic_ELCORR`, `rapic_FREQUENCY`, `rapic_PRODUCT`, `rapic_RXGAIN_H`, `rapic_VOLUMEID`.
- **sweep `/how`** (6 of 12): `pulsewidth`, `polmode`, `peakpwrH`, `rpm`, `scan_index`,
  `scan_count`. (`rpm` is whitelisted at root level only, but this vendor writes it per sweep.)
- **every field `/how`**: `key_labels`, `key_values`, `coarse_key_labels`,
  `coarse_key_values`, `rapic_DBZLVL` (159 floats), `rapic_VIDRES`, `rapic_QC0..QC4`,
  `rapic_SCHEDULE`, `rapic_VIDEOGAIN`, `rapic_VIDEOOFFSET`, `rapic_VIDEOUNITS`.

The field-level loss is the worst of the three. The `CLASS` quality field decodes to
integers 1-15 whose meaning lives entirely in its `how` group:

```
key_labels = "conv,sconv,strat,insect,smoke,chaff,bird,cx_gnd,ap_gnd,ap_sea,cx_sea,2trip,eemit,speck,blip"
key_values = [1 2 3 ... 15]
```

Read it with pyodim today and the field becomes an unlabelled integer array.

### 2c. `what/product`: a whitelist bug

`get_dataset_metadata` looks for `product` in the sweep's `how` group, but ODIM puts it in
`what` (and so does this file). It is therefore never captured, and it is **mandatory** for a
compliant SCAN/PVOL dataset. This is the only mandatory attribute currently missing.

## 3. Field encoding

`gain`, `offset`, `nodata`, `undetect` and the group `id` are kept per field, so re-encoding
is arithmetic. Two findings from the prototype:

**The storage dtype is not recoverable from the retained attributes, and guessing it fails
silently.** The prototype first chose the dtype from the magnitude of `nodata`/`undetect`,
picked `uint8` for a `uint16` field, and wrapped 400 gates of `DBZH_CLEAN`
(raw 512/513 -> 0/1 = `nodata`/`undetect`) with no error. Choosing the dtype from the
*required raw range* instead, `(max(values) - offset)/gain` widened to include the reserved
codes, reproduces every source dtype exactly:

| field | raw range | dtype chosen | dtype in file |
| --- | --- | --- | --- |
| `DBZH` | 0..195 | uint8 | uint8 |
| `QCFLAGS` | 0..23 | uint8 | uint8 |
| `DBZH_CLEAN` | 0..773 | uint16 | uint16 |
| `VRADDH` | 0..3775 | uint16 | uint16 |
| `CLASS` | -2..14 | int8 | int8 |

Either derive it this way or have the reader record `dtype` in the field attrs; deriving is
enough, and keeps working for fields the user created himself.

**`mask_undetect=True` (the default) is lossy for round-trips.** `undetect` and `nodata` both
become NaN and cannot be told apart afterwards, so every undetect gate is written back as
nodata. On `DBZH_CLEAN` in sweep `dataset13` that is 318,359 undetect gates against 54,067
nodata gates. `mask_undetect=False` is the lossless path: an undetect gate keeps
`gain*undetect + offset`, which re-encodes to exactly `undetect`.

## 4. Prototype result

A ~70-line throwaway writer (root/sweep/field groups rebuilt from `attrs` + coordinates,
fields re-encoded, strings through the existing `write_odim_str_attrib`) round-trips the
whole sample volume through `read_odim -> write -> read_odim`:

- 13 sweeps, same field names, same shapes
- `DBZH`, `VRADH` bit-identical; `DBZH_CLEAN` identical once the dtype rule above is used
- `azimuth`, `range`, per-ray `time` identical
- attrs lost in the round trip: `copyright`, `rapic_HIPRF`, `rapic_UNFOLDING` - i.e. only
  what the prototype did not bother to write, not something structurally impossible

**Conclusion: writing compliant ODIM is not blocked by anything in the format or the data
model. It is blocked by the reader's whitelists.**
