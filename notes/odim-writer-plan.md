# Plan: writing compliant ODIM H5 from xarray Datasets

Companion to `odim-metadata-audit.md`, which has the evidence behind every claim here.
Target: two entry points.

1. `write_odim(sweeps: list[xr.Dataset], filename)` - round-trip case. The user read a file
   with `read_odim`, added a field, and wants a new ODIM file out.
2. `create_sweep(...) -> xr.Dataset` - synthesis case. The user has bare arrays and wants a
   skeleton Dataset shaped exactly like `read_sweep` output, which `write_odim` then saves.

Both share one writer, so the second case is mostly validation once the first works.

## Step 1 - stop losing metadata in the reader

This is the prerequisite: `write_odim` can only be as faithful as `read_odim`. Everything
here is in `pyodim/metadata.py` and `pyodim/reader.py`.

### 1.1 Read all of `/how` instead of a whitelist

Replace the curated key lists in `get_root_metadata` and `get_dataset_metadata` with
"read every attribute in the group, converted through `_as_python`", landing them in the
container variables of 1.4 rather than in `Dataset.attrs`. A curated list is the
wrong tool: it cannot anticipate vendor extensions (`rapic_*`, `monitoring_*`), and every
missing entry is a silent loss. The whitelist exists only to keep `attrs` tidy, which is not
worth the cost.

Type fidelity comes free as long as we do not cast: `_as_python` already maps an HDF5
`int64` to a Python `int` and a `float64` to a Python `float`, which map back to ODIM `long`
and `double` on write. Keep the explicit `float()`/`int()` casts only for the geometry values
the reader genuinely needs as numbers.

### 1.2 Read per-field `/how` into the variable's attrs

Open `data*/how` and `quality*/how` and merge into each field's attrs (prefix-free; the
variable scopes them). This is what recovers `key_labels`/`key_values`, without which the
`CLASS` field is an unlabelled integer array.

Watch: `rapic_DBZLVL` is a 159-element float array and `key_values` a 15-element int array.
Array-valued attrs are fine for netCDF and `_as_python` already passes them through.

### 1.3 Fix `what/product`

`get_dataset_metadata` looks for `product` in `how`; ODIM puts it in `what`. It is mandatory
for a compliant dataset group, so this must be fixed before any writer ships.

### 1.4 Where the raw `/how` metadata lands (decided)

**`Dataset.attrs` keeps its current 20 names and its current contract.** The raw `/how`
groups go into variables instead, following the CfRadial2 / xradar convention of scalar
container variables (`radar_parameters`, `radar_calibration`, ...):

| source group | lands in | sample file |
| --- | --- | --- |
| root `/how` | scalar coordinate `odim_how_root`, raw attrs on the variable | 17 attrs |
| `datasetN/how` | scalar coordinate `odim_how_sweep`, raw attrs on the variable | 12 attrs |
| `datasetN/dataM/how` | the field variable's own attrs (`ds["CLASS"].attrs`) | 4-11 attrs |
| `what/product` | `ds.attrs["product"]` - mandatory ODIM, one new string key | 1 attr |
| `startazA`/`stopazA`/`startazT`/`stopazT` | real variables on the `azimuth` dim | absent here |

Three reasons this beats widening `attrs`:

- **The collision problem disappears.** Root `/how/X` and sweep `/how/X` live in different
  containers, so provenance is structural rather than something the writer has to infer.
  (Real case: `rpm` is whitelisted at root level today, but this vendor writes it per sweep.)
- **The serialisability contract holds.** `test_attributes_are_serialisable` asserts no
  ndarray in `ds.attrs`; array-valued `how` attributes such as `rapic_DBZLVL` (159 floats)
  would break it. On a variable's attrs they are valid netCDF and the contract is untouched.
- **Field metadata sits on the field it describes.** `key_labels`/`key_values` belong on
  `ds["CLASS"]`, not in a volume-wide namespace.

Use **coordinates, not data variables**, for the two containers: `set(ds.data_vars)` then
stays exactly as it is today, so neither the frozen tests nor a user's
`for name in ds.data_vars` loop sees a metadata variable appear among the fields.

Verified against the sample file: `ds.attrs` unchanged, `ds.data_vars` unchanged, and all
1449 attributes in the file convert to netCDF-safe types (str / int / float / 1-d numeric
array) through the existing `_as_python`. A file carrying a compound-, enum- or
reference-typed attribute would need `_as_python` to grow a fallback; none exists here.

### 1.5 Per-ray azimuth and time arrays

`startazA`/`stopazA` currently collapse to ray centres, and `startazT`/`stopazT` are never
read at all. Keep them when present, as variables on the `azimuth` dimension rather than
attrs, and use `startazT`/`stopazT` for `time` instead of the linear ramp - which makes the
reader more accurate in its own right, independently of the writer.

The sample file has none of these, so this needs a test file that does before it can be
trusted. Lower priority than 1.1-1.3.

### 1.6 The `ODIM_ATTRIBUTES` table

One table in `metadata.py`, `name -> (group, level, hdf5_type)`, covering the attributes
pyodim derives or promotes itself (`lat`, `lon`, `height`, `nrays`, `nbins`, `a1gate`,
`elangle`, `rstart`, `rscale`, `astart`, `product`, `object`, `source`, `version`,
`date`/`time`, `beamwH`, `beamwV`, `wavelength`, `NI`, `highprf`, `lowprf`, ...).

The writer needs it for two things the data alone cannot tell it: which group an attribute
belongs in, and whether to emit it as `double`, `long` or ODIM string. Without it a file has
the right attribute names with the wrong HDF5 types, which validators reject.

It only has to cover the attributes pyodim derives itself. Everything read verbatim carries
its own placement (which container variable it came from, per 1.4) and its own Python type,
which maps back to ODIM `double`/`long`/string without a lookup.

## Step 2 - `encode_field`, the inverse of `decode_field`

In `pyodim/decode.py`, beside `decode_field`:

```python
encode_field(values, gain, offset, nodata, undetect=None, dtype=None) -> np.ndarray
```

- NaN -> `nodata`; `round((values - offset) / gain)` elsewhere.
- `dtype=None` selects the smallest integer type covering the required raw range *including*
  the reserved codes - see the audit, section 3, for why guessing from `nodata` alone is a
  silent-corruption bug.
- Guard: raise (or nudge) when a valid value rounds onto a reserved code.
- Optional `gain`/`offset` derivation for fields the user created with neither:
  `gain = (max - min) / (2**16 - 3)` with 0 reserved for undetect.

## Step 3 - `write_odim`

In `pyodim/writer.py`, beside the existing `write_odim_str_attrib` and `copy_h5_data`.

- Root groups from `sweeps[0].attrs` (root metadata is duplicated into every sweep by the
  reader): `/what`, `/where`, `/how`, plus the `Conventions` root attribute. `attrs["date"]`
  is ISO 8601, so split it back into `%Y%m%d` / `%H%M%S`.
- Per sweep: `what` from `start_time`/`end_time` (already `"YYYYMMDD_HHMMSS"`, split on `_`),
  `where` and `how/astart` inverted from the coordinates per the audit's table.
- Fields: skip the derived variables `x`, `y`, `z`, `longitude`, `latitude`, `prt`. `prt` is
  synthesised from `highprf` by the reader, not an ODIM quantity.
- Honour each field's `id` attr so a `quality1` group is written back as a quality group, not
  as `dataN`. Renumber only when `id` is absent.
- All strings through `write_odim_str_attrib` (`STR_NULLTERM`, as ODIM requires), all numbers
  typed through `ODIM_ATTRIBUTES`.
- `compression="gzip"` on the data arrays, matching what real files use.
- Warn when the sweeps were read with `mask_undetect=True`, since undetect gates will be
  written as nodata (audit, section 3). Consider recording the flag in `attrs` at read time
  so the writer can detect this rather than guess.

## Step 4 - `create_sweep`

```python
create_sweep(*, azimuth | nrays, range | (nbins, rscale, rstart), elangle,
             longitude, latitude, height, start_time, end_time, source,
             fields=None, object="PVOL", **how) -> xr.Dataset
```

Must emit exactly the structure `read_sweep` emits, so that `write_odim` and a subsequent
`read_odim` round-trip it. That comes free by reusing the existing builders rather than
writing new ones: `coord_from_metadata` for range/azimuth/elevation, `generate_timestamp`
for per-ray time, `radar_coordinates_to_xyz` + `height` for `x`/`y`/`z`.

Fields passed without `gain`/`offset`/`nodata`/`undetect` get them from step 2.

A shared `_validate_sweep(ds)` - dims are `("azimuth", "range")`, shapes consistent, range
monotonic increasing, mandatory attrs present - called both here and at the top of
`write_odim`, so a user's hand-merged Dataset fails with a clear message rather than
producing a broken file.

## Acceptance

- `read_odim(write_odim(read_odim(f, mask_undetect=False)))` matches the original field for
  field, attribute for attribute, including the field-level `how` groups.
- The written file opens in `xradar` / `wradlib` and passes an ODIM validator.
- HDF5 attribute types match the source (`double`/`long`/string), not just the values.
- `create_sweep(...)` -> `write_odim` -> `read_odim` returns what went in.
- Existing test suite still passes unchanged: `Dataset.attrs` and `Dataset.data_vars` keep
  their current contents (1.4), so the frozen-value tests need no edits. A new test asserts
  that every raw attribute in the sample file survives a read.

## Order and effort

| Step | Work | Notes |
| --- | --- | --- |
| 1.1-1.3 | ~0.5 day | Unblocks everything; useful on its own |
| 1.6 table | ~0.5 day | Mostly data entry |
| 2 | ~0.5 day | Self-contained, easy to test |
| 3 | ~1 day | The prototype is ~70 lines and already round-trips |
| 4 | ~0.5 day | Mostly validation |
| 1.5 | ~0.5 day | Needs a test file with `startazA`/`startazT` |

No decisions outstanding; everything above is mechanical.

## Open questions

- Do we need a test file with per-ray azimuth/time arrays before shipping 1.5?
- `test_attributes_are_serialisable` cannot run in the current environment (no `netCDF4`,
  `h5netcdf` or `scipy` installed, so `to_netcdf` raises) - worth adding one of them as a
  test dependency, since that test is the guard for everything in 1.4.
- Should `write_odim` default to `ODIM_H5/V2_4`, or echo the source file's `Conventions`?
