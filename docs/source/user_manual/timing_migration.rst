Timing provenance and ImageCollection migration
===============================================

New Butler metadata records the convention ``butler.visitinfo.utc_midpoint.v1``:
``VisitInfo.date`` is the native exposure midpoint, converted to UTC MJD.
The nominal start is the midpoint minus half the exposure duration; the observing
day uses the Rubin/DECam noon-TAI boundary.

Per-image ``timing_*`` columns record the convention, scale, format, standardizer
timing-contract version (``ButlerStandardizer/1``), KBMOD package version, and the
epoch and dataset ID to which that record applies. The timing-contract version
is independent of the package version. Source checkouts without generated version
information explicitly record the package version as ``unknown``.

These columns survive collection packing, selection, concatenation, and WorkUnit
FITS serialization. Search Results record a summary and a hash of the ordered
epochs actually searched. ``timing_provenance`` properties expose the record on
ImageCollections, WorkUnits, and Results. Missing records remain ``unknown``;
loading or rewriting a historical artifact does not certify or upgrade it.
``current`` describes a recognized record consistent with its stored epoch and
dataset ID, not a fresh independent check of the source. Modified epochs/IDs,
mixed records, or unsupported conventions cannot silently become current.

Audit without Butler access
---------------------------

The installed command follows the argument-parser/``execute`` pattern of the
other KBMOD command-line utilities. By default it only audits recorded metadata::

    kbmod-migrate-imagecollections --input /data/collections --report audit.jsonl

Use ``--input`` with a file or a directory tree. The default discovery patterns
are ``**/*.ecsv``, ``**/*.parquet``, and ``**/*.parq``; repeat ``--glob`` to narrow
the scan, for example ``--glob '**/collection*.ecsv'``. Unrelated readable tables
are skipped during a directory scan. Corrupt candidate files are reported as
errors and processing continues. Reports are JSON Lines on stdout and, optionally,
in a new ``--report`` file. Existing report files are never replaced.

The scan first screens ECSV headers and Parquet footers. Unrelated tables are
skipped and collections missing provenance are flagged without reading rows.
For recorded provenance, row-level consistency is checked one table at a time.
No exposure pixels, standardizer construction, or Butler access are needed.
An unmarked collection is ``unknown``, even if its timestamps happen to be correct.

Verify and upgrade
------------------

Check actual source epochs before deciding which files need changes::

    kbmod-migrate-imagecollections --input /data/collections \
        --verify-source --butler /path/to/butler.yaml --report verified.jsonl

Preview a migration, then write corrected copies::

    kbmod-migrate-imagecollections --input /data/collections \
        --upgrade --dry-run --butler /path/to/butler.yaml

    kbmod-migrate-imagecollections --input /data/collections \
        --upgrade --butler /path/to/butler.yaml \
        --output-dir /data/corrected-collections --report migration.jsonl

Source verification requires a Rubin environment and access to the explicitly
selected repository. The utility uses a read-only Butler and reads only dataset
references and ``visitInfo`` components. Dataset IDs must resolve and any stored
visit/detector identities must agree with the source. The default cache retains
up to 4096 dataset records across files; ``--cache-size`` changes this bound.
Calendar arithmetic and time-scale conversions are vectorized across each table.

Only all-Butler collections are supported for source verification and upgrade.
FITS-standardizer collections are reported but not automatically corrected:
historical DECam FITS readers and manually modified products can have different
conventions. A missing record never authorizes subtracting a guessed offset.

An upgrade corrects ``mjd_mid``, ``mjd_start``, ``obs_day``, and ``exposureTime``
from source metadata, adds provenance, and preserves original changed values in
``timing_previous_*`` columns. Configurations, original WCSs, source identities,
and other columns are retained. Migration history records the original path and
SHA-256, source repository, verification time, previous provenance, changed
fields, and removed columns. Repeating verification of an upgraded file is a
no-op. Already-current source-verified files are reported without making another
copy; the output directory therefore contains only files that needed upgrading.

Directory structure is preserved below ``--output-dir``. Originals and existing
destinations are never overwritten. Each output is staged, read back and checked,
then published by an atomic, non-overwriting hard link on the output filesystem.
A failed source lookup or validation leaves no partially upgraded output. A
multi-file run can succeed for some files and fail for others; inspect its report.
The output filesystem must support hard links.

Derived products
----------------

If timestamps change, known reflex-coordinate columns require explicit removal
with ``--drop-reflex``. They must be recomputed at the corrected epochs. Custom
derived columns may also depend on time; review them before using the output.

This utility upgrades ImageCollections only. It does not relabel or migrate
injection catalogs, WorkUnits, reprojected images, Results, ephemerides, or cached
matches. Rebuild those dependent products consistently before comparing recovery
or linking outcomes. Preserve the original artifacts and audit any downstream
manual time offsets to avoid applying the correction twice.

Exit codes
----------

* ``0``: all processed collections are current, verified current, or upgraded
  (unrelated tables may have been skipped).
* ``1``: at least one error, including unavailable sources, unsupported upgrades,
  or an existing destination.
* ``2``: no errors, but unknown/mixed/inconsistent records or proposed upgrades
  remain. This is expected when an audit or dry run finds work to do.
