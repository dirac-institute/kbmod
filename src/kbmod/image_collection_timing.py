"""Audit and explicitly migrate Butler ImageCollection timestamps without image I/O."""

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import tempfile
import uuid

import astropy.units as u
from astropy.table import Table
from astropy.table.meta import get_header_from_yaml
from astropy.time import Time
import numpy as np
import pyarrow.parquet as pq

from .image_collection import ImageCollection, pack_table, unpack_table
from .standardizers import ButlerStandardizer
from .timing import BUTLER_MIDPOINT_CONVENTION, TIMING_FIELDS, butler_timing_metadata, timing_summary


class ButlerTimingResolver:
    """Reuse one read-only Butler and a bounded dataset cache across input files.

    Only dataset references and VisitInfo components are requested. The repo
    must be explicit: a stored datastore location need not be a Butler config.
    A Butler instance can be supplied for embedded use and tests.
    """

    def __init__(self, repo=None, *, butler=None, cache_size=4096):
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.repo = repo
        self.butler = butler
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __call__(self, row):
        if str(row["std_name"]) != "ButlerStandardizer":
            raise ValueError("Only ButlerStandardizer rows can be upgraded by this utility")
        identifier = uuid.UUID(str(row["dataId"]))
        if self.butler is None:
            if self.repo is None:
                raise ValueError("An explicit Butler repository is required")
            from lsst.daf.butler import Butler

            self.butler = Butler(self.repo, writeable=False)
        if identifier not in self.cache:
            ref = self.butler.get_dataset(identifier, dimension_records=True)
            if ref is None:
                raise ValueError(f"Dataset {identifier} not found in the supplied Butler repository")
            visit = self.butler.get(ref.makeComponentRef("visitInfo"))
            midpoint = ButlerStandardizer._visit_midpoint_utc(visit)
            exposure = float(visit.exposureTime)
            if not np.isfinite(midpoint.mjd) or not np.isfinite(exposure) or exposure < 0:
                raise ValueError(f"Invalid native timing for dataset {identifier}")
            self.cache[identifier] = (ref, midpoint, exposure)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(identifier)
        ref, midpoint, exposure = self.cache[identifier]
        # A valid UUID must not silently certify a row with different identity.
        for key in ("visit", "detector"):
            if key in row.colnames and (np.ma.is_masked(row[key]) or row[key] != ref.dataId[key]):
                raise ValueError(f"Dataset {identifier}: collection {key} differs from Butler")
        return midpoint, exposure


def collection_format(path):
    """Return the supported lossless table format for an ImageCollection path."""
    suffix = Path(path).suffix.lower()
    if suffix == ".ecsv":
        return "ascii.ecsv"
    if suffix in (".parquet", ".parq"):
        return "parquet"
    raise ValueError("Supported ImageCollection formats are .ecsv, .parquet, and .parq")


def collection_columns(path):
    """Screen table headers/Parquet footers without loading table rows."""
    if collection_format(path) == "parquet":
        from .results import Results

        with pq.ParquetFile(path) as parquet:
            names = set(parquet.schema_arrow.names)
            meta = Results._extract_parquet_metadata(parquet)
    else:
        lines = []
        with open(path, encoding="utf-8") as stream:
            first = stream.readline()
            if not first.startswith("# %ECSV"):
                raise ValueError("Invalid ECSV header")
            for line in stream:
                if not line.startswith("#"):
                    break
                lines.append(line[1:].rstrip("\n"))
        header = get_header_from_yaml(lines)
        names = {col["name"] for col in header.get("datatype", [])}
        meta = header.get("meta", {})
    names.update(meta.get("shared_cols", []))
    required = set(ImageCollection.required_metadata + ImageCollection._supporting_metadata)
    if not required.issubset(names):
        raise ValueError("Not an ImageCollection: missing required metadata/support columns")
    return names


def read_collection_table(path):
    """Read metadata only; never construct standardizers or load exposures."""
    collection_columns(path)
    data = Table.read(path, format=collection_format(path))
    if data.meta.get("is_packed", False):
        data = unpack_table(data)
    required = set(ImageCollection.required_metadata + ImageCollection._supporting_metadata)
    if not required.issubset(data.colnames):
        raise ValueError("Not an ImageCollection: missing required metadata/support columns")
    return data


def corrected_collection(data, resolver, *, drop_reflex=False):
    """Return an independently source-verified copy and a compact change report.

    Every row is checked before any file is written. Unknown conventions are
    not reverse-engineered from exposure lengths. Known reflex-coordinate
    caches must be explicitly dropped if epochs change. Other custom derived
    columns require the caller's review, as their semantics are not inferable.
    """
    if not len(data):
        raise ValueError("Empty ImageCollection")
    if "dataId" not in data.colnames or not np.all(data["std_name"] == "ButlerStandardizer"):
        raise ValueError("Only all-Butler ImageCollections with dataset IDs are supported for upgrades")
    if "timing_convention" in data.colnames:
        known = np.ma.asarray(data["timing_convention"]).compressed()
        if any(value not in (BUTLER_MIDPOINT_CONVENTION, "unknown", "") for value in known):
            raise ValueError("Unsupported recorded convention; refusing to downgrade it")
    old_times = np.ma.asarray(data["mjd_mid"], dtype=float).filled(np.nan)
    if not np.all(np.isfinite(old_times)):
        raise ValueError("Stored mjd_mid must be finite and unmasked")
    jd1, jd2, durations = [], [], []
    for row in data:
        midpoint, exposure = resolver(row)
        midpoint = midpoint.utc
        jd1.append(midpoint.jd1)
        jd2.append(midpoint.jd2)
        durations.append(exposure)
    # Vectorize time-scale conversion/calendar arithmetic and retain two-part
    # Julian-date precision when deriving the nominal starts.
    midpoints = Time(jd1, jd2, format="jd", scale="utc")
    durations = np.asarray(durations)
    native = midpoints.mjd
    starts = (midpoints - durations / 2 * u.s).utc.mjd
    days = ButlerStandardizer._mjd_to_obs_day(native)
    delta = (native - old_times) * 86400
    updated = data.copy(copy_data=True)
    new_columns = {"mjd_mid": native, "mjd_start": starts, "obs_day": days, "exposureTime": durations}
    changed_fields = []
    for key, values in new_columns.items():
        if key not in data.colnames or not np.ma.allequal(data[key], values, fill_value=False):
            changed_fields.append(key)
            if key in data.colnames and f"timing_previous_{key}" not in updated.colnames:
                updated[f"timing_previous_{key}"] = data[key].copy()
        updated[key] = values

    reflex = [c for c in data.colnames if re.match(r"^(ra|dec)(_(tl|tr|bl|br))?_[+-]?\d", c)]
    dropped = []
    if "mjd_mid" in changed_fields and reflex:
        if not drop_reflex:
            raise ValueError("Epoch changes invalidate reflex coordinates; use --drop-reflex to remove them")
        updated.remove_columns(reflex)
        dropped = reflex

    previous = timing_summary(data)
    needs_update = bool(changed_fields) or previous["status"] != "current"
    if needs_update:
        records = [butler_timing_metadata(t, row["dataId"]) for t, row in zip(native, data)]
        for field in TIMING_FIELDS:
            updated[field] = [record[field] for record in records]
        history = list(updated.meta.get("timing_migrations", []))
        history.append(
            {
                "schema_version": 1,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "Butler VisitInfo.date (UTC); metadata only",
                "previous_record": previous,
                "convention": BUTLER_MIDPOINT_CONVENTION,
                "changed_fields": changed_fields,
                "dropped_columns": dropped,
                "dependent_products": "Rebuild injection catalogs, WorkUnits/reprojections and time-dependent matches",
            }
        )
        updated.meta["timing_migrations"] = history
    report = dict(
        status="needs_upgrade" if needs_update else "verified_current",
        rows=len(data),
        changed_fields=changed_fields,
        dropped_columns=dropped,
        changed_midpoints=int(np.count_nonzero(delta)),
        min_correction_seconds=float(delta.min()),
        max_correction_seconds=float(delta.max()),
        previous_record=previous,
    )
    return updated, report


def file_sha256(path):
    """Stream a file checksum without holding file bytes in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_upgraded_collection(data, destination):
    """Validate a staged round trip, then publish atomically without overwriting.

    The temporary file and destination share a filesystem. Hard-link creation
    provides atomic no-clobber publication, including concurrent writers.
    """
    destination = Path(destination)
    fmt = collection_format(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".kbmod-timing-", suffix=destination.suffix, dir=destination.parent
    )
    os.close(fd)
    try:
        pack_table(data.copy(copy_data=True)).write(temporary, format=fmt, overwrite=True)
        loaded = read_collection_table(temporary)
        if set(loaded.colnames) != set(data.colnames) or len(loaded) != len(data):
            raise ValueError("Staged migration changed table shape")
        for key in data.colnames:
            np.testing.assert_array_equal(np.ma.getmaskarray(data[key]), np.ma.getmaskarray(loaded[key]))
            np.testing.assert_array_equal(data[key], loaded[key])
        if timing_summary(loaded)["status"] != "current":
            raise ValueError("Staged migration lost timing provenance")
        if loaded.meta.get("timing_migrations") != data.meta.get("timing_migrations"):
            raise ValueError("Staged migration lost audit history")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
