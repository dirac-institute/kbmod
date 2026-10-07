"""Audit or source-verify and migrate ImageCollection timing metadata.

kbmod-migrate-imagecollections --input collections --report audit.jsonl
kbmod-migrate-imagecollections --input collections --verify-source --butler REPO
kbmod-migrate-imagecollections --input collections --upgrade --butler REPO --output-dir corrected
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from kbmod.image_collection_timing import (
    ButlerTimingResolver,
    collection_columns,
    corrected_collection,
    file_sha256,
    read_collection_table,
    write_upgraded_collection,
)
from kbmod.timing import TIMING_FIELDS, timing_summary

logger = logging.getLogger(__name__)


def find_collection_files(input_path, patterns=None):
    """Find candidate metadata files once; do not follow directory symlinks."""
    root = Path(input_path)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    patterns = patterns or ["**/*.ecsv", "**/*.parquet", "**/*.parq"]
    return sorted({p for pattern in patterns for p in root.glob(pattern) if p.is_file()})


def execute(args):
    """Process one metadata table at a time; emit one JSON record per file.

    Exit 0: all candidates current/skipped; 1: errors; 2: unresolved records
    or a dry-run upgrade is needed. A successful upgrade exits 0.
    """
    root = Path(args.input).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else None
    report_path = Path(args.report).resolve() if args.report else None
    resolver = ButlerTimingResolver(args.butler, cache_size=args.cache_size)
    errors = pending = False
    files = find_collection_files(root, args.glob)
    if not files:
        raise ValueError("No candidate ImageCollection files found")
    stream = open(report_path, "x") if report_path else None
    try:
        for path in files:
            if output and output in path.resolve().parents:
                continue
            record = {"path": str(path)}
            try:
                columns = collection_columns(path)
                # Hash before reading as well as before publication: never bind
                # provenance to a concurrently changed input file.
                original_hash = file_sha256(path) if args.upgrade else None
                if not (args.verify_source or args.upgrade) and not set(TIMING_FIELDS).issubset(columns):
                    record.update(schema_version=1, status="unknown")
                else:
                    data = read_collection_table(path)
                    record.update(timing_summary(data))
                if args.verify_source or args.upgrade:
                    updated, verified = corrected_collection(data, resolver, drop_reflex=args.drop_reflex)
                    record.update(verified)
                    if verified["status"] == "needs_upgrade" and args.upgrade and not args.dry_run:
                        if file_sha256(path) != original_hash:
                            raise ValueError("Source file changed during verification")
                        relative = path.relative_to(root) if root.is_dir() else Path(path.name)
                        destination = output / relative
                        if destination.resolve() == path.resolve():
                            raise ValueError("Destination must differ from the input file")
                        updated.meta["timing_migrations"][-1].update(
                            input_path=str(path), input_sha256=original_hash, butler_repo=args.butler
                        )
                        write_upgraded_collection(updated, destination)
                        record.update(
                            status="upgraded", output_path=str(destination), input_sha256=original_hash
                        )
                if record["status"] not in ("current", "verified_current", "upgraded"):
                    pending = True
            except ValueError as exc:
                if str(exc).startswith("Not an ImageCollection") and root.is_dir():
                    record.update(status="skipped", reason=str(exc))
                else:
                    record.update(status="error", error=str(exc))
                    errors = True
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                errors = True
                logger.debug("Failed to process %s", path, exc_info=True)
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            if stream:
                stream.write(line + "\n")
                stream.flush()
    finally:
        if stream:
            stream.close()
    return 1 if errors else 2 if pending else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="kbmod-migrate-imagecollections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Audit timing provenance or upgrade ImageCollections using native Butler midpoints.",
    )
    parser.add_argument("--input", required=True, help="An ImageCollection file or directory tree")
    parser.add_argument("--glob", action="append", help="Directory glob; repeat to scan multiple patterns")
    parser.add_argument(
        "--verify-source", action="store_true", help="Check native Butler metadata without writing"
    )
    parser.add_argument(
        "--upgrade", action="store_true", help="Verify native timing and write corrected copies"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="With --upgrade, report changes without writing"
    )
    parser.add_argument("--butler", help="Explicit Butler repository/config; used read-only")
    parser.add_argument("--output-dir", help="Output tree; existing files and originals are never replaced")
    parser.add_argument("--report", help="New JSONL audit file (also printed to stdout)")
    parser.add_argument("--cache-size", type=int, default=4096, help="Maximum cached Butler dataset records")
    parser.add_argument(
        "--drop-reflex", action="store_true", help="Remove known reflex-coordinate columns if epochs change"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if (args.verify_source or args.upgrade) and not args.butler:
        parser.error("--butler is required for source verification or upgrades")
    if args.upgrade and not args.dry_run and not args.output_dir:
        parser.error("--output-dir is required for upgrades")
    if args.cache_size < 1:
        parser.error("--cache-size must be positive")
    if args.output_dir:
        root, output = Path(args.input).resolve(), Path(args.output_dir).resolve()
        if output == root or output in root.parents:
            parser.error("--output-dir must not equal or contain --input")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    try:
        return execute(args)
    except Exception as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
