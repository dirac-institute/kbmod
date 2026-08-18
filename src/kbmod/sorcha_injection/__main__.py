"""Command line entry point: ``python -m kbmod.sorcha_injection``.

Two subcommands:

``build-index``
    The one-time scan over the raw Sorcha outputs. Pass ``--collections`` with the
    ImageCollections you intend to inject into so the index is restricted to visits
    that can actually be used -- for the DP2 collections that alone removes ~94% of
    the simulated visits.

``inspect``
    Print an existing index's provenance and a coverage summary.

Example::

    python -m kbmod.sorcha_injection build-index \\
        --out /path/to/sorcha_index \\
        --populations cc hc cen de \\
        --mag-max 27 \\
        --collections /path/to/combined_all.collection \\
        --workers 32
"""

import argparse
import json
import logging
import sys

import numpy as np


def _visits_from_collections(paths):
    """Read visit ids out of .collection (ECSV) files without materialising them.

    A full DP2 collection is a multi-GB ECSV whose ``wcs`` column contains spaces, so
    a positional split is only safe up to that column. ``visit`` sits at index 2,
    comfortably before it. Files may also be concatenations of several ECSV blocks,
    hence the repeated header handling.
    """
    visits = set()
    for path in paths:
        n = 0
        with open(path) as fh:
            for line in fh:
                if not line or line[0] == "#":
                    continue
                parts = line.split(None, 3)
                if len(parts) < 3 or parts[0] == "dataId":
                    continue
                try:
                    visits.add(int(parts[2]))
                    n += 1
                except ValueError:
                    continue
        logging.info("  %s: %d rows", path, n)
    return np.array(sorted(visits), dtype=np.int64)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m kbmod.sorcha_injection", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    from .config import DEFAULT_POINTING_DB, DEFAULT_POPULATIONS, DEFAULT_SORCHA_ROOT

    b = sub.add_parser("build-index", help="scan the Sorcha outputs into a partitioned index")
    b.add_argument("--out", required=True, help="output directory for the index")
    b.add_argument("--sorcha-root", default=DEFAULT_SORCHA_ROOT)
    b.add_argument("--populations", nargs="+", default=list(DEFAULT_POPULATIONS))
    b.add_argument("--pointing-db", default=DEFAULT_POINTING_DB)
    b.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="ImageCollection files whose visits form the whitelist (strongly recommended)",
    )
    b.add_argument("--mag-max", type=float, default=27.0)
    b.add_argument("--nside", type=int, default=64)
    b.add_argument("--workers", type=int, default=16)
    b.add_argument("--overwrite", action="store_true")

    i = sub.add_parser("inspect", help="summarise an existing index")
    i.add_argument("index", help="path to an index directory")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "build-index":
        from .index import build_sorcha_index

        whitelist = None
        if args.collections:
            logging.info("Reading visit whitelist from %d collection(s)", len(args.collections))
            whitelist = _visits_from_collections(args.collections)
            logging.info("Whitelist: %d distinct visits", len(whitelist))

        meta = build_sorcha_index(
            out_path=args.out,
            sorcha_root=args.sorcha_root,
            populations=tuple(args.populations),
            pointing_db=args.pointing_db,
            visit_whitelist=whitelist,
            mag_max=args.mag_max,
            nside=args.nside,
            n_workers=args.workers,
            overwrite=args.overwrite,
        )
        print(json.dumps(meta, indent=2))
        return 0

    if args.cmd == "inspect":
        from .index import SorchaIndex

        index = SorchaIndex(args.index)
        print(index)
        print(json.dumps(index.meta, indent=2))
        table = index.read(columns=["population", "night", "visit", "trailedSourceMag"])
        if table.num_rows:
            import collections

            pops = collections.Counter(table["population"].to_pylist())
            nights = np.unique(table["night"].to_numpy(zero_copy_only=False))
            mag = table["trailedSourceMag"].to_numpy(zero_copy_only=False)
            print(f"\nrows per population: {dict(pops)}")
            print(f"nights: {len(nights)} spanning MJD {nights.min()}..{nights.max()}")
            print(f"magnitudes: {mag.min():.2f} .. {mag.max():.2f} (median {np.median(mag):.2f})")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
