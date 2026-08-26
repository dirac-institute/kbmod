"""
Commandline tool for priming an ImageCollection with reflex-corrected columns
once, so that subsequent calls to ``region_searcher.py`` (which read the primed
IC via ``--ic-path``) skip the expensive parallax computation.

The primed IC is just a normal ``.collection`` file with the reflex-corrected
RA/Dec/corner columns already populated. ``RegionSearch.__init__`` detects
those columns and skips ``ic.reflex_correct`` automatically.

Example
-------

>>> python prime_ic.py \
        --ic-path path/to/base.collection \
        --output-path path/to/base_primed_42.collection \
        --guess-distances 42.0 \
        --bands-to-drop u \
        --max-wcs-err 0.1 \
        --obs-site Rubin
"""

import argparse
import os
import time

import kbmod
from astropy.coordinates import EarthLocation


def elapsed_t(start_time, sigfigs=2):
    return f"[{round(time.time() - start_time, sigfigs)} s elapsed]"


def prime_ic(
    ic_path,
    output_path,
    guess_distances,
    site_name="Rubin",
    bands_to_drop=None,
    max_wcs_err=None,
    overwrite=False,
):
    """Read ``ic_path``, apply filters, reflex-correct, and write to ``output_path``.

    Skips the work entirely if ``output_path`` already exists and ``overwrite`` is False.
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"Primed IC already exists at {output_path}. Pass --overwrite to regenerate.")
        return

    start_time = time.time()
    print(f"{elapsed_t(start_time)} Reading base ImageCollection from {ic_path}...")
    ic = kbmod.ImageCollection.read(ic_path)
    print(f"{elapsed_t(start_time)} Loaded {len(ic)} rows.")

    if bands_to_drop:
        print(f"Dropping bands from ImageCollection: {bands_to_drop}.")
        curr_len = len(ic)
        ic.drop_bands(bands_to_drop)
        print(f"Dropped {curr_len - len(ic)} rows due to band filtering.")

    if max_wcs_err is not None:
        print(f"Dropping rows with wcs_err > {max_wcs_err} arcsec.")
        curr_len = len(ic)
        ic["wcs_err"] = [abs(x) for x in ic["wcs_err"]]
        ic.filter_by_wcs_error(max_wcs_err, in_arcsec=True)
        print(f"Dropped {curr_len - len(ic)} rows due to high WCS error.")

    earth_loc = EarthLocation.of_site(site_name)

    needs_correction = [
        d for d in guess_distances
        if ic.reflex_corrected_col("ra", d) not in ic.data.colnames
    ]
    if needs_correction:
        print(f"{elapsed_t(start_time)} Reflex-correcting for guess distances {needs_correction}...")
        ic.reflex_correct(needs_correction, earth_loc)
        print(f"{elapsed_t(start_time)} Reflex correction done.")
    else:
        print(f"{elapsed_t(start_time)} All requested distances already have reflex columns; nothing to do.")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"{elapsed_t(start_time)} Writing primed IC to {output_path}...")
    ic.write(output_path, overwrite=overwrite, validate=False)
    print(f"{elapsed_t(start_time)} Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prime an ImageCollection with reflex-corrected columns so subsequent "
            "region_searcher.py runs skip the parallax computation."
        )
    )
    parser.add_argument("--ic-path", dest="ic_path", required=True, type=str,
                        help="path to the base ImageCollection")
    parser.add_argument("--output-path", dest="output_path", required=True, type=str,
                        help="path to write the primed ImageCollection")
    parser.add_argument("--guess-distances", dest="guess_distances", type=float,
                        nargs="+", required=True,
                        help="guess distances in AU to reflex-correct for")
    parser.add_argument("--obs-site", dest="obs_site", type=str, default="Rubin",
                        help="observatory site name (for EarthLocation)")
    parser.add_argument("--bands-to-drop", dest="bands_to_drop", type=str,
                        default=[], nargs="+",
                        help="list of bands to drop (e.g. --bands-to-drop u y)")
    parser.add_argument("--max-wcs-err", dest="max_wcs_err", type=float, default=None,
                        help="maximum WCS error in arcsec; rows above are dropped")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=False,
                        help="regenerate even if the output file already exists")

    args = parser.parse_args()
    prime_ic(
        ic_path=args.ic_path,
        output_path=args.output_path,
        guess_distances=args.guess_distances,
        site_name=args.obs_site,
        bands_to_drop=args.bands_to_drop,
        max_wcs_err=args.max_wcs_err,
        overwrite=args.overwrite,
    )
