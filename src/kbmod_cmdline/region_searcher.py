"""
Commandline tool for performing Region Search on an ImageCollection.
Generates patches, matches them to the ImageCollection, and generates
ImageCollections for matched patches.

If the output ImageCollections already exist, they will be reused rather than
regenerated, unless the --overwrite flag is provided.

Also generates analysis tables summarizing the results. Additionally, 
any errors in processing patches are logged to an errors.csv file in 
the output directory.

Example Usage:
-------------

The following command takes a base ImageCollection, generates patch grids
for multiple reflex-correction guess distances and patch sizes, and
performs Region Search on each configuration. The resulting ImageCollections
and analysis tables are saved to the specified output directory. 

>>> python region_searcher.py \
        --ic-path path/to/image_collection.collection \
        --guess-distances 1.0 2.0 \
        --patch-side-len 10 20 \
        --obs-site Rubin \
        --pixel-scale 0.2 \
        --patch-overlap-percentage 0.1 \
        --bands-to-drop u y \
        --out-dir output_directory \
        --max-wcs-err 0.5 \
        --overwrite
"""

import argparse
from dateutil import parser
import os
import time
from tqdm import tqdm

import kbmod
from kbmod import ImageCollection
from kbmod.region_search import RegionSearch


from astropy.coordinates import EarthLocation
from astropy.table import Table


def elapsed_t(startTime, sigfigs=2):
    """
    Returns a string representing the elapsed time since startTime.

    Parameters
    ----------
    startTime : float
        The starting time in seconds since the epoch.
    sigfigs : int, optional
        The number of significant figures to round the elapsed time to. Default is 2.
    Returns
    -------
    str
        A string in the format "[X.XX s elapsed]".
    """

    elapsed_time = round(time.time() - startTime, sigfigs)
    return f"[{elapsed_time} s elapsed]"


def dist_patch_size_str(guess_dist, patch_size):
    """
    Returns a string representing the guess distance and patch size.

    Parameters
    ----------
    guess_dist : float
        The guess-correction distance.
    patch_size : int
        The length of a side of a square patch in arcminutes.

    Returns
    -------
    str
        A string in the format "GUESSDIST_PATCHSIZExPATCHSIZE".
    """
    return f"{guess_dist}_{patch_size}X{patch_size}"


def patch_id_to_ic_path(patch_id, guess_distance, patch_size, ic_dir):
    """
    Constructs the file path for the ImageCollection corresponding to a given patch ID.

    Parameters
    ----------
    patch_id : int
        The ID of the patch.
    guess_distance : float
        The guess-correction distance in AU.
    patch_size : int
        The length of a side of a square patch in arcminutes.
    ic_dir : str
        The directory where ImageCollections are stored.

    Returns
    -------
    str
        The file path for the ImageCollection.
    """
    dist_and_patch_size_str = dist_patch_size_str(guess_distance, patch_size)
    ic_path = os.path.join(ic_dir, f"{patch_id}_{dist_and_patch_size_str}.collection")
    return ic_path


def generate_or_load_patch_ic(patch_ids, guess_distance, patch_size, region_search, ic_dir, overwrite=False):
    """
    Fetch or generate ImageCollections for a list of matched patches.

    Parameters
    ----------
    patch_ids : list of int
        List of patch IDs that have matched ephemerides.
    guess_distance : float
        The reflex-correction guess distance used in the search, in AU.
    patch_size : int
        The length of a side of a square patch in arcminutes.
    region_search : kbmod.region_search.RegionSearch
        The RegionSearch object maintaining the patch grid and base ImageCollection.
    ic_dir : str
        Directory to look for existing ImageCollections.
    overwrite : bool, optional
        If True, regenerate all ImageCollections even if they exist on disk. Default is False.

    Returns
    -------
    dict
        A dictionary mapping patch IDs to their corresponding ImageCollections.
    """
    # Which patches need to be newly generated
    patch_ics_to_generate = []
    # Maps patch_id to an ImageCollection
    patch_id_to_ic = {}

    if overwrite:
        patch_ics_to_generate = patch_ids
    else:
        # Check which ImageCollections already exist on disk
        for patch_id in patch_ids:
            ic_file = patch_id_to_ic_path(patch_id, guess_distance, patch_size, ic_dir=ic_dir)
            if os.path.exists(ic_file):
                patch_id_to_ic[patch_id] = ImageCollection.read(ic_file)
            else:
                patch_ics_to_generate.append(patch_id)

    print(
        f"Recycled {len(patch_id_to_ic)} ImageCollections from {ic_dir}. Continuing to generation phase (if needed)..."
    )
    files_written = 0
    error_patch_ids = []
    errors = []
    for patch_id in tqdm(patch_ics_to_generate, desc="Processing patches"):
        try:
            patch_ic = region_search.get_image_collection_from_patch(patch_id, guess_dist=guess_distance)
            patch_id_to_ic[patch_id] = patch_ic
            patch_ic.write(
                patch_id_to_ic_path(patch_id, guess_distance, patch_size, ic_dir=ic_dir),
                overwrite=overwrite,
            )
            files_written += 1
        except ValueError as msg:
            print(f"Error for patch_id {patch_id} : {msg}")
            error_patch_ids.append(patch_id)
            errors.append(msg)

    print(
        f"Wrote {files_written} new ImageCollections to {ic_dir}. {len(patch_ics_to_generate) - files_written} failed to generate."
    )
    # Write out any errors encountered during generation to a CSV.
    error_table = Table({"patch_id": error_patch_ids, "error_msg": errors})
    error_table.write(os.path.join(ic_dir, "errors.csv"), overwrite=True)

    return patch_id_to_ic


def generate_analysis_table(patch_id_to_ic):
    """
    Generate an analysis table summarizing overlap statistics for each patch.

    Parameters
    ----------
    patch_id_to_ic : dict
        A dictionary mapping patch IDs to their corresponding ImageCollections.

    Returns
    -------
    astropy.table.Table
        An analysis table with columns for patch ID, overlap area, visit count,
        unique MJDs, and observation nights spanned.
    """
    patch_ids = []
    overlap_deg = []
    visit_counts = []
    unique_mjds = []
    obs_nights_spanned = []
    for patch_id, patch_ic in patch_id_to_ic.items():
        patch_ids.append(patch_id)
        overlap_deg.append(sum(patch_ic.data["overlap_deg"]))
        visit_counts.append(len(set(patch_ic["visit"])))
        unique_mjds.append(len(set([int(m) for m in patch_ic["mjd_mid"]])))
        obs_nights_spanned.append(patch_ic.obs_nights_spanned())
    t = Table(
        {
            "patch_id": patch_ids,
            "overlap_deg2": overlap_deg,
            "visit_count": visit_counts,
            "unique_mjds": unique_mjds,
            "obs_nights_spanned": obs_nights_spanned,
        }
    )
    t.sort("overlap_deg2", reverse=True)

    return t


def region_searcher(
    ic_path,
    guess_distance,
    site_name,
    patch_size,
    patch_overlap_percentage,
    pixel_scale,
    bands_to_drop,
    max_wcs_err,
    out_dir,
    overwrite,
):
    """
    Perform Region Search on a base ImageCollection for a given guess distance and patch size.

    This base ImageCollection is the metadata for all images we want to perform region search on.
    The function generates patches, matches them to the images in the base ImageCollection, generates
    new ImageCollections for each matched patch. These ImageCollections and an analysis table are saved to disk
    in the specified output directory.

    Parameters
    ----------
    ic_path : str
        Path to the base ImageCollection file.
    guess_distance : float
        The reflex-correction guess distance in AU.
    site_name : str
        The name of the observatory site for EarthLocation.
    patch_size : int
        The length of a side of a square patch in arcminutes.
    patch_overlap_percentage : float
        The percentage overlap between adjacent generated patches in a search (0.0-1.0).
    pixel_scale : float
        The pixel scale of images in arcseconds/pixel.
    bands_to_drop : list of str
        List of bands to drop from the ImageCollection.
    max_wcs_err : float or None
        Maximum WCS error in arcseconds. Rows with higher WCS error will be dropped.
    out_dir : str
        Output directory for generated ImageCollections and analysis tables.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    None
    """
    # The start time for elapsed time tracking
    startTime = time.time()

    # Load a base ImageCollection to which contains all images we want to perform
    # region search on. We will filter for images matching our criteria and then
    # generate subset ImageCollections for the actual searches.
    print(f"{elapsed_t(startTime)} Reading base ImageCollection from {ic_path}...")
    ic = kbmod.ImageCollection.read(ic_path)

    # Filter out unhelpful data from the ImageCollection
    if bands_to_drop:
        print(f"Dropping bands from ImageCollection: {bands_to_drop}.")
        curr_len = len(ic)
        ic.drop_bands(bands_to_drop)
        print(f"Dropped {curr_len - len(ic)} rows due to band filtering.")
    if max_wcs_err is not None:
        print(f"Dropping rows with wcs_err > {max_wcs_err} arcsec.")
        curr_len = len(ic)
        ic.filter_by_wcs_error(max_wcs_err, in_arcsec=True)
        print(f"Dropped {curr_len - len(ic)} rows due to high WCS error.")

    # region search setup including dividing the sky into grid of patches.
    print(f"{elapsed_t(startTime)} Generating {dist_patch_size_str(guess_distance, patch_size)} patches...")
    earth_loc = EarthLocation.of_site(site_name)
    region_search = RegionSearch(ic, guess_dists=[guess_distance], earth_loc=earth_loc)
    region_search.generate_patches(
        arcminutes=patch_size,
        overlap_percentage=patch_overlap_percentage,
        pixel_scale=pixel_scale,
    )
    print(
        f"{elapsed_t(startTime)} Generated {len(region_search.get_patches())} {dist_patch_size_str(guess_distance, patch_size)} patches. Searching ImageCollection..."
    )

    # See which of the generated patches have Images from our base collection.
    found_patches = region_search.match_ic_to_patches(region_search.ic, guess_distance, earth_loc)
    print(f"{elapsed_t(startTime)} Found {len(found_patches)} patches. Running analysis...")

    ic_dir = os.path.join(out_dir, dist_patch_size_str(guess_distance, patch_size))
    if not os.path.exists(ic_dir):
        os.makedirs(ic_dir)

    # For all of the patches that had matches to images in or base ImageCollection,
    # generate or load their ImageCollections (a collection of only the images overlapping that patch).
    patch_id_to_ic = generate_or_load_patch_ic(
        patch_ids=list(found_patches),
        guess_distance=guess_distance,
        patch_size=patch_size,
        region_search=region_search,
        ic_dir=ic_dir,
        overwrite=False,
    )

    # Generate and save an analysis table providing summary statistics for each patch.
    table_csvfile = os.path.join(ic_dir, f"overlap_{dist_patch_size_str(guess_distance, patch_size)}.csv")
    if not overwrite and os.path.exists(table_csvfile):
        print(f"Analysis table {table_csvfile} exists and overwrite is False, not writing.")
    else:
        print(f"{elapsed_t(startTime)} Generating analysis table...")
        t = generate_analysis_table(patch_id_to_ic)
        print(f"{elapsed_t(startTime)} Saving {table_csvfile} to disk.")
        t.write(table_csvfile, overwrite=True)
    print(f"{elapsed_t(startTime)} Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commandline Region Search tool")
    parser.add_argument(
        "--ic-path", dest="ic_path", help="path to main ImageCollection to perform Region Search on", type=str
    )
    parser.add_argument(
        "--guess-distances",
        dest="guess_distances",
        help="guess distances for reflex-correction in AU",
        type=float,
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--obs-site",
        dest="obs_site",
        help="observatory site name (for EarthLocation)",
        type=str,
        default="Rubin",
    )
    parser.add_argument(
        "--patch-side-len",
        dest="patch_side_len",
        help="square patch side length (arcminutes)",
        type=int,
        default=[10],
        nargs="+",
    )
    parser.add_argument(
        "--pixel-scale",
        dest="pixel_scale",
        help="pixel scale of images in arcseconds/pixel",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--patch-overlap-percentage",
        dest="patch_overlap_percentage",
        help="percentage overlap between patches (0.0-1.0)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--bands-to-drop",
        dest="bands_to_drop",
        help="list of bands to drop from ImageCollection, e.g., --bands-to-drop u y",
        type=str,
        default=["u", "y"],
        nargs="+",
    )
    parser.add_argument(
        "--max-wcs-err",
        dest="max_wcs_err",
        help="maximum WCS error (arcseconds). Default is 0.2 arcseconds (~1 pixel in Rubin)",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="output directory for generated ImageCollections",
        type=str,
        default=os.getcwd(),
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        help="whether to overwrite existing IC files",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    for patch_side_len in args.patch_side_len:
        for guess_distance in args.guess_distances:
            region_searcher(
                ic_path=args.ic_path,
                guess_distance=guess_distance,
                site_name=args.obs_site,
                patch_size=patch_side_len,
                patch_overlap_percentage=args.patch_overlap_percentage,
                pixel_scale=args.pixel_scale,
                bands_to_drop=args.bands_to_drop,
                max_wcs_err=args.max_wcs_err,
                out_dir=args.out_dir,
                overwrite=args.overwrite,
            )
