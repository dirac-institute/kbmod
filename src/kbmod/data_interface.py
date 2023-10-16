import os

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

import kbmod.search as kb

from kbmod.configuration import SearchConfiguration
from kbmod.file_utils import *


def load_input_from_individual_files(
    im_filepath,
    time_file,
    psf_file,
    mjd_lims,
    default_psf,
    verbose=False,
):
    """This function loads images and ingests them into an ImageStack.

    Parameters
    ----------
    im_filepath : `str`
        Image file path from which to load images.
    time_file : `str`
        File name containing image times.
    psf_file : `str`
        File name containing the image-specific PSFs.
        If set to None the code will use the provided default psf for
        all images.
    mjd_lims : `list` of ints
        Optional MJD limits on the images to search.
    default_psf : `PSF`
        The default PSF in case no image-specific PSF is provided.
    verbose : `bool`
        Use verbose output (mainly for debugging).

    Returns
    -------
    stack : `kbmod.ImageStack`
        The stack of images loaded.
    wcs_list : `list`
        A list of `astropy.wcs.WCS` objects for each image.
    visit_times : `list`
        A list of MJD times.
    """
    print("---------------------------------------")
    print("Loading Images")
    print("---------------------------------------")

    # Load a mapping from visit numbers to the visit times. This dictionary stays
    # empty if no time file is specified.
    image_time_dict = FileUtils.load_time_dictionary(time_file)
    if verbose:
        print(f"Loaded {len(image_time_dict)} time stamps.")

    # Load a mapping from visit numbers to PSFs. This dictionary stays
    # empty if no time file is specified.
    image_psf_dict = FileUtils.load_psf_dictionary(psf_file)
    if verbose:
        print(f"Loaded {len(image_psf_dict)} image PSFs stamps.")

    # Retrieve the list of visits (file names) in the data directory.
    patch_visits = sorted(os.listdir(im_filepath))

    # Load the images themselves.
    images = []
    visit_times = []
    wcs_list = []
    for visit_file in np.sort(patch_visits):
        # Skip non-fits files.
        if not ".fits" in visit_file:
            if verbose:
                print(f"Skipping non-FITS file {visit_file}")
            continue

        # Compute the full file path for loading.
        full_file_path = os.path.join(im_filepath, visit_file)

        # Try loading information from the FITS header.
        visit_id = None
        with fits.open(full_file_path) as hdu_list:
            curr_wcs = WCS(hdu_list[1].header)

            # If the visit ID is in header (using Rubin tags), use for the visit ID.
            # Otherwise extract it from the filename.
            if "IDNUM" in hdu_list[0].header:
                visit_id = str(hdu_list[0].header["IDNUM"])
            else:
                name = os.path.split(full_file_path)[-1]
                visit_id = FileUtils.visit_from_file_name(name)

        # Skip files without a valid visit ID.
        if visit_id is None:
            if verbose:
                print(f"WARNING: Unable to extract visit ID for {visit_file}.")
            continue

        # Check if the image has a specific PSF.
        psf = default_psf
        if visit_id in image_psf_dict:
            psf = kb.PSF(image_psf_dict[visit_id])

        # Load the image file and set its time.
        if verbose:
            print(f"Loading file: {full_file_path}")
        img = kb.LayeredImage(full_file_path, psf)
        time_stamp = img.get_obstime()

        # Overload the header's time stamp if needed.
        if visit_id in image_time_dict:
            time_stamp = image_time_dict[visit_id]
            img.set_obstime(time_stamp)

        if time_stamp <= 0.0:
            if verbose:
                print(f"WARNING: No valid timestamp provided for {visit_file}.")
            continue

        # Check if we should filter the record based on the time bounds.
        if mjd_lims is not None and (time_stamp < mjd_lims[0] or time_stamp > mjd_lims[1]):
            if verbose:
                print(f"Pruning file {visit_file} by timestamp={time_stamp}.")
            continue

        # Save image, time, and WCS information.
        visit_times.append(time_stamp)
        images.append(img)
        wcs_list.append(curr_wcs)

    print(f"Loaded {len(images)} images")
    stack = kb.ImageStack(images)

    return (stack, wcs_list, visit_times)


def load_input_from_config(config, verbose=False):
    """This function loads images and ingests them into an ImageStack.

    Parameters
    ----------
    config : `SearchConfiguration`
        The configuration with the individual file information.
    verbose : `bool`, optional
        Use verbose output (mainly for debugging).

    Returns
    -------
    stack : `kbmod.ImageStack`
        The stack of images loaded.
    wcs_list : `list`
        A list of `astropy.wcs.WCS` objects for each image.
    visit_times : `list`
        A list of MJD times.
    """
    return load_input_from_individual_files(
        config["im_filepath"],
        config["time_file"],
        config["psf_file"],
        config["mjd_lims"],
        kb.PSF(config["psf_val"]),  # Default PSF.
        verbose=verbose,
    )
