import os

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from pathlib import Path

from kbmod.configuration import SearchConfiguration
from kbmod.file_utils import *
from kbmod.search import ImageStack, LayeredImage, PSF, RawImage, Logging
from kbmod.wcs_utils import append_wcs_to_hdu_header
from kbmod.work_unit import WorkUnit, raw_image_to_hdu


logger = Logging.getLogger(__name__)


def visit_from_file_name(filename):
    """Automatically extract the visit ID from the file name.

    Uses the heuristic that the visit ID is the first numeric
    string of at least length 5 digits in the file name.

    Parameters
    ----------
    filename : str
        The file name

    Returns
    -------
    result : str
        The visit ID string or None if there is no match.
    """
    expr = re.compile(r"\d{4}(?:\d+)")
    res = expr.search(filename)
    if res is None:
        return None
    return res.group()


def load_deccam_layered_image(filename, psf):
    """Load a layered image from the legacy deccam format.

    Parameters
    ----------
    filename : `str`
        The name of the file to load.
    psf : `PSF`
        The PSF to use for the image.

    Returns
    -------
    img : `LayeredImage`
        The loaded image.

    Raises
    ------
    Raises a ``FileNotFoundError`` if the file does not exist.
    Raises a ``ValueError`` if any of the validation checks fail.
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"{filename} not found")

    img = None
    with fits.open(filename) as hdul:
        if len(hdul) < 4:
            raise ValueError("Not enough extensions for legacy deccam format")

        # Extract the obstime trying from a few keys and a few extensions.
        obstime = -1.0
        if "MJD" in hdul[0].header:
            obstime = hdul[0].header["MJD"]
        elif "MJD" in hdul[1].header:
            obstime = hdul[1].header["MJD"]

        img = LayeredImage(
            RawImage(hdul[1].data.astype(np.float32), obstime),  # Science
            RawImage(hdul[3].data.astype(np.float32), obstime),  # Variance
            RawImage(hdul[2].data.astype(np.float32), obstime),  # Mask
            psf,
        )

    return img


def save_deccam_layered_image(img, filename, wcs=None, overwrite=True):
    """Save a layered image to the legacy deccam format.

    Parameters
    ----------
    img : `LayeredImage`
        The image to save.
    filename : `str`
        The name of the file to save.
    wcs : `astropy.wcs.WCS`, optional
        The WCS of the image. If provided appends this information to the header.
    overwrite : `bool`
        Indicates whether to overwrite the current file if it exists.

    Raises
    ------
    Raises a ``ValueError`` if the file exists and ``overwrite`` is ``False``.
    """
    if Path(filename).is_file() and not overwrite:
        raise ValueError(f"{filename} already exists")

    hdul = fits.HDUList()

    # Create the primary header.
    pri = fits.PrimaryHDU()
    pri.header["MJD"] = img.get_obstime()
    if wcs is not None:
        append_wcs_to_hdu_header(wcs, pri.header)
    hdul.append(pri)

    # Append the science layer.
    sci_hdu = raw_image_to_hdu(img.get_science(), wcs)
    sci_hdu.name = f"science"
    hdul.append(sci_hdu)

    # Append the mask layer.
    msk_hdu = raw_image_to_hdu(img.get_mask(), wcs)
    msk_hdu.name = f"mask"
    hdul.append(msk_hdu)

    # Append the variance layer.
    var_hdu = raw_image_to_hdu(img.get_variance(), wcs)
    var_hdu.name = f"variance"
    hdul.append(var_hdu)

    hdul.writeto(filename, overwrite=overwrite)


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
    logger.info("Loading Images")

    # Load a mapping from visit numbers to the visit times. This dictionary stays
    # empty if no time file is specified.
    image_time_dict = FileUtils.load_time_dictionary(time_file)
    logger.info(f"Loaded {len(image_time_dict)} time stamps.")

    # Load a mapping from visit numbers to PSFs. This dictionary stays
    # empty if no time file is specified.
    image_psf_dict = FileUtils.load_psf_dictionary(psf_file)
    logger.info(f"Loaded {len(image_psf_dict)} image PSFs stamps.")

    # Retrieve the list of visits (file names) in the data directory.
    patch_visits = sorted(os.listdir(im_filepath))

    # Load the images themselves.
    images = []
    visit_times = []
    wcs_list = []
    for visit_file in np.sort(patch_visits):
        # Skip non-fits files.
        if not ".fits" in visit_file:
            logger.info(f"Skipping non-FITS file {visit_file}")
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
                visit_id = visit_from_file_name(name)

        # Skip files without a valid visit ID.
        if visit_id is None:
            logger.warning(f"WARNING: Unable to extract visit ID for {visit_file}.")
            continue

        # Check if the image has a specific PSF.
        psf = default_psf
        if visit_id in image_psf_dict:
            psf = PSF(image_psf_dict[visit_id])

        # Load the image file and set its time.
        logger.info(f"Loading file: {full_file_path}")
        img = load_deccam_layered_image(full_file_path, psf)
        time_stamp = img.get_obstime()

        # Overload the header's time stamp if needed.
        if visit_id in image_time_dict:
            time_stamp = image_time_dict[visit_id]
            img.set_obstime(time_stamp)

        if time_stamp <= 0.0:
            logger.warning(f"WARNING: No valid timestamp provided for {visit_file}.")
            continue

        # Check if we should filter the record based on the time bounds.
        if mjd_lims is not None and (time_stamp < mjd_lims[0] or time_stamp > mjd_lims[1]):
            logger.info(f"Pruning file {visit_file} by timestamp={time_stamp}.")
            continue

        # Save image, time, and WCS information.
        visit_times.append(time_stamp)
        images.append(img)
        wcs_list.append(curr_wcs)

    logger.info(f"Loaded {len(images)} images")
    stack = ImageStack(images)

    return (stack, wcs_list, visit_times)


def load_input_from_config(config, verbose=False):
    """This function loads images and ingests them into a WorkUnit.

    Parameters
    ----------
    config : `SearchConfiguration`
        The configuration with the individual file information.
    verbose : `bool`, optional
        Use verbose output (mainly for debugging).

    Returns
    -------
    result : `kbmod.WorkUnit`
        The input data as a ``WorkUnit``.
    """
    stack, wcs_list, _ = load_input_from_individual_files(
        config["im_filepath"],
        config["time_file"],
        config["psf_file"],
        config["mjd_lims"],
        PSF(config["psf_val"]),  # Default PSF.
        verbose=verbose,
    )
    return WorkUnit(stack, config, None, wcs_list)


def load_input_from_file(filename, overrides=None):
    """Build a WorkUnit from a single filename which could point to a WorkUnit
    or configuration file.

    Parameters
    ----------
    filename : `str`
        The path and file name of the data to load.
    overrides : `dict`, optional
        A dictionary of configuration parameters to override. For testing.

    Returns
    -------
    result : `kbmod.WorkUnit`
        The input data as a ``WorkUnit``.

    Raises
    ------
    ``ValueError`` if unable to read the data.
    """
    path_var = Path(filename)
    if not path_var.is_file():
        raise ValueError(f"File {filename} not found.")

    work = None

    path_suffix = path_var.suffix
    if path_suffix == ".yml" or path_suffix == ".yaml":
        # Try loading as a WorkUnit first.
        with open(filename) as ff:
            work = WorkUnit.from_yaml(ff.read(), strict=False)

        # If that load did not work, try loading the file as a configuration
        # and then using that to load the data files.
        if work is None:
            config = SearchConfiguration.from_file(filename, strict=False)
            if overrides is not None:
                config.set_multiple(overrides)
            if config["im_filepath"] is not None:
                return load_input_from_config(config)
    elif ".fits" in filename:
        work = WorkUnit.from_fits(filename)

    # None of the load paths worked.
    if work is None:
        raise ValueError(f"Could not interprete {filename}.")

    if overrides is not None:
        work.config.set_multiple(overrides)
    return work
