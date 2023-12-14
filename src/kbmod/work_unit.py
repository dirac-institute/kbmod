import math

from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
import numpy as np
from pathlib import Path
import warnings

from kbmod.configuration import SearchConfiguration
from kbmod.search import ImageStack, LayeredImage, PSF, RawImage


class WorkUnit:
    """The work unit is a storage and I/O class for all of the data
    needed for a full run of KBMOD, including the: the search parameters,
    data files, and the data provenance metadata.

    Atributes
    ---------
    im_stack : `kbmod.search.ImageStack`
        The image data for the KBMOD run.
    config : `kbmod.configuration.SearchConfiguration`
        The configuration for the KBMOD run.
    wcs : `astropy.wcs.WCS`
        A gloabl WCS for all images in the WorkUnit.
    per_image_wcs : `list`
        A list with one WCS for each image in the WorkUnit. Used for when
        the images have not been standardized to the same pixel space.
    """

    def __init__(self, im_stack=None, config=None, wcs=None, per_image_wcs=None):
        self.im_stack = im_stack
        self.config = config
        self.wcs = wcs

        if per_image_wcs is None:
            self.per_image_wcs = [None] * im_stack.img_count()
        else:
            if len(per_image_wcs) != im_stack.img_count():
                raise ValueError("Incorrect number of WCS provided.")
            self.per_image_wcs = per_image_wcs

    @classmethod
    def from_fits(cls, filename):
        """Create a WorkUnit from a single FITS file.

        The FITS file will have at least the following extensions:

        0. ``PRIMARY`` extension
        1. ``METADATA`` extension containing provenance
        2. ``KBMOD_CONFIG`` extension containing search parameters
        3. (+) any additional image extensions are named ``SCI_i``, ``VAR_i``, ``MSK_i``
        and ``PSF_i`` for the science, variance, mask and PSF of each image respectively,
        where ``i`` runs from 0 to number of images in the `WorkUnit`.

        Parameters
        ----------
        filename : `str`
            The file to load.

        Returns
        -------
        result : `WorkUnit`
            The loaded WorkUnit.
        """
        if not Path(filename).is_file():
            raise ValueError(f"WorkUnit file {filename} not found.")

        imgs = []
        with fits.open(filename) as hdul:
            num_layers = len(hdul)
            if num_layers < 5:
                raise ValueError(f"WorkUnit file has too few extensions {len(hdul)}.")

            # TODO - Read in provenance metadata from extension #1.

            # Read in the search parameters from the 'kbmod_config' extension.
            config = SearchConfiguration.from_hdu(hdul["kbmod_config"])

            # Read in the global WCS from extension 0 if the information exists.
            # We filter the warning that the image dimension does not match the WCS dimension
            # since the primary header does not have an image.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                global_wcs = extract_wcs(hdul[0])

            # Read the size and order information from the primary header.
            num_images = hdul[0].header["NUMIMG"]
            if len(hdul) != 4 * num_images + 3:
                raise ValueError(
                    f"WorkUnit wrong number of extensions. Expected "
                    f"{4 * num_images + 3}. Found {len(hdul)}."
                )

            # Read in all the image files.
            per_image_wcs = []
            for i in range(num_images):
                # Extract the per-image WCS if one exists.
                per_image_wcs.append(extract_wcs(hdul[f"SCI_{i}"]))

                # Read in science, variance, and mask layers.
                sci = hdu_to_raw_image(hdul[f"SCI_{i}"])
                var = hdu_to_raw_image(hdul[f"VAR_{i}"])
                msk = hdu_to_raw_image(hdul[f"MSK_{i}"])

                # Read the PSF layer.
                p = PSF(hdul[f"PSF_{i}"].data)

                imgs.append(LayeredImage(sci, var, msk, p))

        im_stack = ImageStack(imgs)
        result = WorkUnit(im_stack=im_stack, config=config, wcs=global_wcs, per_image_wcs=per_image_wcs)
        return result

    def to_fits(self, filename, overwrite=False):
        """Write the WorkUnit to a single FITS file.

        Uses the following extensions:
            0 - Primary header with overall metadata
            1 or "metadata" - The data provenance metadata
            2 or "kbmod_config" - The search parameters
            3+ - Image extensions for the science layer ("SCI_i"),
                variance layer ("VAR_i"), mask layer ("MSK_i"), and
                PSF ("PSF_i") of each image.

        Parameters
        ----------
        filename : `str`
            The file to which to write the data.
        overwrite : bool
            Indicates whether to overwrite an existing file.
        """
        if Path(filename).is_file() and not overwrite:
            print(f"Warning: WorkUnit file {filename} already exists.")
            return

        # Set up the initial HDU list, including the primary header
        # the metadata (empty), and the configuration.
        hdul = fits.HDUList()
        pri = fits.PrimaryHDU()
        pri.header["NUMIMG"] = self.im_stack.img_count()

        # If the global WCS exists, append the corresponding keys.
        if self.wcs is not None:
            wcs_header = self.wcs.to_header()
            for key in wcs_header:
                pri.header[key] = wcs_header[key]

        hdul.append(pri)

        meta_hdu = fits.BinTableHDU()
        meta_hdu.name = "metadata"
        hdul.append(meta_hdu)

        config_hdu = self.config.to_hdu()
        config_hdu.name = "kbmod_config"
        hdul.append(config_hdu)

        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)

            if i < len(self.per_image_wcs):
                img_wcs = self.per_image_wcs[i]
            else:
                img_wcs = None
            sci_hdu = raw_image_to_hdu(layered.get_science(), img_wcs)
            sci_hdu.name = f"SCI_{i}"
            hdul.append(sci_hdu)

            var_hdu = raw_image_to_hdu(layered.get_variance())
            var_hdu.name = f"VAR_{i}"
            hdul.append(var_hdu)

            msk_hdu = raw_image_to_hdu(layered.get_mask())
            msk_hdu.name = f"MSK_{i}"
            hdul.append(msk_hdu)

            p = layered.get_psf()
            psf_array = np.array(p.get_kernel()).reshape((p.get_dim(), p.get_dim()))
            psf_hdu = fits.hdu.image.ImageHDU(psf_array)
            psf_hdu.name = f"PSF_{i}"
            hdul.append(psf_hdu)

        hdul.writeto(filename)


def extract_wcs(hdu):
    """Read an WCS from the header and does basic validity checking.

    Parameters
    ----------
    hdu : An astropy HDU (Image or Primary)
        The extension

    Returns
    --------
    curr_wcs : `astropy.wcs.WCS`
        The WCS or None if it does not exist.
    """
    # Check that we have (at minimum) the CRVAL and CRPIX keywords.
    # These are necessary (but not sufficient) requirements for the WCS.
    if "CRVAL1" not in hdu.header or "CRVAL2" not in hdu.header:
        return None
    if "CRPIX1" not in hdu.header or "CRPIX2" not in hdu.header:
        return None

    curr_wcs = WCS(hdu.header)
    if curr_wcs is None:
        return None
    if curr_wcs.naxis != 2:
        return None

    return curr_wcs


def raw_image_to_hdu(img, wcs=None):
    """Helper function that creates a HDU out of RawImage.

    Parameters
    ----------
    img : `RawImage`
        The RawImage to convert.
    wcs : `astropy.wcs.WCS`
        An optional WCS to include in the header.

    Returns
    -------
    hdu : `astropy.io.fits.hdu.image.ImageHDU`
        The image extension.
    """
    hdu = fits.hdu.image.ImageHDU(img.image)

    # If the WCS is given, copy each entry into the header.
    if wcs is not None:
        wcs_header = wcs.to_header()
        for key in wcs_header:
            hdu.header[key] = wcs_header[key]

    # Set the time stamp.
    hdu.header["MJD"] = img.obstime
    return hdu


def hdu_to_raw_image(hdu):
    """Helper function that creates a RawImage from a HDU.

    Parameters
    ----------
    hdu : `astropy.io.fits.hdu.image.ImageHDU`
        The image extension.

    Returns
    -------
    img : `RawImage` or None
        The RawImage if there is valid data and None otherwise.
    """
    img = None
    if isinstance(hdu, fits.hdu.image.ImageHDU):
        # This will be a copy whenever dtype != np.single including when
        # endianness doesn't match the native float.
        img = RawImage(hdu.data.astype(np.single))
        if "MJD" in hdu.header:
            img.obstime = hdu.header["MJD"]
    return img
