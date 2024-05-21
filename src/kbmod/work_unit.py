import math

from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
import numpy as np
from pathlib import Path
import warnings
from yaml import dump, safe_load

from kbmod.configuration import SearchConfiguration
from kbmod.search import ImageStack, LayeredImage, PSF, RawImage, Logging
from kbmod.wcs_utils import (
    append_wcs_to_hdu_header,
    extract_wcs_from_hdu_header,
    wcs_fits_equal,
    wcs_from_dict,
    wcs_to_dict,
)


logger = Logging.getLogger(__name__)


class WorkUnit:
    """The work unit is a storage and I/O class for all of the data
    needed for a full run of KBMOD, including the: the search parameters,
    data files, and the data provenance metadata.

    Attributes
    ----------
    im_stack : `kbmod.search.ImageStack`
        The image data for the KBMOD run.
    config : `kbmod.configuration.SearchConfiguration`
        The configuration for the KBMOD run.
    wcs : `astropy.wcs.WCS`
        A global WCS for all images in the WorkUnit. Only exists
        if all images have been projected to same pixel space.
    per_image_wcs : `list`
        A list with one WCS for each image in the WorkUnit. Used for when
        the images have *not* been standardized to the same pixel space.
    """

    def __init__(self, im_stack=None, config=None, wcs=None, per_image_wcs=None):
        self.im_stack = im_stack
        self.config = config

        # Handle WCS input. If both the global and per-image WCS are provided,
        # ensure they are consistent.
        self.wcs = wcs
        if per_image_wcs is None:
            self._per_image_wcs = [None] * im_stack.img_count()
            if self.wcs is None:
                warnings.warn("No WCS provided.", Warning)
        else:
            if len(per_image_wcs) != im_stack.img_count():
                raise ValueError(f"Incorrect number of WCS provided. Expected {im_stack.img_count()}")
            self._per_image_wcs = per_image_wcs

            # Check if all the per-image WCS are None. This can happen during a load.
            all_none = self.per_image_wcs_all_match(None)
            if self.wcs is None and all_none:
                warnings.warn("No WCS provided.", Warning)

            # Check for consistency with the global WCS if needed.
            if self.wcs is not None and not all_none and not self.per_image_wcs_all_match(self.wcs):
                raise ValueError(f"Inconsistent global and per-image WCS provided.")

            # See if we can compress the per-image WCS into a global one.
            if self.wcs is None and not all_none and self.per_image_wcs_all_match(self._per_image_wcs[0]):
                self.wcs = self._per_image_wcs[0]
                self._per_image_wcs = [None] * im_stack.img_count()

    def __len__(self):
        """Returns the size of the WorkUnit in number of images."""
        return self.im_stack.img_count()

    def has_common_wcs(self):
        """Returns whether the WorkUnit has a common WCS for all images."""
        return self.wcs is not None

    def per_image_wcs_all_match(self, target=None):
        """Check if all the per-image WCS are the same as a given target value.

        Parameters
        ----------
        target : `astropy.wcs.WCS`, optional
            The WCS to which to compare the per-image WCS. If None, checks that
            all of the per-image WCS are None.

        Returns
        -------
        result : `bool`
            A Boolean indicating that all the per-images WCS match the target.
        """
        for current in self._per_image_wcs:
            if not wcs_fits_equal(current, target):
                return False
        return True

    def get_wcs(self, img_num):
        """Return the WCS for the a given image. Alway prioritizes
        a global WCS if one exits.

        Parameters
        ----------
        img_num : `int`
            The number of the image.

        Returns
        -------
        wcs : `astropy.wcs.WCS`
            The image's WCS if one exists. Otherwise None.

        Raises
        ------
        IndexError if an invalid index is given.
        """
        if img_num < 0 or img_num >= self.im_stack.img_count():
            raise IndexError(f"Invalid image number {img_num}")

        # Extract the per-image WCS if one exists.
        if self._per_image_wcs is not None and img_num < len(self._per_image_wcs):
            per_img = self._per_image_wcs[img_num]
        else:
            per_img = None

        if self.wcs is not None:
            if per_img is not None and not wcs_fits_equal(self.wcs, per_img):
                warnings.warn("Both a global and per-image WCS given. Using global WCS.", Warning)
            return self.wcs

        return per_img

    def get_all_obstimes(self):
        """Return a list of the observation times."""
        return [self.im_stack.get_obstime(i) for i in range(self.im_stack.img_count())]

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
        logger.info(f"Loading WorkUnit from FITS file {filename}.")
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
                global_wcs = extract_wcs_from_hdu_header(hdul[0].header)

            # Read the size and order information from the primary header.
            num_images = hdul[0].header["NUMIMG"]
            logger.info(f"Loading {num_images} images and {4 * num_images + 3} total layers.")
            if len(hdul) != 4 * num_images + 3:
                raise ValueError(
                    f"WorkUnit wrong number of extensions. Expected "
                    f"{4 * num_images + 3}. Found {len(hdul)}."
                )

            # Read in all the image files.
            per_image_wcs = []
            for i in range(num_images):
                # Extract the per-image WCS if one exists.
                per_image_wcs.append(extract_wcs_from_hdu_header(hdul[f"SCI_{i}"].header))

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

    @classmethod
    def from_dict(cls, workunit_dict):
        """Create a WorkUnit from a combined dictionary.

        Parameters
        ----------
        workunit_dict : `dict`
            The dictionary of information.

        Returns
        -------
        `WorkUnit`

        Raises
        ------
        Raises a ``ValueError`` for any invalid parameters.
        """
        num_images = workunit_dict["num_images"]
        logger.debug(f"Creating WorkUnit from dictionary with {num_images} images.")

        width = workunit_dict["width"]
        height = workunit_dict["height"]
        if width <= 0 or height <= 0:
            raise ValueError(f"Illegal image dimensions width={width}, height={height}")

        # Load the configuration supporting both dictionary and SearchConfiguration.
        if type(workunit_dict["config"]) is dict:
            config = SearchConfiguration.from_dict(workunit_dict["config"])
        elif type(workunit_dict["config"]) is SearchConfiguration:
            config = workunit_dict["config"]
        else:
            raise ValueError("Unrecognized type for WorkUnit config parameter.")

        # Load the global WCS if one exists.
        if "wcs" in workunit_dict:
            if type(workunit_dict["wcs"]) is dict:
                global_wcs = wcs_from_dict(workunit_dict["wcs"])
            else:
                global_wcs = workunit_dict["wcs"]
        else:
            global_wcs = None

        imgs = []
        per_image_wcs = []
        for i in range(num_images):
            obs_time = workunit_dict["times"][i]

            if type(workunit_dict["sci_imgs"][i]) is RawImage:
                sci_img = workunit_dict["sci_imgs"][i]
            else:
                sci_arr = np.array(workunit_dict["sci_imgs"][i], dtype=np.float32).reshape(height, width)
                sci_img = RawImage(img=sci_arr, obs_time=obs_time)

            if type(workunit_dict["var_imgs"][i]) is RawImage:
                var_img = workunit_dict["var_imgs"][i]
            else:
                var_arr = np.array(workunit_dict["var_imgs"][i], dtype=np.float32).reshape(height, width)
                var_img = RawImage(img=var_arr, obs_time=obs_time)

            # Masks are optional.
            if workunit_dict["msk_imgs"][i] is None:
                msk_arr = np.zeros(height, width)
                msk_img = RawImage(img=msk_arr, obs_time=obs_time)
            elif type(workunit_dict["msk_imgs"][i]) is RawImage:
                msk_img = workunit_dict["msk_imgs"][i]
            else:
                msk_arr = np.array(workunit_dict["msk_imgs"][i], dtype=np.float32).reshape(height, width)
                msk_img = RawImage(img=msk_arr, obs_time=obs_time)

            # PSFs are optional.
            if workunit_dict["psfs"][i] is None:
                p = PSF()
            elif type(workunit_dict["psfs"][i]) is PSF:
                p = workunit_dict["psfs"][i]
            else:
                p = PSF(np.array(workunit_dict["psfs"][i], dtype=np.float32))

            imgs.append(LayeredImage(sci_img, var_img, msk_img, p))

            # Read a per_image_wcs if one exists.
            current_wcs = workunit_dict["per_image_wcs"][i]
            if type(current_wcs) is dict:
                current_wcs = wcs_from_dict(current_wcs)
            per_image_wcs.append(current_wcs)

        im_stack = ImageStack(imgs)
        return WorkUnit(im_stack=im_stack, config=config, wcs=global_wcs, per_image_wcs=per_image_wcs)

    @classmethod
    def from_yaml(cls, work_unit, strict=False):
        """Load a configuration from a YAML string.

        Parameters
        ----------
        work_unit : `str` or `_io.TextIOWrapper`
            The serialized YAML data.
        strict : `bool`
            Raise an error if the file is not a WorkUnit.

        Returns
        -------
        result : `WorkUnit` or `None`
            Returns the extracted WorkUnit. If the file did not contain a WorkUnit and
            strict=False the function will return None.

        Raises
        ------
        Raises a ``ValueError`` for any invalid parameters.
        """
        yaml_dict = safe_load(work_unit)

        # Check if this a WorkUnit yaml file by checking it has the required fields.
        required_fields = ["config", "height", "num_images", "sci_imgs", "times", "var_imgs", "width"]
        for name in required_fields:
            if name not in yaml_dict:
                if strict:
                    raise ValueError(f"Missing required field {name}")
                else:
                    return None

        return WorkUnit.from_dict(yaml_dict)

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
            # are you sure you did not want to raise an error here?
            logger.error(f"Warning: WorkUnit file {filename} already exists.")
            return

        # Set up the initial HDU list, including the primary header
        # the metadata (empty), and the configuration.
        hdul = fits.HDUList()
        pri = fits.PrimaryHDU()
        pri.header["NUMIMG"] = self.im_stack.img_count()

        # If the global WCS exists, append the corresponding keys.
        if self.wcs is not None:
            append_wcs_to_hdu_header(self.wcs, pri.header)

        hdul.append(pri)

        meta_hdu = fits.BinTableHDU()
        meta_hdu.name = "metadata"
        hdul.append(meta_hdu)

        config_hdu = self.config.to_hdu()
        config_hdu.name = "kbmod_config"
        hdul.append(config_hdu)

        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)

            img_wcs = self.get_wcs(i)
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

    def to_yaml(self):
        """Serialize the WorkUnit as a YAML string.

        Returns
        -------
        result : `str`
            The serialized YAML string.
        """
        workunit_dict = {
            "num_images": self.im_stack.img_count(),
            "width": self.im_stack.get_width(),
            "height": self.im_stack.get_height(),
            "config": self.config._params,
            "wcs": wcs_to_dict(self.wcs),
            # Per image data
            "times": [],
            "sci_imgs": [],
            "var_imgs": [],
            "msk_imgs": [],
            "psfs": [],
            "per_image_wcs": [],
        }

        # Fill in the per-image data.
        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)
            workunit_dict["times"].append(layered.get_obstime())
            p = layered.get_psf()

            workunit_dict["sci_imgs"].append(layered.get_science().image.tolist())
            workunit_dict["var_imgs"].append(layered.get_variance().image.tolist())
            workunit_dict["msk_imgs"].append(layered.get_mask().image.tolist())

            psf_array = np.array(p.get_kernel()).reshape((p.get_dim(), p.get_dim()))
            workunit_dict["psfs"].append(psf_array.tolist())

            if self.wcs is not None:
                img_wcs = self._per_image_wcs[i]
            else:
                img_wcs = None
            workunit_dict["per_image_wcs"].append(wcs_to_dict(img_wcs))

        return dump(workunit_dict)


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
        append_wcs_to_hdu_header(wcs, hdu.header)

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
