import math

from astropy.io import fits
from astropy.table import Table
import numpy as np
from pathlib import Path

from kbmod.configuration import SearchConfiguration
from kbmod.search import ImageStack, LayeredImage, PSF, RawImage


class WorkUnit:
    """The work unit is a storage and I/O class for all of the data
    needed for a full run of KBMOD, including the: the search parameters,
    data files, and the data provenance metadata.
    """

    def __init__(self, im_stack=None, config=None):
        self.im_stack = im_stack
        self.config = config

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

            # Read the size and order information from the primary header.
            num_images = hdul[0].header["NUMIMG"]
            if len(hdul) != 4 * num_images + 3:
                raise ValueError(
                    f"WorkUnit wrong number of extensions. Expected "
                    f"{4 * num_images + 3}. Found {len(hdul)}."
                )

            # Read in all the image files.
            for i in range(num_images):
                # Read in science, variance, and mask layers.
                sci = hdu_to_raw_image(hdul[f"SCI_{i}"])
                var = hdu_to_raw_image(hdul[f"VAR_{i}"])
                msk = hdu_to_raw_image(hdul[f"MSK_{i}"])

                # Read the PSF layer.
                p = PSF(hdul[f"PSF_{i}"].data)

                imgs.append(LayeredImage(sci, var, msk, p))

        im_stack = ImageStack(imgs)
        return WorkUnit(im_stack=im_stack, config=config)

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

        imgs = []
        for i in range(num_images):
            obs_time = workunit_dict["times"][i]

            if type(workunit_dict["sci_imgs"][i]) is RawImage:
                sci_img = workunit_dict["sci_imgs"][i]
            else:
                sci_arr = workunit_dict["sci_imgs"][i].reshape(height, width)
                sci_img = RawImage(img=sci_arr, obs_time=obs_time)

            if type(workunit_dict["var_imgs"][i]) is RawImage:
                var_img = workunit_dict["var_imgs"][i]
            else:
                var_arr = workunit_dict["var_imgs"][i].reshape(height, width)
                var_img = RawImage(img=var_arr, obs_time=obs_time)

            # Masks are optional.
            if workunit_dict["msk_imgs"][i] is None:
                msk_arr = np.zeros(height, width)
                msk_img = RawImage(img=msk_arr, obs_time=obs_time)
            elif type(workunit_dict["msk_imgs"][i]) is RawImage:
                msk_img = workunit_dict["msk_imgs"][i]
            else:
                msk_arr = workunit_dict["msk_imgs"][i].reshape(height, width)
                msk_img = RawImage(img=msk_arr, obs_time=obs_time)

            # PSFs are optional.
            if workunit_dict["psfs"][i] is None:
                p = PSF()
            elif type(workunit_dict["psfs"][i]) is PSF:
                p = workunit_dict["psfs"][i]
            else:
                p = PSF(workunit_dict["psfs"][i])

            imgs.append(LayeredImage(sci_img, var_img, msk_img, p))

        im_stack = ImageStack(imgs)
        return WorkUnit(im_stack=im_stack, config=config)

    @classmethod
    def from_yaml(cls, work_unit):
        """Load a configuration from a YAML string.

        Parameters
        ----------
        work_unit : `str` or `_io.TextIOWrapper`
            The serialized YAML data.

        Raises
        ------
        Raises a ``ValueError`` for any invalid parameters.
        """
        yaml_dict = safe_load(work_unit)
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
            print(f"Warning: WorkUnit file {filename} already exists.")
            return

        # Set up the initial HDU list, including the primary header
        # the metadata (empty), and the configuration.
        hdul = fits.HDUList()
        pri = fits.PrimaryHDU()
        pri.header["NUMIMG"] = self.im_stack.img_count()
        hdul.append(pri)

        meta_hdu = fits.BinTableHDU()
        meta_hdu.name = "metadata"
        hdul.append(meta_hdu)

        config_hdu = self.config.to_hdu()
        config_hdu.name = "kbmod_config"
        hdul.append(config_hdu)

        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)

            sci_hdu = raw_image_to_hdu(layered.get_science())
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

    def to_dict(self, use_python_types=True):
        """Create a single dictionary representing the WorkUnit.

        Parameters
        ----------
        use_python_types: `bool`
            Convert everything to python types (instead of kbmod types).
            Used for serialization.

        Returns
        -------
        workunit_dict : `dict`
            The dictionary
        """
        # Fill in the per-image data.
        workunit_dict = {
            "num_images": self.im_stack.img_count(),
            "width": self.im_stack.get_width(),
            "height": self.im_stack.get_height(),
            "config": self.config._params,
            # Per image data
            "times": [],
            "sci_imgs": [],
            "var_imgs": [],
            "msk_imgs": [],
            "psfs": [],
        }

        # Fill in the per-image data.
        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)
            workunit_dict["times"].append(layered.get_obstime())
            p = layered.get_psf()

            if use_python_types:
                workunit_dict["sci_imgs"].append(layered.get_science().image)
                workunit_dict["var_imgs"].append(layered.get_variance().image)
                workunit_dict["msk_imgs"].append(layered.get_mask().image)

                psf_array = np.array(p.get_kernel()).reshape((p.get_dim(), p.get_dim()))
                workunit_dict["psfs"].append(psf_array)
            else:
                workunit_dict["sci_imgs"].append(layered.get_science())
                workunit_dict["var_imgs"].append(layered.get_variance())
                workunit_dict["msk_imgs"].append(layered.get_mask())                
                workunit_dict["psfs"].append(p)

        return workunit_dict

    def to_yaml(self):
        """Serialize the WorkUnit as a YAML string.

        Returns
        -------
        result : `str`
            The serialized YAML string.
        """
        return dump(self.to_dict())


def raw_image_to_hdu(img):
    """Helper function that creates a HDU out of RawImage.

    Parameters
    ----------
    img : `RawImage`
        The RawImage to convert.

    Returns
    -------
    hdu : `astropy.io.fits.hdu.image.ImageHDU`
        The image extension.
    """
    hdu = fits.hdu.image.ImageHDU(img.image)
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
