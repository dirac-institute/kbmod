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

    A WorkUnit file is a FITS file with the following extensions:
        0 - Primary header with overall metadata
        1 - The data provenance metadata
        2 - The search parameters
        3 through X - Image layers with either alternating layers of science, variance,
            mask, and PSF. Layers may be empty if no data is provided.
    """

    def __init__(self, im_stack=None, config=None):
        self.im_stack = im_stack
        self.config = config

    @classmethod
    def from_file(cls, filename):
        """Create a WorkUnit from a single file.

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

        result = None
        with fits.open(filename) as hdul:
            num_layers = len(hdul)
            if num_layers < 5:
                raise ValueError(f"WorkUnit file has too few extensions {len(hdul)}.")

            # TODO - Read in provenance metadata from extension #1.

            # Read in the search parameters from extension #2.
            config = SearchConfiguration()
            config_table = Table(hdul[2].data)
            config.set_from_table(config_table)

            # Read the size and order information from the primary header.
            num_images = hdul[0].header["NUMIMG"]
            if len(hdul) != 4 * num_images + 3:
                raise ValueError(
                    f"WorkUnit wrong number of extensions. Expected "
                    f"{4 * num_images + 3}. Found {len(hdul)}."
                )

            # Read in all the image files.
            imgs = []
            for i in range(num_images):
                ext_num = 3 + 4 * i

                # Read in science and variance layers.
                sci = hdu_to_raw_image(hdul[ext_num])
                var = hdu_to_raw_image(hdul[ext_num + 1])

                # Read the mask layer if it exists.
                msk = hdu_to_raw_image(hdul[ext_num + 2])
                if msk is None:
                    msk = RawImage(np.zeros((sci.get_height(), sci.get_width())))

                # Check if the PSF layer exists.
                if hdul[ext_num + 3].header["NAXIS"] == 2:
                    p = PSF(hdul[ext_num + 3].data)
                else:
                    p = PSF(1e-8)

                imgs.append(LayeredImage(sci, var, msk, p))

            im_stack = ImageStack(imgs)

            result = WorkUnit(im_stack=im_stack, config=config)
        return result

    def write_to_file(self, filename, overwrite=False):
        """Write the WorkUnit to a single file.

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
        hdul.append(fits.BinTableHDU())
        hdul.append(fits.BinTableHDU(self.config.to_table(make_fits_safe=True)))

        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)
            hdul.append(raw_image_to_hdu(layered.get_science()))
            hdul.append(raw_image_to_hdu(layered.get_variance()))
            hdul.append(raw_image_to_hdu(layered.get_mask()))

            p = layered.get_psf()
            psf_array = np.array(p.get_kernel()).reshape((p.get_dim(), p.get_dim()))
            hdul.append(fits.hdu.image.ImageHDU(psf_array))

        hdul.writeto(filename)


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
    # Expensive copy. To be removed with RawImage refactor.
    np_pixels = np.array(img.get_all_pixels()).astype("float32", casting="same_kind")
    np_array = np_pixels.reshape((img.get_height(), img.get_width()))
    hdu = fits.hdu.image.ImageHDU(np_array)
    hdu.header["MJD"] = img.get_obstime()
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
    if hdu.header["NAXIS"] == 2:
        # Expensive copy. To be removed with RawImage refactor.
        img = RawImage(hdu.data)
        if "MJD" in hdu.header:
            img.set_obstime(hdu.header["MJD"])
    return img
