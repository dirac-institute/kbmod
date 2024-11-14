import os
import warnings
from pathlib import Path

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u

import numpy as np
from tqdm import tqdm

from kbmod import is_interactive
from kbmod.configuration import SearchConfiguration
from kbmod.reprojection_utils import invert_correct_parallax
from kbmod.search import ImageStack, LayeredImage, PSF, RawImage, Logging
from kbmod.util_functions import get_matched_obstimes
from kbmod.wcs_utils import (
    append_wcs_to_hdu_header,
    calc_ecliptic_angle,
    deserialize_wcs,
    extract_wcs_from_hdu_header,
    serialize_wcs,
    wcs_fits_equal,
)


_DEFAULT_WORKUNIT_TQDM_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"


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
    n_images : `int`
        The number of current images.
    n_constituents : `int`
        The number of original images making up the data in this WorkUnit. This might be
        different from the number of images stored in memory if the WorkUnit has been
        reprojected.
    img_meta : `astropy.table.Table`
        The meta data for each of the current images. These might differ from the
        constituent images if the WorkUnit has been filtered or reprojected.
          * wcs - The WCS of the image. This is set even if all image share a global WCS.
          * per_image_indices - A lists containing the indicies of `constituent_images`
            for each current image. Used for finding corresponding original images when we
            stitch images together during reprojection.
    org_img_meta : `astropy.table.Table`
        The meta data for each constituent image. Includes columns:
        * data_loc - the original location of the image
        * ebd_wcs - Used to reproject the images into EBD space.
        * geocentric_distance - The best fit geocentric distances used when creating
          the per image EBD WCS.
        * original_wcs - The original WCS of the image.
    wcs : `astropy.wcs.WCS`
        A global WCS for all images in the WorkUnit. Only exists
        if all images have been projected to same pixel space.
    heliocentric_distance : `float`
        The heliocentric distance that was used when creating the `per_image_ebd_wcs`.
    reprojected : `bool`
        Whether or not the WorkUnit image data has been reprojected.
    per_image_indices : `list` of `list`
        A list of lists containing the indicies of `constituent_images` at each layer
        of the `ImageStack`. Used for finding corresponding original images when we
        stitch images together during reprojection.
    lazy : `bool`
        Whether or not to load the image data for the `WorkUnit`.
    file_paths : `list[str]`
        The paths for the shard files, only created if the `WorkUnit` is loaded
        in lazy mode.
    obstimes : `list[float]`
        The MJD obstimes of the images.

    Parameters
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
    reprojected : `bool`
        Whether or not the WorkUnit image data has been reprojected.
    heliocentric_distance : `float`
        The heliocentric distance that was used when creating the `per_image_ebd_wcs`.
    lazy : `bool`
        Whether or not to load the image data for the `WorkUnit`.
    file_paths : `list[str]`
        The paths for the shard files, only created if the `WorkUnit` is loaded
        in lazy mode.
    obstimes : `list[float]`
        The MJD obstimes of the images.
    img_meta : `dict` or `astropy.table.Table`, optional
        The meta data for each of the current images.
    org_image_meta : `dict` or `astropy.table.Table`, optional
        A table of per-image data for the constituent images.
    """

    def __init__(
        self,
        im_stack=None,
        config=None,
        wcs=None,
        per_image_wcs=None,
        reprojected=False,
        heliocentric_distance=None,
        lazy=False,
        file_paths=None,
        obstimes=None,
        image_meta=None,
        org_image_meta=None,
    ):
        self.im_stack = im_stack
        self.config = config
        self.lazy = lazy
        self.file_paths = file_paths
        self._obstimes = obstimes

        # Track the metadata for each of the current images.
        self.img_meta = WorkUnit.create_meta(
            constituent=False,
            data=image_meta,
            n_images=im_stack.img_count(),
        )

        # Base the number of current images on the metadata because in a lazy load,
        # the ImageStack might be empty.
        self.n_images = len(self.img_meta)

        # Track the metadata for each constituent image in the WorkUnit. If no constituent
        # data is provided, this will create an empty array the same size as the original.
        no_org_img_meta_given = org_image_meta is None
        self.org_img_meta = WorkUnit.create_meta(
            constituent=True,
            data=org_image_meta,
            n_images=self.n_images,
        )
        self.n_constituents = len(self.org_img_meta)

        # Handle WCS input.
        self.wcs = wcs
        if per_image_wcs is not None:
            # If we are given explicit per-image WCS, use those. Overwrite the values
            # the current image metadata.
            if len(per_image_wcs) != self.n_images:
                raise ValueError(f"Incorrect number of WCS provided. Expected {self.n_images}")
            self.img_meta["wcs"] = np.array(per_image_wcs)
        elif "wcs" not in self.img_meta.colnames or np.all(self.img_meta["wcs"] == None):
            # If we have no per-image WCS already, use the global one (which might be None).
            self.img_meta["wcs"] = np.array([self.wcs] * self.n_images)

        # If no constituent data was provided, then save the current image's WCS as the original.
        # This is needed to ensure that we always have a correct original WCS.
        if no_org_img_meta_given:
            for i in range(self.n_images):
                self.org_img_meta["original_wcs"][i] = self.img_meta["wcs"][i]

        # If both the global and per-image WCS are provided, ensure they are consistent.
        if self.wcs is not None and not np.all(self.img_meta["wcs"].value == None):
            for idx in range(im_stack.img_count()):
                if not wcs_fits_equal(self.wcs, self.img_meta["wcs"][idx]):
                    raise ValueError(f"Inconsistent WCS at index {idx}.")
        if self.wcs is None and np.any(self.img_meta["wcs"].value == None):
            logger.warning("No WCS provided for at least one image.")

        # Set the global metadata for reprojection.
        self.reprojected = reprojected
        self.heliocentric_distance = heliocentric_distance

    def __len__(self):
        """Returns the size of the WorkUnit in number of images."""
        return self.im_stack.img_count()

    @staticmethod
    def create_meta(constituent=False, n_images=None, data=None):
        """Create an img_meta table, filling in default values
        for any unspecified columns.

        Parameters
        ----------
        constituent : `bool`
            Indicates the type of table. True indicates a table of constituent (original)
            images. False indicates a table of current images.
        data : `dict`, `astropy.table.Table`, or None
            The data from which to seed the table.
        n_images : `int`, optional
            The number of images to include. Only use when no data is
            provided in order to fill in defaults.

        Returns
        -------
        img_meta : `astropy.table.Table`
            The empty table of org_img_meta.
        """
        if data is None:
            if n_images is None or n_images <= 0:
                raise ValueError("If no data provided 'n_images' must be >= 1. Is {n_images}")

            # Add a place holder column of the correct size.
            data = Table({"_index": np.arange(n_images)})
        elif isinstance(data, dict):
            data = Table(data)
        elif isinstance(data, Table):
            data = data.copy()
        else:
            raise TypeError("Unsupported type for data table.")
        n_images = len(data)

        if constituent:
            # Fill in the defaults for the original/constituent images.
            for colname in ["data_loc", "ebd_wcs", "geocentric_distance", "original_wcs"]:
                if colname not in data.colnames:
                    data[colname] = np.full(n_images, None)
        else:
            # Fill in the defaults for the current image.
            if "per_image_indices" not in data.colnames:
                data["per_image_indices"] = [[i] for i in range(n_images)]
            if "wcs" not in data.colnames:
                data["wcs"] = np.full(n_images, None)
        return data

    def get_constituent_meta(self, column):
        """Get the meta data values of a given column for all the constituent images.

        Parameters
        ----------
        column : `str`
            The column name to fetch.

        Returns
        -------
        data : `list`
            A list of the meta-data for each constituent image.
        """
        return list(self.org_img_meta[column].data)

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
        """
        if self.wcs:
            return self.wcs
        else:
            return self.img_meta["wcs"][img_num]

    def get_pixel_coordinates(self, ra, dec, times=None):
        """Get the pixel coordinates for pairs of (RA, dec) coordinates. Uses the global
        WCS if one exists. Otherwise uses the per-image WCS. If times is provided, uses those values
        to choose the per-image WCS.

        Parameters
        ----------
        ra : `numpy.ndarray`
            The right ascension coordinates in degrees.
        dec : `numpy.ndarray`
            The declination coordinates in degrees.
        times : `numpy.ndarray` or `None`, optional
            The times to match.

        Returns
        -------
        x_pos, y_pos: `numpy.ndarray`
            Arrays of the X and Y pixel positions respectively.
        """
        num_pts = len(ra)
        if num_pts != len(dec):
            raise ValueError(f"Mismatched array sizes RA={len(ra)} and dec={len(dec)}.")
        if times is not None and len(times) != num_pts:
            raise ValueError(f"Mismatched array sizes RA={len(ra)} and times={len(times)}.")

        if self.wcs is not None:
            # If we have a single global WCS, we can use it for all the conversions. No time matching needed.
            x_pos, y_pos = self.wcs.world_to_pixel(SkyCoord(ra=ra * u.degree, dec=dec * u.degree))
        else:
            if times is None:
                if len(self._obstimes) == num_pts:
                    inds = np.arange(num_pts)
                else:
                    raise ValueError("No time information for a WorkUnit without a global WCS.")
            elif self._obstimes is not None:
                inds = get_matched_obstimes(self._obstimes, times, threshold=0.02)
            else:
                raise ValueError("No times provided for images in WorkUnit.")

            # TODO: Determine if there is a way to vectorize.
            x_pos = np.zeros(num_pts)
            y_pos = np.zeros(num_pts)
            for i, index in enumerate(inds):
                if index == -1:
                    raise ValueError(f"Unmatched time {times[i]}.")
                current_wcs = self.img_meta["wcs"][index]
                curr_x, curr_y = current_wcs.world_to_pixel(
                    SkyCoord(ra=ra[i] * u.degree, dec=dec[i] * u.degree)
                )
                x_pos[i] = curr_x
                y_pos[i] = curr_y

        return x_pos, y_pos

    def compute_ecliptic_angle(self):
        """Return the ecliptic angle (in radians in pixel space) derived from the
        images and WCS.

        Returns
        -------
        ecliptic_angle : `float` or `None`
            The computed ecliptic_angle or ``None`` if data is missing.
        """

        wcs = self.get_wcs(0)
        if wcs is None or self.im_stack is None:
            logger.warning(f"A valid wcs and ImageStack is needed to compute the ecliptic angle.")
            return None
        center_pixel = (self.im_stack.get_width() / 2, self.im_stack.get_height() / 2)
        return calc_ecliptic_angle(wcs, center_pixel)

    def get_all_obstimes(self):
        """Return a list of the observation times.
        If the `WorkUnit` was lazily loaded, then the obstimes
        have already been preloaded. Otherwise, grab them from
        the `ImageStack."""
        if self._obstimes is not None:
            return self._obstimes

        self._obstimes = [self.im_stack.get_obstime(i) for i in range(self.im_stack.img_count())]
        return self._obstimes

    def get_unique_obstimes_and_indices(self):
        """Returns the unique obstimes and the list of indices that they are associated with."""
        all_obstimes = self.get_all_obstimes()
        unique_obstimes = np.unique(all_obstimes)
        unique_indices = [list(np.where(all_obstimes == time)[0]) for time in unique_obstimes]
        return unique_obstimes, unique_indices

    @classmethod
    def from_fits(cls, filename, show_progress=None):
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
        show_progress : `bool` or `None`, optional
            If `None` use default settings, when a boolean forces the progress bar to be
            displayed or hidden.

        Returns
        -------
        result : `WorkUnit`
            The loaded WorkUnit.
        """
        show_progress = is_interactive() if show_progress is None else show_progress
        logger.info(f"Loading WorkUnit from FITS file {filename}.")
        if not Path(filename).is_file():
            raise ValueError(f"WorkUnit file {filename} not found.")

        im_stack = ImageStack()
        with fits.open(filename) as hdul:
            num_layers = len(hdul)
            if num_layers < 5:
                raise ValueError(f"WorkUnit file has too few extensions {len(hdul)}.")

            # Read in the search parameters from the 'kbmod_config' extension.
            config = SearchConfiguration.from_hdu(hdul["kbmod_config"])

            # Read the size and order information from the primary header.
            num_images = hdul[0].header["NUMIMG"]
            n_constituents = hdul[0].header["NCON"] if "NCON" in hdul[0].header else num_images
            logger.info(f"Loading {num_images} images.")

            # Read in the per-image metadata for the current images and the constituent images.
            if "IMG_META" in hdul:
                logger.debug("Reading image metadata from IMG_META.")
                per_image_meta = hdu_to_metadata_table(hdul["IMG_META"])
            else:
                per_image_meta = WorkUnit.create_meta(constituent=True, data=None, n_images=num_images)

            if "ORG_META" in hdul:
                logger.debug("Reading original image metadata from ORG_META.")
                org_image_meta = hdu_to_metadata_table(hdul["ORG_META"])
            else:
                org_image_meta = WorkUnit.create_meta(constituent=True, data=None, n_images=n_constituents)

            # Read in the global WCS from extension 0 if the information exists.
            # We filter the warning that the image dimension does not match the WCS dimension
            # since the primary header does not have an image.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                global_wcs = extract_wcs_from_hdu_header(hdul[0].header)

            # Misc. reprojection metadata
            reprojected = hdul[0].header["REPRJCTD"]
            heliocentric_distance = hdul[0].header["HELIO"]
            if np.all(org_image_meta["geocentric_distance"] == None):
                # If the metadata table does not have the geocentric_distance, try
                # loading it from the primary header's GEO_i fields (legacy approach).
                for i in range(n_constituents):
                    if f"GEO_{i}" in hdul[0].header:
                        org_image_meta["geocentric_distance"][i] = hdul[0].header[f"GEO_{i}"]

            # Read in all the image files.
            for i in tqdm(
                range(num_images),
                bar_format=_DEFAULT_WORKUNIT_TQDM_BAR,
                desc="Loading images",
                disable=not show_progress,
            ):
                sci_hdu = hdul[f"SCI_{i}"]

                # Read in the layered image from different extensions.
                img = LayeredImage(
                    sci_hdu.data.astype(np.single),
                    hdul[f"VAR_{i}"].data.astype(np.single),
                    hdul[f"MSK_{i}"].data.astype(np.single),
                    PSF(hdul[f"PSF_{i}"].data),
                    sci_hdu.header["MJD"],
                )

                # force_move destroys img object, but avoids a copy.
                im_stack.append_image(img, force_move=True)

                # Check if we need to load the map of current images to constituent images
                # from the (legacy) headers.
                if "NIND" in sci_hdu.header:
                    n_indices = sci_hdu.header["NIND"]
                    sub_indices = []
                    for j in range(n_indices):
                        sub_indices.append(sci_hdu.header[f"IND_{j}"])
                    per_image_meta["per_image_indices"][i] = sub_indices

            # Extract the per-image data from header information if needed. This happens
            # when the WorkUnit was saved before metadata tables were saved as layers and
            # all the information is in header values.
            for i in tqdm(
                range(n_constituents),
                bar_format=_DEFAULT_WORKUNIT_TQDM_BAR,
                desc="Loading WCS",
                disable=not show_progress,
            ):
                if f"WCS_{i}" in hdul:
                    org_image_meta["original_wcs"][i] = extract_wcs_from_hdu_header(hdul[f"WCS_{i}"].header)
                    org_image_meta["data_loc"][i] = hdul[f"WCS_{i}"].header["ILOC"]
                if f"EBD_{i}" in hdul:
                    org_image_meta["original_wcs"][i] = extract_wcs_from_hdu_header(hdul[f"EBD_{i}"].header)

        result = WorkUnit(
            im_stack=im_stack,
            config=config,
            wcs=global_wcs,
            heliocentric_distance=heliocentric_distance,
            reprojected=reprojected,
            image_meta=per_image_meta,
            org_image_meta=org_image_meta,
        )
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
        logger.info(f"Writing WorkUnit with {self.im_stack.img_count()} images to file {filename}")
        if Path(filename).is_file() and not overwrite:
            raise FileExistsError(f"WorkUnit file {filename} already exists.")

        # Create an HDU list with the metadata layers.
        hdul = self.metadata_to_hdul(include_wcs=False)

        # Create each image layer.
        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)
            obstime = layered.get_obstime()

            img_wcs = self.get_wcs(i)
            sci_hdu = raw_image_to_hdu(layered.get_science(), obstime, img_wcs)
            sci_hdu.name = f"SCI_{i}"
            hdul.append(sci_hdu)

            var_hdu = raw_image_to_hdu(layered.get_variance(), obstime)
            var_hdu.name = f"VAR_{i}"
            hdul.append(var_hdu)

            msk_hdu = raw_image_to_hdu(layered.get_mask(), obstime)
            msk_hdu.name = f"MSK_{i}"
            hdul.append(msk_hdu)

            p = layered.get_psf()
            psf_array = np.array(p.get_kernel()).reshape((p.get_dim(), p.get_dim()))
            psf_hdu = fits.hdu.image.ImageHDU(psf_array)
            psf_hdu.name = f"PSF_{i}"
            hdul.append(psf_hdu)

        hdul.writeto(filename, overwrite=overwrite)

    def to_sharded_fits(self, filename, directory, overwrite=False):
        """Write the WorkUnit to a multiple FITS files.
        Will create:
            - One "primary" file, containing the main WorkUnit metadata
            (see below) as well as the per_image_wcs information for
            the whole set. This will have the given filename.
            -One image fits file containing all of the image data for
            every LayeredImage in the ImageStack. This will have the
            image index infront of the given filename, e.g.
            "0_filename.fits".


        Primary File:
            0 - Primary header with overall metadata
            1 or "metadata" - The data provenance metadata
            2 or "kbmod_config" - The search parameters
        Individual Image File:
            Image extensions for the science layer ("SCI_i"),
            variance layer ("VAR_i"), mask layer ("MSK_i"), and
            PSF ("PSF_i") of each image.

        Parameters
        ----------
        filename : `str`
            The base filename to which to write the data.
        directory: `str`
            The directory to place all of the FITS files.
            Recommended that you have one directory per
            sharded file to avoid confusion.
        overwrite : bool
            Indicates whether to overwrite an existing file.
        """
        logger.info(
            f"Writing WorkUnit shards with {self.im_stack.img_count()} images with main file {filename} in {directory}"
        )
        primary_file = os.path.join(directory, filename)
        if Path(primary_file).is_file() and not overwrite:
            raise FileExistsError(f"WorkUnit file {filename} already exists.")
        if self.lazy:
            raise ValueError(
                "WorkUnit was lazy loaded, must load all ImageStack data to output new WorkUnit."
            )

        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)
            obstime = layered.get_obstime()
            sub_hdul = fits.HDUList()

            img_wcs = self.get_wcs(i)
            sci_hdu = raw_image_to_hdu(layered.get_science(), obstime, img_wcs)
            sci_hdu.name = f"SCI_{i}"
            sub_hdul.append(sci_hdu)

            var_hdu = raw_image_to_hdu(layered.get_variance(), obstime)
            var_hdu.name = f"VAR_{i}"
            sub_hdul.append(var_hdu)

            msk_hdu = raw_image_to_hdu(layered.get_mask(), obstime)
            msk_hdu.name = f"MSK_{i}"
            sub_hdul.append(msk_hdu)

            p = layered.get_psf()
            psf_array = np.array(p.get_kernel()).reshape((p.get_dim(), p.get_dim()))
            psf_hdu = fits.hdu.image.ImageHDU(psf_array)
            psf_hdu.name = f"PSF_{i}"
            sub_hdul.append(psf_hdu)
            sub_hdul.writeto(os.path.join(directory, f"{i}_{filename}"))

        # Create a primary file with all of the metadata
        hdul = self.metadata_to_hdul(include_wcs=True)
        hdul.writeto(os.path.join(directory, filename), overwrite=overwrite)

    @classmethod
    def from_sharded_fits(cls, filename, directory, lazy=False):
        """Create a WorkUnit from multiple FITS files.
        Pointed towards the result of WorkUnit.to_sharded_fits.

        The FITS files will have the following extensions:

        Primary File:
            0 - Primary header with overall metadata
            1 or "metadata" - The data provenance metadata
            2 or "kbmod_config" - The search parameters
        Individual Image File:
            Image extensions for the science layer ("SCI_i"),
            variance layer ("VAR_i"), mask layer ("MSK_i"), and
            PSF ("PSF_i") of each image.

        Parameters
        ----------
        filename : `str`
            The primary file to load.
        directory : `str`
            The directory where the sharded file is located.
        lazy : `bool`
            Whether or not to lazy load, i.e. whether to load
            all of the image data into the WorkUnit or just
            the metadata.

        Returns
        -------
        result : `WorkUnit`
            The loaded WorkUnit.
        """
        logger.info(f"Loading WorkUnit from primary FITS file {filename} in {directory}.")
        if not Path(os.path.join(directory, filename)).is_file():
            raise ValueError(f"WorkUnit file {filename} not found.")

        im_stack = ImageStack()

        # open the main header
        with fits.open(os.path.join(directory, filename)) as primary:
            config = SearchConfiguration.from_hdu(primary["kbmod_config"])

            # Read the size and order information from the primary header.
            num_images = primary[0].header["NUMIMG"]
            n_constituents = primary[0].header["NCON"] if "NCON" in primary[0].header else num_images
            logger.info(f"Loading {num_images} images.")

            # Read in the per-image metadata for the current images and the constituent images.
            if "IMG_META" in primary:
                logger.debug("Reading image metadata from IMG_META.")
                per_image_meta = hdu_to_metadata_table(primary["IMG_META"])
            else:
                per_image_meta = WorkUnit.create_meta(constituent=True, data=None, n_images=num_images)

            if "ORG_META" in primary:
                logger.debug("Reading original image metadata from ORG_META.")
                org_image_meta = hdu_to_metadata_table(primary["ORG_META"])
            else:
                org_image_meta = WorkUnit.create_meta(constituent=True, data=None, n_images=n_constituents)

            # Read in the global WCS from extension 0 if the information exists.
            # We filter the warning that the image dimension does not match the WCS dimension
            # since the primary header does not have an image.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                global_wcs = extract_wcs_from_hdu_header(primary[0].header)

            # Misc. reprojection metadata
            reprojected = primary[0].header["REPRJCTD"]
            heliocentric_distance = primary[0].header["HELIO"]
            for i in range(n_constituents):
                if f"GEO_{i}" in primary[0].header:
                    org_image_meta["geocentric_distance"][i] = primary[0].header[f"GEO_{i}"]

            # Extract the per-image data from header innformation if needed.
            # This happens with when the WorkUnit was saved before metadata tables were
            # saved as layers.that we will fill in from the headers.
            for i in range(n_constituents):
                if f"WCS_{i}" in primary:
                    org_image_meta["original_wcs"][i] = extract_wcs_from_hdu_header(
                        primary[f"WCS_{i}"].header
                    )
                    org_image_meta["data_loc"][i] = primary[f"WCS_{i}"].header["ILOC"]
                if f"EBD_{i}" in primary:
                    org_image_meta["original_wcs"][i] = extract_wcs_from_hdu_header(
                        primary[f"EBD_{i}"].header
                    )

        file_paths = []
        obstimes = []
        for i in range(num_images):
            shard_path = os.path.join(directory, f"{i}_{filename}")
            if not Path(shard_path).is_file():
                raise ValueError(f"No shard provided for index {i} for {filename}")
            with fits.open(shard_path) as hdul:
                # Read in the image file.
                sci_hdu = hdul[f"SCI_{i}"]
                obstimes.append(sci_hdu.header["MJD"])

                # Read in the layered image from different extensions.
                if not lazy:
                    img = load_layered_image_from_shard(shard_path)
                    # force_move destroys img object, but avoids a copy.
                    im_stack.append_image(img, force_move=True)
                else:
                    file_paths.append(shard_path)

                if "NIND" in sci_hdu.header:
                    n_indices = sci_hdu.header["NIND"]
                    sub_indices = []
                    for j in range(n_indices):
                        sub_indices.append(sci_hdu.header[f"IND_{j}"])
                    per_image_meta["per_image_indices"][i] = sub_indices

        file_paths = None if not lazy else file_paths
        result = WorkUnit(
            im_stack=im_stack,
            config=config,
            wcs=global_wcs,
            reprojected=reprojected,
            heliocentric_distance=heliocentric_distance,
            file_paths=file_paths,
            lazy=lazy,
            image_meta=per_image_meta,
            org_image_meta=org_image_meta,
        )
        return result

    def metadata_to_hdul(self, include_wcs=True):
        """Creates the metadata fits headers.

        Parameters
        ----------
        include_wcs : `bool`
            whether or not to append all the per image wcses
            to the header (optional for the serial `to_fits`
            case so that we can maintain the same indexing
            as before).
        Returns
        -------
        hdul : `astropy.io.fits.HDUList`
            The HDU List.
        """
        # Set up the initial HDU list, including the primary header
        # the metadata (empty), and the configuration.
        hdul = fits.HDUList()
        pri = fits.PrimaryHDU()
        pri.header["NUMIMG"] = self.n_images
        pri.header["NCON"] = self.n_constituents
        pri.header["REPRJCTD"] = self.reprojected
        pri.header["HELIO"] = self.heliocentric_distance

        # If the global WCS exists, append the corresponding keys.
        if self.wcs is not None:
            append_wcs_to_hdu_header(self.wcs, pri.header)
        hdul.append(pri)

        # Add the configuration layer.
        config_hdu = self.config.to_hdu()
        config_hdu.name = "kbmod_config"
        hdul.append(config_hdu)

        # Save the additional metadata tables into HDUs
        hdul.append(metadata_table_to_hdu(self.img_meta, "IMG_META"))
        hdul.append(metadata_table_to_hdu(self.org_img_meta, "ORG_META"))

        return hdul

    def image_positions_to_original_icrs(
        self, image_indices, positions, input_format="xy", output_format="xy", filter_in_frame=True
    ):
        """Method to transform image positions in EBD reprojected images
        into coordinates in the orignal ICRS frame of reference.

        Parameters
        ----------
        image_indices : `numpy.array`
            The `ImageStack` indices to transform coordinates.
        positions : `list` of `astropy.coordinates.SkyCoord`s or `tuple`s
            The positions to be transformed.
        input_format : `str`
            The input format for the positions. Either 'xy' or 'radec'.
            If 'xy' is given, positions must be in the format of a
            `tuple` with two float or integer values, like (x, y).
            If 'radec' is given, positions must be in the format of
            a `astropy.coordinates.SkyCoord`.
        output_format : `str`
            The output format for the positions. Either 'xy' or 'radec'.
            If 'xy' is given, positions will be returned in the format of a
            `tuple` with two `int`s, like (x, y).
            If 'radec' is given, positions will be returned in the format of
            a `astropy.coordinates.SkyCoord`.
        filter_in_frame : `bool`
            Whether or not to filter the output based on whether they fit within the
            original `constituent_image` frame. If `True`, only results that fall within
            the bounds of the original WCS will be returned.

        Returns
        -------
        positions : `list` of `astropy.coordinates.SkyCoord`s or `tuple`s
            The transformed positions. If `filter_in_frame` is true, each
            element of the result list will also be a tuple with the
            URI string of the constituent image matched to the position.
        """
        if input_format not in ["xy", "radec"]:
            raise ValueError(f"input format must be 'xy' or 'radec' , '{input_format}' provided")
        if input_format == "xy":
            if not all(isinstance(i, tuple) and len(i) == 2 for i in positions):
                raise ValueError("positions in incorrect format for input_format='xy'")
        if input_format == "radec" and not all(isinstance(i, SkyCoord) for i in positions):
            raise ValueError("positions in incorrect format for input_format='radec'")
        if len(positions) != len(image_indices):
            raise ValueError(f"wrong number of inputs, expected {len(image_indices)}, got {len(positions)}")

        if output_format not in ["xy", "radec"]:
            raise ValueError(f"output format must be 'xy' or 'radec' , '{output_format}' provided")

        position_ebd_coords = positions

        if input_format == "xy":
            radec_coords = []
            for pos, ind in zip(positions, image_indices):
                ebd_wcs = self.get_wcs(ind)
                ra, dec = ebd_wcs.all_pix2world(pos[0], pos[1], 0)
                radec_coords.append(SkyCoord(ra=ra, dec=dec, unit="deg"))
            position_ebd_coords = radec_coords

        helio_dist = self.heliocentric_distance
        geo_dists = [self.org_img_meta["geocentric_distance"][i] for i in image_indices]
        all_times = self.get_all_obstimes()
        obstimes = [all_times[i] for i in image_indices]

        # this should be part of the WorkUnit metadata
        location = EarthLocation.of_site("ctio")

        inverted_coords = []
        for coord, ind, obstime, geo_dist in zip(position_ebd_coords, image_indices, obstimes, geo_dists):
            inverted_coord = invert_correct_parallax(
                coord=coord,
                obstime=Time(obstime, format="mjd"),
                point_on_earth=location,
                heliocentric_distance=helio_dist,
                geocentric_distance=geo_dist,
            )
            inverted_coords.append(inverted_coord)

        if output_format == "radec" and not filter_in_frame:
            return inverted_coords

        positions = []
        for i in image_indices:
            inds = self.img_meta["per_image_indices"][i]
            coord = inverted_coords[i]
            pos = []
            for j in inds:
                con_image = self.org_img_meta["data_loc"][j]
                con_wcs = self.org_img_meta["original_wcs"][j]
                height, width = con_wcs.array_shape
                x, y = skycoord_to_pixel(coord, con_wcs)
                x, y = float(x), float(y)
                if output_format == "xy":
                    result_coord = (x, y)
                else:
                    result_coord = coord
                to_allow = (y >= 0.0 and y <= height and x >= 0 and x <= width) or (not filter_in_frame)
                if to_allow:
                    pos.append((result_coord, con_image))
            if len(pos) == 0:
                positions.append(None)
            elif len(pos) > 1:
                positions.append(pos)
                if filter_in_frame:
                    warnings.warn(
                        f"ambiguous image origin for coordinate {i}, including all potential constituent images.",
                        Warning,
                    )
            else:
                positions.append(pos[0])
        return positions

    def load_images(self):
        """Function for loading in `ImageStack` data when `WorkUnit`
        was created lazily.
        """
        if not self.lazy:
            raise ValueError("ImageStack has already been loaded.")
        im_stack = ImageStack()

        for file_path in self.file_paths:
            img = load_layered_image_from_shard(file_path)
            # force_move destroys img object, but avoids a copy.
            im_stack.append_image(img, force_move=True)

        self.im_stack = im_stack
        self.lazy = False


def load_layered_image_from_shard(file_path):
    """Function for loading a `LayeredImage` from
    a `WorkUnit` shard.

    Parameters
    ----------
    file_path : `str`
        The location of the shard file.

    Returns
    -------
    img : `LayeredImage`
        The materialized `LayeredImage`.
    """
    if not Path(file_path).is_file():
        raise ValueError(f"provided file_path '{file_path}' is not an existing file.")

    index = int(file_path.split("/")[-1].split("_")[0])
    with fits.open(file_path) as hdul:
        img = LayeredImage(
            hdul[f"SCI_{index}"].data.astype(np.single),
            hdul[f"VAR_{index}"].data.astype(np.single),
            hdul[f"MSK_{index}"].data.astype(np.single),
            PSF(hdul[f"PSF_{index}"].data),
            hdul[f"SCI_{index}"].header["MJD"],
        )
        return img


def raw_image_to_hdu(img, obstime, wcs=None):
    """Helper function that creates a HDU out of RawImage.

    Parameters
    ----------
    img : `RawImage`
        The RawImage to convert.
    obstime : `float`
        The time of the observation.
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
    hdu.header["MJD"] = obstime

    return hdu


# ------------------------------------------------------------------
# --- Utility functions for the metadata table ---------------------
# ------------------------------------------------------------------


def metadata_table_to_hdu(data, layer_name=None):
    """Create a HDU layer from an astropy table with custom
    encodings for some columns (such as WCS).

    Parameters
    ----------
    data : `astropy.table.Table`
        The table of the data to save.
    layer_name : `str`, optional
        The name of the layer in which to save the table.
    """
    num_rows = len(data)
    if num_rows == 0:
        # No data to encode. Just use the current table.
        save_table = data
    else:
        # Create a new table to save with the correct column
        # values/names for the serialized information.
        save_table = Table()
        for colname in data.colnames:
            col_data = data[colname].value

            if np.all(col_data == None):
                # The entire column is filled with Nones (probably from a default value).
                save_table[f"_EMPTY_{colname}"] = np.full(num_rows, "None")
            elif isinstance(col_data[0], WCS):
                # Serialize WCS objects and use a custom tag so we can unserialize them.
                values = np.array([serialize_wcs(entry) for entry in data[colname]])
                save_table[f"_WCSSTR_{colname}"] = values
            else:
                save_table[colname] = data[colname]

    # Format the metadata as a single HDU
    meta_hdu = fits.BinTableHDU(save_table)
    if layer_name is not None:
        meta_hdu.name = layer_name
    return meta_hdu


def hdu_to_metadata_table(hdu):
    """Load a HDU layer with custom encodings for some columns (such as WCS)
    to an astropy table.

    Parameters
    ----------
    hdu : `astropy.io.fits.BinTableHDU`
        The HDUList for the fits file.

    Returns
    -------
    data : `astropy.table.Table`
        The table of loaded data.
    """
    if hdu is None:
        # Nothing to decode. Return an empty table.
        return Table()

    data = Table(hdu.data)
    all_cols = set(data.colnames)

    # Check if there are any columns we need to decode. If so: decode them, add a new column,
    # and delete the old column.
    for colname in all_cols:
        if colname.startswith("_WCSSTR_"):
            data[colname[8:]] = np.array([deserialize_wcs(entry) for entry in data[colname]])
            data.remove_column(colname)
        elif colname.startswith("_EMPTY_"):
            data[colname[7:]] = np.array([None for _ in data[colname]])
            data.remove_column(colname)

    return data
