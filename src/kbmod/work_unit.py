import math
import os

from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import numpy as np
from pathlib import Path
import warnings
from yaml import dump, safe_load
from tqdm import tqdm

from kbmod.configuration import SearchConfiguration
from kbmod.search import ImageStack, LayeredImage, PSF, RawImage, Logging
from kbmod.wcs_utils import (
    append_wcs_to_hdu_header,
    extract_wcs_from_hdu_header,
    wcs_fits_equal,
    wcs_from_dict,
    wcs_to_dict,
)
from kbmod.reprojection_utils import invert_correct_parallax
from kbmod.tqdm_utils import TQDMUtils


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
    constituent_images: `list`
        A list of strings with the original locations of images used
        to construct the WorkUnit. This is necessary to maintain as metadata
        because after reprojection we may stitch multiple images into one.
    per_image_wcs : `list`
        A list with one WCS for each image in the WorkUnit. Used for when
        the images have *not* been standardized to the same pixel space.
    per_image_ebd_wcs : `list`
        A list with one WCS for each image in the WorkUnit. Used to reproject the images
        into EBD space.
    heliocentric_distance : `float`
        The heliocentric distance that was used when creating the `per_image_ebd_wcs`.
    geocentric_distances : `list`
        The best fit geocentric distances used when creating the `per_image_ebd_wcs`.
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
    """

    def __init__(
        self,
        im_stack=None,
        config=None,
        wcs=None,
        constituent_images=None,
        per_image_wcs=None,
        per_image_ebd_wcs=None,
        heliocentric_distance=None,
        geocentric_distances=None,
        reprojected=False,
        per_image_indices=None,
        lazy=False,
        file_paths=None,
        obstimes=None,
    ):
        self.im_stack = im_stack
        self.config = config
        self.lazy = lazy
        self.file_paths = file_paths
        self._obstimes = obstimes

        # Handle WCS input. If both the global and per-image WCS are provided,
        # ensure they are consistent.
        self.wcs = wcs
        if constituent_images is None:
            n_constituents = im_stack.img_count()
            self.constituent_images = [None] * n_constituents
        else:
            n_constituents = len(constituent_images)
            self.constituent_images = constituent_images

        if per_image_wcs is None:
            self._per_image_wcs = [None] * n_constituents
            if self.wcs is None and per_image_ebd_wcs is None:
                warnings.warn("No WCS provided.", Warning)
        else:
            if len(per_image_wcs) != n_constituents:
                raise ValueError(f"Incorrect number of WCS provided. Expected {n_constituents}")
            self._per_image_wcs = per_image_wcs

            # Check if all the per-image WCS are None. This can happen during a load.
            all_none = self.per_image_wcs_all_match(None)
            if self.wcs is None and all_none:
                warnings.warn("No WCS provided.", Warning)

            # See if we can compress the per-image WCS into a global one.
            if self.wcs is None and not all_none and self.per_image_wcs_all_match(self._per_image_wcs[0]):
                self.wcs = self._per_image_wcs[0]
                self._per_image_wcs = [None] * im_stack.img_count()

        # TODO: Refactor all of this code to make it cleaner

        if per_image_ebd_wcs is None:
            self._per_image_ebd_wcs = [None] * n_constituents
        else:
            if len(per_image_ebd_wcs) != n_constituents:
                raise ValueError(f"Incorrect number of EBD WCS provided. Expected {n_constituents}")
            self._per_image_ebd_wcs = per_image_ebd_wcs

        if geocentric_distances is None:
            self.geocentric_distances = [None] * n_constituents
        else:
            self.geocentric_distances = geocentric_distances

        self.heliocentric_distance = heliocentric_distance
        self.reprojected = reprojected
        if per_image_indices is None:
            self._per_image_indices = [[i] for i in range(len(self.constituent_images))]
        else:
            self._per_image_indices = per_image_indices

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
        if img_num < 0 or img_num >= len(self._per_image_wcs):
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

    def get_num_images(self):
        return len(self._per_image_indices)

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

        im_stack = ImageStack()
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
            n_constituents = hdul[0].header["NCON"]
            expected_num_images = (4 * num_images) + (2 * n_constituents) + 3
            if len(hdul) != expected_num_images:
                raise ValueError(f"WorkUnit wrong number of extensions. Expected " f"{expected_num_images}.")
            logger.info(f"Loading {num_images} images and {expected_num_images} total layers.")

            # Misc. reprojection metadata
            reprojected = hdul[0].header["REPRJCTD"]
            heliocentric_distance = hdul[0].header["HELIO"]
            geocentric_distances = []
            for i in range(num_images):
                geocentric_distances.append(hdul[0].header[f"GEO_{i}"])

            per_image_indices = []
            # Read in all the image files.
            for i in tqdm(
                range(num_images), bar_format=TQDMUtils.DEFAULT_TQDM_BAR_FORMAT, desc="Loading images"
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

                n_indices = sci_hdu.header["NIND"]
                sub_indices = []
                for j in range(n_indices):
                    sub_indices.append(sci_hdu.header[f"IND_{j}"])
                per_image_indices.append(sub_indices)

            per_image_wcs = []
            per_image_ebd_wcs = []
            constituent_images = []
            for i in tqdm(
                range(n_constituents), bar_format=TQDMUtils.DEFAULT_TQDM_BAR_FORMAT, desc="Loading WCS"
            ):
                # Extract the per-image WCS if one exists.
                per_image_wcs.append(extract_wcs_from_hdu_header(hdul[f"WCS_{i}"].header))
                per_image_ebd_wcs.append(extract_wcs_from_hdu_header(hdul[f"EBD_{i}"].header))
                constituent_images.append(hdul[f"WCS_{i}"].header["ILOC"])

        result = WorkUnit(
            im_stack=im_stack,
            config=config,
            wcs=global_wcs,
            constituent_images=constituent_images,
            per_image_wcs=per_image_wcs,
            per_image_ebd_wcs=per_image_ebd_wcs,
            heliocentric_distance=heliocentric_distance,
            geocentric_distances=geocentric_distances,
            reprojected=reprojected,
            per_image_indices=per_image_indices,
        )
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

        constituent_images = workunit_dict["constituent_images"]
        heliocentric_distance = workunit_dict["heliocentric_distance"]
        geocentric_distances = workunit_dict["geocentric_distances"]
        reprojected = workunit_dict["reprojected"]
        per_image_indices = workunit_dict["per_image_indices"]

        imgs = []
        per_image_wcs = []
        per_image_ebd_wcs = []
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

        n_constituents = len(constituent_images)
        for i in range(n_constituents):
            # Read a per_image_wcs if one exists.
            current_wcs = workunit_dict["per_image_wcs"][i]
            if type(current_wcs) is dict:
                current_wcs = wcs_from_dict(current_wcs)
            per_image_wcs.append(current_wcs)

            current_ebd = workunit_dict["per_image_ebd_wcs"][i]
            if type(current_ebd) is dict:
                current_ebd = wcs_from_dict(current_ebd)
            per_image_ebd_wcs.append(current_ebd)

        im_stack = ImageStack(imgs)
        return WorkUnit(
            im_stack=im_stack,
            config=config,
            wcs=global_wcs,
            constituent_images=constituent_images,
            per_image_wcs=per_image_wcs,
            per_image_ebd_wcs=per_image_ebd_wcs,
            heliocentric_distance=heliocentric_distance,
            geocentric_distances=geocentric_distances,
            reprojected=reprojected,
            per_image_indices=per_image_indices,
        )

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
        logger.info(f"Writing WorkUnit with {self.im_stack.img_count()} images to file {filename}")
        if Path(filename).is_file() and not overwrite:
            raise FileExistsError(f"WorkUnit file {filename} already exists.")

        hdul = self.metadata_to_primary_header(include_wcs=False)

        for i in range(self.im_stack.img_count()):
            layered = self.im_stack.get_single_image(i)
            c_indices = self._per_image_indices[i]
            n_indices = len(c_indices)

            img_wcs = self.get_wcs(i)
            sci_hdu = raw_image_to_hdu(layered.get_science(), img_wcs)
            sci_hdu.name = f"SCI_{i}"
            sci_hdu.header["NIND"] = n_indices
            for j in range(n_indices):
                sci_hdu.header[f"IND_{j}"] = c_indices[j]
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

        self.append_all_wcs(hdul)

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
            c_indices = self._per_image_indices[i]
            n_indices = len(c_indices)
            sub_hdul = fits.HDUList()

            img_wcs = self.get_wcs(i)
            sci_hdu = raw_image_to_hdu(layered.get_science(), img_wcs)
            sci_hdu.name = f"SCI_{i}"
            sci_hdu.header["NIND"] = n_indices
            for j in range(n_indices):
                sci_hdu.header[f"IND_{j}"] = c_indices[j]
            sub_hdul.append(sci_hdu)

            var_hdu = raw_image_to_hdu(layered.get_variance())
            var_hdu.name = f"VAR_{i}"
            sub_hdul.append(var_hdu)

            msk_hdu = raw_image_to_hdu(layered.get_mask())
            msk_hdu.name = f"MSK_{i}"
            sub_hdul.append(msk_hdu)

            p = layered.get_psf()
            psf_array = np.array(p.get_kernel()).reshape((p.get_dim(), p.get_dim()))
            psf_hdu = fits.hdu.image.ImageHDU(psf_array)
            psf_hdu.name = f"PSF_{i}"
            sub_hdul.append(psf_hdu)
            sub_hdul.writeto(os.path.join(directory, f"{i}_{filename}"))

        hdul = self.metadata_to_primary_header(include_wcs=True)
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

            # Read in the global WCS from extension 0 if the information exists.
            # We filter the warning that the image dimension does not match the WCS dimension
            # since the primary header does not have an image.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                global_wcs = extract_wcs_from_hdu_header(primary[0].header)

            # Read the size and order information from the primary header.
            num_images = primary[0].header["NUMIMG"]
            n_constituents = primary[0].header["NCON"]
            expected_num_images = (4 * num_images) + (2 * n_constituents) + 3

            # Misc. reprojection metadata
            reprojected = primary[0].header["REPRJCTD"]
            heliocentric_distance = primary[0].header["HELIO"]
            geocentric_distances = []
            for i in range(n_constituents):
                geocentric_distances.append(primary[0].header[f"GEO_{i}"])

            per_image_wcs = []
            per_image_ebd_wcs = []
            constituent_images = []
            for i in range(n_constituents):
                # Extract the per-image WCS if one exists.
                per_image_wcs.append(extract_wcs_from_hdu_header(primary[f"WCS_{i}"].header))
                per_image_ebd_wcs.append(extract_wcs_from_hdu_header(primary[f"EBD_{i}"].header))
                constituent_images.append(primary[f"WCS_{i}"].header["ILOC"])
        per_image_indices = []
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

                n_indices = sci_hdu.header["NIND"]
                sub_indices = []
                for j in range(n_indices):
                    sub_indices.append(sci_hdu.header[f"IND_{j}"])
                per_image_indices.append(sub_indices)

        file_paths = None if not lazy else file_paths
        result = WorkUnit(
            im_stack=im_stack,
            config=config,
            wcs=global_wcs,
            constituent_images=constituent_images,
            per_image_wcs=per_image_wcs,
            per_image_ebd_wcs=per_image_ebd_wcs,
            heliocentric_distance=heliocentric_distance,
            geocentric_distances=geocentric_distances,
            reprojected=reprojected,
            per_image_indices=per_image_indices,
            lazy=lazy,
            file_paths=file_paths,
            obstimes=obstimes,
        )
        return result

    def metadata_to_primary_header(self, include_wcs=True):
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
        pri.header["NUMIMG"] = self.get_num_images()
        pri.header["REPRJCTD"] = self.reprojected
        pri.header["HELIO"] = self.heliocentric_distance
        for i in range(len(self.constituent_images)):
            pri.header[f"GEO_{i}"] = self.geocentric_distances[i]

        # If the global WCS exists, append the corresponding keys.
        if self.wcs is not None:
            append_wcs_to_hdu_header(self.wcs, pri.header)

        pri.header["NCON"] = len(self.constituent_images)

        hdul.append(pri)

        meta_hdu = fits.BinTableHDU()
        meta_hdu.name = "metadata"
        hdul.append(meta_hdu)

        config_hdu = self.config.to_hdu()
        config_hdu.name = "kbmod_config"
        hdul.append(config_hdu)

        if include_wcs:
            self.append_all_wcs(hdul)

        return hdul

    def append_all_wcs(self, hdul):
        """Append the `_per_image_wcs` and
        `_per_image_ebd_wcs` elements to a header.

        Parameters
        ----------
        hdul : `astropy.io.fits.HDUList`
            The HDU list.
        """
        n_constituents = len(self.constituent_images)
        for i in range(n_constituents):
            img_location = self.constituent_images[i]

            orig_wcs = self._per_image_wcs[i]
            wcs_hdu = fits.TableHDU()
            append_wcs_to_hdu_header(orig_wcs, wcs_hdu.header)
            wcs_hdu.name = f"WCS_{i}"
            wcs_hdu.header["ILOC"] = img_location
            hdul.append(wcs_hdu)

            im_ebd_wcs = self._per_image_ebd_wcs[i]
            ebd_hdu = fits.TableHDU()
            append_wcs_to_hdu_header(im_ebd_wcs, ebd_hdu.header)
            ebd_hdu.name = f"EBD_{i}"
            hdul.append(ebd_hdu)

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
            "constituent_images": self.constituent_images,
            "per_image_wcs": [],
            "per_image_ebd_wcs": [],
            "heliocentric_distance": self.heliocentric_distance,
            "geocentric_distances": self.geocentric_distances,
            "reprojected": self.reprojected,
            "per_image_indices": self._per_image_indices,
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

        for i in range(len(self._per_image_wcs)):
            workunit_dict["per_image_wcs"].append(wcs_to_dict(self._per_image_wcs[i]))
            workunit_dict["per_image_ebd_wcs"].append(wcs_to_dict(self._per_image_ebd_wcs[i]))

        return dump(workunit_dict)

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
        geo_dists = [self.geocentric_distances[i] for i in image_indices]
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
            inds = self._per_image_indices[i]
            coord = inverted_coords[i]
            pos = []
            for j in inds:
                con_image = self.constituent_images[j]
                con_wcs = self._per_image_wcs[j]
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
        raise ValueError("file_path must be an existing file.")

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
