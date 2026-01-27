from collections.abc import Iterable
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
from kbmod.core.image_stack_py import ImageStackPy, LayeredImagePy
from kbmod.reprojection_utils import invert_correct_parallax, image_positions_to_original_icrs
from kbmod.search import Logging
from kbmod.util_functions import get_matched_obstimes
from kbmod.wcs_utils import (
    append_wcs_to_hdu_header,
    calc_ecliptic_angle,
    deserialize_wcs,
    extract_wcs_from_hdu_header,
    serialize_wcs,
)

_DEFAULT_WORKUNIT_TQDM_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"


logger = Logging.getLogger(__name__)


class WorkUnit:
    """The work unit is a storage and I/O class for all of the data
    needed for a full run of KBMOD, including the: the search parameters,
    data files, and the data provenance metadata.

    Attributes
    ----------
    im_stack : `ImageStackPy`
        The image data for the KBMOD run.
    config : `kbmod.configuration.SearchConfiguration`
        The configuration for the KBMOD run.
    n_constituents : `int`
        The number of original images making up the data in this WorkUnit. This might be
        different from the number of images stored in memory if the WorkUnit has been
        reprojected.
    org_img_meta : `astropy.table.Table`
        The meta data for each constituent image. Columns are all optional and can include:
        * data_loc - the original location of the image data.
        * ebd_wcs - Used to reproject the images into EBD space.
        * geocentric_distance - The best fit geocentric distances used when creating
          the per image EBD WCS.
        * original_wcs - The original per-image WCS of the image.
        * visit - The visit number of the image (if known).
        * filter - The filter used for the image.
    wcs : `astropy.wcs.WCS`
        A global WCS for all images in the WorkUnit. Only exists
        if all images have been projected to same pixel space.
    barycentric_distance : `float`
        The barycentric distance that was used when creating the `per_image_ebd_wcs` (in AU).
    reprojected : `bool`
        Whether or not the WorkUnit image data has been reprojected.
    per_image_indices : `list` of `list`
        A list of lists containing the indicies of `constituent_images` at each layer
        of the image stack. Used for finding corresponding original images when we
        stitch images together during reprojection.
    lazy : `bool`
        Whether or not to load the image data for the `WorkUnit`.
    file_paths : `list[str]`
        The paths for the shard files, only created if the `WorkUnit` is loaded
        in lazy mode.
    obstimes : `list[float]`
        The MJD obstimes of the midpoint of the images (in UTC).

    Parameters
    ----------
    im_stack : `ImageStackPy`
        The image data for the KBMOD run.
    config : `kbmod.configuration.SearchConfiguration`
        The configuration for the KBMOD run.
    wcs : `astropy.wcs.WCS`, optional
        A global WCS for all images in the WorkUnit. Only exists
        if all images have been projected to same pixel space.
    per_image_wcs : `list`, optional
        A list with one WCS for each image in the WorkUnit. Used for when
        the images have *not* been standardized to the same pixel space. If provided
        this will overwrite the WCS values in org_image_meta
    reprojected : `bool`, optional
        Whether or not the WorkUnit image data has been reprojected.
    reprojection_frame : `str`, optional
        Which coordinate frame the WorkUnit has been reprojected into, either
        "original" or "ebd" for a parallax corrected reprojection.
    per_image_indices : `list` of `list`, optional
        A list of lists containing the indicies of `constituent_images` at each layer
        of the image stack. Used for finding corresponding original images when we
        stitch images together during reprojection.
    barycentric_distance : `float`, optional
        The barycentric distance that was used when creating the `per_image_ebd_wcs` (in AU).
    lazy : `bool`, optional
        Whether or not to load the image data for the `WorkUnit`.
    file_paths : `list[str]`, optional
        The paths for the shard files, only created if the `WorkUnit` is loaded
        in lazy mode.
    obstimes : `list[float]`
        The MJD obstimes of the midpoint of the images (in UTC).
    org_image_meta : `astropy.table.Table`, optional
        A table of per-image data for the constituent images.
    """

    def __init__(
        self,
        im_stack,
        config,
        wcs=None,
        per_image_wcs=None,
        reprojected=False,
        reprojection_frame=None,
        per_image_indices=None,
        barycentric_distance=None,
        lazy=False,
        file_paths=None,
        obstimes=None,
        org_image_meta=None,
        observatory=None,
    ):
        # Assign the core components.
        self.im_stack = im_stack
        self.config = config
        self.lazy = lazy
        self.file_paths = file_paths
        self._obstimes = obstimes

        # Validate the image stack (in warning only mode).
        if not lazy:
            im_stack.validate()

        # Determine the number of constituent images. If we are given metadata for the
        # of constituent_images, use that. Otherwise use the size of the image stack.
        if org_image_meta is not None:
            self.n_constituents = len(org_image_meta)
        elif per_image_wcs is not None:
            self.n_constituents = len(per_image_wcs)
        else:
            self.n_constituents = im_stack.num_times

        # Track the metadata for each constituent image in the WorkUnit. If no constituent
        # data is provided, this will create a table of default values the correct size.
        self.org_img_meta = create_image_metadata(self.n_constituents, data=org_image_meta)

        # Handle WCS input. If per_image_wcs is provided as an argument, use that.
        # If no per_image_wcs values are provided, use the global one.
        self.wcs = wcs
        if per_image_wcs is not None:
            if len(per_image_wcs) != self.n_constituents:
                raise ValueError(f"Incorrect number of WCS provided. Expected {self.n_constituents}")
            self.org_img_meta["per_image_wcs"] = per_image_wcs
        if np.all(self.org_img_meta["per_image_wcs"] == None):
            self.org_img_meta["per_image_wcs"] = np.full(self.n_constituents, self.wcs)
        if np.any(self.org_img_meta["per_image_wcs"] == None):
            warnings.warn("At least one image does not have a WCS.", Warning)

        # Set the global metadata for reprojection.
        self.reprojected = reprojected
        self.reprojection_frame = reprojection_frame
        self.barycentric_distance = barycentric_distance

        # Set the observatory location (defaults to Rubin Observatory).
        if observatory is None:
            logger.warning("No observatory location provided, defaulting to Rubin Observatory.")
            # Rubin Observatory coordinates (site name not available in all astropy versions)
            self.observatory = EarthLocation(
                lat=-30.24463333 * u.deg, lon=-70.74941667 * u.deg, height=2662.75 * u.m
            )
        else:
            self.observatory = observatory

        # If we have mosaicked images, each image in the stack could link back
        # to more than one constituents image. Build a mapping of image stack index
        # to needed original image indices.
        if per_image_indices is None:
            self._per_image_indices = [[i] for i in range(self.n_constituents)]
        else:
            self._per_image_indices = per_image_indices

        # Run some basic validity checks.
        if self.reprojected and self.wcs is None:
            raise ValueError("Global WCS required for reprojected data.")
        for inds in self._per_image_indices:
            if np.max(inds) >= self.n_constituents:
                raise ValueError(
                    f"Found pointer to constituents image {np.max(inds)} of {self.n_constituents}"
                )

    def __len__(self):
        """Returns the size of the WorkUnit in number of images."""
        return self.im_stack.num_times

    def get_num_images(self):
        return len(self._per_image_indices)

    def print_stats(self):
        print("WorkUnit:")
        print(f"  Num Constituent Images ({self.n_constituents}):")
        print(f"  Reprojected: {self.reprojected}")
        if self.reprojected:
            print(f"  Reprojected Frame: {self.reprojection_frame}")
            print(f"  Barycentric Distance: {self.barycentric_distance}")

        self.im_stack.print_stats()

    def get_constituent_meta(self, column):
        """Get the metadata values of a given column or a list of columns
        for all the constituent images.

        Parameters
        ----------
        column : `str`, or Iterable
            The column name(s) to fetch.

        Returns
        -------
        data : `list` or `dict`
            If a single string column name is provided, the function returns the
            values in a list. Otherwise it returns a dictionary, mapping
            each column name to its values.
        """
        if isinstance(column, str):
            return self.org_img_meta[column].data.tolist()
        elif isinstance(column, Iterable):
            results = {}
            for col in column:
                if col in self.org_img_meta.colnames:
                    results[col] = self.org_img_meta[col].data.tolist()
            return results
        else:
            raise TypeError(f"Unsupported column type {type(column)}")

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
        if self.wcs is not None:
            return self.wcs
        else:
            # If there is no common WCS, use the original per-image one.
            return self.org_img_meta["per_image_wcs"][img_num]

    def get_pixel_coordinates(self, ra, dec, times=None):
        """Get the pixel coordinates for pairs of (RA, dec) coordinates. Uses the global
        WCS if one exists. Otherwise uses the per-image WCS. If times is provided, uses those values
        to choose the per-image WCS.

        Parameters
        ----------
        ra : `numpy.ndarray`
            The right ascension coordinates (in degrees.
        dec : `numpy.ndarray`
            The declination coordinates in degrees.
        times : `numpy.ndarray` or `None`, optional
            The times to match in MJD.

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
                current_wcs = self.org_img_meta["per_image_wcs"][index]
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
            The computed ecliptic_angle in radians in pixel space or
            ``None`` if data is missing.
        """

        wcs = self.get_wcs(0)
        if wcs is None or self.im_stack is None:
            logger.warning(f"A valid wcs and ImageStackPy is needed to compute the ecliptic angle.")
            return None
        center_pixel = (self.im_stack.width / 2, self.im_stack.height / 2)
        return calc_ecliptic_angle(wcs, center_pixel)

    def get_all_obstimes(self):
        """Return a list of the observation times in MJD.

        If the `WorkUnit` was lazily loaded, then the obstimes have already been preloaded.
        Otherwise, grab them from the `ImageStackPy`.

        Returns
        -------
        obs_times : `list`
            The list of observation times in MJD.
        """
        if self._obstimes is not None:
            return self._obstimes

        self._obstimes = np.copy(self.im_stack.times)
        return self._obstimes

    def get_unique_obstimes_and_indices(self):
        """Returns the unique obstimes and the list of indices that they are associated with.

        Returns
        -------
        unique_obstimes : `list`
            The list of unique observation times in MJD.
        unique_indices : `list`
            The list of the indices corresponding to each observation time.
        """
        all_obstimes = self.get_all_obstimes()
        unique_obstimes = np.unique(all_obstimes)
        unique_indices = [list(np.where(all_obstimes == time)[0]) for time in unique_obstimes]
        return unique_obstimes, unique_indices

    def filter_images(self, mask):
        """Filter the images in the WorkUnit by a Boolean mask, keeping
        only the images where the mask is True.  If the images have been
        reprojected all constituent metadata will be kept.

        Parameters
        ----------
        mask : `list` or `numpy.ndarray`
            A Boolean array of length equal to the number of images in the
            WorkUnit. True values indicate that the image should be kept.
        """
        if len(mask) != self.im_stack.num_times:
            raise ValueError(
                f"Mask length {len(mask)} does not match number of images {self.im_stack.num_times}."
            )
        if np.all(mask):
            return  # Nothing to do.

        # Filter the image stack.
        self.im_stack.filter_images(mask)
        if self.file_paths is not None:
            self.file_paths = [val for idx, val in enumerate(self.file_paths) if mask[idx]]
        if self._obstimes is not None:
            self._obstimes = [val for idx, val in enumerate(self._obstimes) if mask[idx]]

        # If the stack has not been reprojected, we are filtering the original images.
        if not self.reprojected:
            self.org_img_meta = self.org_img_meta[mask]
            self.n_constituents = len(self.org_img_meta)
            self._per_image_indices = [[i] for i in range(self.n_constituents)]
        else:
            self._per_image_indices = [val for idx, val in enumerate(self._per_image_indices) if mask[idx]]

    def apply_color_scale(self, color_scale=None):
        """Apply color scaling to the images in the WorkUnit.

        Parameters
        ----------
        color_scale : `dict`, optional
            A dictionary mapping filter names to their scaling factors.
        """
        if color_scale is None:
            return  # Nothing to do.

        # Check that we have the information to make the filters to their scaling factors.
        if "filter" not in self.org_img_meta.colnames:
            raise ValueError("No filter information in WorkUnit to apply color scaling.")
        filters = np.asarray(self.org_img_meta["filter"])

        num_img = len(self.im_stack)
        scaling_factors = np.zeros(num_img)
        for i in range(num_img):
            const_indices = np.asarray(self._per_image_indices[i])
            const_filters = filters[const_indices]
            if len(np.unique(const_filters)) != 1:
                raise ValueError(
                    "More than one filter found for a single image in the WorkUnit. "
                    f"Image {i} has filters {const_filters}."
                )
            if const_filters[0] not in color_scale:
                raise ValueError(f"Color scale missing entry for filter {const_filters[0]}.")

            scaling_factors[i] = color_scale[const_filters[0]]
        self.im_stack.scale_images(scaling_factors, is_magnitude=True)

        # Save the scaling factors to the metadata. These need to be per-constituent image
        # (instead of per-current image).
        self.org_img_meta["color_scale"] = np.array([color_scale[filt] for filt in filters])

    def disorder_obstimes(self):
        """Reorders the timestamps in the WorkUnit to be random. Random offsets
        are chosen for each unique obstime and added to the original obstime.
        The maximum offset is the number of images/times in the image stack or
        the difference between the maximum and minimum obstime.

        The offsets are applied such that images will have a shared
        obstime if they did so before this method was called.
        The WorkUnit's image stack is then sorted in ascending order of the
        updated obstimes.

        This is useful for testing and ML training purposes where we might
        want to perform a search on a WorkUnit that would produce unlikely
        KBMOD results.
        """
        unique_obstimes = np.unique(self.get_all_obstimes())
        if len(unique_obstimes) == 0:
            raise ValueError("No obstimes provided for WorkUnit.")

        # Randomly select an offset between 0 and the max time difference
        # which can be added to the minimum time. This should be randomly
        # sampled *without* replacement so that we don't have duplicate times. Note
        # if the max time difference is less than the number of times in the im_stack,
        # we will use the number of times in the im_stack as the max offset.
        max_offset = max(np.max(unique_obstimes) - np.min(unique_obstimes) + 1, self.im_stack.num_times)
        random_offsets = np.random.choice(
            np.arange(0, max_offset),
            len(unique_obstimes),  # Generate an offset for each unique obstime
            replace=False,  # Sample without to avoid changing uniqueness
        )

        # Map each unique obstime to a given offset
        new_obstimes_map = {}
        for i, obstime in enumerate(unique_obstimes):
            new_obstimes_map[obstime] = obstime + random_offsets[i]

        # Apply the mapping of offsets to obstimes for all timestamps in the workunit.
        new_obstimes = [new_obstimes_map[obstime] for obstime in self.get_all_obstimes()]
        self.im_stack.times = np.asanyarray(new_obstimes)

        # Sort our image stack by our updated obstimes. This WorkUnit may have already
        # been sorted so we do this to preserve that expectation after reordering.
        sorted_indices = self.im_stack.sort_by_time()

        # Sort our metadata in the same way to match the new image order.
        self.org_img_meta = self.org_img_meta[sorted_indices]
        self._per_image_indices = [self._per_image_indices[i] for i in sorted_indices]

        # Reset the WorkUnit's cached obstimes.
        self._obstimes = None

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

        im_stack = ImageStackPy()
        with fits.open(filename) as hdul:
            # Read the configuration from the HDU.
            if "kbmod_config" not in hdul:
                logger.debug("No kbmod_config found in WorkUnit, creating empty configuration.")
                config = SearchConfiguration()
            else:
                logger.debug("Reading kbmod_config from HDU.")
                config = SearchConfiguration.from_hdu(hdul["kbmod_config"])

            # Read the size and order information from the primary header.
            num_images = hdul[0].header["NUMIMG"]
            n_constituents = hdul[0].header["NCON"] if "NCON" in hdul[0].header else num_images
            logger.info(f"Loading {num_images} images (with {n_constituents} constituents).")

            # Read in the per-image metadata for the constituent images.
            if "IMG_META" in hdul:
                logger.debug("Reading original image metadata from IMG_META.")
                hdu_meta = hdu_to_image_metadata_table(hdul["IMG_META"])
            else:
                logger.debug("No IMG_META found, creating empty metadata table.")
                hdu_meta = None
            org_image_meta = create_image_metadata(n_constituents, data=hdu_meta)

            # Read in the global WCS from extension 0 if the information exists.
            # We filter the warning that the image dimension does not match the WCS dimension
            # since the primary header does not have an image.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                global_wcs = extract_wcs_from_hdu_header(hdul[0].header)

            # Misc. reprojection metadata
            reprojected = hdul[0].header["REPRJCTD"]
            if "BARY" in hdul[0].header:
                barycentric_distance = hdul[0].header["BARY"]
            else:
                # No reprojection
                barycentric_distance = None

            # ensure backwards compatibility
            if "REPFRAME" in hdul[0].header.keys():
                reprojection_frame = hdul[0].header["REPFRAME"]
            else:
                reprojection_frame = None

            # Read observatory location if available
            if "OBS_LAT" in hdul[0].header:
                observatory = EarthLocation(
                    lat=hdul[0].header["OBS_LAT"] * u.deg,
                    lon=hdul[0].header["OBS_LON"] * u.deg,
                    height=hdul[0].header["OBS_ELEV"] * u.m,
                )
            else:
                observatory = None  # Will default to Rubin in __init__

            # Read in all the image files.
            per_image_indices = []
            for i in tqdm(
                range(num_images),
                bar_format=_DEFAULT_WORKUNIT_TQDM_BAR,
                desc="Loading images",
                disable=not show_progress,
            ):
                logger.debug(f"Loading image {i} from WorkUnit file.")
                sci_hdu = hdul[f"SCI_{i}"]

                # Read in the layered image from different extensions.
                sci, var, mask, obstime, psf_kernel, _ = read_image_data_from_hdul(hdul, i)
                im_stack.append_image(obstime, sci, var, mask=mask, psf=psf_kernel)

                # Read the mapping of current image to constituent image from the header info.
                # TODO: Serialize this into its own table.
                n_indices = sci_hdu.header["NIND"]
                sub_indices = []
                for j in range(n_indices):
                    sub_indices.append(sci_hdu.header[f"IND_{j}"])
                per_image_indices.append(sub_indices)

        result = WorkUnit(
            im_stack=im_stack,
            config=config,
            wcs=global_wcs,
            barycentric_distance=barycentric_distance,
            reprojected=reprojected,
            reprojection_frame=reprojection_frame,
            per_image_indices=per_image_indices,
            org_image_meta=org_image_meta,
            observatory=observatory,
        )
        return result

    def to_fits(
        self,
        filename,
        overwrite=False,
        compression_type="RICE_1",
        quantize_level=-0.01,
    ):
        """Write the WorkUnit to a single FITS file.

        Uses the following extensions:
            0 - Primary header with overall metadata
            1 or "metadata" - The data provenance metadata
            2 or "kbmod_config" - The search parameters
            3+ - Image extensions for the science layer ("SCI_i"),
                variance layer ("VAR_i"), mask layer ("MSK_i"), and
                PSF ("PSF_i") of each image.

        Note
        ----
        The function will automatically compress the fits file
        based on the filename suffix (".gz", ".zip" or ".bz2").

        Parameters
        ----------
        filename : `str`
            The file to which to write the data.
        overwrite : bool
            Indicates whether to overwrite an existing file.
        compression_type : `str`
            The compression type to use for the image layers (sci and var). Must be
            one of "NOCOMPRESS", "RICE_1", "GZIP_1", "GZIP_2", or "HCOMPRESS_1".
            Default: "RICE_1"
        quantize_level : `float`
            The level at which to quantize the floats before compression.
            See https://docs.astropy.org/en/stable/io/fits/api/images.html for details.
            Default: -0.01
        """
        logger.info(f"Writing WorkUnit with {self.im_stack.num_times} images to file {filename}")
        if Path(filename).is_file() and not overwrite:
            raise FileExistsError(f"WorkUnit file {filename} already exists.")

        # Create an HDU list with the metadata layers, including all the WCS info.
        hdul = self.metadata_to_hdul()

        # Create each image layer.
        for i in range(self.im_stack.num_times):
            obstime = self.im_stack.times[i]
            c_indices = self._per_image_indices[i]
            n_indices = len(c_indices)

            # Append all of the image data to the main hdu list. We create
            # the mask layer because we do not store it in the image stack.
            add_image_data_to_hdul(
                hdul,
                i,
                self.im_stack.sci[i],
                self.im_stack.var[i],
                self.im_stack.get_mask(i),
                obstime,
                psf_kernel=self.im_stack.psfs[i],
                wcs=self.get_wcs(i),
                compression_type=compression_type,
                quantize_level=quantize_level,
            )

            # Append the index values onto the science header.
            # TODO: Move this to its own table.
            sci_hdu = hdul[f"SCI_{i}"]
            sci_hdu.header["NIND"] = n_indices
            for j in range(n_indices):
                sci_hdu.header[f"IND_{j}"] = c_indices[j]

        hdul.writeto(filename, overwrite=overwrite)

    def to_sharded_fits(
        self,
        filename,
        directory,
        overwrite=False,
        compression_type="RICE_1",
        quantize_level=-0.01,
    ):
        """Write the WorkUnit to a multiple FITS files.

        Will create:
            - One "primary" file, containing the main WorkUnit metadata
            (see below) as well as the per_image_wcs information for
            the whole set. This will have the given filename.
            - One image fits file containing all of the image data for
            every time step in the image stack. This will have the
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

        Note
        ----
        The function will automatically compress the fits file
        based on the filename suffix (".gz", ".zip" or ".bz2").

        Parameters
        ----------
        filename : `str`
            The base filename to which to write the data.
        directory: `str`
            The directory to place all of the FITS files.
            Recommended that you have one directory per
            sharded file to avoid confusion.
        overwrite : `bool`
            Indicates whether to overwrite an existing file.
        compression_type : `str`
            The compression type to use for the image layers (sci and var). Must be
            one of "NOCOMPRESS", "RICE_1", "GZIP_1", "GZIP_2", or "HCOMPRESS_1".
            Default: "RICE_1"
        quantize_level : `float`
            The level at which to quantize the floats before compression.
            See https://docs.astropy.org/en/stable/io/fits/api/images.html for details.
            Default: -0.01
        """
        logger.info(
            f"Writing WorkUnit shards with {self.im_stack.num_times} images with main file {filename} in {directory}"
        )
        primary_file = os.path.join(directory, filename)
        if Path(primary_file).is_file() and not overwrite:
            raise FileExistsError(f"WorkUnit file {filename} already exists.")
        if self.lazy:
            raise ValueError(
                "WorkUnit was lazy loaded, must load all ImageStackPy data to output new WorkUnit."
            )

        for i in range(self.im_stack.num_times):
            obstime = self.im_stack.times[i]
            c_indices = self._per_image_indices[i]
            n_indices = len(c_indices)
            sub_hdul = fits.HDUList()

            # Append all of the image data to the sub_hdul. We create
            # the mask layer because we do not store it in the image stack.
            add_image_data_to_hdul(
                sub_hdul,
                i,
                self.im_stack.sci[i],
                self.im_stack.var[i],
                self.im_stack.get_mask(i),
                obstime,
                psf_kernel=self.im_stack.psfs[i],
                wcs=self.get_wcs(i),
                compression_type=compression_type,
                quantize_level=quantize_level,
            )

            # Append the index values onto the science header.
            # TODO: Move this to its own table.
            sci_hdu = sub_hdul[f"SCI_{i}"]
            sci_hdu.header["NIND"] = n_indices
            for j in range(n_indices):
                sci_hdu.header[f"IND_{j}"] = c_indices[j]

            sub_hdul.writeto(os.path.join(directory, f"{i}_{filename}"), overwrite=overwrite)

        # Create a primary file with all of the metadata, including all the WCS info.
        hdul = self.metadata_to_hdul()
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

        im_stack = ImageStackPy()

        # open the main header
        with fits.open(os.path.join(directory, filename)) as primary:
            config = SearchConfiguration.from_hdu(primary["kbmod_config"])

            # Read the size and order information from the primary header.
            num_images = primary[0].header["NUMIMG"]
            n_constituents = primary[0].header["NCON"] if "NCON" in primary[0].header else num_images
            logger.info(f"Loading {num_images} images.")

            # Read in the per-image metadata for the constituent images.
            if "IMG_META" in primary:
                logger.debug("Reading original image metadata from IMG_META.")
                hdu_meta = hdu_to_image_metadata_table(primary["IMG_META"])
            else:
                hdu_meta = None
            org_image_meta = create_image_metadata(n_constituents, data=hdu_meta)

            # Read in the global WCS from extension 0 if the information exists.
            # We filter the warning that the image dimension does not match the WCS dimension
            # since the primary header does not have an image.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                global_wcs = extract_wcs_from_hdu_header(primary[0].header)

            # Misc. reprojection metadata
            reprojected = primary[0].header["REPRJCTD"]
            if "BARY" in primary[0].header:
                barycentric_distance = primary[0].header["BARY"]
            else:
                # No reprojection
                barycentric_distance = None

            # ensure backwards compatibility
            if "REPFRAME" in primary[0].header.keys():
                reprojection_frame = primary[0].header["REPFRAME"]
            else:
                reprojection_frame = None

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
                    im_stack.append_layered_image(img)
                else:
                    file_paths.append(shard_path)

                # Load the mapping of current image to constituent image.
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
            reprojected=reprojected,
            reprojection_frame=reprojection_frame,
            lazy=lazy,
            barycentric_distance=barycentric_distance,
            per_image_indices=per_image_indices,
            file_paths=file_paths,
            obstimes=obstimes,
            org_image_meta=org_image_meta,
        )
        return result

    def metadata_to_hdul(self):
        """Creates the metadata fits headers.

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
        pri.header["NCON"] = self.n_constituents
        pri.header["REPRJCTD"] = self.reprojected
        pri.header["REPFRAME"] = self.reprojection_frame
        pri.header["BARY"] = self.barycentric_distance

        # Serialize observatory location
        if self.observatory is not None:
            pri.header["OBS_LAT"] = self.observatory.lat.deg
            pri.header["OBS_LON"] = self.observatory.lon.deg
            pri.header["OBS_ELEV"] = self.observatory.height.to(u.m).value

        # If the global WCS exists, append the corresponding keys to the primary header.
        if self.wcs is not None:
            append_wcs_to_hdu_header(self.wcs, pri.header)
        hdul.append(pri)

        # Add the configuration layer.
        config_hdu = self.config.to_hdu()
        config_hdu.name = "kbmod_config"
        hdul.append(config_hdu)

        # Save the additional metadata table into HDUs
        hdul.append(image_metadata_table_to_hdu(self.org_img_meta, "IMG_META"))

        return hdul

    def image_positions_to_original_icrs(
        self, image_indices, positions, input_format="xy", output_format="xy", filter_in_frame=True
    ):
        """Method to transform image positions in EBD reprojected images
        into coordinates in the orignal ICRS frame of reference.

        Parameters
        ----------
        image_indices : `numpy.array`
            The `ImageStackPy` indices to transform coordinates.
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
        # input value validation
        if not self.reprojected:
            raise ValueError("`WorkUnit` not reprojected. This method is purpose built \
                for handling post reproject coordinate tranformations.")

        original_wcses = [w for w in self.org_img_meta["per_image_wcs"]]

        return image_positions_to_original_icrs(
            image_indices,
            positions,
            self.wcs,
            original_wcses,
            self.get_all_obstimes(),
            self.observatory,
            input_format=input_format,
            output_format=output_format,
            filter_in_frame=filter_in_frame,
            reprojection_frame=self.reprojection_frame,
            barycentric_distance=self.barycentric_distance,
            geocentric_distances=self.org_img_meta["geocentric_distance"],
            per_image_indices=self._per_image_indices,
            image_locations=self.org_img_meta["data_loc"],
        )

    def load_images(self):
        """Function for loading in `ImageStackPy` data when `WorkUnit`
        was created lazily.
        """
        if not self.lazy:
            raise ValueError("ImageStackPy has already been loaded.")
        im_stack = ImageStackPy()

        for file_path in self.file_paths:
            img = load_layered_image_from_shard(file_path)
            im_stack.append_layered_image(img)

        self.im_stack = im_stack
        self.lazy = False

    def write_config(self, overwrite=False):
        """Create the provenance directory and writes the `SearchConfiguration` out to disk."""
        result_filename = Path(self.config["result_filename"])
        if not os.path.isabs(result_filename):
            raise ValueError("result_filename must be absolute to use `write_config`")

        result_dir = result_filename.parent.absolute()
        base_filename = os.path.basename(result_filename).split(".ecsv")[0]
        provenance_dir = f"{base_filename}_provenance"
        provenance_dir_path = result_dir.joinpath(provenance_dir)

        if not os.path.exists(provenance_dir_path) or overwrite:
            os.makedirs(provenance_dir_path)
        else:
            raise ValueError(f"{provenance_dir} directory already exists")

        config_filename = f"{base_filename}_config.yaml"
        config_path = provenance_dir_path.joinpath(config_filename)
        self.config.to_file(config_path, overwrite)


def load_layered_image_from_shard(file_path):
    """Function for loading a `LayeredImagePy` from a `WorkUnit` shard.

    Parameters
    ----------
    file_path : `str`
        The location of the shard file.

    Returns
    -------
    img : `LayeredImagePy`
        The materialized `LayeredImagePy`.
    """
    if not Path(file_path).is_file():
        raise ValueError(f"provided file_path '{file_path}' is not an existing file.")

    index = int(file_path.split("/")[-1].split("_")[0])
    with fits.open(file_path) as hdul:
        sci, var, mask, obstime, psf_kernel, _ = read_image_data_from_hdul(hdul, index)
        img = LayeredImagePy(sci, var, mask, time=obstime, psf=psf_kernel)
        return img


# ------------------------------------------------------------------
# --- Utility functions for saving/loading image data --------------
# ------------------------------------------------------------------


def add_image_data_to_hdul(
    hdul,
    idx,
    sci,
    var,
    mask,
    obstime,
    psf_kernel=None,
    wcs=None,
    compression_type="RICE_1",
    quantize_level=-0.01,
):
    """Add the image data for a single time step to a fits file's HDUL as individual
    layers for science, variance, etc.  Masked pixels in the science and variance
    layers are added to the masked bits.

    Parameters
    ----------
    hdul : HDUList
        The HDUList for the fits file.
    idx : `int`
        The time step number (index of the layer).
    sci : `np.ndarray`
        The pixels of the science image.
    var : `np.ndarray`
        The pixels of the variance image.
    mask : `np.ndarray`
        The pixels of the mask image.
    obstime : `float`
        The observation time of the image in UTC MJD.
    psf_kernel : `np.ndarray`, optional
        The kernel values of the PSF.
    wcs : `astropy.wcs.WCS`, optional
        An optional WCS to include in the header.
    compression_type : `str`
        The compression type to use for the image layers (sci and var). Must be
        one of "NOCOMPRESS", "RICE_1", "GZIP_1", "GZIP_2", or "HCOMPRESS_1".
        Default: "RICE_1"
    quantize_level : `float`
        The level at which to quantize the floats before compression.
        See https://docs.astropy.org/en/stable/io/fits/api/images.html for details.
        Default: -0.01
    """
    # Use a high quantize_level to preserve most of the image information.
    # A value of -0.01 indicates that we have at least 0.01 difference between
    # quantized values.
    sci_hdu = fits.CompImageHDU(
        sci,
        compression_type=compression_type,
        quantize_level=quantize_level,
    )
    sci_hdu.name = f"SCI_{idx}"
    sci_hdu.header["MJD"] = obstime

    var_hdu = fits.CompImageHDU(
        var,
        compression_type=compression_type,
        quantize_level=quantize_level,
    )
    var_hdu.name = f"VAR_{idx}"
    var_hdu.header["MJD"] = obstime

    # The saved mask is a binarized version of which pixels are valid.
    mask_full = (mask > 0) | (~np.isfinite(sci)) | (~np.isfinite(var))
    mask_hdu = fits.ImageHDU(mask_full.astype(np.int8))
    mask_hdu.name = f"MSK_{idx}"
    mask_hdu.header["MJD"] = obstime

    # If a WCS is provided, copy it into the headers.
    if wcs is not None:
        append_wcs_to_hdu_header(wcs, sci_hdu.header)
        append_wcs_to_hdu_header(wcs, var_hdu.header)
        append_wcs_to_hdu_header(wcs, mask_hdu.header)

    # If the PSF is not provided, use an identity kernel.
    if psf_kernel is None:
        psf_kernel = np.array([[1.0]])
    psf_hdu = fits.hdu.ImageHDU(psf_kernel)
    psf_hdu.name = f"PSF_{idx}"

    # Append everything to the hdul
    hdul.append(sci_hdu)
    hdul.append(var_hdu)
    hdul.append(mask_hdu)
    hdul.append(psf_hdu)


def read_image_data_from_hdul(hdul, idx):
    """Read the image data for a single time step to a fits file's HDUL.
    The mask is auto-applied to the science and variance layers.

    Parameters
    ----------
    hdul : HDUList
        The HDUList for the fits file.
    idx : `int`
        The time step number (index of the layer).

    Returns
    -------
    sci : `np.ndarray`
        The pixels of the science image.
    var : `np.ndarray`
        The pixels of the variance image.
    mask : `np.ndarray`
        The pixels of the mask image.
    obstime : `float`
        The observation time of the image in UTC MJD.
    psf_kernel : `np.ndarray`
        The kernel values of the PSF.
    wcs : `astropy.wcs.WCS`
        An optional WCS to include in the header.  May be None
        if no WCS is found.
    """
    # Get the science layer and everything from it.
    sci_layer = hdul[f"SCI_{idx}"]
    sci = sci_layer.data.astype(np.single)
    obstime = sci_layer.header["MJD"]
    wcs = extract_wcs_from_hdu_header(sci_layer.header)

    # Get the variance layer.
    var = hdul[f"VAR_{idx}"].data.astype(np.single)

    # Allow the mask to be optional. Apply the mask if it is present
    # and use an empty mask if there is no mask layer.
    if f"MSK_{idx}" in hdul:
        mask = hdul[f"MSK_{idx}"].data.astype(np.float32)
        sci[mask > 0] = np.nan
        var[mask > 0] = np.nan
    else:
        mask = np.zeros_like(sci, dtype=np.float32)

    # Allow the PSF to be optional. Use an identity PSF if none is present.
    if f"PSF_{idx}" in hdul:
        psf_kernel = hdul[f"PSF_{idx}"].data.astype(np.single)
    else:
        psf_kernel = np.ones([[1.0]])

    return sci, var, mask, obstime, psf_kernel, wcs


# ------------------------------------------------------------------
# --- Utility functions for the metadata table ---------------------
# ------------------------------------------------------------------


def create_image_metadata(n_images, data=None):
    """Create an empty img_meta table, filling in default values
    for any unspecified columns.

    Parameters
    ----------
    n_images : `int`
        The number of images to include.
    data : `astropy.table.Table`
        An existing table from which to fill in some of the data.

    Returns
    -------
    img_meta : `astropy.table.Table`
        The empty table of org_img_meta.
    """
    if n_images <= 0:
        raise ValueError("Invalid metadata size: {n_images}")
    img_meta = Table()

    # Fill in the defaults.
    for colname in ["data_loc", "ebd_wcs", "geocentric_distance", "per_image_wcs"]:
        img_meta[colname] = np.full(n_images, None)

    # Fill in any values from the given table. This overwrites the defaults.
    if data is not None and len(data) > 0:
        if len(data) != n_images:
            raise ValueError(f"Metadata size mismatch. Expected {n_images}. Found {len(data)}")
        for colname in data.colnames:
            img_meta[colname] = data[colname]

    return img_meta


def image_metadata_table_to_hdu(data, layer_name=None):
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

            if data[colname].dtype != "O":
                # If this is something we know how to encode (float, int, string), just add the column.
                save_table[colname] = data[colname]
            elif np.all(col_data == None):
                # Skip completely empty columns.
                logger.debug("Skipping empty metadata column {colname}")
            elif isinstance(col_data[0], WCS):
                # Serialize WCS objects and use a custom tag so we can unserialize them.
                values = np.array([serialize_wcs(entry) for entry in data[colname]], dtype=str)
                save_table[f"_WCSSTR_{colname}"] = values
            else:
                # Try converting to a string.
                save_table[colname] = data[colname].data.astype(str)

    # Format the metadata as a single HDU
    meta_hdu = fits.BinTableHDU(save_table)
    if layer_name is not None:
        meta_hdu.name = layer_name
    return meta_hdu


def hdu_to_image_metadata_table(hdu):
    """Load a HDU layer with custom encodings for some columns (such as WCS)
    to an astropy table.

    Parameters
    ----------
    hdu : `astropy.io.fits.TableHDU`
        The HDUList for the fits file.

    Returns
    -------
    data : `astropy.table.Table`
        The table of loaded data.
    """
    data = Table(hdu.data)
    all_cols = set(data.colnames)

    # Check if there are any columns we need to decode. If so: decode them, add a new column,
    # and delete the old column.
    for colname in all_cols:
        if colname.startswith("_WCSSTR_"):
            data[colname[8:]] = np.array([deserialize_wcs(entry) for entry in data[colname]])
            data.remove_column(colname)

    return data
