"""A data structure for storing image data for multiple times along with
helper functions to operate on these stacks of images.

Note: This is a numpy-based implementation of KBMOD's ImageStack.
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

from kbmod.core.psf import PSF


class ImageStackPy:
    """A class for storing science and variance image data along
    with corresponding metadata.

    Notes
    -----
    The images are not required to be in sorted time order, but the first
    image is used for t=0.0 when computing zeroed times (which might make
    some times negative).

    Attributes
    ----------
    times : np.array
        The length T array of time stamps (in UTC MJD).
    sci : np.array
        The T x H x W array of science data.
    var : np.array
        The T x H x W array of variance data.
    mask : np.array
        The T x H x W array of Boolean mask data.
    psfs : list of numpy arrays
        The length T array of PSF information.  This is a list instead of a
        numpy array because the PSFs can be different sizes. If a list of PSF
        objects is provided, only the kernels are stored.
    wcs : list of astropy.wcs.WCS
        The length T array of WCS information.
    num_times : int
        The number of times in the stack.
    zeroed_times : np.array
        The length T array of zeroed times.
    """

    def __init__(self, times, sci, var, mask=None, psfs=None, wcs=None):
        if times is None or len(times) == 0:
            raise ValueError("Cannot create an ImageStack with no times.")
        self.num_times = len(times)
        self.times = np.asarray(times, dtype=float)
        self.zeroed_times = self.times - self.times[0]

        self.sci = np.asarray(sci, dtype=float)
        self.var = np.asarray(var, dtype=float)
        if len(self.sci.shape) != 3:
            raise ValueError("3d (T x H x W) numpy array of science data required to build stack.")
        if self.sci.shape[0] != self.num_times:
            raise ValueError(f"Science data must have {self.num_times} images.")
        if self.sci.shape != self.var.shape:
            raise ValueError("Science and variance data must have the same shape.")

        # If a mask is given, apply it now. Save it as a boolean array.
        if mask is not None:
            if mask.shape != sci.shape:
                raise ValueError("Science and Mask data must have the same shape.")
            mask = np.asarray(mask) > 0
            sci[mask] = np.nan
            var[mask] = np.nan
            self.mask = np.asarray(mask > 0, dtype=bool)
        else:
            self.mask = np.full_like(self.sci, False, dtype=bool)

        # Checks (and creates defaults) for the PSF input.
        if psfs is None:
            self.psfs = [np.ones((1, 1)) for i in range(self.num_times)]
        elif len(psfs) != self.num_times:
            raise ValueError(f"PSF data must have {self.num_times} entries.")
        else:
            self.psfs = psfs

            # Only save the kernels (not the PSF objects).
            for i, psf in enumerate(self.psfs):
                if isinstance(psf, PSF):
                    self.psfs[i] = psf.kernel
                elif isinstance(psf, np.ndarray):
                    self.psfs[i] = psf
                else:
                    raise ValueError("PSF data must be a PSF object or a numpy array.")

        # Preprocess the WCS data.
        if wcs is None:
            self.wcs = [None for i in range(self.num_times)]
        elif len(wcs) != self.num_times:
            raise ValueError(f"WCS data must have {self.num_times} entries.")
        else:
            self.wcs = wcs

    def __len__(self):
        return self.num_times

    @property
    def height(self):
        """Return each image's height."""
        return self.sci.shape[1]

    @property
    def width(self):
        """Return each image's width."""
        return self.sci.shape[2]

    @property
    def num_images(self):
        """Return the number of images."""
        return self.num_times

    @property
    def npixels(self):
        """Return the number of pixels in each image."""
        return self.sci.shape[1] * self.sci.shape[2]

    @property
    def total_pixels(self):
        """Return the total number of pixels in the stack."""
        return self.sci.shape[0] * self.sci.shape[1] * self.sci.shape[2]

    @classmethod
    def make_empty(cls, num_times, height, width):
        """Create an empty ImageStack with the given times, height, and width.

        Parameters
        ----------
        num_times : int
            The number of times in the stack.
        height : int
            The height of the images in pixels.
        width : int
            The width of the images in pixels.

        Returns
        -------
        ImageStackPy
            An empty ImageStack with the given parameters.
        """
        return cls(
            np.zeros(num_times),
            np.zeros((num_times, height, width), dtype=np.single),
            np.zeros((num_times, height, width), dtype=np.single),
        )

    def num_masked_pixels(self):
        """Compute the number of masked pixels."""
        return np.sum(np.isnan(self.sci))

    def set_images_at_time(self, time_idx, sci, var, mask=None):
        """Set the images at a specific time index.

        Parameters
        ----------
        time_idx : int
            The time index to set.
        sci : np.array
            The science image data (H x W).
        var : np.array
            The variance image data (H x W).
        mask : np.array, optional
            The mask image data (H x W). If None, no mask is applied.
        """
        if time_idx < 0 or time_idx >= self.num_times:
            raise ValueError("Time index out of bounds.")
        if sci.shape != self.sci[time_idx].shape:
            raise ValueError("Science image must have the same shape as the stack.")
        if var.shape != self.var[time_idx].shape:
            raise ValueError("Variance image must have the same shape as the stack.")

        # Set the science and variance images.
        self.sci[time_idx] = np.asarray(sci, dtype=float)
        self.var[time_idx] = np.asarray(var, dtype=float)

        # Apply the mask if provided.
        if mask is not None:
            if mask.shape != sci.shape:
                raise ValueError("Science and Mask data must have the same shape.")
            mask = np.asarray(mask) > 0
            self.sci[time_idx, mask] = np.nan
            self.var[time_idx, mask] = np.nan
            self.mask[time_idx] = np.asarray(mask > 0, dtype=bool)
        else:
            self.mask[time_idx, :, :] = False

    def get_matched_obstimes(self, query_times, threshold=0.0007):
        """Given a list of times, returns the indices of images that are close
        enough to the query times.

        Parameters
        ----------
        query_times : list-like
            The query times (in MJD).
        threshold : float
            The match threshold (in days)
            Default: 0.0007 = ~1 minute

        Returns
        -------
        match_indices : np.array
            The matching index for each obs time. Set to -1 if there is no
            obstime within the given threshold.
        """
        # Create a version of the data times bounded by -inf and inf.
        all_times = np.insert(self.times, [0, self.num_times], [-np.inf, np.inf])

        # Find each query time's insertion point in the sorted array.  Because we inserted
        # -inf and inf we have 0 < sorted_inds <= len(all_times).
        sorted_inds = np.searchsorted(all_times, query_times, side="left")
        right_dist = np.abs(all_times[sorted_inds] - query_times)
        left_dist = np.abs(all_times[sorted_inds - 1] - query_times)

        min_dist = np.where(left_dist > right_dist, right_dist, left_dist)
        min_inds = np.where(left_dist > right_dist, sorted_inds, sorted_inds - 1)

        # Filter out matches that exceed the threshold.
        # Shift back to account for the -inf inserted at the start.
        min_inds = np.where(min_dist <= threshold, min_inds - 1, -1)

        return min_inds

    def world_to_pixel(self, time_idx, ra, dec):
        """Get the pixel coordinates for a given time index and RA/Dec.

        Parameters
        ----------
        time_idx : int
            The time index to use.
        ra : `float`
            The right ascension coordinates (in degrees).
        dec : `float`
            The declination coordinates (in degrees).

        Returns
        -------
        x_pos : `float`
            The X pixel position.
        y_pos : `float`
            The Y pixel position.
        """
        if self.wcs[time_idx] is None:
            raise ValueError("No WCS information for this time index.")
        x_pos, y_pos = self.wcs[time_idx].world_to_pixel(SkyCoord(ra=ra * u.degree, dec=dec * u.degree))
        return x_pos, y_pos

    def pixel_to_world(self, time_idx, x_pos, y_pos):
        """Get the pixel coordinates for a given time index and RA/Dec.

        Parameters
        ----------
        time_idx : int
            The time index to use.
        x_pos : `float`
            The X pixel position.
        y_pos : `float`
            The Y pixel position.

        Returns
        -------
        ra : `float`
            The right ascension coordinates (in degrees).
        dec : `float`
            The declination coordinates (in degrees).
        """
        if self.wcs[time_idx] is None:
            raise ValueError("No WCS information for this time index.")
        coord = self.wcs[time_idx].pixel_to_world(x_pos, y_pos)
        return coord.ra.deg, coord.dec.deg


def make_fake_image_stack(width, height, times, noise_level=2.0, psf_val=0.5, rng=None):
    """Create a fake ImageStack for testing.

    Parameters
    ----------
    width : int
        The width of the images in pixels.
    height : int
        The height of the images in pixels.
    times : list
        A list of time stamps.
    noise_level : float
        The level of the background noise.
        Default: 2.0
    psf_val : float
        The value of the default PSF.
        Default: 0.5
    rng : np.random.Generator
        The random number generator to use. If None creates a new random generator.
        Default: None
    """
    if rng is None:
        rng = np.random.default_rng()
    times = np.asarray(times)

    # Create the science and variance images.
    sci = rng.normal(0.0, noise_level, (len(times), height, width)).astype(np.float32)
    var = np.full((len(times), height, width), noise_level**2).astype(np.float32)

    # Create the PSF information.
    psf_kernel = PSF.make_gaussian_kernel(psf_val)
    psfs = [psf_kernel for i in range(len(times))]

    return ImageStackPy(times, sci, var, psfs=psfs)


def image_stack_add_fake_object(stack, x, y, vx, vy, flux):
    """Insert a fake object given the trajectory.

    Parameters
    ----------
    stack : ImageStackPy
        The image stack to modify.
    x : int
        The x-coordinate of the object at the first time (in pixels).
    y : int
        The y-coordinate of the object at the first time (in pixels).
    vx : float
        The x-velocity of the object (in pixels per day).
    vy : float
        The y-velocity of the object (in pixels per day).
    flux : float
        The flux of the object.
    """
    for idx, t in enumerate(stack.zeroed_times):
        psf_kernel = stack.psfs[idx]
        psf_dim = psf_kernel.shape[0]
        psf_radius = psf_dim // 2

        px = int(x + vx * t + 0.5)
        py = int(y + vy * t + 0.5)
        for psf_y in range(psf_dim):
            for psf_x in range(psf_dim):
                img_x = px + psf_x - psf_radius
                img_y = py + psf_y - psf_radius
                if img_x >= 0 and img_x < stack.width and img_y >= 0 and img_y < stack.height:
                    stack.sci[idx, img_y, img_x] += flux * psf_kernel[psf_y, psf_x]
