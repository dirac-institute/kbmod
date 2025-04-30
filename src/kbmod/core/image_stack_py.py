"""A data structure for storing image data for multiple times along with
helper functions to operate on these stacks of images.

Note: This is a numpy-based implementation of KBMOD's ImageStack.
"""

import logging
import numpy as np

from kbmod.core.psf import PSF


class LayeredImagePy:
    """A data class for storing all of the image components for a single
    time step.  This is primarily used to ease the transition between
    the numpy-based ImageStackPy and the C++ ImageStack.

    Attributes
    ----------
    sci : np.array
        The H x W array of science data.
    var : np.array
        The H x W array of variance data.
    time : float, optional
        The time stamp (in UTC MJD).
    mask : np.array, optional
        The H x W array of mask data.
    psf : np.array, optional
        The kernel of the PSF function.
    """

    def __init__(self, sci, var, mask=None, time=0.0, psf=None):
        self.time = time
        self.sci = np.asanyarray(sci, dtype=np.float32)
        self.var = np.asanyarray(var, dtype=np.float32)

        if psf is None:
            self.psf = np.ones((1, 1), dtype=np.float32)
        else:
            self.psf = np.asanyarray(psf, dtype=np.float32)

        if mask is None:
            self.mask = np.zeros_like(sci, dtype=np.float32)
        else:
            self.mask = mask

    @property
    def width(self):
        """The width of the image."""
        return self.sci.shape[1]

    @property
    def height(self):
        """The height of the image."""
        return self.sci.shape[0]


class ImageStackPy:
    """A class for storing science and variance image data along
    with corresponding metadata.

    Notes
    -----
    The images in the stack must all be the same shape (height and width).
    They are not required to be in sorted time order, but the first image
    is used for t=0.0 when computing zeroed times (which might make some
    times negative).

    Attributes
    ----------
    times : np.array
        The length T array of time stamps (in UTC MJD).
    sci : list of np.array
        A length T list of H x W arrays of science data.
    var : list of np.array
        A length T list of H x W arrays of variance data.
    psfs : list of numpy arrays
        The length T array of PSF information.  This is a list instead of a
        numpy array because the PSFs can be different sizes. If a list of PSF
        objects is provided, only the kernels are stored.
    num_times : int
        The number of times in the stack.
    height : int
        The height of each image in pixels.
    width : int
        The width of each image in pixels.
    zeroed_times : np.array
        The length T array of zeroed times.
    """

    def __init__(self, times, sci, var, mask=None, psfs=None):
        if times is None or len(times) == 0:
            raise ValueError("Cannot create an ImageStack with no times.")
        self.num_times = len(times)
        self.times = np.asarray(times, dtype=float)
        self.zeroed_times = self.times - self.times[0]

        # Get and validate the image size information.
        if len(sci) != self.num_times:
            raise ValueError(
                f"Incorrect number of science images. Expected {self.num_times}. Received {len(sci)}."
            )
        if len(var) != self.num_times:
            raise ValueError(
                f"Incorrect number of variance images. Expected {self.num_times}. Received {len(var)}."
            )
        if mask is not None and len(mask) != self.num_times:
            raise ValueError(
                f"Incorrect number of mask images. Expected {self.num_times}. Received {len(mask)}."
            )
        self.height = len(sci[0])
        self.width = len(sci[0][0])

        # Validate and save each of the science images.
        self.sci = []
        for img in sci:
            self.sci.append(self._standardize_image(img))

        # Validate and save each of the variance images.
        self.var = []
        for img in var:
            self.var.append(self._standardize_image(img))

        # If a mask is given, apply it now.
        if mask is not None:
            for idx in range(self.num_times):
                current_mask = np.asanyarray(mask[idx])
                if current_mask.shape != (self.height, self.width):
                    raise ValueError("Science and Mask data must have the same shape.")
                masked_pixels = current_mask > 0
                self.sci[idx][masked_pixels] = np.nan
                self.var[idx][masked_pixels] = np.nan

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

    def _standardize_image(self, img):
        """Validate that an image is in the form that is expected,
        transforming it if not.

        Parameters
        ----------
        img : np.ndarray
            The incoming image.

        Returns
        -------
        img : np.ndarray
            The standardized image.
        """
        # All images should be a numpy array in single precision floating point.
        img = np.asanyarray(img)
        if img.dtype != np.single:
            img = img.astype(np.single)

        # Check that the image is the correct size.
        if img.shape[1] != self.width:
            raise ValueError(f"Incorrect image width. Expected {self.width}. Received {img.shape[1]}")
        if img.shape[0] != self.height:
            raise ValueError(f"Incorrect image height. Expected {self.height}. Received {img.shape[0]}")

        return img

    def __len__(self):
        return self.num_times

    @property
    def npixels(self):
        """Return the number of pixels in each image."""
        return self.height * self.width

    @property
    def total_pixels(self):
        """Return the total number of pixels in the stack."""
        return self.height * self.width * self.num_times

    def num_masked_pixels(self):
        """Compute the number of masked pixels."""
        total = 0
        for img in self.sci:
            total += np.sum(np.isnan(img))
        return total

    def get_single_image(self, index):
        """Get a single image from the stack.

        Parameters
        ----------
        index : int
            The index of the image to get.

        Returns
        -------
        LayeredImagePy
            The image data at the given index.
        """
        if index < 0 or index >= self.num_times:
            raise IndexError(f"Index {index} out of range for ImageStack.")
        return LayeredImagePy(self.sci[index], self.var[index], time=self.times[index], psf=self.psfs[index])

    def set_single_image(self, index, img):
        """Set a single image in the stack.

        Parameters
        ----------
        index : int
            The index of the image to set.
        img : LayeredImagePy
            The image data to set.
        """
        if index < 0 or index >= self.num_times:
            raise IndexError(f"Index {index} out of range for ImageStack.")
        if img.width != self.width or img.height != self.height:
            raise ValueError(
                f"Image shape does not match the ImageStack size. Expected ({self.width},{self.height}). "
                f"Received ({img.width}, {img.height})."
            )

        new_sci = self._standardize_image(img.sci)
        new_var = self._standardize_image(img.var)

        # Do any masking needed.
        masked_pixels = img.mask > 0
        if np.any(masked_pixels):
            new_sci[masked_pixels] = np.nan
            new_var[masked_pixels] = np.nan

        # Set the image data.
        self.sci[index] = new_sci
        self.var[index] = new_var
        self.psfs[index] = img.psf
        self.times[index] = img.time
        self.zeroed_times[index] = img.time - self.times[0]

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

    def validate(
        self,
        masked_fraction=0.5,
        min_flux=-1e8,
        max_flux=1e8,
        min_var=1e-20,
        max_var=1e8,
    ):
        """Run basic validation checks on an image stack. If any of the checks fail,
        the code will log a warning and return False.

        Parameters
        ----------
        masked_fraction: `float`
            The maximum fraction of masked pixels allowed.
            Default: 0.5
        min_flux : `float`
            The minimum flux value allowed.
            Default: -1e8
        max_flux : `float`
            The maximum flux value allowed.
            Default: 1e8
        min_var : `float`
            The minimum variance value allowed.
            Default: -1e8
        max_var : `float`
            The maximum variance value allowed.
            Default: 1e-20 (no zero or negative variance)
        """
        logger = logging.getLogger(__name__)

        is_valid = True

        if self.total_pixels == 0 or self.num_times == 0:
            logger.warning("Image stack is empty.")
            return False

        for idx in range(self.num_times):
            sci = self.sci[idx]
            var = self.var[idx]

            # Count masked pixels.
            is_masked = np.isnan(sci) | np.isnan(var)
            percent_masked = np.count_nonzero(is_masked) / (self.height * self.width)
            if percent_masked > masked_fraction:
                logger.warning(f"Image {idx} has {percent_masked * 100.0} percent masked pixels.")
                is_valid = False

            # Check for valid flux and variance values.  We only do this is the layer has at least
            # one unmasked value.
            if percent_masked < 1.0:
                if np.nanmin(sci) < min_flux:
                    logger.warning(f"Image {idx} has invalid flux values: {np.nanmin(sci)} < {min_flux}")
                    is_valid = False
                if np.nanmax(sci) > max_flux:
                    logger.warning(f"Image {idx} has invalid flux values: {np.nanmax(sci)} > {max_flux}")
                    is_valid = False
                if np.nanmin(var) < min_var:
                    logger.warning(f"Image {idx} has invalid flux values: {np.nanmin(var)} < {min_var}")
                    is_valid = False
                if np.nanmax(var) > max_var:
                    logger.warning(f"Image {idx} has invalid flux values: {np.nanmax(var)} > {max_var}")
                    is_valid = False

        return is_valid


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
                    stack.sci[idx][img_y, img_x] += flux * psf_kernel[psf_y, psf_x]
