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

    def __init__(self, times=None, sci=None, var=None, mask=None, psfs=None):
        # If nothing is provided, create an empty data structure.
        if times is None or len(times) == 0:
            if sci is not None or var is not None:
                raise ValueError("Cannot create an ImageStackPy without times")
            self.num_times = 0
            self.times = np.array([])
            self.sci = []
            self.var = []
            self.psfs = []
            self.height = -1
            self.width = -1
            self.zeroed_times = np.array([])
            return

        self.num_times = len(times)
        self.times = np.asarray(times, dtype=float)
        self.zeroed_times = self.times - self.times[0]

        # Get and validate the image size information.
        if sci is None:
            raise ValueError("Missing science data.")
        elif len(sci) != self.num_times:
            raise ValueError(f"Expected {self.num_times} science images. Received {len(sci)}.")
        if var is None:
            raise ValueError("Missing variance data.")
        elif len(var) != self.num_times:
            raise ValueError(f"Expected {self.num_times} science images. Received {len(var)}.")
        if mask is not None and len(mask) != self.num_times:
            raise ValueError(f"Expected {self.num_times} mask images. Received {len(mask)}.")
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

        # If we have not seen any data before (empty stack) use these image dimensions.
        if self.num_times == 0:
            self.width = img.shape[1]
            self.height = img.shape[0]

        # Check that the image is the correct size.
        if img.shape[1] != self.width:
            raise ValueError(f"Incorrect image width. Expected {self.width}. Received {img.shape[1]}")
        if img.shape[0] != self.height:
            raise ValueError(f"Incorrect image height. Expected {self.height}. Received {img.shape[0]}")

        return img

    def __len__(self):
        return self.num_times

    def __eq__(self, other):
        if self.num_times != other.num_times:
            return False
        if self.height != other.height or self.width != other.width:
            return False
        if not np.allclose(self.times, other.times):
            return False
        if not np.allclose(self.zeroed_times, other.zeroed_times):
            return False

        # Check the image data at each time.
        for i in range(self.num_times):
            if not np.allclose(self.sci[i], other.sci[i]):
                return False
            if not np.allclose(self.var[i], other.var[i]):
                return False
            if not np.allclose(self.psfs[i], other.psfs[i]):
                return False
        return True

    @property
    def npixels(self):
        """Return the number of pixels in each image."""
        return self.height * self.width

    @property
    def total_pixels(self):
        """Return the total number of pixels in the stack."""
        return self.height * self.width * self.num_times

    def get_total_pixels(self):
        """Return the total number of pixels in the stack."""
        return self.height * self.width * self.num_times

    def get_obstime(self, index):
        """Retrieve the time stamp for a given image.

        Notes
        -----
        This is a temporary bridge function to make the Python code look
        more like the C++ code.

        Parameters
        ----------
        index : int
            The index of the image.

        Returns
        -------
        obstime : float
            The time stamp (in UTC MJD).
        """
        if index < 0 or index >= self.num_times:
            raise IndexError(f"Index {index} out of range for ImageStack.")
        return self.times[index]

    def copy(self):
        """Make a deep copy of the image stack."""
        new_stack = ImageStackPy(
            times=[self.times[i] for i in range(self.num_times)],
            sci=[np.copy(self.sci[i]) for i in range(self.num_times)],
            var=[np.copy(self.var[i]) for i in range(self.num_times)],
            mask=None,
            psfs=[np.copy(self.psfs[i]) for i in range(self.num_times)],
        )
        return new_stack

    def num_masked_pixels(self):
        """Compute the number of masked pixels."""
        total = 0
        for img in self.sci:
            total += np.sum(np.isnan(img))
        return total

    def get_mask(self, index):
        """Get the mask for a given time step.  Creates the mask on the fly.

        Parameters
        ----------
        index : int
            The index of the image.

        Returns
        -------
        mask : np.array
            The mask for the image.
        """
        if index < 0 or index >= self.num_times:
            raise IndexError(f"Index {index} out of range for image stack.")
        return np.isnan(self.sci[index]) | np.isnan(self.var[index])

    def append_image(self, time, sci, var, mask=None, psf=None):
        """Append an image onto the back of the stack.

        Parameters
        ----------
        time : float
            The observation time (in UTC MJD).
        sci : np.array
            A H x W array of science data.
        var : np.array
            A H x W array of variance data.
        mask : np.array, optional
            A H x W array of mask data. If not provided, nothing is masked.
        psfs : np.array or PSF
            The PSF information for this time.
        """
        current_idx = self.num_times
        self.sci.append(self._standardize_image(sci))
        self.var.append(self._standardize_image(var))
        if psf is None:
            psf = np.array([[1.0]])
        elif isinstance(psf, PSF):
            psf = psf.kernel
        self.psfs.append(psf)

        # Apply the mask if it is provided.
        if mask is not None:
            mask = np.asanyarray(mask)
            if mask.shape != (self.height, self.width):
                raise ValueError("Science and Mask data must have the same shape.")
            masked_pixels = mask > 0
            self.sci[current_idx][masked_pixels] = np.nan
            self.var[current_idx][masked_pixels] = np.nan

        # Set the time information.
        self.num_times += 1
        self.times = np.append(self.times, time)
        self.zeroed_times = self.times - self.times[0]

    def append_layered_image(self, layered_image):
        """Append a LayeredImagePy object to the stack.
        This is a wrapper for append_image.

        Parameters
        ----------
        layered_image : LayeredImagePy
            The image data to append.
        """
        self.append_image(
            layered_image.time,
            layered_image.sci,
            layered_image.var,
            mask=layered_image.mask,
            psf=layered_image.psf,
        )

    def filter_images(self, mask):
        """Remove images from the stack according to a mask.

        Parameters
        ----------
        mask : list or np.ndarray
            A Boolean array of the images to keep.
        """
        mask = np.asanyarray(mask)

        new_sci = []
        new_var = []
        new_psfs = []
        for idx in range(self.num_times):
            if mask[idx]:
                new_sci.append(self.sci[idx])
                new_var.append(self.var[idx])
                new_psfs.append(self.psfs[idx])
        self.sci = new_sci
        self.var = new_var
        self.psfs = new_psfs

        self.num_times = len(self.sci)
        self.times = self.times[mask]
        if self.num_times > 0:
            self.zeroed_times = self.times - self.times[0]
        else:
            self.zeroed_times = []

    def get_masked_fractions(self):
        """Compute the fraction of masked pixels for each image.

        Returns
        -------
        masked_fraction : np.ndarray
            An array of the fraction of pixels that are masked.
        """
        masked_fraction = np.zeros(self.num_times)
        total_pixels = float(self.width * self.height)

        # Iterate through the list checking each image.
        for idx in range(self.num_times):
            is_masked = np.isnan(self.sci[idx]) | np.isnan(self.var[idx])
            masked_fraction[idx] = np.count_nonzero(is_masked) / total_pixels
        return masked_fraction

    def mask_by_science_bounds(self, min_val=-1e20, max_val=1e20):
        """Mask pixels whose value in the science layer lies outside the given bounds.
        Applies mask to both science and variance layer.

        Parameter
        ---------
        min_val : float
            The minimum acceptable flux. Default: -1e20
        max_val : float
            The maximum acceptable flux. Default: 1e20
        """
        for idx in range(self.num_times):
            bad_values = (self.sci[idx] < min_val) | (self.sci[idx] > max_val)
            self.sci[idx][bad_values] = np.nan
            self.var[idx][bad_values] = np.nan

    def mask_by_variance_bounds(self, min_val=1e-20, max_val=1e20):
        """Mask pixels whose value in the variance layer lies outside the given bounds.
        Applies mask to both science and variance layer.

        Parameter
        ---------
        min_val : float
            The minimum acceptable variance (should always be > 0.0 since negative
            variance and 0.0 variance are both invalid).
            Default: 1e-20
        max_val : float
            The maximum acceptable variance. Default: 1e20
        """
        for idx in range(self.num_times):
            bad_values = (self.var[idx] < min_val) | (self.var[idx] > max_val)
            self.sci[idx][bad_values] = np.nan
            self.var[idx][bad_values] = np.nan

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

    def sort_by_time(self):
        """Sort the images in the stack by time."""
        # Get the sorted indices.
        sorted_indices = np.argsort(self.times)

        # Sort the images.
        self.sci = [self.sci[i] for i in sorted_indices]
        self.var = [self.var[i] for i in sorted_indices]
        self.psfs = [self.psfs[i] for i in sorted_indices]
        self.times = self.times[sorted_indices]
        self.zeroed_times = self.times - self.times[0]

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

    def print_stats(self):
        """Compute the basic statistics of an image stack and display in a table."""
        total_pixels = self.height * self.width
        num_times = self.num_times

        print("Image Stack Statistics:")
        print(f"  Image Count: {num_times}")
        print(f"  Image Size: {self.height} x {self.width} = {total_pixels}")

        sep_line = (
            "+------+------------+------------+------------+------------+----------+"
            "----------+----------+--------+"
        )
        print(sep_line)
        print(
            "|  idx |     Time   |  Flux Min  |  Flux Max  |  Flux Mean |  Var Min |"
            "  Var Max | Var Mean | Masked |"
        )
        print(sep_line)

        for idx in range(self.num_times):
            time = self.times[idx]

            # Count the masked pixels.
            is_masked = np.isnan(self.sci[idx]) | np.isnan(self.var[idx]) | (self.var[idx] <= 0.0)
            percent_masked = (np.count_nonzero(is_masked) / total_pixels) * 100.0

            # Compute the basic statistics.
            flux_min = np.nanmin(self.sci[idx])
            flux_max = np.nanmax(self.sci[idx])
            flux_mean = np.nanmean(self.sci[idx])
            var_min = np.nanmin(self.var[idx])
            var_max = np.nanmax(self.var[idx])
            var_mean = np.nanmean(self.var[idx])

            print(
                f"| {idx:4d} | {time:10.3f} | {flux_min:10.2f} | {flux_max:10.2f} | {flux_mean:10.2f} "
                f"| {var_min:8.2f} | {var_max:8.2f} | {var_mean:8.2f} | {percent_masked:6.2f} |"
            )
            print(sep_line)


def make_fake_image_stack(height, width, times, noise_level=2.0, psf_val=0.5, psfs=None, rng=None):
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
        The value of the default PSF.  Used if individual psfs are not specified.
        Default: 0.5
    psfs : `list` of `numpy.ndarray`, optional
        A list of PSF kernels. If none, Gaussian PSFs from with std=psf_val are used.
    rng : np.random.Generator
        The random number generator to use. If None creates a new random generator.
        Default: None
    """
    if rng is None:
        rng = np.random.default_rng()
    times = np.asarray(times)

    # Create the science and variance images.
    sci = [rng.normal(0.0, noise_level, (height, width)).astype(np.float32) for i in range(len(times))]
    var = [np.full((height, width), noise_level**2).astype(np.float32) for i in range(len(times))]

    # Create the PSF information.
    if psfs is None:
        psf_kernel = PSF.make_gaussian_kernel(psf_val)
        psfs = [psf_kernel for i in range(len(times))]
    elif len(psfs) != len(times):
        raise ValueError(f"The number of PSFs ({len(psfs)}) must be the same as times ({len(times)}).")

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
