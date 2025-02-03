import jax.numpy as jnp
import numpy as np
from scipy.signal import convolve2d

from kbmod.search import KB_NO_DATA


class PSF:
    """A class to represent a Point Spread Function (PSF).

    Attributes
    ----------
    kernel : np.ndarray
        A 2D numpy array representing the PSF.
    """

    def __init__(self, kernel):
        if len(kernel.shape) != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError(f"PSF kernel must be a 2D square array (shape={kernel.shape}).")

        # Compute the width and radius. Check there is a center pixel and an equal radius
        # on either side of the center (the width is odd).
        self.width = kernel.shape[0]
        if self.width % 2 == 0:
            raise ValueError(f"PSF kernel must have an odd width (width={self.width}).")
        self.radius = (self.width - 1) // 2

        # Check the kernel values themselves are valid.
        if np.any(kernel < 0):
            raise ValueError("PSF kernel values must be non-negative.")
        if not np.all(np.isfinite(kernel)):
            raise ValueError("PSF kernel values must be finite.")

        # Set the kernel.
        self.kernel = kernel
        self._normalize()

    @property
    def shape(self):
        """Returns the shape of the PSF."""
        return self.kernel.shape

    @classmethod
    def from_gaussian(cls, stddev):
        """Creates a PSF with a symmetric Gaussian kernel.

        Parameters
        ----------
        stddev : `float`
            The standard deviation of the Gaussian kernel.
        """
        if stddev < 0:
            raise ValueError("Standard deviation must be non-negative.")

        radius = int(3 * stddev)
        x = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(x, x)
        kernel = np.exp(-0.5 * (xx**2 + yy**2) / stddev**2)
        return cls(kernel)

    def _normalize(self):
        """Normalizes the PSF so that it sums to 1."""
        self.kernel /= np.sum(self.kernel)

    def make_square(self):
        """Returns the PSF corresponding to the square of the current PSF.

        Returns
        -------
        PSF
            A new PSF object with the kernel equal to the square of the current kernel.
        """
        new_kernel = self.kernel**2
        return PSF(new_kernel)

    def convolve_image(self, image):
        """Perform the 2D convolution where NO_DATA or NaN values are masked.

        Parameters
        ----------
        image : `numpy.ndarray`
            A 2D array of image data.
        psf : `numpy.ndarray`
            A 2D array with the PSF values.

        Returns
        -------
        result : `numpy.ndarray`
            A 2D array of the same shape as the image data.
        """
        if len(image.shape) != 2:
            raise ValueError("Image data must be a 2D array.")

        # Determine the location of the masked pixels in the image and set those to zero,
        # then do the convolution.
        data_mask = np.isfinite(image) & (image != KB_NO_DATA)
        safe_image = np.where(data_mask, image, 0.0)
        convolved = convolve2d(safe_image, self.kernel, mode="full")

        # The "full" convolution mode will pad the result to account for the width of
        # of the kernel, so we remove the padding and just get the points under
        # the center of the PSF.
        if self.radius > 0:
            convolved = convolved[self.radius : -self.radius, self.radius : -self.radius]

        # Reapply the mask. We use np.where so the result in an numpy array.
        result = np.where(data_mask, convolved, KB_NO_DATA)
        return result
