import numpy as np
import torch


class PSF:
    """A class to represent a Point Spread Function (PSF).

    Attributes
    ----------
    kernel : np.ndarray
        A 2D numpy array representing the PSF.
    width : int
        The width of the PSF kernel.
    radius : int
        The radius of the PSF kernel.
    """

    def __init__(self, kernel):
        if np.isscalar(kernel):
            # If we are given a scalar, assume it is the standard deviation of a Gaussian
            # kernel and create the kernel.
            kernel = self.make_gaussian_kernel(kernel)
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

    @staticmethod
    def make_gaussian_kernel(stddev, normalize=True):
        """Creates a symmetric Gaussian kernel with a given standard deviation.

        Parameters
        ----------
        stddev : `float`
            The standard deviation of the Gaussian kernel.
        normalize : `bool`
            Whether to normalize the kernel so the sum of points is 1.0.

        Returns
        -------
        kernel : `np.ndarray`
            A 2D numpy array representing the Gaussian kernel.
        """
        if stddev < 0:
            raise ValueError("Standard deviation must be non-negative.")

        radius = int(3 * stddev)
        x = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(x, x)
        kernel = np.exp(-0.5 * (xx**2 + yy**2) / stddev**2)

        if normalize:
            kernel /= np.sum(kernel)
        return kernel.astype(np.float32)

    @classmethod
    def from_gaussian(cls, stddev):
        """Creates a PSF with a symmetric Gaussian kernel.

        Parameters
        ----------
        stddev : `float`
            The standard deviation of the Gaussian kernel.
        """
        kernel = cls.make_gaussian_kernel(stddev)
        return cls(kernel)

    def copy(self):
        """Returns a copy of the PSF."""
        return PSF(np.copy(self.kernel.copy()))

    def _normalize(self):
        """Normalizes the PSF so that it sums to 1."""
        self.kernel /= np.sum(self.kernel)

    def convolve_image(self, image, scale_by_masked=True, in_place=False, device=None):
        """Perform the 2D convolution where NO_DATA or NaN values are masked.

        Parameters
        ----------
        image : `numpy.ndarray`
            A 2D array of image data.
        scale_by_masked : `bool`
            The convolution is scaled to account for masked pixels so as to preserve to the
            flux in the unmasked pixels.
            Default is True.
        in_place : `bool`, optional
            If True, the convolution is performed in place, modifying the input image.
            If False, a new array is created and returned.
            Default is False.
        device : `torch.device`, optional
            The device to use for the convolution.
            If None, the default device is used.
            Default is None.

        Returns
        -------
        result : `numpy.ndarray`
            A 2D array of the same shape as the image data.
        """
        return convolve_psf_and_image(
            image,
            self.kernel,
            scale_by_masked=scale_by_masked,
            in_place=in_place,
            device=device,
        )


def convolve_psf_and_image(image, kernel, scale_by_masked=True, in_place=False, device=None):
    """Perform the 2D convolution where NO_DATA or NaN values are masked.

    Parameters
    ----------
    image : `numpy.ndarray`
        A 2D array of image data.
    kernel : `numpy.ndarray`
        A 2D array representing the PSF kernel. Must be square.
    scale_by_masked : `bool`
        The convolution is scaled to account for masked pixels so as to preserve to the
        flux in the unmasked pixels.
        Default is True.
    in_place : `bool`, optional
        If True, the convolution is performed in place, modifying the input image.
        If False, a new array is created and returned.
        Default is False.
    device : `torch.device`, optional
        The device to use for the convolution.
        If None, the default device is used.
        Default is None.

    Returns
    -------
    result : `numpy.ndarray`
        A 2D array of the same shape as the image data.
    """
    if len(image.shape) != 2:
        raise ValueError("Image data must be a 2D array.")
    if len(kernel.shape) != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("PSF kernel must be a 2D square array.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Flip the kernel so we are performing correlation.
    flipped_kernel = np.flip(kernel)

    # Convert the image and kernel to PyTorch tensors.
    image_tensor = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    kernel_total = torch.sum(kernel_tensor)

    # Determine the location of the masked pixels in the image and set those to zero.
    data_mask = torch.isfinite(image_tensor)
    image_tensor[~data_mask] = 0.0

    # Perform the convolution. Using padding="same" to effectively zero-pad the image.
    convolved_image = torch.nn.functional.conv2d(image_tensor, kernel_tensor, padding="same")

    if scale_by_masked:
        # To account for the masked pixels, we divide by the sum of kernel values that
        # landed on unmasked pixels.
        bin_tensor = torch.where(data_mask, 1.0, 0.0)
        scale = torch.clamp(
            torch.nn.functional.conv2d(bin_tensor, kernel_tensor, padding="same"),
            min=1e-24,  # Avoid divide by zero.
        )

        # Divide by the fraction of the kernel that was used.
        convolved_image = convolved_image * (kernel_total / scale)

    # Re-mask the masked points.
    convolved_image[~data_mask] = float("nan")

    if in_place:
        image[:] = convolved_image.squeeze().cpu().numpy()
        return image
    else:
        return convolved_image.squeeze().cpu().numpy()
