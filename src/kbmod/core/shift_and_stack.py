"""Functions for performing core shift and stack functionality."""

import numpy as np

from kbmod.core.psf import convolve_psf_and_image, PSF


def generate_psi_phi_images(sci, var, psf):
    """Generate the PSI and PHI images from a given science image,
    variance image, and psf.

    Parameters
    ----------
    sci : `numpy.ndarray`
        The H x W matrix of image pixels.
    var : `numpy.ndarray`
        The H x W matrix of variance pixels.
    psf : `numpy.ndarray` or PSF.
        The PSF data as a PSF object or a matrix of kernel values.

    Returns
    -------
    psi : `numpy.ndarray`
        The H x W matrix of the PSI image.
    phi : `numpy.ndarray`
        The H x W matrix of the PHI image.
    """
    psi = np.full_like(sci, np.nan)
    phi = np.full_like(sci, np.nan)
    valid_mask = ~(np.isnan(sci) | np.isnan(var) | (var <= 0.0))

    psi[valid_mask] = sci[valid_mask] / var[valid_mask]
    phi[valid_mask] = 1.0 / var[valid_mask]

    # Convolve psi with the PSF and phi with the square of the PSF.
    # Note that convolve_image correctly preserves NaNs (and uses them
    # for scaling).
    if isinstance(psf, PSF):
        psf = psf.kernel
    psi = convolve_psf_and_image(psi, psf, scale_by_masked=True)

    psf_sq = psf**2
    phi = convolve_psf_and_image(phi, psf_sq, scale_by_masked=True)

    return psi, phi


def generate_all_psi_phi_images(sci, var, psf):
    """Generate the list of PSI and PHI images from corresponding lists of
    science images, variance images, and PSFs.

    Parameters
    ----------
    sci : `list` of `numpy.ndarray`
        The list of H x W matrices of image pixels.
    var : `list` of `numpy.ndarray`
        The list of H x W matrices of variance pixels.
    psf : `list` of `numpy.ndarray`
        The list of matrices of the PSF kernel values.

    Returns
    -------
    psi : `list` of `numpy.ndarray`
        The list of H x W matrices of the PSI images.
    phi : `list` of `numpy.ndarray`
        The list of H x W matrices of the PHI images.
    """
    psi = []
    phi = []
    for sci_img, var_img, psf_img in zip(sci, var, psf):
        psi_img, phi_img = generate_psi_phi_images(sci_img, var_img, psf_img)
        psi.append(psi_img)
        phi.append(phi_img)

    return psi, phi


def generate_psi_phi_from_image_stack(stack):
    """Generate the PSI and PHI images from a given image stack.

    Parameters
    ----------
    stack : `ImageStackPy`
        The ImageStackPy object containing the stack of images.

    Returns
    -------
    psi : `list` of `numpy.ndarray`
        The list of H x W matrices of the PSI images.
    phi : `list` of `numpy.ndarray`
        The list of H x W matrices of the PHI images.
    """
    return generate_all_psi_phi_images(stack.sci, stack.var, stack.psfs)
