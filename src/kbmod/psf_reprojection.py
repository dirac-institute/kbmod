"""Measurement utilities for comparing PSFs across reprojection.

Reprojecting an image changes the point-source response in two distinct ways,
and telling them apart is the whole point of this module:

1. **Geometry.** If the output pixels subtend a different angle than the input
   pixels, or the frame is rotated or sheared, the PSF changes shape *in pixel
   units* without any loss of information. A source that is 5 px wide in a
   0.2"/px frame and 3.8 px wide in a 0.26"/px frame has not been blurred.
2. **Interpolation.** The resampling kernel itself adds width. This is real
   information loss and it is what the effective PSF must capture.

Reporting a single pixel FWHM cannot separate these. Every comparison here is
therefore available in angular units via the local WCS Jacobian, and the
geometric-only prediction is available separately so the interpolation residual
can be isolated.

Conventions, which are a common source of half-pixel and transpose errors:

- Images are NumPy arrays indexed ``[y, x]``.
- Pixel coordinates are ``(x, y)`` floats in the Astropy convention, where
  ``(0, 0)`` is the *center* of the first pixel.
- Covariance matrices are ordered ``[[xx, xy], [xy, yy]]``.
- Angular quantities are arcseconds, with the right-ascension axis already
  multiplied by ``cos(dec)`` so the units are true angle on the sky.
"""

from dataclasses import dataclass

import numpy as np
from astropy.wcs.utils import local_partial_pixel_derivatives

__all__ = [
    "MomentMeasurement",
    "ShapeSummary",
    "measure_moments",
    "local_wcs_jacobian",
    "transform_covariance",
    "summarize_shape",
    "encircled_energy_radius",
    "predict_geometric_covariance",
]

# FWHM of a Gaussian in units of its standard deviation.
_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class MomentMeasurement:
    """Flux-weighted moments of an image.

    Attributes
    ----------
    flux : `float`
        Sum of the (finite, non-negative-weighted) pixel values used.
    centroid_x, centroid_y : `float`
        Flux-weighted centroid in pixel coordinates.
    covariance : `numpy.ndarray`
        The 2x2 second-moment matrix ``[[xx, xy], [xy, yy]]`` in pixel units.
    n_pixels : `int`
        Number of pixels that contributed.
    """

    flux: float
    centroid_x: float
    centroid_y: float
    covariance: np.ndarray
    n_pixels: int


@dataclass(frozen=True)
class ShapeSummary:
    """Gaussian-equivalent shape derived from a second-moment matrix.

    The FWHM values are those of the Gaussian with the same second moments.
    For a non-Gaussian PSF they remain a well-defined shape summary but are not
    a substitute for the full profile, and they must not be combined in
    quadrature to model blur.

    Attributes
    ----------
    fwhm_major, fwhm_minor : `float`
        Gaussian-equivalent FWHM along the principal axes, in the units of the
        covariance matrix supplied (pixels or arcseconds).
    position_angle : `float`
        Orientation of the major axis in degrees, counterclockwise from the
        ``+x`` axis.
    """

    fwhm_major: float
    fwhm_minor: float
    position_angle: float


def measure_moments(image, threshold=0.0):
    """Measure flux-weighted centroid and second moments of an image.

    NaN pixels are excluded rather than propagated, so a clipped or masked
    output can still be measured. Note that excluding them biases the result if
    the excluded region carries real flux; check the footprint separately.

    Parameters
    ----------
    image : `numpy.ndarray`
        A 2D image, indexed ``[y, x]``.
    threshold : `float`
        Ignore pixels at or below this value. The default of 0.0 excludes
        negative pixels, which would otherwise produce a meaningless
        (potentially negative-determinant) covariance.

    Returns
    -------
    `MomentMeasurement`

    Raises
    ------
    ValueError
        If no pixel exceeds ``threshold``.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

    weights = np.where(np.isfinite(image) & (image > threshold), image, 0.0).astype(np.float64)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("No pixels above threshold; cannot measure moments.")

    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)

    dx = xx - cx
    dy = yy - cy
    mxx = float((weights * dx * dx).sum() / total)
    myy = float((weights * dy * dy).sum() / total)
    mxy = float((weights * dx * dy).sum() / total)

    return MomentMeasurement(
        flux=float(total),
        centroid_x=cx,
        centroid_y=cy,
        covariance=np.array([[mxx, mxy], [mxy, myy]]),
        n_pixels=int((weights > 0).sum()),
    )


def local_wcs_jacobian(wcs, x, y):
    """Return the local Jacobian mapping pixel offsets to angular offsets.

    The returned matrix ``J`` satisfies ``[dRA*cos(dec), dDec] = J @ [dx, dy]``
    with the angular offsets in arcseconds, so it captures pixel scale,
    rotation, shear, and local distortion together. A pure rotation leaves
    ``det(J)`` unchanged; a scale change does not.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        The WCS to differentiate.
    x, y : `float`
        Pixel coordinate at which to evaluate.

    Returns
    -------
    `numpy.ndarray`
        A 2x2 matrix in arcseconds per pixel.
    """
    # Rows are world axes (RA, Dec), columns are pixel axes (x, y), in degrees.
    derivatives = local_partial_pixel_derivatives(wcs, x, y)

    _, dec = wcs.pixel_to_world_values(x, y)
    cos_dec = np.cos(np.deg2rad(float(dec)))

    jacobian = np.array(
        [
            [derivatives[0, 0] * cos_dec, derivatives[0, 1] * cos_dec],
            [derivatives[1, 0], derivatives[1, 1]],
        ]
    )
    return jacobian * 3600.0


def transform_covariance(covariance, jacobian):
    """Transform a covariance matrix through a locally affine map.

    Computes ``J C J^T``. Used both to convert pixel moments to angular
    moments and to predict output moments from input moments on geometry alone.

    Parameters
    ----------
    covariance : `numpy.ndarray`
        A 2x2 covariance matrix.
    jacobian : `numpy.ndarray`
        A 2x2 linear map.

    Returns
    -------
    `numpy.ndarray`
        The transformed 2x2 covariance matrix.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    return jacobian @ covariance @ jacobian.T


def predict_geometric_covariance(covariance_native, original_wcs, common_wcs, x, y):
    """Predict output moments from input moments using geometry alone.

    The prediction is what the second moments would be if resampling were a
    perfect coordinate change that added no width of its own. Comparing a
    measured output covariance against this isolates the interpolation
    contribution: attribute a residual to blur only when the *matrix*
    difference is meaningful, not merely a scalar FWHM difference.

    Parameters
    ----------
    covariance_native : `numpy.ndarray`
        Measured 2x2 pixel covariance in the original frame.
    original_wcs, common_wcs : `astropy.wcs.WCS`
        The input and output WCS.
    x, y : `float`
        Pixel coordinate in the *original* frame at which to linearize.

    Returns
    -------
    `numpy.ndarray`
        Predicted 2x2 pixel covariance in the common frame.
    """
    native_jacobian = local_wcs_jacobian(original_wcs, x, y)

    sky = original_wcs.pixel_to_world_values(x, y)
    out_x, out_y = common_wcs.world_to_pixel_values(*sky)
    common_jacobian = local_wcs_jacobian(common_wcs, float(out_x), float(out_y))

    # native pixels -> angle -> common pixels
    pixel_to_pixel = np.linalg.inv(common_jacobian) @ native_jacobian
    return transform_covariance(covariance_native, pixel_to_pixel)


def summarize_shape(covariance):
    """Summarize a second-moment matrix as Gaussian-equivalent axes and angle.

    Parameters
    ----------
    covariance : `numpy.ndarray`
        A 2x2 covariance matrix, in pixel or angular units.

    Returns
    -------
    `ShapeSummary`

    Raises
    ------
    ValueError
        If the matrix is not positive semi-definite, which indicates the
        moments were measured over a region containing negative weight.
    """
    covariance = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-12:
        raise ValueError(f"Covariance is not positive semi-definite: eigenvalues {eigenvalues}.")
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # eigh returns ascending eigenvalues; the major axis is the last.
    major, minor = eigenvalues[1], eigenvalues[0]
    major_vector = eigenvectors[:, 1]
    angle = float(np.rad2deg(np.arctan2(major_vector[1], major_vector[0])))

    return ShapeSummary(
        fwhm_major=float(_FWHM_PER_SIGMA * np.sqrt(major)),
        fwhm_minor=float(_FWHM_PER_SIGMA * np.sqrt(minor)),
        position_angle=(angle + 180.0) % 180.0,
    )


def encircled_energy_radius(image, center_x, center_y, fraction=0.5):
    """Return the radius enclosing a given fraction of the total flux.

    Unlike a Gaussian-equivalent FWHM this is sensitive to the wings, which is
    what determines how much kernel support a PSF actually needs.

    Parameters
    ----------
    image : `numpy.ndarray`
        A 2D image, indexed ``[y, x]``. NaN pixels are ignored.
    center_x, center_y : `float`
        Center to measure about, in pixel coordinates.
    fraction : `float`
        Enclosed flux fraction, in ``(0, 1]``.

    Returns
    -------
    `float`
        Radius in pixels, linearly interpolated between the bracketing radii.

    Raises
    ------
    ValueError
        If ``fraction`` is out of range or the image has no positive flux.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")

    image = np.asarray(image)
    values = np.where(np.isfinite(image) & (image > 0.0), image, 0.0).astype(np.float64)
    total = values.sum()
    if total <= 0.0:
        raise ValueError("Image has no positive flux; cannot measure encircled energy.")

    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    radii = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2).ravel()
    flat = values.ravel()

    order = np.argsort(radii)
    radii = radii[order]
    cumulative = np.cumsum(flat[order]) / total

    index = int(np.searchsorted(cumulative, fraction))
    if index == 0:
        return float(radii[0])
    if index >= len(radii):
        return float(radii[-1])

    # Linear interpolation between the bracketing samples.
    low, high = cumulative[index - 1], cumulative[index]
    if high == low:
        return float(radii[index])
    weight = (fraction - low) / (high - low)
    return float(radii[index - 1] + weight * (radii[index] - radii[index - 1]))
