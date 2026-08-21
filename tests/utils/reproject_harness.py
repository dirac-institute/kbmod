"""Generated isolated-source harness for reprojection validation.

Phase 0 of the PSF/reprojection work needs controlled data with known truth,
because the shipped reprojection fixture cannot support PSF acceptance gates:
it stores four byte-identical 3x3 kernels, so a test built on it can pass while
the PSF is badly wrong.

This harness generates an isolated source with an analytic profile at a chosen
subpixel phase, reprojects it through the *same* configured operator the science
path uses, and measures both frames. Sources are injected before reprojection,
never directly into common-frame pixels, so the measured output includes the
resampling the real pipeline applies.
"""

from dataclasses import dataclass

import numpy as np
from astropy.wcs import WCS

from kbmod.psf_reprojection import (
    encircled_energy_radius,
    local_wcs_jacobian,
    measure_moments,
    predict_geometric_covariance,
    summarize_shape,
    transform_covariance,
)
from kbmod.reprojection import reproject_image
from kbmod.reprojection_config import LEGACY_CONFIG

__all__ = [
    "TrialResult",
    "make_tan_wcs",
    "gaussian_source",
    "coma_source",
    "empirical_source",
    "moffat_source",
    "elliptical_gaussian_source",
    "render_source",
    "run_trial",
    "matched_filter_sums",
    "measure_matched_flux",
    "measure_trajectory_flux",
]


def make_tan_wcs(crval, pixel_scale_deg, shape, rot_deg=0.0, scale_y=None):
    """Build a deterministic TAN WCS.

    Parameters
    ----------
    crval : `tuple`
        ``(ra, dec)`` of the reference point, in degrees.
    pixel_scale_deg : `float`
        Pixel scale in degrees. Negative follows the usual RA convention.
    shape : `tuple`
        Array shape ``(ny, nx)``.
    rot_deg : `float`
        Rotation in degrees.
    scale_y : `float`, optional
        Separate y scale, for anisotropic/shear cases. Defaults to
        ``pixel_scale_deg``.

    Returns
    -------
    `astropy.wcs.WCS`
    """
    if scale_y is None:
        scale_y = pixel_scale_deg

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2.0 + 0.5, shape[0] / 2.0 + 0.5]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    rot = np.deg2rad(rot_deg)
    rotation = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    wcs.wcs.cd = rotation @ np.diag([pixel_scale_deg, scale_y])
    wcs.array_shape = shape
    return wcs


def gaussian_source(fwhm):
    """Return a circular Gaussian profile callable of ``(dx, dy)``."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def profile(dx, dy):
        return np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))

    return profile


def elliptical_gaussian_source(fwhm_major, fwhm_minor, position_angle_deg):
    """Return an elliptical Gaussian profile callable of ``(dx, dy)``.

    An asymmetric profile is essential for catching transpose and axis-swap
    errors, which a circular source cannot detect.
    """
    to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_major = fwhm_major * to_sigma
    sigma_minor = fwhm_minor * to_sigma
    angle = np.deg2rad(position_angle_deg)

    def profile(dx, dy):
        u = dx * np.cos(angle) + dy * np.sin(angle)
        v = -dx * np.sin(angle) + dy * np.cos(angle)
        return np.exp(-(u**2 / (2.0 * sigma_major**2) + v**2 / (2.0 * sigma_minor**2)))

    return profile


def moffat_source(fwhm, beta=2.5):
    """Return a Moffat profile callable of ``(dx, dy)``.

    Moffat wings are much heavier than a Gaussian's, which is what makes fixed
    "3 sigma" kernel support invalid for realistic PSFs.
    """
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))

    def profile(dx, dy):
        return (1.0 + (dx**2 + dy**2) / alpha**2) ** (-beta)

    return profile


def coma_source(fwhm, asymmetry=0.45):
    """Return a coma-like asymmetric profile callable of ``(dx, dy)``.

    A bright core plus a displaced, broader, fainter lobe, giving the profile a
    genuine third moment. Unlike an ellipse this is not symmetric under a
    180-degree rotation, so it detects sign and reflection errors that an
    elliptical Gaussian cannot.
    """
    to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    core = fwhm * to_sigma
    tail = 1.8 * core
    offset = asymmetry * fwhm

    def profile(dx, dy):
        main = np.exp(-(dx**2 + dy**2) / (2.0 * core**2))
        lobe = 0.35 * np.exp(-((dx - offset) ** 2 + (dy - 0.5 * offset) ** 2) / (2.0 * tail**2))
        return main + lobe

    return profile


def empirical_source(array, pixel_scale=1.0):
    """Return a profile callable interpolating a tabulated PSF array.

    Stands in for a Rubin model supplied as pixels rather than an analytic form,
    which is the realistic case. ``array`` must be odd-sized and square; its
    center pixel is the profile origin.
    """
    from scipy.ndimage import map_coordinates

    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Empirical PSF must be a 2D square array, got {array.shape}.")
    if array.shape[0] % 2 == 0:
        raise ValueError("Empirical PSF must have an odd width so its center is well defined.")
    radius = array.shape[0] // 2

    def profile(dx, dy):
        rows = np.asarray(dy, dtype=np.float64) / pixel_scale + radius
        cols = np.asarray(dx, dtype=np.float64) / pixel_scale + radius
        values = map_coordinates(array, [rows.ravel(), cols.ravel()], order=1, mode="constant", cval=0.0)
        return values.reshape(np.shape(dx))

    return profile


def render_source(shape, x, y, profile, oversample=5):
    """Render a source at a subpixel position.

    Each pixel is integrated by averaging ``oversample**2`` sub-samples, so the
    result reflects a pixel's response to a source at a fractional position
    rather than a point evaluation at the pixel center. That distinction is what
    makes pixel-phase tests meaningful.

    Parameters
    ----------
    shape : `tuple`
        Output shape ``(ny, nx)``.
    x, y : `float`
        Source position in pixel coordinates.
    profile : `callable`
        Function of ``(dx, dy)`` returning intensity.
    oversample : `int`
        Sub-samples per pixel per axis.

    Returns
    -------
    `numpy.ndarray`
        A float32 image, indexed ``[y, x]``.
    """
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)

    offsets = (np.arange(oversample) + 0.5) / oversample - 0.5
    accumulator = np.zeros(shape, dtype=np.float64)
    for sub_y in offsets:
        for sub_x in offsets:
            accumulator += profile((xx + sub_x) - x, (yy + sub_y) - y)
    accumulator /= oversample**2

    return accumulator.astype(np.float32)


@dataclass
class TrialResult:
    """Everything one reprojection trial produced.

    Attributes
    ----------
    science, variance : `numpy.ndarray`
        Reprojected planes in the common frame.
    footprint : `numpy.ndarray`
        Coverage of the common frame.
    native_image : `numpy.ndarray`
        The injected source before reprojection.
    native_moments, common_moments : `MomentMeasurement`
        Measured moments in each frame.
    native_shape_pixels, common_shape_pixels : `ShapeSummary`
        Gaussian-equivalent shape in pixel units.
    native_shape_angular, common_shape_angular : `ShapeSummary`
        The same shape in arcseconds, comparable across frames.
    geometric_prediction : `ShapeSummary`
        Common-frame shape predicted from geometry alone. The difference
        between this and ``common_shape_pixels`` is the interpolation
        contribution.
    flux_ratio : `float`
        Recovered flux divided by injected flux.
    centroid_error : `float`
        Distance in common-frame pixels between the measured centroid and the
        mapped truth position.
    clipped : `bool`
        Whether any NaN appeared inside the footprint, meaning the boundary
        mode discarded data.
    """

    science: np.ndarray
    variance: np.ndarray
    footprint: np.ndarray
    native_image: np.ndarray
    native_moments: object
    common_moments: object
    native_shape_pixels: object
    common_shape_pixels: object
    native_shape_angular: object
    common_shape_angular: object
    geometric_prediction: object
    flux_ratio: float
    centroid_error: float
    clipped: bool


def run_trial(
    profile,
    original_wcs,
    common_wcs,
    source_x,
    source_y,
    config=LEGACY_CONFIG,
    variance_level=1.0,
    oversample=5,
):
    """Inject an isolated source, reproject it, and measure both frames.

    Returns
    -------
    `TrialResult`
    """
    shape = original_wcs.array_shape
    native = render_source(shape, source_x, source_y, profile, oversample=oversample)
    native_variance = np.full(shape, variance_level, dtype=np.float32)

    science, footprint = reproject_image(native, original_wcs, common_wcs, config=config)
    variance, _ = reproject_image(native_variance, original_wcs, common_wcs, config=config)

    native_moments = measure_moments(native)
    common_moments = measure_moments(science)

    native_jacobian = local_wcs_jacobian(original_wcs, native_moments.centroid_x, native_moments.centroid_y)
    common_jacobian = local_wcs_jacobian(common_wcs, common_moments.centroid_x, common_moments.centroid_y)

    # Where the injected source should land in the common frame.
    sky = original_wcs.pixel_to_world_values(source_x, source_y)
    truth_x, truth_y = common_wcs.world_to_pixel_values(*sky)
    centroid_error = float(
        np.hypot(common_moments.centroid_x - float(truth_x), common_moments.centroid_y - float(truth_y))
    )

    inside = footprint > 0
    clipped = bool(np.any(np.isnan(science[inside]))) if inside.any() else True

    return TrialResult(
        science=science,
        variance=variance,
        footprint=footprint,
        native_image=native,
        native_moments=native_moments,
        common_moments=common_moments,
        native_shape_pixels=summarize_shape(native_moments.covariance),
        common_shape_pixels=summarize_shape(common_moments.covariance),
        native_shape_angular=summarize_shape(
            transform_covariance(native_moments.covariance, native_jacobian)
        ),
        common_shape_angular=summarize_shape(
            transform_covariance(common_moments.covariance, common_jacobian)
        ),
        geometric_prediction=summarize_shape(
            predict_geometric_covariance(
                native_moments.covariance,
                original_wcs,
                common_wcs,
                native_moments.centroid_x,
                native_moments.centroid_y,
            )
        ),
        flux_ratio=float(common_moments.flux / native_moments.flux),
        centroid_error=centroid_error,
        clipped=clipped,
    )


def encircled_energy(image, moments, fraction=0.5):
    """Convenience wrapper measuring encircled energy about measured moments."""
    return encircled_energy_radius(image, moments.centroid_x, moments.centroid_y, fraction)


# ---------------------------------------------------------------------------
# Matched-filter photometry
#
# These drive KBMOD's own `generate_psi_phi_images` and reproduce the flux and
# likelihood definitions from `Results._update_likelihood`:
#
#     flux       = sum(psi) / sum(phi)
#     likelihood = sum(psi) / sqrt(sum(phi))
#
# so what is measured here is the estimator the search actually uses, not a
# re-derivation of it.
#
# Why the kernel choice decides the answer: for an image f * P with constant
# variance and a filter kernel K,
#
#     flux_hat = f * sum(P * K) / sum(K * K)
#
# which equals f exactly when K == P and is biased otherwise. The effective PSF
# is the K that matches the P actually present in reprojected pixels, so the
# flux bias measured below is a direct read-out of kernel mismatch.
# ---------------------------------------------------------------------------


def matched_filter_sums(science, variance, kernel, x, y):
    """Return ``(psi, phi)`` sampled at the pixel nearest ``(x, y)``.

    Sampling at an integer pixel is deliberate: the search evaluates candidate
    trajectories at integer pixels too, so the phase-dependent component of the
    bias this exposes is one KBMOD genuinely experiences.
    """
    from kbmod.core.shift_and_stack import generate_psi_phi_images

    psi, phi = generate_psi_phi_images(science, variance, np.asarray(kernel, dtype=np.float32))
    row, col = int(round(y)), int(round(x))
    if not (0 <= row < psi.shape[0] and 0 <= col < psi.shape[1]):
        return np.nan, np.nan
    return float(psi[row, col]), float(phi[row, col])


def measure_matched_flux(science, variance, kernel, x, y):
    """Matched-filter flux and likelihood for a stationary source."""
    psi, phi = matched_filter_sums(science, variance, kernel, x, y)
    if not np.isfinite(psi) or not np.isfinite(phi) or phi <= 0:
        return np.nan, np.nan
    return psi / phi, psi / np.sqrt(phi)


def measure_trajectory_flux(science_list, variance_list, kernels, xs, ys):
    """Matched-filter flux and likelihood summed along a trajectory.

    Mirrors the search: psi and phi are accumulated across epochs at the
    trajectory's position in each, then combined once at the end.
    """
    psi_total = 0.0
    phi_total = 0.0
    used = 0
    for science, variance, kernel, x, y in zip(science_list, variance_list, kernels, xs, ys):
        psi, phi = matched_filter_sums(science, variance, kernel, x, y)
        if np.isfinite(psi) and np.isfinite(phi) and phi > 0:
            psi_total += psi
            phi_total += phi
            used += 1
    if phi_total <= 0 or used == 0:
        return np.nan, np.nan, used
    return psi_total / phi_total, psi_total / np.sqrt(phi_total), used
