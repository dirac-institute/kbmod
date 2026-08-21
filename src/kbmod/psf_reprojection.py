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

from dataclasses import dataclass, field

import numpy as np
from astropy.wcs.utils import local_partial_pixel_derivatives

__all__ = [
    "EffectivePsf",
    "EffectivePsfError",
    "make_effective_psf",
    "DEFAULT_MAX_SUPPORT",
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

#: Default cap on the effective-PSF support radius, giving a 51x51 kernel.
#: Convergence alone is not a sufficient stopping rule: a noiseless Moffat's
#: wings keep contributing flux well past any support affordable to convolve
#: with. The cap trades a reported, quantified truncation for a usable kernel.
DEFAULT_MAX_SUPPORT = 25


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


# ---------------------------------------------------------------------------
# Effective PSF generation
#
# The PSF that matters for a search is the one present in the pixels actually
# searched, which is the native model *after* the same resampling the science
# image went through. Generating it by any route other than the real science
# operator -- fitting a wider Gaussian, adding widths in quadrature -- produces
# a kernel that is close enough to look right and wrong enough to bias
# photometry. So the native stamp is pasted into a frame and pushed through the
# identical configured operator, with the identical options.
# ---------------------------------------------------------------------------


@dataclass
class EffectivePsf:
    """A PSF resampled into the common frame by the science operator.

    Attributes
    ----------
    kernel : `numpy.ndarray`
        Normalized, odd-sized, square, float32. This is the search-facing
        representation.
    centroid_x, centroid_y : `float`
        Centroid of the resampled PSF in common-frame pixels.
    sum_before_normalization : `float`
        Total inside the support before normalizing. Comparing this against the
        reprojected flux of a real source is how truncation is detected.
    lost_fraction : `float`
        Fraction of the resampled stamp's flux falling outside the support.
        Reported before normalization, because normalizing hides it.
    native_x, native_y : `float`
        Where the stamp sat in the original frame.
    output_x, output_y : `float`
        Where that position maps to in the common frame.
    support_radius : `int`
        Half-width of the kernel; the width is ``2 * support_radius + 1``.
    covariance : `numpy.ndarray`
        Second-moment matrix of the kernel, in common-frame pixels.
    provenance : `dict`
        Reprojection preset, config hash, ``reproject`` version, and method.
    warnings : `list`
        Non-fatal issues worth surfacing, e.g. support that hit its cap before
        converging.
    """

    kernel: np.ndarray
    centroid_x: float
    centroid_y: float
    sum_before_normalization: float
    lost_fraction: float
    native_x: float
    native_y: float
    output_x: float
    output_y: float
    support_radius: int
    covariance: np.ndarray
    provenance: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def width(self):
        """Width of the kernel in pixels."""
        return self.kernel.shape[0]


class EffectivePsfError(ValueError):
    """Raised when an effective PSF cannot be generated safely."""


def _enclosed(image, cx, cy, radius):
    """Sum and second moments within a square support, ignoring NaN."""
    y0, y1 = int(round(cy)) - radius, int(round(cy)) + radius + 1
    x0, x1 = int(round(cx)) - radius, int(round(cx)) + radius + 1
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return None
    patch = image[y0:y1, x0:x1]
    if not np.all(np.isfinite(patch)):
        return None
    return patch, (x0, y0)


def _required_padding(config, scale_ratio):
    """Native-pixel padding needed so the resampler never samples off-canvas.

    The adaptive resampler reads a region ``sample_region_width`` output pixels
    across for each output pixel. With ``boundary_mode="strict"`` any output
    pixel whose region falls off the input becomes NaN, so the canvas has to
    extend past the stamp by that region expressed in native pixels, plus slack
    for the interpolation kernel itself.

    Measured rather than guessed: the generated kernel is bit-identical for any
    padding from 2 to 30 px on the default configuration, because the stamp is
    already near zero at its own edge and the support crop sits well inside the
    canvas. Over-padding is not free -- it shrinks the region of the native
    frame where a PSF can be evaluated at all, which matters for mosaic tiles
    that only clip the common frame. Insufficient padding fails loudly, since
    the support crop refuses to include NaN.
    """
    ratio = max(1.0, float(scale_ratio))
    region = float(config.sample_region_width) * ratio
    tail = 2.0 * float(config.kernel_width) * ratio
    return int(np.ceil(region + tail)) + 2


def _local_scale_ratio(original_wcs, common_wcs, x, y):
    """Native pixels per output pixel at a position (linear, not area)."""
    native = local_wcs_jacobian(original_wcs, x, y)
    sky = original_wcs.pixel_to_world_values(x, y)
    out_x, out_y = common_wcs.world_to_pixel_values(*sky)
    common = local_wcs_jacobian(common_wcs, float(out_x), float(out_y))
    native_scale = np.sqrt(abs(np.linalg.det(native)))
    common_scale = np.sqrt(abs(np.linalg.det(common)))
    return native_scale / common_scale if common_scale > 0 else 1.0


def make_effective_psf(
    stamp,
    stamp_origin,
    original_wcs,
    common_wcs,
    config=None,
    method="cutout",
    max_support=None,
    flux_tolerance=1e-5,
    moment_tolerance=1e-3,
):
    """Resample a native PSF stamp into the common frame.

    Parameters
    ----------
    stamp : `numpy.ndarray`
        The native PSF stamp. Need not be normalized.
    stamp_origin : `tuple`
        ``(x0, y0)``, the native pixel coordinate of ``stamp[0, 0]``. This is
        the Rubin bounding-box origin; supplying it wrong shifts the result by
        exactly that error, which is why it is required rather than guessed.
    original_wcs, common_wcs : `astropy.wcs.WCS`
        Input and output WCS. Both must carry an ``array_shape``.
    config : `AdaptiveReprojectionConfig`, optional
        The reprojection options. **Must be the same object used for the
        science image**; a different configuration produces a different
        effective PSF. Defaults to the legacy preset.
    method : `str`
        ``"full"`` pastes into a full-size native frame -- the simple reference
        implementation. ``"cutout"`` works on a padded sub-frame and is the
        default; the two are required to agree.
    max_support : `int`, optional
        Cap on the support radius. Defaults to `DEFAULT_MAX_SUPPORT`. Heavy
        wings converge slowly -- a noiseless Moffat can formally demand a radius
        past 40, an 80-pixel-wide kernel that would dominate search cost for a
        negligible amount of flux. When the cap binds before convergence the
        truncated fraction is reported in ``lost_fraction`` and a warning is
        recorded, rather than the loss being normalized away.
    flux_tolerance, moment_tolerance : `float`
        Convergence thresholds for growing the support.

    Returns
    -------
    `EffectivePsf`

    Raises
    ------
    EffectivePsfError
        If the stamp cannot be placed, the resampled PSF is clipped by a
        boundary, or the support cannot be established without hitting NaN.
    """
    from kbmod.reprojection import reproject_image
    from kbmod.reprojection_config import LEGACY_CONFIG

    if config is None:
        config = LEGACY_CONFIG
    if method not in ("full", "cutout"):
        raise ValueError(f"method must be 'full' or 'cutout', got {method!r}.")

    stamp = np.asarray(stamp, dtype=np.float64)
    if stamp.ndim != 2:
        raise EffectivePsfError(f"PSF stamp must be 2D, got shape {stamp.shape}.")
    if not np.all(np.isfinite(stamp)):
        raise EffectivePsfError("PSF stamp contains non-finite values.")
    if stamp.sum() <= 0:
        raise EffectivePsfError("PSF stamp has no positive flux.")

    origin_x, origin_y = int(stamp_origin[0]), int(stamp_origin[1])
    stamp_moments = measure_moments(stamp)
    native_x = origin_x + stamp_moments.centroid_x
    native_y = origin_y + stamp_moments.centroid_y

    sky = original_wcs.pixel_to_world_values(native_x, native_y)
    output_x, output_y = (float(v) for v in common_wcs.world_to_pixel_values(*sky))

    scale_ratio = _local_scale_ratio(original_wcs, common_wcs, native_x, native_y)
    pad = _required_padding(config, scale_ratio)

    height, width = stamp.shape
    warnings_out = []

    if method == "full":
        native_shape = original_wcs.array_shape
        canvas = np.zeros(native_shape, dtype=np.float32)
        if (
            origin_x < 0
            or origin_y < 0
            or origin_x + width > native_shape[1]
            or origin_y + height > native_shape[0]
        ):
            raise EffectivePsfError(
                f"PSF stamp at origin ({origin_x}, {origin_y}) with shape {stamp.shape} does not fit "
                f"inside the native frame {native_shape}. The evaluation point is too close to the edge; "
                "a clipped stamp must not be silently normalized."
            )
        canvas[origin_y : origin_y + height, origin_x : origin_x + width] = stamp
        canvas_wcs = original_wcs
        out_wcs = common_wcs
        out_offset = (0, 0)
    else:
        # Native cutout around the stamp, padded so the resampler never reads
        # off the edge of what we hand it.
        native_shape = original_wcs.array_shape

        # The *stamp* must fit inside the real detector: a stamp clipped by the
        # frame edge describes a PSF whose wings were genuinely cut off, and
        # normalizing that would hide the loss.
        if (
            origin_x < 0
            or origin_y < 0
            or origin_x + width > native_shape[1]
            or origin_y + height > native_shape[0]
        ):
            raise EffectivePsfError(
                f"PSF stamp at origin ({origin_x}, {origin_y}) with shape {stamp.shape} does not fit "
                f"inside the native frame {native_shape}. A stamp clipped by the detector edge must "
                "not be silently normalized."
            )

        # The *padding* may extend past the detector. It exists so the resampler
        # never reads off the edge of the canvas, and the canvas is synthetic --
        # zeros with the stamp pasted in -- so those samples are zero whether or
        # not real data exists there. Requiring the padding to fit inside the
        # native frame would leave a mosaic tile that only clips the common
        # frame with no measurable PSF at all, forcing a choice between
        # discarding its real pixels and searching them with a neighbour's
        # kernel. WCS.slice handles out-of-range starts by CRPIX arithmetic, and
        # the full-frame reference path cross-checks the result.
        nx0 = origin_x - pad
        ny0 = origin_y - pad
        nx1 = origin_x + width + pad
        ny1 = origin_y + height + pad
        canvas = np.zeros((ny1 - ny0, nx1 - nx0), dtype=np.float32)
        canvas[origin_y - ny0 : origin_y - ny0 + height, origin_x - nx0 : origin_x - nx0 + width] = stamp
        canvas_wcs = original_wcs.slice((slice(ny0, ny1), slice(nx0, nx1)))

        # Output cutout: the mapped position plus enough room for the support
        # search, in output pixels.
        out_pad = int(np.ceil(max(width, height) / max(scale_ratio, 1e-6))) + pad
        ox0 = int(np.floor(output_x)) - out_pad
        oy0 = int(np.floor(output_y)) - out_pad
        ox1 = int(np.floor(output_x)) + out_pad + 1
        oy1 = int(np.floor(output_y)) + out_pad + 1
        common_shape = common_wcs.array_shape
        ox0_c, oy0_c = max(ox0, 0), max(oy0, 0)
        ox1_c, oy1_c = min(ox1, common_shape[1]), min(oy1, common_shape[0])
        if ox1_c - ox0_c < 3 or oy1_c - oy0_c < 3:
            raise EffectivePsfError(
                f"The PSF maps to ({output_x:.2f}, {output_y:.2f}), which leaves no usable region "
                f"inside the common frame {common_shape}."
            )
        if (ox0_c, oy0_c, ox1_c, oy1_c) != (ox0, oy0, ox1, oy1):
            warnings_out.append("output cutout was clipped by the common frame edge")
        out_wcs = common_wcs.slice((slice(oy0_c, oy1_c), slice(ox0_c, ox1_c)))
        out_offset = (ox0_c, oy0_c)

    resampled, _ = reproject_image(canvas, canvas_wcs, out_wcs, config=config)

    finite = np.isfinite(resampled)
    if not np.any(finite & (resampled > 0)):
        raise EffectivePsfError(
            "Resampling the PSF stamp produced no positive finite pixels. The evaluation point is "
            "probably outside the common frame's footprint."
        )

    usable = np.where(finite, resampled, 0.0)
    local = measure_moments(usable)
    cx, cy = local.centroid_x, local.centroid_y

    # Grow the support until enclosed flux and second moments stop changing.
    # Cropping to a fixed radius would silently truncate a Moffat-like wing.
    limit = min(
        int(min(usable.shape) // 2) - 1,
        int(min(cx, cy, usable.shape[1] - 1 - cx, usable.shape[0] - 1 - cy)),
    )
    limit = min(limit, DEFAULT_MAX_SUPPORT if max_support is None else int(max_support))
    if limit < 2:
        raise EffectivePsfError(
            f"Only {limit} px of room around the resampled PSF centroid; cannot establish a support."
        )

    start = max(2, int(np.ceil(2.0 * np.sqrt(max(local.covariance[0, 0], local.covariance[1, 1])))))
    start = min(start, limit)

    chosen = None
    previous = None
    for radius in range(start, limit + 1):
        window = _enclosed(usable, cx, cy, radius)
        if window is None:
            # Hit a NaN or an edge before converging.
            break
        patch, _ = window
        total = float(patch.sum())
        if total <= 0:
            continue
        patch_moments = measure_moments(patch)
        # The full covariance, including the xy cross-term. Comparing only the
        # diagonal would let a rotated asymmetric PSF appear converged while its
        # position angle was still moving.
        current = (total, patch_moments.covariance.copy())
        if previous is not None:
            flux_change = abs(current[0] - previous[0]) / max(current[0], 1e-30)
            scale = max(np.abs(current[1]).max(), 1e-30)
            moment_change = float(np.abs(current[1] - previous[1]).max() / scale)
            if flux_change < flux_tolerance and moment_change < moment_tolerance:
                chosen = radius
                break
        previous = current

    if chosen is None:
        # Convergence failed. Fall back to the *largest* safe support, not the
        # smallest: the whole reason convergence failed is that flux is still
        # arriving, so the widest affordable kernel truncates least.
        chosen = limit
        warnings_out.append(
            f"support did not converge within radius {limit}; using {chosen}. Check lost_fraction: "
            "heavy PSF wings converge slowly and the kernel is truncated here."
        )

    window = _enclosed(usable, cx, cy, chosen)
    if window is None:
        raise EffectivePsfError(
            "The resampled PSF is clipped by a boundary or contains NaN within its support. "
            "Normalizing a clipped kernel would hide the lost flux."
        )
    patch, (px0, py0) = window

    total_resampled = float(usable.sum())
    enclosed_sum = float(patch.sum())
    lost_fraction = max(0.0, 1.0 - enclosed_sum / total_resampled) if total_resampled > 0 else 1.0

    patch_moments = measure_moments(patch)
    kernel = (patch / enclosed_sum).astype(np.float32)

    return EffectivePsf(
        kernel=kernel,
        centroid_x=out_offset[0] + px0 + patch_moments.centroid_x,
        centroid_y=out_offset[1] + py0 + patch_moments.centroid_y,
        sum_before_normalization=enclosed_sum,
        lost_fraction=lost_fraction,
        native_x=native_x,
        native_y=native_y,
        output_x=output_x,
        output_y=output_y,
        support_radius=chosen,
        covariance=patch_moments.covariance,
        provenance={
            "method": method,
            "preset": config.preset_name,
            "config_hash": config.hexdigest,
            "reproject_version": config.provenance["reproject_version"],
            "scale_ratio": scale_ratio,
            "padding": pad,
        },
        warnings=warnings_out,
    )
