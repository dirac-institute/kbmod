"""Extraction of native Rubin ``Exposure.psf`` models into KBMOD kernels.

KBMOD historically replaced the Rubin PSF with a fixed Gaussian, discarding the
per-exposure, spatially varying model the Science Pipelines had already fitted.
This module renders the real model instead.

The Rubin import is deferred to call time, so importing KBMOD never requires the
Science Pipelines to be installed.

Two Rubin rendering entry points exist and they are **not** interchangeable:

``computeKernelImage(position)``
    Returns the PSF centered on the origin, which is what a convolution or
    matched filter requires. This is the correct source for KBMOD's search
    kernel and is the default here.

``computeImage(position)``
    Returns the PSF as it would appear in the image, retaining the fractional
    pixel phase of ``position`` and carrying a bounding box that locates the
    stamp in exposure coordinates. This is the correct source when the stamp
    must be placed in the native frame and reprojected, because the pixel phase
    is part of what reprojection acts on.

Using the image-mode stamp as a search kernel would bake a sub-pixel offset into
the matched filter and bias recovered positions. Using the kernel-mode stamp as
a reprojection source would discard the pixel phase the reprojection needs.
Neither is a safe default for the other's job, so the mode is explicit and the
choice is recorded in the returned stamp's provenance.
"""

import importlib
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "NativePsfStamp",
    "RubinPsfError",
    "render_rubin_psf",
    "average_position",
]

#: Negative pixels at or above this fraction of the peak (in magnitude) are
#: treated as interpolation noise and clipped to zero. Anything more negative is
#: a real signed lobe and is refused rather than silently flattened, because
#: clipping it would change the kernel's normalization and its meaning.
NEGATIVE_TOLERANCE = 1e-6


class RubinPsfError(ValueError):
    """Raised when a Rubin PSF model is missing or cannot be used."""


@dataclass
class NativePsfStamp:
    """A PSF stamp rendered from a Rubin ``Exposure.psf``.

    Attributes
    ----------
    array : `numpy.ndarray`
        The float32 stamp, normalized to sum to 1.
    origin_x, origin_y : `int`
        The Rubin bounding-box origin (``xy0``) in exposure pixel coordinates.
        Discarding this loses the stamp's placement, which is the most common
        source of one-pixel errors in this code.
    eval_x, eval_y : `float`
        The floating-point coordinate the model was evaluated at.
    centroid_x, centroid_y : `float`
        Flux-weighted centroid *within* ``array``.
    native_sum : `float`
        Sum of the stamp before normalization. Rubin models are usually close
        to unity but are not guaranteed to be exactly so.
    provenance : `str`
        Which Rubin entry point produced this, ``"computeKernelImage"`` or
        ``"computeImage"``.
    clipped_negative_sum : `float`
        Total magnitude of negative pixels clipped as interpolation noise. Non
        zero values are worth reporting; large ones raise instead.
    diagnostics : `dict`
        Optional shape measurements, populated when requested.
    """

    array: np.ndarray
    origin_x: int
    origin_y: int
    eval_x: float
    eval_y: float
    centroid_x: float
    centroid_y: float
    native_sum: float
    provenance: str
    clipped_negative_sum: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    @property
    def width(self):
        """Width of the stamp in pixels."""
        return self.array.shape[1]

    @property
    def offset_x(self):
        """Centroid in exposure coordinates along x."""
        return self.origin_x + self.centroid_x

    @property
    def offset_y(self):
        """Centroid in exposure coordinates along y."""
        return self.origin_y + self.centroid_y


def _point2d(x, y):
    """Construct an ``lsst.geom.Point2D``, importing Rubin only when called."""
    try:
        geom = importlib.import_module("lsst.geom")
    except ImportError as err:  # pragma: no cover - exercised only without LSST
        raise RubinPsfError(
            "Evaluating a Rubin Exposure.psf requires the LSST Science Pipelines "
            "(lsst.geom). Install them, or set the 'psf_fallback_std' config option "
            "to opt in to a Gaussian fallback."
        ) from err
    return geom.Point2D(float(x), float(y))


def _stamp_origin(image, x, y, width, height):
    """Return the stamp's ``xy0`` origin, falling back to a centered guess."""
    getter = getattr(image, "getBBox", None)
    if getter is not None:
        try:
            bbox = getter()
            return int(bbox.getMinX()), int(bbox.getMinY())
        except Exception:
            pass

    getter = getattr(image, "getXY0", None)
    if getter is not None:
        try:
            xy0 = getter()
            return int(xy0.getX()), int(xy0.getY())
        except Exception:
            pass

    # No placement information available. Assume the stamp is centered on the
    # evaluation pixel, and say so rather than pretending it is authoritative.
    return int(np.floor(x)) - width // 2, int(np.floor(y)) - height // 2


def _measure_centroid(array):
    """Flux-weighted centroid within a stamp, in stamp pixel coordinates."""
    weights = np.where(array > 0.0, array, 0.0).astype(np.float64)
    total = weights.sum()
    if total <= 0.0:
        raise RubinPsfError("Rubin PSF stamp has no positive flux.")
    yy, xx = np.mgrid[0 : array.shape[0], 0 : array.shape[1]]
    return float((weights * xx).sum() / total), float((weights * yy).sum() / total)


def _resolve_psf_model(source):
    """Return the PSF model from an ``Exposure`` or from a bare model.

    Accepting both shapes matters because the Butler can serve the ``psf``
    component on its own, which is far cheaper than materializing a full
    ``Exposure`` when only the model is needed.

    The presence of a ``psf`` attribute is what identifies an exposure, and it
    is checked first even when its value is `None` -- an exposure whose model is
    missing must resolve to `None` so the caller raises, rather than falling
    through to treat the exposure itself as a model. Probing for the rendering
    methods first would misidentify any permissive stand-in (a bare `Mock`,
    say) as its own PSF model.
    """
    if hasattr(source, "psf"):
        return source.psf
    if hasattr(source, "computeKernelImage") or hasattr(source, "computeImage"):
        return source
    return None


def render_rubin_psf(source, x, y, mode="kernel", diagnostics=False):
    """Render a Rubin PSF model at a position.

    Parameters
    ----------
    source : Rubin ``Exposure`` or ``Psf``
        Either an exposure whose ``psf`` attribute holds the model, or the
        model itself (as returned by the Butler's ``psf`` component).
    x, y : `float`
        Pixel coordinate at which to evaluate. May be fractional.
    mode : `str`
        ``"kernel"`` uses ``computeKernelImage``, giving an origin-centered
        stamp suitable as a convolution kernel. ``"image"`` uses
        ``computeImage``, retaining pixel phase and placement for reprojection.
    diagnostics : `bool`
        Measure and attach second-moment shape diagnostics.

    Returns
    -------
    `NativePsfStamp`

    Raises
    ------
    RubinPsfError
        If the model is missing, raises, or produces an unusable stamp.
    """
    if mode not in ("kernel", "image"):
        raise ValueError(f"mode must be 'kernel' or 'image', got {mode!r}.")

    psf_model = _resolve_psf_model(source)
    if psf_model is None:
        raise RubinPsfError(
            "Exposure has no PSF model. KBMOD will not silently substitute a Gaussian; "
            "set the 'psf_fallback_std' config option to opt in to one."
        )

    method_name = "computeKernelImage" if mode == "kernel" else "computeImage"
    method = getattr(psf_model, method_name, None)
    if method is None:
        raise RubinPsfError(f"Exposure PSF model does not implement {method_name}.")

    try:
        rubin_image = method(_point2d(x, y))
    except RubinPsfError:
        raise
    except Exception as err:
        raise RubinPsfError(
            f"Rubin PSF model raised while evaluating {method_name} at ({x}, {y}): {err}"
        ) from err

    array = getattr(rubin_image, "array", None)
    if array is None:
        raise RubinPsfError(f"{method_name} returned an object without an 'array' attribute.")

    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2:
        raise RubinPsfError(f"Rubin PSF stamp must be 2D, got shape {array.shape}.")
    if array.size == 0:
        raise RubinPsfError("Rubin PSF stamp is empty.")
    if not np.all(np.isfinite(array)):
        raise RubinPsfError("Rubin PSF stamp contains non-finite values (NaN or inf).")

    height, width = array.shape
    if width != height:
        raise RubinPsfError(f"Rubin PSF stamp must be square, got shape {array.shape}.")
    if width % 2 == 0:
        # Padding an even stamp to odd would move the centroid by half a pixel,
        # silently biasing every position the kernel is used to measure. Refuse.
        raise RubinPsfError(
            f"Rubin PSF stamp has even width {width}; an odd width is required so the "
            "kernel has a well-defined center. Padding would shift the centroid by half "
            "a pixel."
        )

    peak = float(array.max())
    if peak <= 0.0:
        raise RubinPsfError("Rubin PSF stamp has no positive peak.")

    minimum = float(array.min())
    clipped_negative_sum = 0.0
    if minimum < 0.0:
        if minimum < -NEGATIVE_TOLERANCE * peak:
            raise RubinPsfError(
                f"Rubin PSF stamp has a materially negative pixel ({minimum:.6g}, peak {peak:.6g}). "
                "KBMOD search kernels must be non-negative, and clipping a real signed lobe would "
                "change the kernel's normalization and meaning. Investigate the model rather than "
                "flattening it."
            )
        clipped_negative_sum = float(-array[array < 0.0].sum())
        array = np.clip(array, 0.0, None)

    native_sum = float(array.sum())
    if native_sum <= 0.0:
        raise RubinPsfError(f"Rubin PSF stamp sums to {native_sum}; cannot normalize.")

    centroid_x, centroid_y = _measure_centroid(array)
    origin_x, origin_y = _stamp_origin(rubin_image, x, y, width, height)

    normalized = (array / native_sum).astype(np.float32)

    stamp = NativePsfStamp(
        array=normalized,
        origin_x=origin_x,
        origin_y=origin_y,
        eval_x=float(x),
        eval_y=float(y),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        native_sum=native_sum,
        provenance=method_name,
        clipped_negative_sum=clipped_negative_sum,
    )

    if diagnostics:
        from kbmod.psf_reprojection import measure_moments, summarize_shape

        moments = measure_moments(normalized)
        shape = summarize_shape(moments.covariance)
        stamp.diagnostics = {
            "fwhm_major": shape.fwhm_major,
            "fwhm_minor": shape.fwhm_minor,
            "position_angle": shape.position_angle,
            "covariance": moments.covariance.tolist(),
        }

    return stamp


def average_position(source):
    """Return the model's average position as ``(x, y)``, or `None`.

    Rubin models expose ``getAveragePosition`` as the canonical reference point.
    Returns `None` when the model does not provide a usable one, so the caller
    can fall back to the exposure center deliberately rather than by accident.

    Accepts an ``Exposure`` or a bare PSF model, like `render_rubin_psf`.
    """
    psf_model = _resolve_psf_model(source)
    if psf_model is None:
        return None

    getter = getattr(psf_model, "getAveragePosition", None)
    if getter is None:
        return None

    try:
        position = getter()
        return float(position.getX()), float(position.getY())
    except Exception:
        return None
