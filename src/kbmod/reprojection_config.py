"""Explicit, serializable configuration for KBMOD's adaptive reprojection.

KBMOD reprojects science, variance, and mask planes with
`reproject.reproject_adaptive`. Historically it passed only two options
explicitly (``bad_value_mode`` and ``roundtrip_coords``) and inherited the rest
from the installed ``reproject`` release. That is a scientific hazard for two
reasons:

1. The inherited values materially change the result. ``conserve_flux``,
   ``kernel_width``, ``sample_region_width``, ``boundary_mode`` and the Jacobian
   options all alter the interpolation and therefore the effective PSF, the
   recovered flux, and the noise correlation of the output.
2. Library defaults drift between releases, so the numbers a run produces are
   not reproducible from the KBMOD source alone.

This module makes every numerically relevant option explicit and records it as
run provenance. The effective PSF must be produced by the *same* operator, with
the *same* options, as the science image, so a single configuration object is
shared by both paths rather than each re-specifying its own arguments.

Only options that affect the numerical result are modeled here. Plumbing
arguments (``shape_out``, ``output_array``, ``output_footprint``, ``block_size``,
``parallel``, ...) stay at the call site.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import reproject

__all__ = [
    "AdaptiveReprojectionConfig",
    "LEGACY_CONFIG",
    "CONSERVE_FLUX_CONFIG",
    "PRESETS",
]


@dataclass(frozen=True)
class AdaptiveReprojectionConfig:
    """Every ``reproject_adaptive`` option that changes the numerical result.

    The field defaults reproduce KBMOD's historical behavior exactly, so
    ``AdaptiveReprojectionConfig()`` is the legacy operator. Two of them
    (``bad_value_mode``, ``roundtrip_coords``) deliberately differ from the
    ``reproject`` library defaults because KBMOD has always overridden them; see
    the notes on those fields before "correcting" either one.

    Attributes
    ----------
    conserve_flux : `bool`
        Scale samples by the Jacobian determinant so total flux, rather than
        surface brightness, is preserved. ``False`` is the historical KBMOD
        behavior and remains the default until the Phase 0 flux audit selects a
        production value on the evidence. Do not flip this casually: it changes
        photometry, and it composes with the zeropoint scaling that
        `ButlerStandardizer` already applies to science and variance.
    kernel : `str`
        Interpolation kernel, ``"gaussian"`` or ``"hann"``.
    kernel_width : `float`
        Gaussian kernel width in output pixels. Ignored for the Hann kernel.
    sample_region_width : `float`
        Width of the sampling region in output pixels. Documented by
        ``reproject`` as a double, so a float is equivalent to the library's
        integer default.
    center_jacobian : `bool`
        Recompute the Jacobian at each pixel center rather than interpolating
        it across the output. More accurate for strongly varying transforms and
        considerably slower.
    despike_jacobian : `bool`
        Suppress spurious Jacobian spikes, e.g. at a projection's poles or
        wrap points.
    boundary_mode : `str`
        Handling of samples falling outside the input image: ``"strict"``,
        ``"constant"``, ``"grow"``, or ``"ignore"``. ``"strict"`` can clip a PSF
        stamp or yield NaN, which must be reported rather than normalized away.
    boundary_fill_value : `float`
        Fill value used when ``boundary_mode="constant"``.
    boundary_ignore_threshold : `float`
        Fraction of a sampling region permitted to fall outside the input before
        the output pixel is rejected, for ``boundary_mode="ignore"``.
    bad_value_mode : `str`
        Handling of NaN/inf input samples. KBMOD uses ``"ignore"`` rather than
        the library's ``"strict"``. Note that ignoring bad values renormalizes
        the local interpolation weights, which perturbs both flux and noise near
        masks; regions within a PSF radius of a mask deserve separate treatment.
    bad_fill_value : `float`
        Fill value used when ``bad_value_mode="constant"``.
    roundtrip_coords : `bool`
        Verify coordinate transforms by round-tripping them. KBMOD uses
        ``False`` where the library defaults to ``True``. **The legacy preset
        must keep ``False``** — restoring the library default here would change
        existing results.
    x_cyclic, y_cyclic : `bool`
        Treat the respective axis as wrapping. False for KBMOD's tangent-plane
        work.
    """

    conserve_flux: bool = False
    kernel: str = "gaussian"
    kernel_width: float = 1.3
    sample_region_width: float = 4.0
    center_jacobian: bool = False
    despike_jacobian: bool = False
    boundary_mode: str = "strict"
    boundary_fill_value: float = 0.0
    boundary_ignore_threshold: float = 0.5
    bad_value_mode: str = "ignore"
    bad_fill_value: float = 0.0
    roundtrip_coords: bool = False
    x_cyclic: bool = False
    y_cyclic: bool = False

    def as_kwargs(self):
        """Return the options as keyword arguments for `reproject_adaptive`.

        Returns
        -------
        `dict`
            A fresh dictionary, safe for the caller to mutate or extend with
            call-site plumbing such as ``shape_out``.
        """
        return asdict(self)

    def evolve(self, **changes):
        """Return a new config with ``changes`` applied.

        Named ``evolve`` rather than ``replace`` so it does not read as an
        in-place mutation of a frozen object.

        Returns
        -------
        `AdaptiveReprojectionConfig`
        """
        return replace(self, **changes)

    @property
    def provenance(self):
        """Return a JSON-serializable record of this configuration.

        Includes the ``reproject`` version, because identical options can still
        produce different numbers across library releases.

        Returns
        -------
        `dict`
        """
        record = {"reproject_version": reproject.__version__, "config_hash": self.hexdigest}
        record.update(asdict(self))
        return record

    @property
    def hexdigest(self):
        """Return a short stable hash of the options and `reproject` version.

        Suitable as a cache key and as a compact provenance tag in metadata.

        Returns
        -------
        `str`
            The first 16 characters of a SHA-256 hex digest.
        """
        payload = {"reproject_version": reproject.__version__, "options": asdict(self)}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def preset_name(self):
        """Return the name of the matching preset, or ``"custom"``.

        Returns
        -------
        `str`
        """
        for name, preset in PRESETS.items():
            if self == preset:
                return name
        return "custom"


#: KBMOD's historical operator. Reproduces the behavior of every release up to
#: and including the commit that introduced this module. This is the default.
LEGACY_CONFIG = AdaptiveReprojectionConfig()

#: Flux-conserving variant, the leading candidate for point-source photometry.
#: Selected as production only on the evidence of the Phase 0 flux audit.
CONSERVE_FLUX_CONFIG = LEGACY_CONFIG.evolve(conserve_flux=True)

PRESETS = {
    "legacy": LEGACY_CONFIG,
    "conserve_flux": CONSERVE_FLUX_CONFIG,
}
