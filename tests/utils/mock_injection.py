"""Mock implementations of lsst.source.injection types for testing
kbmod.injection.inject_sources_into_ic without requiring the full LSST
Science Pipelines stack.

The MockVisitInjectTask stamps tiny 2D Gaussian PSFs into the exposure's
image array at the catalog (ra, dec) positions, using the exposure's WCS
to convert sky coordinates to pixel positions.
"""

import copy
from dataclasses import dataclass, field
from unittest import mock

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

__all__ = [
    "MockVisitInjectConfig",
    "MockVisitInjectTask",
    "MockVisitInjectResult",
]


class MockVisitInjectConfig:
    """Drop-in replacement for ``lsst.source.injection.VisitInjectConfig``."""

    pass


@dataclass
class MockVisitInjectResult:
    """Container matching the ``lsst.pipe.base.Struct`` returned by
    ``VisitInjectTask.run``."""

    output_exposure: object = None
    output_catalog: Table = field(default_factory=Table)


def _stamp_gaussian(image_array, cx, cy, flux, sigma=1.5):
    """Stamp a tiny normalised 2D Gaussian into *image_array* at (cx, cy).

    Parameters
    ----------
    image_array : ndarray
        2-D array to modify **in place**.
    cx, cy : float
        Centre of the Gaussian in pixel coordinates.
    flux : float
        Total integrated flux of the Gaussian.
    sigma : float
        Standard deviation in pixels.
    """
    hw = int(4 * sigma)  # half-width of stamp
    iy, ix = int(round(cy)), int(round(cx))
    ylo, yhi = max(0, iy - hw), min(image_array.shape[0], iy + hw + 1)
    xlo, xhi = max(0, ix - hw), min(image_array.shape[1], ix + hw + 1)
    if ylo >= yhi or xlo >= xhi:
        return  # completely off-image

    yy, xx = np.mgrid[ylo:yhi, xlo:xhi]
    stamp = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma**2)
    norm = stamp.sum()
    if norm > 0:
        stamp *= flux / norm
    image_array[ylo:yhi, xlo:xhi] += stamp.astype(image_array.dtype)


class MockVisitInjectTask:
    """Drop-in replacement for ``lsst.source.injection.VisitInjectTask``.

    Stamps tiny Gaussians into the image array at the catalog positions.
    Determines pixel positions by interpreting the exposure's WCS mock
    (``wcs.getFitsMetadata()`` → astropy WCS → ``world_to_pixel``).
    """

    def __init__(self, config=None):
        self.config = config or MockVisitInjectConfig()

    def run(self, injection_catalogs, input_exposure, psf=None, photo_calib=None, wcs=None):
        """Inject sources from *injection_catalogs* into *input_exposure*.

        Parameters
        ----------
        injection_catalogs : `astropy.table.Table`
            Must contain ``ra``, ``dec``, ``mag`` columns.
        input_exposure : mock Exposure
            Must have ``image.array`` (ndarray) and ``wcs``.
        psf, photo_calib, wcs : object
            Accepted for API compatibility; ``wcs`` from the exposure is used.

        Returns
        -------
        result : MockVisitInjectResult

        Raises
        ------
        RuntimeError
            If no sources fall within the image bounds (mirrors real behaviour).
        """
        if wcs is None:
            wcs = input_exposure.wcs

        # Build an astropy WCS from the mock SkyWcs header
        try:
            hdr = wcs.getFitsMetadata()
            astropy_wcs = WCS(hdr)
        except Exception:
            astropy_wcs = None

        out_exposure = copy.deepcopy(input_exposure)
        injected_count = 0

        for row in injection_catalogs:
            ra, dec, mag = row["ra"], row["dec"], row["mag"]
            # Convert magnitude to a simple flux proxy (arbitrary zero-point)
            flux = 10 ** ((25.0 - mag) / 2.5)

            if astropy_wcs is not None:
                from astropy.coordinates import SkyCoord

                coord = SkyCoord(ra, dec, unit="deg")
                px, py = astropy_wcs.world_to_pixel(coord)
            else:
                # Fallback: treat ra/dec as pixel coords (for very simple mocks)
                px, py = ra, dec

            _stamp_gaussian(out_exposure.image.array, float(px), float(py), flux)
            injected_count += 1

        if injected_count == 0:
            raise RuntimeError("No sources were injected within bounds.")

        return MockVisitInjectResult(
            output_exposure=out_exposure,
            output_catalog=injection_catalogs,
        )
