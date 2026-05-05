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
from astropy.coordinates import SkyCoord
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


def _stamp_gaussian(image_array, x_center, y_center, flux, sigma=1.5):
    """Stamp a tiny normalised 2D Gaussian into *image_array* at (x_center, y_center).

    Modifies the image_array in place.

    Parameters
    ----------
    image_array : ndarray
        2-D array to modify **in place**.
    x_center, y_center : float
        Centre of the Gaussian in pixel coordinates.
    flux : float
        Total integrated flux of the Gaussian.
    sigma : float
        Standard deviation in pixels.

    Returns
    -------
    bool
        True if the stamp was applied (at least partially in-bounds), False otherwise.
    """
    # Use the boundaries of the image array to determine the bounds of the stamp
    hw = int(4 * sigma)  # half-width of stamp
    y_center_int, x_center_int = int(round(y_center)), int(round(x_center))
    y_low, y_high = max(0, y_center_int - hw), min(image_array.shape[0], y_center_int + hw + 1)
    x_low, x_high = max(0, x_center_int - hw), min(image_array.shape[1], x_center_int + hw + 1)
    if y_low >= y_high or x_low >= x_high:
        return False  # completely off-image

    yy, xx = np.mgrid[y_low:y_high, x_low:x_high]
    # Here we create a 2D Gaussian PSF and stamp it into the image array
    stamp = np.exp(-0.5 * ((xx - x_center) ** 2 + (yy - y_center) ** 2) / sigma**2)
    norm = stamp.sum()
    if norm > 0:
        stamp *= flux / norm
    image_array[y_low:y_high, x_low:x_high] += stamp.astype(image_array.dtype)
    return True


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
                coord = SkyCoord(ra, dec, unit="deg")
                px, py = astropy_wcs.world_to_pixel(coord)
            else:
                # Fallback: treat ra/dec as pixel coords (for very simple mocks)
                px, py = ra, dec

            if _stamp_gaussian(out_exposure.image.array, float(px), float(py), flux):
                injected_count += 1

        if injected_count == 0:
            raise RuntimeError("No sources were injected within bounds.")

        return MockVisitInjectResult(
            output_exposure=out_exposure,
            output_catalog=injection_catalogs,
        )
