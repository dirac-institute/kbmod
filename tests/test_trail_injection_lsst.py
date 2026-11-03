"""Optional rendering regression; run in an LSST Science Pipelines environment.

Unlike catalog/mocking tests, this checks Rubin's actual trail normalization,
PSF convolution, and pixel-frame orientation with a rotated, reflected WCS.
"""

import unittest

import numpy as np
from astropy.table import Table

from kbmod.injection import HAS_LSST, prepare_injection_catalog

if HAS_LSST:
    from lsst.afw.geom import makeCdMatrix, makeSkyWcs
    from lsst.afw.image import ExposureF, PhotoCalib
    from lsst.geom import Point2D, SpherePoint, arcseconds, degrees
    from lsst.meas.algorithms import GaussianPsf
    from lsst.source.injection import VisitInjectConfig, VisitInjectTask


@unittest.skipUnless(HAS_LSST, "Requires LSST Science Pipelines and source_injection")
class TestTrailRendering(unittest.TestCase):
    def render(self, length=None, angle=0.0):
        exposure = ExposureF(129, 129)
        yy, xx = np.indices(exposure.image.array.shape)
        exposure.image.array[:] = 100.0 + xx + yy
        exposure.variance.array[:] = exposure.image.array + 10.0
        exposure.setPsf(GaussianPsf(31, 31, 1.5))
        exposure.setPhotoCalib(PhotoCalib(1.0))
        exposure.setWcs(
            makeSkyWcs(
                crpix=Point2D(64, 64),
                crval=SpherePoint(10.0, -20.0, degrees),
                cdMatrix=makeCdMatrix(scale=0.2 * arcseconds, orientation=27 * degrees, flipX=True),
            )
        )
        catalog = Table(
            {
                "ra": [10.0],
                "dec": [-20.0],
                "mag_total": [22.0],
                "source_type": ["Star" if length is None else "Trail"],
            }
        )
        if length is not None:
            catalog["trail_length"] = [length]
            catalog["beta"] = [angle]
        config = VisitInjectConfig()
        config.add_noise = False
        config.inject_variance = False
        original = exposure.image.array.copy()
        result = VisitInjectTask(config=config).run(
            injection_catalogs=[prepare_injection_catalog(catalog)],
            input_exposure=exposure,
            psf=exposure.getPsf(),
            photo_calib=exposure.getPhotoCalib(),
            wcs=exposure.getWcs(),
        )
        self.assertEqual(len(result.output_catalog), 1)
        self.assertEqual(result.output_catalog["injection_flag"][0], 0)
        image = result.output_exposure.image.array.astype(float) - original
        flux = image.sum()
        expected = exposure.getPhotoCalib().magnitudeToInstFlux(22.0, Point2D(64, 64))
        self.assertAlmostEqual(flux / expected, 1.0, delta=0.005)
        center = np.array([(image * xx).sum(), (image * yy).sum()]) / flux
        np.testing.assert_allclose(center, [64, 64], atol=0.02)
        dx, dy = xx - center[0], yy - center[1]
        moments = (
            np.array(
                [
                    [(image * dx**2).sum(), (image * dx * dy).sum()],
                    [(image * dx * dy).sum(), (image * dy**2).sum()],
                ]
            )
            / flux
        )
        return image, moments

    def test_trail_flux_length_orientation_and_point_source_limit(self):
        star, star_moments = self.render()
        for length, angle in [(0.1, 0), (10, 0), (20, 90), (20, 35)]:
            with self.subTest(length=length, angle=angle):
                image, moments = self.render(length, angle)
                direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                # A uniform line adds L^2/12 variance along its direction.
                expected = (length**2 / 12) * np.outer(direction, direction)
                np.testing.assert_allclose(moments - star_moments, expected, atol=0.1, rtol=0.02)
                if length == 0.1:
                    self.assertLess(np.abs(image - star).sum() / star.sum(), 0.01)


if __name__ == "__main__":
    unittest.main()
