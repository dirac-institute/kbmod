import unittest

import numpy as np
from utils.utils_for_tests import get_absolute_data_path

from kbmod.reprojection import reproject_work_unit
from kbmod.search import KB_NO_DATA
from kbmod.work_unit import ImageStack, WorkUnit


class test_reprojection(unittest.TestCase):
    def setUp(self):
        self.data_path = get_absolute_data_path("shifted_wcs_diff_dimms_tiled.fits")
        self.test_wunit = WorkUnit.from_fits(self.data_path)
        self.common_wcs = self.test_wunit.per_image_wcs[0]

    def test_reproject(self):
        reprojected_wunit = reproject_work_unit(self.test_wunit, self.common_wcs)

        assert reprojected_wunit.wcs != None
        assert reprojected_wunit.im_stack.get_width() == 60
        assert reprojected_wunit.im_stack.get_height() == 50

        images = reprojected_wunit.im_stack.get_images()

        # will be 3 as opposed to the four in the original `WorkUnit`,
        # as the last two images have the same obstime and therefore
        # get condensed to one image.
        assert len(images) == 3

        data = [[i.get_science().image, i.get_variance().image, i.get_mask().image] for i in images]

        for img in data:
            for i in img:
                assert not np.any(np.isnan(i))
            # test that mask values are binary
            assert np.all(np.array(img[2] == 1.0) | np.array(img[2] == 0.0))

        test_vals = np.array(
            [
                231.61615,
                113.59214,
                166.82635,
                KB_NO_DATA,
                4.0,
                1.0,
            ]
        ).astype("float32")
        # make sure the PSF for the object hasn't been warped
        # in the no-op case
        assert data[0][0][10][43] == test_vals[0]

        # test other object locations
        assert data[1][0][15][46] == test_vals[1]
        assert data[2][0][21][49] == test_vals[2]

        # test variance
        assert data[2][1][25][0] == test_vals[3]
        assert data[2][1][25][9] == test_vals[4]

        # test that mask values are projected without interpolation/bleeding
        assert np.all(data[2][2][35] == test_vals[5])
        assert np.all(data[2][2][9] == test_vals[5])
        assert len(data[2][2][36][data[2][2][36] == 1.0]) == 7
        assert len(data[2][2][34][data[2][2][34] == 1.0]) == 7

    def test_except_no_per_image_wcs(self):
        """Make sure we fail when we don't have all the provided WCS."""
        self.test_wunit.per_image_wcs = self.test_wunit.per_image_wcs[:-1]
        try:
            reproject_work_unit(self.test_wunit, self.common_wcs)
        except ValueError as e:
            assert str(e) == "per_image_wcs not provided for all WorkUnit"

    def test_except_add_overlapping_images(self):
        """Make sure that the reprojection fails when images at the same time
        have overlapping pixels."""
        images = self.test_wunit.im_stack.get_images()
        images[1].set_obstime(images[0].get_obstime())
        new_im_stack = ImageStack(images)
        self.test_wunit.im_stack = new_im_stack

        try:
            reproject_work_unit(self.test_wunit, self.common_wcs)
        except ValueError as e:
            assert str(e) == "Images with the same obstime are overlapping."
