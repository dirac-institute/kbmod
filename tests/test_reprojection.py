import unittest
import numpy as np
from utils.utils_for_tests import get_absolute_data_path

from kbmod.reprojection import reproject_work_unit, _get_first_psf_at_time, _validate_original_wcs
from kbmod.search import pixel_value_valid
from kbmod.work_unit import ImageStack, WorkUnit


class test_reprojection(unittest.TestCase):
    def setUp(self):
        self.data_path = get_absolute_data_path("shifted_wcs_diff_dimms_tiled.fits")
        self.test_wunit = WorkUnit.from_fits(self.data_path)
        self.common_wcs = self.test_wunit.get_wcs(0)

    def test_reproject(self):
        for parallelize in [True, False]:
            with self.subTest(parallelize=parallelize):
                reprojected_wunit = reproject_work_unit(
                    self.test_wunit, self.common_wcs, parallelize=parallelize
                )

                assert reprojected_wunit.wcs != None
                assert reprojected_wunit.im_stack.get_width() == 60
                assert reprojected_wunit.im_stack.get_height() == 50
                assert reprojected_wunit.geocentric_distances == self.test_wunit.geocentric_distances
                images = reprojected_wunit.im_stack.get_images()

                # will be 3 as opposed to the four in the original `WorkUnit`,
                # as the last two images have the same obstime and therefore
                # get condensed to one image.
                assert len(images) == 3

                data = [[i.get_science().image, i.get_variance().image, i.get_mask().image] for i in images]

                for img in data:
                    # test that mask values are binary
                    assert np.all(np.array(img[2] == 1.0) | np.array(img[2] == 0.0))

                test_vals = np.array(
                    [
                        115.519264,
                        94.1921,
                        114.12677,
                        4.0,
                        1.0,
                    ]
                ).astype("float32")
                # make sure the PSF for the object hasn't been warped
                # in the no-op case
                assert data[0][0][5][53] == test_vals[0]

                # test other object locations
                assert data[1][0][30][36] == test_vals[1]
                assert data[2][0][4][18] == test_vals[2]

                # test variance
                assert not pixel_value_valid(data[2][1][25][0])
                assert data[2][1][25][9] == test_vals[3]

                # test that mask values are projected without interpolation/bleeding
                assert len(data[2][2][36][data[2][2][36] == 1.0]) == 9
                assert len(data[2][2][34][data[2][2][34] == 1.0]) == 9

                assert len(reprojected_wunit._per_image_indices) == 3
                assert reprojected_wunit._per_image_indices[2] == [2, 3]

    def test_except_add_overlapping_images(self):
        """Make sure that the reprojection fails when images at the same time
        have overlapping pixels."""
        images = self.test_wunit.im_stack.get_images()
        images[1].set_obstime(images[0].get_obstime())
        new_im_stack = ImageStack(images)
        self.test_wunit.im_stack = new_im_stack

        for parallelize in [True, False]:
            with self.subTest(parallelize=parallelize):
                try:
                    reproject_work_unit(self.test_wunit, self.common_wcs, parallelize=parallelize)
                except ValueError as e:
                    assert str(e) == "Images with the same obstime are overlapping."

    def test_get_first_psf_at_time(self):
        """Make sure that the expected PSF is returned for a given time."""
        obstimes = np.array(self.test_wunit.get_all_obstimes())
        psf = self.test_wunit.im_stack.get_images()[0].get_psf()

        _psf = _get_first_psf_at_time(self.test_wunit, obstimes[0])
        assert np.isclose(psf.get_sum(), _psf.get_sum())
        assert np.isclose(psf.get_size(), _psf.get_size())

    def test_get_first_psf_at_time_exception(self):
        """Make sure that an exception is raised when the obstime is not found."""
        obstimes = np.array(self.test_wunit.get_all_obstimes())
        time = obstimes[0] - 1
        try:
            _get_first_psf_at_time(self.test_wunit, time)
        except ValueError as e:
            assert str(e) == f"Observation time {time} not found in work unit."

    def test_validate_original_wcs(self):
        """Make sure that the original WCS is validated correctly."""
        wcs = self.test_wunit.get_wcs(0)
        _wcs = _validate_original_wcs(self.test_wunit, [0])
        assert np.all(wcs.pixel_scale_matrix == _wcs[0].pixel_scale_matrix)


if __name__ == "__main__":
    unittest.main()
