import unittest
import numpy as np
from utils.utils_for_tests import get_absolute_data_path
import tempfile

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.reprojection import (
    reproject_work_unit,
    _get_first_psf_at_time,
    _validate_original_wcs,
)
from kbmod.search import pixel_value_valid
from kbmod.work_unit import WorkUnit


class test_reprojection(unittest.TestCase):
    def setUp(self):
        self.data_path = get_absolute_data_path("shifted_wcs_diff_dimms_tiled.fits")
        self.test_wunit = WorkUnit.from_fits(self.data_path, show_progress=False)
        self.common_wcs = self.test_wunit.get_wcs(0)

        # Set the data_loc metadata to make sure it propagates correctly.
        self.num_org_images = len(self.test_wunit.im_stack)
        self.data_locs = [f"test_data_loc_{i}" for i in range(self.num_org_images)]
        self.test_wunit.org_img_meta["data_loc"] = self.data_locs

    def test_reproject(self):
        # test exception conditions
        self.assertRaises(
            ValueError,
            reproject_work_unit,
            work_unit=self.test_wunit,
            common_wcs=self.common_wcs,
            write_output=True,
            show_progress=False,
        )

        self.test_wunit.lazy = True
        self.assertRaises(
            ValueError,
            reproject_work_unit,
            work_unit=self.test_wunit,
            common_wcs=self.common_wcs,
            show_progress=False,
        )
        self.test_wunit.lazy = False

        test_conditions = [
            (True, False, True),
            (True, False, True),
            (True, True, True),
            (False, False, True),
            (False, False, False),
        ]
        for parallelize, lazy, write_out in test_conditions:
            with self.subTest(parallelize=parallelize, lazy=lazy, write_out=write_out):
                if write_out:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        if lazy:
                            self.test_wunit.to_sharded_fits("test_wunit.fits", tmpdir)
                            wunit = WorkUnit.from_sharded_fits("test_wunit.fits", tmpdir, lazy=True)
                        else:
                            wunit = self.test_wunit
                        reproject_work_unit(
                            wunit,
                            self.common_wcs,
                            "original",
                            parallelize=parallelize,
                            write_output=write_out,
                            directory=tmpdir,
                            filename="repr_wu.fits",
                            show_progress=False,
                        )
                        reprojected_wunit = WorkUnit.from_sharded_fits("repr_wu.fits", tmpdir)
                else:
                    reprojected_wunit = reproject_work_unit(
                        self.test_wunit,
                        self.common_wcs,
                        parallelize=parallelize,
                        show_progress=False,
                    )

                assert reprojected_wunit.wcs != None
                assert reprojected_wunit.im_stack.width == 60
                assert reprojected_wunit.im_stack.height == 50

                test_dists = self.test_wunit.get_constituent_meta("geocentric_distance")
                reproject_dists = reprojected_wunit.get_constituent_meta("geocentric_distance")
                assert test_dists == reproject_dists

                # Make sure the data_loc metadata is propagated correctly.
                loaded_data_locs = reprojected_wunit.get_constituent_meta("data_loc")
                for i in range(self.num_org_images):
                    assert loaded_data_locs[i] == self.data_locs[i]

                # will be 3 as opposed to the four in the original `WorkUnit`,
                # as the last two images have the same obstime and therefore
                # get condensed to one image.
                assert len(reprojected_wunit.im_stack) == 3
                data = [
                    [
                        reprojected_wunit.im_stack.sci[i],
                        reprojected_wunit.im_stack.var[i],
                        reprojected_wunit.im_stack.get_mask(i),
                    ]
                    for i in range(3)
                ]

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

                # Make sure the PSF for the object hasn't been warped in the no-op case.
                # We allow a little error in case the result is compressed as it is written
                # to a file.
                self.assertAlmostEqual(data[0][0][5][53], test_vals[0], delta=0.05)

                # test other object locations
                self.assertAlmostEqual(data[1][0][30][36], test_vals[1], delta=0.05)
                self.assertAlmostEqual(data[2][0][4][18], test_vals[2], delta=0.05)

                # test variance
                assert not pixel_value_valid(data[2][1][25][0])
                self.assertAlmostEqual(data[2][1][25][9], test_vals[3], delta=0.05)

                # test that mask values are projected without interpolation/bleeding
                assert len(data[2][2][36][data[2][2][36] == 1.0]) == 9
                assert len(data[2][2][34][data[2][2][34] == 1.0]) == 9

                assert len(reprojected_wunit._per_image_indices) == 3
                assert reprojected_wunit._per_image_indices[2] == [2, 3]

    def test_except_add_overlapping_images(self):
        """Make sure that the reprojection fails when images at the same time
        have overlapping pixels."""
        new_times = np.copy(self.test_wunit.im_stack.times)
        new_times[1] = new_times[0]
        new_stack = ImageStackPy(
            new_times,
            self.test_wunit.im_stack.sci,
            self.test_wunit.im_stack.var,
            psfs=self.test_wunit.im_stack.psfs,
        )
        self.test_wunit.im_stack = new_stack

        for parallelize in [True, False]:
            with self.subTest(parallelize=parallelize):
                try:
                    reproject_work_unit(
                        self.test_wunit,
                        self.common_wcs,
                        parallelize=parallelize,
                        show_progress=False,
                    )
                except ValueError as e:
                    assert str(e) == "Images with the same obstime are overlapping."

    def test_get_first_psf_at_time(self):
        """Make sure that the expected PSF is returned for a given time."""
        obstimes = np.array(self.test_wunit.get_all_obstimes())
        psf = self.test_wunit.im_stack.psfs[0]

        _psf = _get_first_psf_at_time(self.test_wunit, obstimes[0])
        assert np.allclose(psf, _psf)

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
