import numpy as np
import unittest

from kbmod.configuration import SearchConfiguration
from kbmod.fake_data_creator import add_fake_object, FakeDataSet
from kbmod.filters.stamp_filters import *
from kbmod.result_list import *
from kbmod.search import *


class test_stamp_filters(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

    def _create_row(self, stamp):
        row = ResultRow(Trajectory(), self.num_times)
        row.stamp = stamp.image
        return row

    def test_peak_filtering_name(self):
        self.assertEqual(StampPeakFilter(5, 1.0, 2.0).get_filter_name(), "StampPeakFilter_1.0_2.0")

    def test_skip_invalid_stamp(self):
        # No stamp
        row = ResultRow(Trajectory(), self.num_times)
        self.assertFalse(StampPeakFilter(5, 100, 100).keep_row(row))

        # Wrong sized stamp
        stamp = RawImage(5, 5)
        row = self._create_row(stamp)
        self.assertFalse(StampPeakFilter(5, 100, 100).keep_row(row))

    def test_peak_filtering(self):
        stamp = RawImage(11, 11)
        stamp.set_all(1.0)
        stamp.set_pixel(3, 4, 10.0)
        row = self._create_row(stamp)

        self.assertTrue(StampPeakFilter(5, 5, 5).keep_row(row))
        self.assertTrue(StampPeakFilter(5, 3, 2).keep_row(row))
        self.assertFalse(StampPeakFilter(5, 2, 1).keep_row(row))
        self.assertFalse(StampPeakFilter(5, 3, 1).keep_row(row))
        self.assertFalse(StampPeakFilter(5, 2, 2).keep_row(row))

    def test_peak_filtering_furthest(self):
        stamp = RawImage(9, 9)
        stamp.set_all(1.0)
        stamp.set_pixel(3, 4, 10.0)
        stamp.set_pixel(8, 1, 10.0)  # Use furthest from center.
        row = self._create_row(stamp)

        self.assertTrue(StampPeakFilter(4, 5, 5).keep_row(row))
        self.assertTrue(StampPeakFilter(4, 5, 4).keep_row(row))
        self.assertFalse(StampPeakFilter(4, 3, 2).keep_row(row))
        self.assertFalse(StampPeakFilter(4, 2, 1).keep_row(row))
        self.assertFalse(StampPeakFilter(4, 4, 4).keep_row(row))
        self.assertFalse(StampPeakFilter(4, 3, 5).keep_row(row))

    def test_central_moments_filtering_name(self):
        self.assertEqual(
            StampMomentsFilter(5, 1.0, 2.0, 3.0, 4.0, 5.0).get_filter_name(),
            "StampMomentsFilter_m01_1.0_m10_2.0_m11_3.0_m02_4.0_m20_5.0",
        )

    def test_central_moments_filtering(self):
        stamp = RawImage(11, 11)
        stamp.set_all(0.0)
        row = self._create_row(stamp)

        # An empty image should have zero moments.
        self.assertTrue(StampMomentsFilter(5, 0.001, 0.001, 0.001, 0.001, 0.001).keep_row(row))

        # A single peak pixel should have zero moments.
        stamp.set_pixel(5, 5, 10.0)
        row = self._create_row(stamp)
        self.assertTrue(StampMomentsFilter(5, 0.001, 0.001, 0.001, 0.001, 0.001).keep_row(row))

        # A symmetric stamop should have zero first order moments and
        # non-zero second order moments.
        stamp.set_pixel(5, 4, 5.0)
        stamp.set_pixel(4, 5, 5.0)
        stamp.set_pixel(5, 6, 5.0)
        stamp.set_pixel(6, 5, 5.0)
        row = self._create_row(stamp)
        self.assertFalse(StampMomentsFilter(5, 0.001, 0.001, 0.001, 0.001, 0.001).keep_row(row))
        self.assertTrue(StampMomentsFilter(5, 0.001, 0.001, 0.001, 0.5, 0.5).keep_row(row))

        # A non symmetric stamp will not pass the filter.
        stamp.set_pixel(0, 0, 50.0)
        stamp.set_pixel(3, 0, 10.0)
        stamp.set_pixel(1, 2, 50.0)
        row = self._create_row(stamp)
        self.assertFalse(StampMomentsFilter(5, 1.0, 1.0, 1.0, 1.0, 1.0).keep_row(row))

    def test_center_filtering_name(self):
        self.assertEqual(
            StampCenterFilter(5, True, 0.05).get_filter_name(),
            "StampCenterFilter_True_0.05",
        )

    def test_center_filtering(self):
        stamp = RawImage(7, 7)
        stamp.set_all(0.0)
        row = self._create_row(stamp)

        # An empty image should fail.
        self.assertFalse(StampCenterFilter(3, True, 0.01).keep_row(row))

        # No local maxima (should fail).
        stamp = RawImage(7, 7)
        stamp.set_all(0.01)
        row = self._create_row(stamp)
        self.assertFalse(StampCenterFilter(3, True, 0.01).keep_row(row))

        # Single strong central peak
        stamp = RawImage(7, 7)
        stamp.set_all(0.05)
        stamp.set_pixel(3, 3, 10.0)
        row = self._create_row(stamp)
        self.assertTrue(StampCenterFilter(3, True, 0.5).keep_row(row))
        self.assertFalse(StampCenterFilter(3, True, 1.0).keep_row(row))

        # Less than 50% in the center pixel.
        stamp = RawImage(7, 7)
        stamp.set_all(0.05)
        stamp.set_pixel(3, 3, 10.0)
        stamp.set_pixel(3, 4, 9.0)
        row = self._create_row(stamp)
        self.assertFalse(StampCenterFilter(3, True, 0.5).keep_row(row))
        self.assertTrue(StampCenterFilter(3, True, 0.4).keep_row(row))

        # Center is not the maximum value.
        stamp = RawImage(7, 7)
        stamp.set_all(0.05)
        stamp.set_pixel(1, 2, 10.0)
        stamp.set_pixel(3, 3, 9.0)
        row = self._create_row(stamp)
        self.assertFalse(StampCenterFilter(3, True, 0.5).keep_row(row))
        self.assertFalse(StampCenterFilter(3, True, 0.4).keep_row(row))
        self.assertTrue(StampCenterFilter(3, False, 0.2).keep_row(row))

    def test_extract_search_parameters_from_config(self):
        config_dict = {
            "center_thresh": 0.05,
            "do_stamp_filter": True,
            "mom_lims": [50.0, 51.0, 1.0, 2.0, 3.0],
            "peak_offset": [1.5, 3.5],
            "stamp_type": "median",
            "stamp_radius": 7,
        }
        config = SearchConfiguration.from_dict(config_dict)

        params = extract_search_parameters_from_config(config)
        self.assertEqual(params.radius, 7)
        self.assertEqual(params.stamp_type, StampType.STAMP_MEDIAN)
        self.assertEqual(params.do_filtering, True)
        self.assertAlmostEqual(params.center_thresh, 0.05)
        self.assertAlmostEqual(params.peak_offset_x, 1.5)
        self.assertAlmostEqual(params.peak_offset_y, 3.5)
        self.assertAlmostEqual(params.m20_limit, 50.0)
        self.assertAlmostEqual(params.m02_limit, 51.0)
        self.assertAlmostEqual(params.m11_limit, 1.0)
        self.assertAlmostEqual(params.m10_limit, 2.0)
        self.assertAlmostEqual(params.m01_limit, 3.0)

        # Test bad configurations
        config.set("stamp_radius", -1)
        self.assertRaises(ValueError, extract_search_parameters_from_config, config)
        config.set("stamp_radius", 7)

        config.set("stamp_type", "broken")
        self.assertRaises(ValueError, extract_search_parameters_from_config, config)
        config.set("stamp_type", "median")

        config.set("peak_offset", [50.0])
        self.assertRaises(ValueError, extract_search_parameters_from_config, config)
        config.set("peak_offset", [1.5, 3.5])

        config.set("mom_lims", [50.0, 51.0, 1.0, 3.0])
        self.assertRaises(ValueError, extract_search_parameters_from_config, config)
        config.set("mom_lims", [50.0, 51.0, 1.0, 2.0, 3.0])

    @unittest.skipIf(not HAS_GPU, "Skipping test (no GPU detected)")
    def test_get_coadds_and_filter(self):
        image_count = 10
        ds = FakeDataSet(
            25,  # width
            35,  # height
            image_count,  # time steps
            1.0,  # noise level
            0.5,  # psf value
            1,  # observations per day
            True,  # Use a fixed seed for testing
        )

        # Insert a single fake object with known parameters.
        trj = make_trajectory(8, 7, 2.0, 1.0, flux=250.0)
        ds.insert_object(trj)

        # Second Trajectory that isn't any good.
        trj2 = make_trajectory(1, 1, 0.0, 0.0)

        # Third Trajectory that is close to good, but offset.
        trj3 = make_trajectory(trj.x + 2, trj.y + 2, trj.vx, trj.vy)

        # Create a fourth Trajectory that is just close enough
        trj4 = make_trajectory(trj.x + 1, trj.y + 1, trj.vx, trj.vy)

        # Create the ResultList.
        keep = ResultList(ds.times)
        keep.append_result(ResultRow(trj, image_count))
        keep.append_result(ResultRow(trj2, image_count))
        keep.append_result(ResultRow(trj3, image_count))
        keep.append_result(ResultRow(trj4, image_count))

        # Create the stamp parameters we need.
        config_dict = {
            "center_thresh": 0.03,
            "do_stamp_filter": True,
            "mom_lims": [35.5, 35.5, 1.0, 1.0, 1.0],
            "peak_offset": [1.5, 1.5],
            "stamp_type": "cpp_mean",
            "stamp_radius": 5,
        }
        config = SearchConfiguration.from_dict(config_dict)

        # Do the filtering.
        get_coadds_and_filter(keep, ds.stack, config, chunk_size=1, debug=False)

        # The check that the correct indices and number of stamps are saved.
        self.assertEqual(keep.num_results(), 2)
        self.assertEqual(keep.results[0].trajectory.x, trj.x)
        self.assertEqual(keep.results[1].trajectory.x, trj.x + 1)
        self.assertIsNotNone(keep.results[0].stamp)
        self.assertIsNotNone(keep.results[1].stamp)

    def test_append_all_stamps(self):
        image_count = 10
        ds = FakeDataSet(
            25,  # width
            35,  # height
            image_count,  # time steps
            1.0,  # noise level
            0.5,  # psf value
            1,  # observations per day
            True,  # Use a fixed seed for testing
        )

        # Make a few results with different trajectories.
        keep = ResultList(ds.times)
        keep.append_result(ResultRow(make_trajectory(8, 7, 2.0, 1.0), image_count))
        keep.append_result(ResultRow(make_trajectory(10, 22, -2.0, -1.0), image_count))
        keep.append_result(ResultRow(make_trajectory(8, 7, -2.0, -1.0), image_count))

        append_all_stamps(keep, ds.stack, 5)
        for row in keep.results:
            self.assertIsNotNone(row.all_stamps)
            self.assertEqual(len(row.all_stamps), image_count)
            for i in range(image_count):
                self.assertEqual(np.shape(row.all_stamps[i])[0], 11)
                self.assertEqual(np.shape(row.all_stamps[i])[1], 11)


if __name__ == "__main__":
    unittest.main()
