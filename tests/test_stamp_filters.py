import unittest

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


if __name__ == "__main__":
    unittest.main()
