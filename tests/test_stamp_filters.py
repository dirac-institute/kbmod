import unittest

from kbmod.filters.stamp_filters import *
from kbmod.result_list import *
from kbmod.search import *


class test_stamp_filters(unittest.TestCase):
    def setUp(self):
        self.times = [(10.0 + 0.1 * float(i)) for i in range(20)]
        self.num_times = len(self.times)

    def _create_row(self, stamp):
        row = ResultRow(trajectory(), self.num_times)
        row.stamp = np.array(stamp.get_all_pixels())
        return row

    def test_peak_filtering(self):
        stamp = raw_image(11, 11)
        stamp.set_all(1.0)
        stamp.set_pixel(3, 4, 10.0)
        row = self._create_row(stamp)

        self.assertTrue(StampPeakFilter(5, 5, 5).keep_row(row))
        self.assertTrue(StampPeakFilter(5, 3, 2).keep_row(row))
        self.assertFalse(StampPeakFilter(5, 2, 1).keep_row(row))
        self.assertFalse(StampPeakFilter(5, 3, 1).keep_row(row))
        self.assertFalse(StampPeakFilter(5, 2, 2).keep_row(row))

    def test_peak_filtering_furthest(self):
        stamp = raw_image(9, 9)
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


if __name__ == "__main__":
    unittest.main()
