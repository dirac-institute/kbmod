import unittest

import numpy as np

from kbmod.analysis.precovery_utils import make_stamps_from_emphems
from kbmod.fake_data.fake_data_creator import add_fake_object, FakeDataSet
from kbmod.search import PSF
from kbmod.wcs_utils import make_fake_wcs


class test_precovery_utils(unittest.TestCase):
    def test_make_stamps_from_emphems(self):
        fake_times = [0.0, 1.0, 2.0, 3.0]
        fake_wcs = make_fake_wcs(25.0, -31.0, 256, 256, 10.0 / 3600.0)
        ds = FakeDataSet(256, 256, fake_times)
        ds.set_wcs(fake_wcs)
        workunit = ds.get_work_unit()

        # Insert a fake object at a few points.
        # T=1.0, x=50, y=75
        sci_image1 = workunit.im_stack.get_single_image(1).get_science()
        add_fake_object(sci_image1, 50, 75, 500.0, psf=PSF(0.1))
        # T=2.0, x=75, y=85
        sci_image2 = workunit.im_stack.get_single_image(2).get_science()
        add_fake_object(sci_image2, 75, 85, 500.0, psf=PSF(0.1))

        # Create some query points.
        query_times = [0.0, 1.0, 1.5, 2.00001, 5.0, 6.0]
        query_x = [25, 50, 62.5, 75, 125, 150]
        query_y = [65, 75, 80, 85, 105, 115]

        # Transform the query pixel coordinates to RA and dec using the workunit's wcs.
        query_ra = []
        query_dec = []
        for qx, qy in zip(query_x, query_y):
            coord = fake_wcs.pixel_to_world(qx, qy)
            query_ra.append(coord.ra.deg)
            query_dec.append(coord.dec.deg)

        # Create the stamps. Only 3 of the times have a match in the data.
        times, stamps = make_stamps_from_emphems(query_times, query_ra, query_dec, workunit, radius=5)
        self.assertEqual(len(times), 3)
        self.assertTrue(np.allclose(times, [0.0, 1.0, 2.0]))

        self.assertEqual(len(stamps), 3)
        for stamp in stamps:
            self.assertEqual(stamp.width, 11)
            self.assertEqual(stamp.height, 11)

        # Two of the three stamps returned should have a bright spot in the middle.
        self.assertLess(stamps[0].get_pixel(5, 5), 50.0)
        self.assertGreater(stamps[1].get_pixel(5, 5), 50.0)
        self.assertGreater(stamps[2].get_pixel(5, 5), 50.0)


if __name__ == "__main__":
    unittest.main()
