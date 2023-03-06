import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from kbmod.image_info import *
from kbmod.search import *


def create_fake_fits_file(fname, x_dim, y_dim):
    # Create a primary HDU with just the date/time
    # and observatory info in the header.
    hdr0 = fits.Header()
    hdr0["DATE-AVG"] = "2022-08-15T06:00:00.000000000"
    hdr0["OBS-LAT"] = -30.166060
    hdr0["OBS-LONG"] = 70.814890
    hdr0["OBS-ELEV"] = 2215.000000
    hdu0 = fits.PrimaryHDU(header=hdr0)

    # Create and image HDU with a header containing
    # minimal celestial information for the WCS.
    data1 = np.ones((y_dim, x_dim))
    hdr1 = fits.Header()
    hdr1["WCSAXES"] = 2

    # (0,0) corner is at RA=201.614 and Dec=-10.788
    # with 0.001 degrees per pixel.
    hdr1["CRPIX1"] = 1.0
    hdr1["CRVAL1"] = 201.614
    hdr1["CDELT1"] = 0.001
    hdr1["CTYPE1"] = "RA"

    hdr1["CRPIX2"] = 1.0
    hdr1["CRVAL2"] = -10.788
    hdr1["CDELT2"] = 0.001
    hdr1["CTYPE2"] = "DEC"
    hdu1 = fits.ImageHDU(data1, header=hdr1)

    # Write both HDUs to the given file.
    h = fits.HDUList([hdu0, hdu1])
    h.writeto(fname, overwrite=True)


class test_image_info(unittest.TestCase):
    def test_unset(self):
        img_info = ImageInfo()
        self.assertEqual(img_info.obs_loc_set, False)
        self.assertEqual(img_info.wcs, None)
        self.assertEqual(img_info.center, None)
        self.assertEqual(img_info.obs_code, "")

    def test_set_obscode(self):
        img_info = ImageInfo()
        self.assertEqual(img_info.obs_loc_set, False)

        img_info.set_obs_code("568")
        self.assertEqual(img_info.obs_loc_set, True)
        self.assertEqual(img_info.obs_code, "568")

    def test_set_obs_position(self):
        img_info = ImageInfo()
        self.assertEqual(img_info.obs_loc_set, False)

        img_info.set_obs_position(-30.2, 70.8, 2000.0)
        self.assertEqual(img_info.obs_loc_set, True)
        self.assertEqual(img_info.obs_code, "")
        self.assertAlmostEqual(img_info.obs_lat, -30.2)
        self.assertAlmostEqual(img_info.obs_long, 70.8)
        self.assertAlmostEqual(img_info.obs_alt, 2000.0)

    def test_load_files(self):
        with tempfile.TemporaryDirectory() as dir_name:
            # Create two fake files in the temporary directory.
            fname1 = "%s/tmp1.fits" % dir_name
            fname2 = "%s/tmp2.fits" % dir_name
            create_fake_fits_file(fname1, 20, 30)
            create_fake_fits_file(fname2, 20, 30)

            # Load the fake files into an ImageInfoSet.
            img_info = ImageInfoSet()
            img_info.load_image_info_from_files([fname1, fname2])

            # Check that we have extracted the image size information.
            self.assertEqual(img_info.num_images, 2)
            self.assertEqual(img_info.get_x_size(), 20)
            self.assertEqual(img_info.get_y_size(), 30)

            # Check that we have extracted the image time information.
            times = img_info.get_all_mjd()
            self.assertEqual(len(times), 2)
            self.assertAlmostEqual(times[0], 59806.25)
            self.assertAlmostEqual(times[1], 59806.25)

            # Check that we can overwrite the times.
            img_info.set_times_mjd([59805.25, 59807.25])
            times2 = img_info.get_all_mjd()
            self.assertEqual(len(times2), 2)
            self.assertAlmostEqual(times2[0], 59805.25)
            self.assertAlmostEqual(times2[1], 59807.25)

            # Check the observatory's position.
            for i in range(img_info.num_images):
                self.assertEqual(img_info.stats[i].obs_loc_set, True)
                self.assertEqual(img_info.stats[i].obs_code, "")
                self.assertAlmostEqual(img_info.stats[i].obs_lat, -30.166060)
                self.assertAlmostEqual(img_info.stats[i].obs_long, 70.814890)
                self.assertAlmostEqual(img_info.stats[i].obs_alt, 2215.000000)

            # The (0, 0) pixel should be the same as the RA, DEC
            # provided in the fits header.
            pos00 = pixel_pos()
            pos00.x = 0.0
            pos00.y = 0.0
            sky_pos00 = img_info.stats[0].pixels_to_skycoords(pos00)
            self.assertAlmostEqual(sky_pos00.ra.degree, 201.614)
            self.assertAlmostEqual(sky_pos00.dec.degree, -10.788)

            # The (10, 20) pixel should be moved by 0.001 per pixel.
            pos2 = pixel_pos()
            pos2.x = 10.0
            pos2.y = 20.0
            sky_pos2 = img_info.stats[0].pixels_to_skycoords(pos2)
            self.assertAlmostEqual(sky_pos2.ra.degree, 201.624)
            self.assertAlmostEqual(sky_pos2.dec.degree, -10.768)

            # A trajectory of x=0, y=0, x_v=5.0, y_v=10.0 should produce
            # the same results as above.
            trj = trajectory()
            trj.x = 0
            trj.y = 0
            trj.x_v = 5.0
            trj.y_v = 10.0
            sky_pos_mult = img_info.trajectory_to_skycoords(trj)
            self.assertAlmostEqual(sky_pos_mult[0].ra.degree, 201.614)
            self.assertAlmostEqual(sky_pos_mult[0].dec.degree, -10.788)
            self.assertAlmostEqual(sky_pos_mult[1].ra.degree, 201.624)
            self.assertAlmostEqual(sky_pos_mult[1].dec.degree, -10.768)

    def test_load_files_with_time(self):
        with tempfile.TemporaryDirectory() as dir_name:
            os.mkdir(f"{dir_name}/data")

            # Create two fake files in the temporary directory.
            fname1 = f"{dir_name}/data/001.fits"
            fname2 = f"{dir_name}/data/002.fits"
            fname3 = f"{dir_name}/data/005.fits"
            create_fake_fits_file(fname1, 20, 30)
            create_fake_fits_file(fname2, 20, 30)
            create_fake_fits_file(fname3, 20, 30)

            # Load the fake files into an ImageInfoSet.
            img_info = ImageInfoSet()
            img_info.load_image_info_from_files([fname1, fname2, fname3])

            # Check that we have extracted the image time information.
            times = img_info.get_all_mjd()
            self.assertEqual(len(times), 3)
            self.assertAlmostEqual(times[0], 59806.25)
            self.assertAlmostEqual(times[1], 59806.25)
            self.assertAlmostEqual(times[2], 59806.25)

            # Create a fake time file with time stamps for 2 of the images.
            time_file = f"{dir_name}/times.dat"
            with open(time_file, "w") as file:
                file.write("# visit_id mean_julian_date\n")
                file.write(f"001 59804.25\n")
                file.write(f"002 59805.25\n")
                file.write(f"003 59805.75\n")  # No FITS file

            # Load the updated times.
            img_info.load_times_from_file(time_file)

            # Check that we have extracted the image time information.
            times = img_info.get_all_mjd()
            self.assertEqual(len(times), 3)
            self.assertAlmostEqual(times[0], 59804.25)
            self.assertAlmostEqual(times[1], 59805.25)
            self.assertAlmostEqual(times[2], 59806.25)  # Time not overwritten


if __name__ == "__main__":
    unittest.main()
