import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from kbmod.image_info import *
from kbmod.search import *


def create_fake_fits_header_file(fname, ra, dec, x_dim, y_dim, id_str=None):
    """Create a primary HDU with the date/time and observatory info in the header and
    optionally save the image information.

    Parameters
    ----------
    fname : `str`
        The filename of the FITs file to create.
    ra : `float`
        The RA location of the (0, 0) pixel.
    dec : `float`
        The dec location of the (0, 0) pixel.
    x_dim : `int`
        The width of the image in pixels.
    y_dim : `int`
        The height of the image in pixels.
    id_str : `str`, optional
        The ID string for the image.
    """
    hdr0 = fits.Header()
    hdr0["DATE-AVG"] = "2022-08-15T06:00:00.000000000"
    hdr0["OBS-LAT"] = -30.166060
    hdr0["OBS-LONG"] = 70.814890
    hdr0["OBS-ELEV"] = 2215.000000
    if id_str is not None:
        hdr0["IDNUM"] = id_str
    hdu0 = fits.PrimaryHDU(header=hdr0)

    # Create and image HDU with a header containing
    # minimal celestial information for the WCS.
    data1 = np.ones((y_dim, x_dim))
    hdr1 = fits.Header()
    hdr1["WCSAXES"] = 2

    # (0,0) corner is at RA=201.614 and Dec=-10.788
    # with 0.001 degrees per pixel.
    hdr1["CRPIX1"] = 1.0
    hdr1["CRVAL1"] = ra
    hdr1["CDELT1"] = 0.001
    hdr1["CTYPE1"] = "RA"

    hdr1["CRPIX2"] = 1.0
    hdr1["CRVAL2"] = dec
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
        self.assertIsNone(img_info.wcs)
        self.assertIsNone(img_info.center)
        self.assertEqual(img_info.obs_code, "")

        # Check that get_epoch raises an error or returns
        # None depending on the settings.
        with self.assertRaises(ValueError):
            _ = img_info.get_epoch()
        self.assertIsNone(img_info.get_epoch(none_if_unset=True))

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

    def test_load_image(self):
        img_info = ImageInfo()
        img_info.populate_from_fits_file("./data/fake_images/000000.fits", load_image=True, p=psf(1.0))
        self.assertIsNotNone(img_info.image)
        self.assertEqual(img_info.image.get_width(), 64)
        self.assertEqual(img_info.image.get_height(), 64)

    def test_load_image_no_psf(self):
        img_info = ImageInfo()
        with self.assertRaises(ValueError):
            img_info.populate_from_fits_file("./data/fake_images/000000.fits", load_image=True)

    def test_load_files(self):
        with tempfile.TemporaryDirectory() as dir_name:
            # Create two fake files in the temporary directory.
            fname1 = "%s/tmp1.fits" % dir_name
            fname2 = "%s/tmp2.fits" % dir_name
            create_fake_fits_header_file(fname1, 201.614, -10.788, 20, 30, id_str="00002")
            create_fake_fits_header_file(fname2, 201.614, -10.788, 20, 30, id_str="00003")

            # Load the fake files into an ImageInfoSet.
            img_info = ImageInfoSet()
            img_info.load_image_info_from_files([fname1, fname2])

            # Check that we go the correct filename and visit ID.
            self.assertEqual(img_info.stats[0].filename, fname1)
            self.assertEqual(img_info.stats[1].filename, fname2)
            self.assertEqual(img_info.stats[0].visit_id, "00002")
            self.assertEqual(img_info.stats[1].visit_id, "00003")

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

            # Test that we can map the sky positions back to the coordinates.
            pixel_pos00 = img_info.stats[0].skycoords_to_pixels(sky_pos00)
            self.assertAlmostEqual(pixel_pos00.x, 0.0)
            self.assertAlmostEqual(pixel_pos00.y, 0.0)

            pixel_pos2 = img_info.stats[0].skycoords_to_pixels(sky_pos2)
            self.assertAlmostEqual(pixel_pos2.x, 10.0)
            self.assertAlmostEqual(pixel_pos2.y, 20.0)

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

            # Create three fake files in the temporary directory.
            fname1 = f"{dir_name}/data/00001.fits"
            fname2 = f"{dir_name}/data/00002.fits"
            fname3 = f"{dir_name}/data/00005.fits"
            create_fake_fits_header_file(fname1, 201.614, -10.788, 20, 30)
            create_fake_fits_header_file(fname2, 201.614, -10.788, 20, 30)
            create_fake_fits_header_file(fname3, 201.614, -10.788, 20, 30)

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
                file.write(f"00001 59804.25\n")
                file.write(f"00002 59805.25\n")
                file.write(f"00003 59805.75\n")  # No FITS file

            # Load the updated times.
            img_info.load_times_from_file(time_file)

            # Check that we have extracted the image time information.
            times = img_info.get_all_mjd()
            self.assertEqual(len(times), 3)
            self.assertAlmostEqual(times[0], 59804.25)
            self.assertAlmostEqual(times[1], 59805.25)
            self.assertAlmostEqual(times[2], 59806.25)  # Time not overwritten

    def test_load_and_sample(self):
        w = 100
        h = 120

        with tempfile.TemporaryDirectory() as dir_name:
            # Create the fake header file and load it.
            fname1 = "%s/tmp1.fits" % dir_name
            create_fake_fits_header_file(fname1, 180.0, 15.0, w, h, id_str="00001")
            img_info = ImageInfo()
            img_info.populate_from_fits_file(fname1)

            # Create a fake image to use for the sampling.
            sci = raw_image(w, h)
            for y in range(h):
                for x in range(w):
                    sci.set_pixel(x, y, x + y / float(h))

            var = raw_image(w, h)
            var.set_all(1.0)

            msk = raw_image(w, h)
            msk.set_all(0.0)

            image = layered_image(sci, var, msk, 1.0, psf(1.0))
            img_info.set_layered_image(image)

            # Sample the image centered on pixel (5, 10) into a 21 x 21 stamp.
            # Use the fact we know each pixel is 0.001 degrees to choose the angular width.
            pos = pixel_pos()
            pos.x = 5
            pos.y = 10
            sky_pos = img_info.pixels_to_skycoords(pos)
            sub_image = img_info.make_resampled_aligned_image(sky_pos, 0.001 * 3600.0 * 10, 10)

            # Check that we recover the correctly sampled science image, including going off the edge of
            # the image (and getting NO_DATA).
            new_sci = sub_image.get_science()
            self.assertEqual(new_sci.get_width(), 21)
            self.assertEqual(new_sci.get_height(), 21)
            for y in range(21):
                for x in range(21):
                    self.assertAlmostEqual(new_sci.get_pixel(x, y), sci.get_pixel(5 + x - 11, 10 + y - 11))


if __name__ == "__main__":
    unittest.main()
