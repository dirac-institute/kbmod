from astropy.io import fits
from astropy.wcs import WCS
from image_info import *
from kbmod import *
import tempfile
import unittest
import numpy as np
        
def create_fake_fits_file(fname, x_dim, y_dim):
    # Create a primary HDU with just the date/time in the header.
    hdr0 = fits.Header()
    hdr0['DATE-AVG'] = '2022-08-15T06:00:00.000000000'
    hdu0 = fits.PrimaryHDU(header=hdr0)

    # Create and image HDU with a header containing
    # minimal celestial information for the WCS.
    data1 = np.ones((y_dim, x_dim))
    hdr1 = fits.Header()
    hdr1['WCSAXES'] = 2

    # (0,0) corner is at RA=201.614 and Dec=-10.788
    # with 0.001 degrees per pixel.
    hdr1['CRPIX1'] = 1.0
    hdr1['CRVAL1'] = 201.614
    hdr1['CDELT1'] = 0.001
    hdr1['CTYPE1'] = 'RA'

    hdr1['CRPIX2'] = 1.0
    hdr1['CRVAL2'] = -10.788
    hdr1['CDELT2'] = 0.001
    hdr1['CTYPE2'] = 'DEC'
    hdu1 = fits.ImageHDU(data1, header=hdr1)
    
    # Write both HDUs to the given file.
    h = fits.HDUList([hdu0, hdu1])
    h.writeto(fname, overwrite=True)
    

class test_image_info(unittest.TestCase):

    def test_load_files(self):
        with tempfile.TemporaryDirectory() as dir_name:
            # Create two fake files in the temporary directory.
            fname1 = ('%s/tmp1.fits' % dir_name)
            fname2 = ('%s/tmp2.fits' % dir_name)
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
            times = img_info.get_file_epoch_times()
            self.assertEqual(len(times), 2)
            self.assertAlmostEqual(times[0].mjd, 59806.25)
            self.assertAlmostEqual(times[1].mjd, 59806.25)
            self.assertEqual(len(img_info.mjd), 0)

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
            
            
if __name__ == '__main__':
   unittest.main()

