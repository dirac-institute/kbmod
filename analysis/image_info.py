from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

# ImageInfo is a helper class tha wraps basic data extracted from a
# fits image file.
class ImageInfo():
    def __init__(self):
        pass
    
    def populate_from_fits_file(self, filename):
        """
        Read the file stats information from a fits file.
        
        Arguments:
            filename : string
                The path and name of the fits file.
        """
        self.filename = filename
        with fits.open(filename) as hdulist:
            self.wcs = WCS(hdulist[1].header)
            self.width = hdulist[1].header["NAXIS1"]
            self.height = hdulist[1].header["NAXIS2"]
            self.epoch = Time(hdulist[0].header["DATE-AVG"], format='isot')

            # Compute the center of the image in sky coordinates.
            self.center = self.wcs.pixel_to_world(self.width/2,
                                                  self.height/2)

    def pixels_to_skycoords(self, pos):
        """
        Transform the pixel position within an image
        to a SkyCoord.
        
        Arguments:
            pos : pixel_pos
                A pixel_pos object containing the x and y
                coordinates on the pixel.

        Returns:
            A SkyCoord with the transformed location.
        """
        return self.wcs.pixel_to_world(pos.x, pos.y)

    def approximate_radius(self):
        """
        Compute an approximate radius of the image.
        
        Arguments: NONE

        Returns:
            A radius in degrees.
        """
        corner = self.wcs.pixel_to_world(0.0, 0.0)
        radius = self.center.separation(corner)
        return radius
    
class ImageInfoSet():
    def __init__(self):
        self.stats = []
        self.num_images = 0
        self.mjd = []

    def get_file_epoch_times(self):
        """
        Returns a list of all times.
        """
        return [self.stats[i].epoch for i in range(self.num_images)]
    
    def get_x_size(self):
        """
        Returns the x_size from the first image.
        """
        if self.num_images == 0:
            return 0
        return self.stats[0].width

    def get_y_size(self):
        """
        Returns the y_size from the first image.
        """
        if self.num_images == 0:
            return 0
        return self.stats[0].height

    def load_image_info_from_files(self, filenames):
        """
        Fills an ImageInfoSet from a list of
        fits filenames.

        Arguments:
           filenames : A list of strings
           A list of filenames (including paths) for the
           fits files.
        """
        self.stats = []
        self.num_images = len(filenames)
        for f in filenames:
            s = ImageInfo()
            s.populate_from_fits_file(f)
            self.stats.append(s)
            
    def pixels_to_skycoords(self, pos):
        """
        Transform the pixel positions to SkyCoords.
        
        Arguments:
            pos : a list of pixel_pos objects

        Returns:
            A list of SkyCoords with the transformed locations.
        """
        assert(self.num_images == len(pos))
        
        results = []
        for i in range(self.num_images):
            results.append(self.stats[i].pixels_to_skycoords(pos[i]))
        return results