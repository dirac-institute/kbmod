from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

# ImageInfo is a helper class that wraps basic data extracted from a
# FITS (Flexible Image Transport System) file.
class ImageInfo:
    def __init__(self):
        self.obs_loc_set = False
        self.epoch_set_ = False
        self.wcs = None
        self.center = None
        self.obs_code = ""

    def populate_from_fits_file(self, filename):
        """
        Read the file stats information from a FITS file.

        Arguments:
            filename : string
                The path and name of the FITS file.
        """
        self.filename = filename
        with fits.open(filename) as hdu_list:
            self.wcs = WCS(hdu_list[1].header)
            self.width = hdu_list[1].header["NAXIS1"]
            self.height = hdu_list[1].header["NAXIS2"]

            if "DATE-AVG" in hdu_list[0].header:
                self.epoch_ = Time(hdu_list[0].header["DATE-AVG"], format="isot")
                self.epoch_set_ = True

            # Extract information about the location of the observatory.
            # Since this doesn't seem to be standardized, we try some
            # documented versions.
            if "OBSERVAT" in hdu_list[0].header:
                self.obs_code = hdu_list[0].header["OBSERVAT"]
                self.obs_loc_set = True
            elif "OBS-LAT" in hdu_list[0].header:
                self.obs_lat = float(hdu_list[0].header["OBS-LAT"])
                self.obs_long = float(hdu_list[0].header["OBS-LONG"])
                self.obs_alt = float(hdu_list[0].header["OBS-ELEV"])
                self.obs_loc_set = True
            elif "LAT_OBS" in hdu_list[0].header:
                self.obs_lat = float(hdu_list[0].header["LAT_OBS"])
                self.obs_long = float(hdu_list[0].header["LONG_OBS"])
                self.obs_alt = float(hdu_list[0].header["ALT_OBS"])
                self.obs_loc_set = True
            else:
                self.obs_loc_set = False

            # Compute the center of the image in sky coordinates.
            self.center = self.wcs.pixel_to_world(self.width / 2, self.height / 2)

    def set_obs_code(self, obs_code):
        """
        Manually set the observatory code.

        Arguments:
            obs_code : string
               The Observatory code.
        """
        self.obs_code = obs_code
        self.obs_loc_set = True

    def set_obs_position(self, lat, long, alt):
        """
        Manually set the observatory location and clear
        the observatory code.

        Arguments:
            lat : float - Observatory latitude.
            long : float - Observatory longitude.
            alt : float - Observatory altitude.
        """
        self.obs_code = ""
        self.obs_lat = lat
        self.obs_long = long
        self.obs_alt = alt
        self.obs_loc_set = True

    def set_epoch(self, epoch):
        """
        Manually set the epoch for this image.

        Arguments:
            epoch : astropy Time object.
        """
        self.epoch_ = epoch
        self.epoch_set_ = True

    def get_epoch(self):
        """
        Get the epoch for this image.

        Returns:
            epoch : astropy Time object.
        """
        assert self.epoch_set_
        return self.epoch_

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

    def ra_radius(self):
        edge = self.wcs.pixel_to_world(0.0, self.height / 2)
        radius = self.center.separation(edge)
        return radius

    def dec_radius(self):
        edge = self.wcs.pixel_to_world(self.width / 2, 0.0)
        radius = self.center.separation(edge)
        return radius


class ImageInfoSet:
    def __init__(self):
        self.stats = []
        self.num_images = 0

    def set_times_mjd(self, mjd):
        """
        Manually sets the image times.

        Arguments:
            mjd : List of floats
                Image times in MJD.
        """
        assert len(mjd) == self.num_images
        for i in range(self.num_images):
            self.stats[i].set_epoch(Time(mjd[i], format="mjd"))

    def get_image_mjd(self, index):
        """
        Return the MJD of a single image.

        Argument:
            index : integer
                The index of the image.

        Returns:
            float : timestamp in MJD.
        """
        self.stats[index].get_epoch().mjd

    def get_all_mjd(self):
        """
        Returns a list of all times in mjd.
        """
        return [self.stats[i].get_epoch().mjd for i in range(self.num_images)]

    def get_duration(self):
        """
        Returns the difference in times between the first and last image.

        Returns:
            float : difference in times (JD or MJD).
        """
        return self.stats[-1].get_epoch().mjd - self.stats[0].get_epoch().mjd

    def get_zero_shifted_times(self):
        """
        Returns a list of timestamps such that the first image
        is at time 0.

        Returns:
            List of floats : zero-shifted times (JD or MJD).
        """
        first = self.stats[0].get_epoch().mjd
        mjds = [(self.stats[i].get_epoch().mjd - first) for i in range(self.num_images)]
        return mjds

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
        FILES filenames.

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
        assert self.num_images == len(pos)

        results = []
        for i in range(self.num_images):
            results.append(self.stats[i].pixels_to_skycoords(pos[i]))
        return results
