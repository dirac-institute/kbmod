"""Classes for working with the input files for KBMOD.

The ``ImageInfo`` class stores additional information for the
input FITS files that is used during a variety of analysis.
"""

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from kbmod.file_utils import FileUtils
from kbmod.search import pixel_pos, LayeredImage


# ImageInfo is a helper class that wraps basic data extracted from a
# FITS (Flexible Image Transport System) file.
class ImageInfo:
    def __init__(self):
        self.obs_loc_set = False
        self.epoch_set_ = False
        self.wcs = None
        self.center = None
        self.obs_code = ""
        self.filename = None
        self.visit_id = None
        self.image = None

    def populate_from_fits_file(self, filename, load_image=False, p=None):
        """Read the file stats information from a FITS file.

        Parameters
        ----------
        filename : string
            The path and name of the FITS file.
        load_image : bool
            Load the image data into a LayeredImage object.
        p : `PSF`
            The PSF for this layered image. Optional when load `load_image`
            is False. Otherwise this is required.
        """
        # Skip non-FITs files.
        if ".fits" not in filename:
            return

        # Load the image itself.
        if load_image:
            if p is None:
                raise ValueError("Loading image without a PSF.")
            self.image = LayeredImage(filename, p)

        self.filename = filename
        with fits.open(filename) as hdu_list:
            self.wcs = WCS(hdu_list[1].header)
            self.width = hdu_list[1].header["NAXIS1"]
            self.height = hdu_list[1].header["NAXIS2"]

            # If the visit ID is in header (using Rubin tags), use for the visit ID.
            # Otherwise extract it from the filename.
            if "IDNUM" in hdu_list[0].header:
                self.visit_id = str(hdu_list[0].header["IDNUM"])
            else:
                name = filename.rsplit("/")[-1]
                self.visit_id = FileUtils.visit_from_file_name(name)

            # Load the time. Try the "DATE-AVG" header entry first, then "MJD".
            if "DATE-AVG" in hdu_list[0].header:
                self.set_epoch(Time(hdu_list[0].header["DATE-AVG"], format="isot"))
            elif "MJD" in hdu_list[0].header:
                self.set_epoch(Time(hdu_list[0].header["MJD"], format="mjd", scale="utc"))

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

    def set_LayeredImage(self, image):
        """Manually set the layered image.

        Parameters
        ----------
        image : `LayeredImage`
            The layered image to use.
        """
        self.image = image

    def set_obs_code(self, obs_code):
        """Manually set the observatory code.

        Parameters
        ----------
        obs_code : string
            The Observatory code.
        """
        self.obs_code = obs_code
        self.obs_loc_set = True

    def set_obs_position(self, lat, long, alt):
        """Manually set the observatory location and clear
        the observatory code.

        Parameters
        ----------
        lat : float
            The observatory's latitude.
        long : float
            The observatory's longitude.
        alt : float
            The observatory's altitude.
        """
        self.obs_code = ""
        self.obs_lat = lat
        self.obs_long = long
        self.obs_alt = alt
        self.obs_loc_set = True

    def set_epoch(self, epoch):
        """Manually set the epoch for this image.

        Parameters
        ----------
        epoch : astropy Time object
            The new epoch.
        """
        self.epoch_ = Time(epoch)
        if self.image is not None:
            self.image.set_obstime(self.epoch_.mjd)
        self.epoch_set_ = True

    def get_epoch(self, none_if_unset=False):
        """Get the epoch for this image.

        Parameters
        ----------
        none_if_unset : bool
            A bool indicating that the function should return None
            if the epoch is not set.

        Returns
        -------
        epoch : astropy Time object.
        """
        if not self.epoch_set_:
            if none_if_unset:
                return None
            else:
                raise ValueError("Epoch unset.")
        return self.epoch_

    def skycoords_to_pixels(self, pos):
        """Transform sky coordinates to the pixel locations within the image.

        Parameters
        ----------
        pos : `SkyCoord`
            The location of the query.

        Returns
        -------
        result : `pixel_pos`
            A `pixel_pos` object with the (x, y) pixel location.
        """
        result = pixel_pos()
        result.x, result.y = self.wcs.world_to_pixel(pos)
        return result

    def pixels_to_skycoords(self, pos):
        """Transform the pixel position within an image to a SkyCoord.

        Parameters
        ----------
        pos : `pixel_pos`
            A `pixel_pos` object containing the x and y
            coordinates on the pixel.

        Returns
        -------
        `SkyCoord`
            A `SkyCoord` with the transformed location.
        """
        return self.wcs.pixel_to_world(pos.x, pos.y)


class ImageInfoSet:
    def __init__(self):
        self.stats = []
        self.num_images = 0

    def append(self, row):
        """Add an ImageInfo to the list.

        Parameters
        ----------
        row : ImageInfo
            The new ImageInfo to add.
        """
        self.stats.append(row)
        self.num_images += 1

    def set_times_mjd(self, mjd):
        """Manually sets the image times.

        Parameters
        ----------
        mjd : List of floats
            The image times in MJD.
        """
        if len(mjd) != self.num_images:
            raise ValueError(f"Incorrect number of times given. Expected {self.num_images}.")
        for i in range(self.num_images):
            self.stats[i].set_epoch(Time(mjd[i], format="mjd", scale="utc"))

    def load_times_from_file(self, time_file):
        """Load the image times from from an auxiliary file.

        The code works by matching the visit IDs in the time file
        with part of the file name. In order to be a match, the
        visit ID string must occur in the file name.

        Parameters
        ----------
        time_file : str
            The full path and filename of the times file.
        """
        image_time_dict = FileUtils.load_time_dictionary(time_file)

        # Check each visit ID against the dictionary.
        for img in self.stats:
            if img.visit_id is not None and img.visit_id in image_time_dict:
                mjd = image_time_dict[img.visit_id]
                img.set_epoch(Time(mjd, format="mjd", scale="utc"))

    def get_image_mjd(self, index):
        """Return the MJD of a single image.

        Parameters
        ----------
        index : int
            The index of the image.

        Returns
        -------
        float
            The timestamp in MJD.
        """
        self.stats[index].get_epoch().mjd

    def get_all_mjd(self):
        """Returns a list of all times in MJD.

        Returns
        -------
        list of floats
            A list of the images times in MJD.
        """
        return [self.stats[i].get_epoch().mjd for i in range(self.num_images)]

    def get_duration(self):
        """Returns the difference in times between the first and last image.

        Returns
        -------
            float : difference in times (JD or MJD).
        """
        return self.stats[-1].get_epoch().mjd - self.stats[0].get_epoch().mjd

    def get_zero_shifted_times(self):
        """Returns a list of timestamps such that the first image
        is at time 0.

        Returns
        -------
        List of floats
            A list of zero-shifted times (JD or MJD).
        """
        first = self.stats[0].get_epoch().mjd
        mjds = [(self.stats[i].get_epoch().mjd - first) for i in range(self.num_images)]
        return mjds

    def get_x_size(self):
        """Returns the x_size from the first image.

        Returns
        -------
        int
            The width of the first image.
        """
        if self.num_images == 0:
            return 0
        return self.stats[0].width

    def get_y_size(self):
        """
        Returns the y_size from the first image.

        Returns
        -------
        int
            The height of the first image.
        """
        if self.num_images == 0:
            return 0
        return self.stats[0].height

    def load_image_info_from_files(self, filenames):
        """Fills an `ImageInfoSet` from a list of FILES filenames.

        Parameters
        ----------
        filenames : A list of strings
           The list of filenames (including paths) for the FITS files.
        """
        self.stats = []
        for f in filenames:
            s = ImageInfo()
            s.populate_from_fits_file(f)
            self.append(s)

    def pixels_to_skycoords(self, pos):
        """Transform the pixel positions to SkyCoords.

        Parameters
        ----------
        pos : a list of `pixel_pos` objects
            The positions in pixel coordinates.

        Returns
        -------
        list of `SkyCoords`
            The transformed locations in (RA, Dec).
        """

        results = []
        for i in range(self.num_images):
            results.append(self.stats[i].pixels_to_skycoords(pos[i]))
        return results

    def trajectory_to_skycoords(self, trj):
        """Transform the trajectory into a list of SkyCoords
        for each time step.

        Parameters
        ----------
        trj: trajectory
            The trajectory struct with the object's initial position
            and velocities in pixel space.

        Returns
        -------
        list of `SkyCoords`
            The trajectory's (RA, Dec) coordinates at each time.
        """
        t0 = self.stats[0].get_epoch().mjd
        results = []
        for i in range(self.num_images):
            dt = self.stats[i].get_epoch().mjd - t0
            pos_x = trj.x + dt * trj.vx
            pos_y = trj.y + dt * trj.vy
            results.append(self.stats[i].wcs.pixel_to_world(pos_x, pos_y))
        return results
