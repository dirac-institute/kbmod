"""Classes for working with the input files for KBMOD.

The ``ImageInfo`` class stores additional information for the
input FITS files that is used during a variety of analysis.
"""
import os
import glob

from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.wcs import WCS

from kbmod.file_utils import FileUtils
from kbmod.search import pixel_pos, layered_image
from kbmod.standardizer import Standardizer


# This is named ImageInfoSet to still merge in with the rest of the
# codebase. It doesn't have to be that and I'd preffer it didn't, the
# Hungarian notation is not neccessary thing to float up to UI
class ImageInfoSet:
    """A collection of basic pointing, file paths, names, timestamps
    and other metadata that facilitate, and make easier the
    construction of ImageStack and execution of KBMOD search.

    When constructed from one of the provided factory methods, methods
    named ``from*``, instantiates an interface that extracts the
    required metadata. 

    Parameters
    ----------
    metadata : `~astropy.table.Table`
        A table of exposure metadata properties
    
    Attributes
    ----------
    data : `~astropy.table.Table`
        Table of exposure metadata properties.   
    """
    def __init__(self, path=None, metadata=None, wcs=None, standardizers=None):
        # we don't allow metadata and path at the same time, path, i.e.
        # serialized ImageInfoSet object takes precendence when given
        if path is not None:
            self.__initFromPath(path)
        elif metadata is not None:
            self.__initFromTable(metadata)

        # It makes no sense forcing the existance of WCS and
        # standardizer objects when the thing is loaded from
        # hand-crafted metadata table, but we need it if we support
        # pixel_to_sky and vice-versa functionality. But this isn't
        # your everyday functionality that everyone would use in the
        # interface. This is a leftover from having to support the
        # previous interface format really. Ideally getting this better
        # removes these instances of nonsensicality, for example, why
        # does ImageInfoSet perform the functionality of ImageStack
        # here? ButlerInterface would probably not be able to return
        # a full WCS for example...
        self.wcs = None
        self.standardizers = None
        
        if wcs is not None:
            self.wcs = wcs
        if standardizers is not None:
            self.standardizers = standardizers
            if self.wcs is None:
                self.wcs = [std.wcs for std in self.standardizers]

    def __initFromPath(self, path):
        # pickle, json, yaml, whatever serializer, goes here
        raise NotImplemented()

    def __initFromTable(self, metadata):
        if isinstance(metadata, Table):
            self.data = metadata
        else:
            raise ValueError(
                "Expected an Table, got {type(metadata)} instead."
                "Use one of the constructors: `fromLocation`, `fromRows` or `fromColumns`"
            )
        if "process" not in self.data.columns:
            self.data.add_column([True]*len(self.data), name="process")
        
    @classmethod 
    def fromLocation(cls, location, **kwargs):
        """Instantiate a ImageInfoSet class from a collection of
        locations, file paths or URIs to FITS files.

        Parameters
        ----------
        locations : `str`, `list`, `tuple`, `array`
            Collection of file-paths, or URIs, to FITS files or a
            path to a directory of FITS files or a butler repository.
        recursive : `bool`
            Scan path to directory recursively in order to look for
            FITS files.
        **kwargs : `dict`
            Remaining kwargs, not listed here, are passed onwards to
            the underlying interface that parses the metadata from the
            location (f.e. `~Standardizer` or `~ButlerInterface`).

        Returns
        -------
        imageInfoSet : `~ImageInfoSet`
            `ImageInfoSet` object

        Raises
        ------
        ValueError: when location is not recognized as a file, directory
            or an URI
        """        
        # somewhere here we also need a check for an URI schema, or
        # punt it this whole thing to lsst.resources in which we don't
        # need all the hidden _fromURI/DIR/FLOCATION constructors
        if os.path.isdir(location):
            butler_config = os.path.join(location, "butler.yaml")
            if os.path.exists(butler_config):
                instance = cls._fromButlerRepo(location, **kwargs)
            instance = cls._fromDir(location, **kwargs)
        elif os.path.isfile(location) and "yaml" in location:
            instance = cls._fromButlerRepo(location, **kwargs)
        elif os.path.isfile(location):
            instance = cls._fromFilepaths([location, ])
        else:
            raise ValueError(
                f"Unrecognized local filesystem path: {location}"
            )

        return instance

    # these two could probably be merged, but I have to support
    # load_image_info_from_files right now so this is how to
    # monkeypatch this for now
    @classmethod
    def _fromURIs(cls, uris):
        """Instantiate ImageInfoSet from collection of URIs by parsing
        metadata from them, downloading the files as neccessary.
        """
        raise NotImplemented()

    @classmethod
    def _fromFilepaths(cls, paths, **kwargs):
        """Instantiate ImageInfoSet from collection of filesystem
        paths to FITS files, by parsing metadata from them.

        Parameters
        ----------
        path : `iterable`
            Collection of paths to fits files.
        """
        # parallelizable for large N_files
        standardizers = [Standardizer.fromFile(path, **kwargs) for path in paths]

        # The datatypes will not be an issue if we transition to using
        # dataclasses instead of dicts, but the Table instantiation
        # might be slower because we have to build a dict again. Good
        # thing would be that the unravelling of stdFits that have
        # multiple detectors into individual rows could be handled
        # much nicer. I am not happy with the monkeypatching of WCS
        # into this unravelling like is being done here, because
        # what if a standardizer can not guarantee a WCS? Also, IDs
        # get confusing, many duplicated rows with BBox being the
        # only clear distinction between them. I wonder if anyone has
        # a clever way of creating unique int IDs for these
        # collections?
        unravelledStdMetadata = []
        for stdFits in standardizers:
            stdMeta = stdFits.standardizeMetadata()
            # this is kind of silly and not good, means our table needs
            # more than the standardizer can provide - i.e. something's wrong
            # with the abstraction but I have confused myself and can not
            # resolve it untill I think about it more.
            # A P.S. - the issue is class functionality, see comments in init 
            for bbox, wcs in zip(stdMeta["bbox"], stdFits.wcs):
                row = {}
                row["wcs"] = wcs
                row["bbox"] = bbox
                for key in stdMeta.keys():
                    if key != "bbox":
                        row[key] = stdMeta[key]
                unravelledStdMetadata.append(row)
                
        metadata = Table(rows=unravelledStdMetadata)
        return cls(metadata=metadata, standardizers=standardizers)

    @classmethod
    def _fromDir(cls, path, recursive=False, **kwargs):
        """Instantiate ImageInfoSet from a path to a directory
        containing FITS files. 

        Parameters
        ----------
        path : `str`
            Path to directory containing fits files.
        """
        # imagine only dir of FITS files
        fits_files = glob.glob(os.path.join(path, "*fits*"), recursive=recursive)
        return cls._fromFilepaths(fits_files, **kwargs)

    def _fromButlerRepo(self, path, **kwargs):
        # imagine only a butler.yaml
        raise NotImplemented()

    def __getitem__(self, key):
        # ideally we'd return ImageInfo here to be compatible with the
        # codebase but like, that class only recognizes the existance
        # of files and all it really does is enumerate properties.
        return self.data[key]

    def __str__(self):
        cleanCols = list(self.data.columns)
        cleanCols.remove("wcs")
        cleanCols.remove("bbox")
        cleanCols.remove("mjd")
        return str(self.data[cleanCols])

    def __repr__(self):
        cleanCols = list(self.data.columns)
        cleanCols.remove("wcs")
        cleanCols.remove("bbox")
        cleanCols.remove("mjd")
        #indicate in at least some way it's not an astropy Table obj
        return repr(self.data[cleanCols]).replace("Table", "ImageInfoSet")

    def __len__(self):
        return len(self.data)

    def append(self, metadata):
        """Append a row, or a table of metadata."""
        # this could still be useful to remove row-type related
        # annoyances from the equation. A perfect case for pattern
        # matching in the wild(!), sadly it's only supported by >=3.10
        if isinstance(metadata, Table):
            self.data = vstack(self.data, metadata)
        elif isinstance(metadata, (tuple, list)):
            # check if we're adding a single row or a collection
            # let astropy table complain if it can't
            if len(metadata) == len(self.data.columns):
                self.data.add_row(metadata)
            else:
                self.data = vstack(self.data, Table(metadata))
        elif isinstance(metadata, ImageInfo):
            # I guess for transition we could provide a ImageInfo
            # dictifier here, I'm just hardcoding this in now, I'd get
            # rid of it. If we need simple property enumeration class
            # probably we should look into dataclasses
            
            m = metadata
            width, height = metadata.image.shape
            t = m.get_epoch()
            center = wcs.pixel_to_world(centerX, centerY)
            corner = wcs.pixel_to_world(0, 0)

            self.data.add_row([
                width,
                height,
                m.visit_id,
                t,
                t.mjd,
                m.obs_code,
                m.obs_lat,
                m.obs_long,
                m.obs_alt,
                center.ra,
                center.dec,
                corner.ra,
                corner.dec
            ])
        else:
            raise ValueError("Can't append row, unknown type")

    def get_zero_shifted_times(self):
        """Returns a list of timestamps such that the first image
        is at time 0.

        Returns
        -------
        List of floats
            A list of zero-shifted times (JD or MJD).
        """
        # what's the deal here - are we required to be sorted?
        return self["mjd"] - self["mjd"][0]
        pass

    def get_duration(self):
        # maybe timespan?
        return self["mjd"][-1] - self["mjd"][0]

###
#    I would get rid of the setters/getters from this point on
#    this is just backwards compatibility because I don't want to
#    redo all the code
##  
    def set_times_mjd(self, mjd):
        self["mjd"] = mjd
    
    def load_times_from_file(self, time_file):
        # we need to get rid of this, I hate it :)
        # this is faster than checking for each individual image
        times = Table(time_file, format="ascii")
        for row in times:
            self[row["visit_id"]] = row["timestamp"]

    def get_image_mjd(self, index):
        return self["mjd"][index]
    
    def get_all_mjd(self):
        return self["mjd"]
        
    def get_x_size(self):
        """Returns the x_size from the first image.

        Returns
        -------
        int
            The width of the first image.
        """
        return self["width"][0]
    
    def get_y_size(self):
        """
        Returns the y_size from the first image.

        Returns
        -------
        int
            The height of the first image.
        """
        return self["height"][0]
    
    def load_image_info_from_files(self, filenames):
        """Fills an `ImageInfoSet` from a list of FILES filenames.

        Parameters
        ----------
        filenames : A list of strings
           The list of filenames (including paths) for the FITS files.
        """
        return self._fromFilepaths(filename)
    
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
        if "wcs" not in self.data.columns:
            raise ValueError("Metadata does not contain a WCS column")
        return (wcs.pixel_to_world(pos.x, pos.y) for wcs in self["wcs"])

    
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
        if "wcs" not in self.data.columns:
            raise ValueError("Metadata does not contain a WCS column")
        return (wcs.world_to_pixel(pos.x, pos.y) for wcs in self["wcs"])



#####
#    Remove everything past this point
##### 

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
        p : `psf`
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
            self.image = layered_image(filename, p)

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

    def set_layered_image(self, image):
        """Manually set the layered image.

        Parameters
        ----------
        image : `layered_image`
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
            self.image.set_time(self.epoch_.mjd)
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

# Left for implementation reference
#class ImageInfoSet:
#    def __init__(self):
#        self.stats = []
#        self.num_images = 0
#
#    def append(self, row):
#        """Add an ImageInfo to the list.
#
#        Parameters
#        ----------
#        row : ImageInfo
#            The new ImageInfo to add.
#        """
#        self.stats.append(row)
#        self.num_images += 1
#
#    def set_times_mjd(self, mjd):
#        """Manually sets the image times.
#
#        Parameters
#        ----------
#        mjd : List of floats
#            The image times in MJD.
#        """
#        if len(mjd) != self.num_images:
#            raise ValueError(f"Incorrect number of times given. Expected {self.num_images}.")
#        for i in range(self.num_images):
#            self.stats[i].set_epoch(Time(mjd[i], format="mjd", scale="utc"))
#
#    def load_times_from_file(self, time_file):
#        """Load the image times from from an auxiliary file.
#
#        The code works by matching the visit IDs in the time file
#        with part of the file name. In order to be a match, the
#        visit ID string must occur in the file name.
#
#        Parameters
#        ----------
#        time_file : str
#            The full path and filename of the times file.
#        """
#        image_time_dict = FileUtils.load_time_dictionary(time_file)
#
#        # Check each visit ID against the dictionary.
#        for img in self.stats:
#            if img.visit_id is not None and img.visit_id in image_time_dict:
#                mjd = image_time_dict[img.visit_id]
#                img.set_epoch(Time(mjd, format="mjd", scale="utc"))
#
#    def get_image_mjd(self, index):
#        """Return the MJD of a single image.
#
#        Parameters
#        ----------
#        index : int
#            The index of the image.
#
#        Returns
#        -------
#        float
#            The timestamp in MJD.
#        """
#        self.stats[index].get_epoch().mjd
#
#    def get_all_mjd(self):
#        """Returns a list of all times in MJD.
#
#        Returns
#        -------
#        list of floats
#            A list of the images times in MJD.
#        """
#        return [self.stats[i].get_epoch().mjd for i in range(self.num_images)]
#
#    def get_duration(self):
#        """Returns the difference in times between the first and last image.
#
#        Returns
#        -------
#            float : difference in times (JD or MJD).
#        """
#        return self.stats[-1].get_epoch().mjd - self.stats[0].get_epoch().mjd
#
#    def get_zero_shifted_times(self):
#        """Returns a list of timestamps such that the first image
#        is at time 0.
#
#        Returns
#        -------
#        List of floats
#            A list of zero-shifted times (JD or MJD).
#        """
#        first = self.stats[0].get_epoch().mjd
#        mjds = [(self.stats[i].get_epoch().mjd - first) for i in range(self.num_images)]
#        return mjds
#
#    def get_x_size(self):
#        """Returns the x_size from the first image.
#
#        Returns
#        -------
#        int
#            The width of the first image.
#        """
#        if self.num_images == 0:
#            return 0
#        return self.stats[0].width
#
#    def get_y_size(self):
#        """
#        Returns the y_size from the first image.
#
#        Returns
#        -------
#        int
#            The height of the first image.
#        """
#        if self.num_images == 0:
#            return 0
#        return self.stats[0].height
#
#    def load_image_info_from_files(self, filenames):
#        """Fills an `ImageInfoSet` from a list of FILES filenames.
#
#        Parameters
#        ----------
#        filenames : A list of strings
#           The list of filenames (including paths) for the FITS files.
#        """
#        self.stats = []
#        for f in filenames:
#            s = ImageInfo()
#            s.populate_from_fits_file(f)
#            self.append(s)
#
#    def pixels_to_skycoords(self, pos):
#        """Transform the pixel positions to SkyCoords.
#
#        Parameters
#        ----------
#        pos : a list of `pixel_pos` objects
#            The positions in pixel coordinates.
#
#        Returns
#        -------
#        list of `SkyCoords`
#            The transformed locations in (RA, Dec).
#        """
#
#        results = []
#        for i in range(self.num_images):
#            results.append(self.stats[i].pixels_to_skycoords(pos[i]))
#        return results
#
#    def trajectory_to_skycoords(self, trj):
#        """Transform the trajectory into a list of SkyCoords
#        for each time step.
#
#        Parameters
#        ----------
#        trj: trajectory
#            The trajectory struct with the object's initial position
#            and velocities in pixel space.
#
#        Returns
#        -------
#        list of `SkyCoords`
#            The trajectory's (RA, Dec) coordinates at each time.
#        """
#        t0 = self.stats[0].get_epoch().mjd
#        results = []
#        for i in range(self.num_images):
#            dt = self.stats[i].get_epoch().mjd - t0
#            pos_x = trj.x + dt * trj.x_v
#            pos_y = trj.y + dt * trj.y_v
#            results.append(self.stats[i].wcs.pixel_to_world(pos_x, pos_y))
#        return results
