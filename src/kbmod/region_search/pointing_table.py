"""A class for loading, storing and querying the pointings data."""

import astropy.units as u
from astropy.coordinates import get_body, get_sun, SkyCoord
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time
import numpy as np

from kbmod.region_search.geometry import ang2unitvec, angular_distance


class PointingTable:
    """PointingTable is a wrapper around astropy's Table that enforces
    required columns, name remapping, and a variety of specific query functions.

    Note: The current code builds PointingTable from a CSV, but we can extend this
    to read from a sqllite database, Butler, parquet files, etc.

    Attributes
    ----------
    pointings: `astropy.table.Table`
        The table of all the pointings.
    """

    def __init__(self, pointings):
        self.pointings = pointings

    @classmethod
    def from_dict(cls, data):
        """Create the pointings data from a dictionary.

        Parameters
        ----------
        data : `dict`
            The dictionary to use.
        """
        pointing_table = Table(data)
        return PointingData(pointing_table)

    @classmethod
    def from_csv(cls, filename, alt_names={}):
        """Read the pointings data from a CSV file.

        Parameters
        ----------
        filename : `str`
            The name of the CSV file.
        alt_names : dict
            A dictionary mapping final column names to known alternatives.
            Example: If we know the `ra` column could be called
            `RightAscension (deg)` or `RA (deg)` we would pass
            alt_names={"ra": ["RightAscension (deg)", "RA (deg)"]}
        """
        t = ascii.read(filename, format="csv")
        data = PointingData(t)
        data._validate_and_standardize(alt_names)
        return data

    def _check_and_rename_column(self, col_name, alt_names, required=True):
        """Check if the column is included using multiple possible names
        and renaming to a single canonical name.

        Parameters
        ----------
        col_name: str
            The canonical name we will use to represent the column.
        alt_names: list of str
            Others things a column could be called. The column will be
            renamed to `col_name` in place.
        required: bool
            Is the column required.

        Returns
        -------
        bool
            Whether the column is in the dataframe.

        Raises
        ------
        KeyError is the column is required and not present.
        """
        if col_name in self.pointings.columns:
            return True

        # Check if the column is using an alternate name and, if so, rename.
        for alt in alt_names:
            if alt in self.pointings.columns:
                self.pointings.rename_column(alt, col_name)
                return True

        if required:
            raise KeyError(f"Required column `{col_name}` missing.")
        return False

    def _validate_and_standardize(self, alt_names={}):
        """Validate that the data has the required columns expected operations.

        Parameters
        ----------
        alt_names : dict
            A dictionary mapping final column names to known alternatives.
            Example: If we know the `ra` column could be called
            `RightAscension (deg)` or `RA (deg)` we would pass
            alt_names={"ra": ["RightAscension (deg)", "RA (deg)"]}

        Raises
        ------
        KeyError is the column is required and not present.
        """
        self._check_and_rename_column("ra", alt_names.get("ra", ["RA"]), required=True)
        self._check_and_rename_column("dec", alt_names.get("dec", ["DEC", "Dec"]), required=True)
        self._check_and_rename_column(
            "obstime", alt_names.get("obstime", ["time", "mjd", "MJD"]), required=True
        )
        for key in alt_names.keys():
            self._check_and_rename_column(key, alt_names[key], required=False)

    def append_sun_pos(self, precise=False, recompute=False):
        """Compute an approximate position of the sun (relative to Earth) at
        each time and append this the table. Caches the result within the table
        so that this computation only needs to be performed once.

        Parameters
        ----------
        precise : `bool`
            Use the (slower) get_body() function instead of get_sun()'s polynomial
            approximation.
        recompute : `bool`
            If the column already exists, recompute it and overwrite.
        """
        if "sun_pos" in self.pointings.columns and not recompute:
            return

        # Compute and save the (RA, dec, dist) coordinates.
        times = Time(self.pointings["obstime"], format="mjd")
        if precise:
            sun_pos = get_body("sun", times)
        else:
            sun_pos = get_sun(times)
        self.pointings["sun_pos"] = sun_pos

        # Compute and save the geocentric cartesian coordinates in AU.
        (unit_x, unit_y, unit_z) = ang2unitvec(sun_pos.ra.degree, sun_pos.dec.degree)
        self.pointings["sun_vec"] = np.array(
            [
                unit_x * sun_pos.distance.value,
                unit_y * sun_pos.distance.value,
                unit_z * sun_pos.distance.value,
            ]
        ).T

    def append_unit_vector(self, recompute=False):
        """Add the unit vector to the pointing data. Caches the result within the table
        so that this computation only needs to be performed once.

        Parameters
        ----------
        recompute : `bool`
            If the column already exists, recompute it and overwrite.
        """
        if not "unit_vec" in self.pointings.columns or recompute:
            (x, y, z) = ang2unitvec(self.pointings["ra"].data, self.pointings["dec"].data)
            self.pointings["unit_vec"] = np.array([x, y, z]).T

    def angular_dist_3d_heliocentric(self, pt):
        """Compute the angular offset (in degrees) between the pointing and
        the 3-d location (specified as heliocentric coordinates and AU) of a point.

        Parameters
        ----------
        pt : tuple, list, or array
            The point represented as (x, y, z) on which to compute the distances.

        Returns
        -------
        ang_dist : numpy array
            A length K numpy array with the angular distances in degrees.

        Raises
        ------
        ValueError if the list of points is not width=3.
        """
        if len(pt) != 3:
            raise ValueError(f"Expected 3 dimensions, found {len(pt)}")

        if "sun_vec" not in self.pointings.columns:
            self.append_sun_pos()
        if "unit_vec" not in self.pointings.columns:
            self.append_unit_vector(self)

        # Compute the geocentric position of the point.
        geo_pt = np.array(
            [
                self.pointings["sun_vec"][:, 0] + pt[0],
                self.pointings["sun_vec"][:, 1] + pt[1],
                self.pointings["sun_vec"][:, 2] + pt[2],
            ]
        ).T

        # Compute the angular distance of the point with each pointing.
        ang_dist = angular_distance(geo_pt, self.pointings["unit_vec"])
        return ang_dist * (180.0 / np.pi)

    def search_heliocentric_pointing(self, ra, dec, dist, fov=None):
        """Search for pointings that would overlap a given heliocentric
        pointing and estimated distance. Allows a single field of view
        or per pointing field of views.

        Note
        ----
        Currently uses a linear algorithm that computes all distances. It
        is likely we can accelerate this with better indexing.

        Parameters
        ----------
        ra : `float`
            The heliocentric pointing's right ascension in degrees.
        dec : `float`
            The heliocentric pointing's declination in degrees.
        dist : `float`
            The point's distance in AU
        fov : `float` (optional)
            The field of view of the individual pointings. If None
            tries to retrieve from table.

        Returns
        -------
        An astropy table with information for the matching pointings.

        Raises
        ------
        ValueError if no field of view is provided.
        """
        if fov is None and "fov" not in self.pointings.columns:
            raise ValueError("No field of view provided.")

        # Create the query point in 3-d heliocentric cartesian space.
        (unit_x, unit_y, unit_z) = ang2unitvec(ra, dec)
        helio_pt = [unit_x * dist, unit_y * dist, unit_z * dist]

        # Compare the angular distance of the query point to each pointing.
        ang_dist = self.angular_dist_3d_heliocentric(helio_pt)
        if fov is None:
            inds = ang_dist < self.pointings["fov"]
        else:
            inds = ang_dist < fov
        return self.pointings[inds]

    def to_csv(self, filename, overwrite=False):
        """Write the pointings data to a CSV file.

        Parameters
        ----------
        filename : `str`
            The name of the CSV file.
        overwrite: `bool`
            Whether to overwrite an existing file.
        """
        self.pointings.write(filename, overwrite=overwrite)
