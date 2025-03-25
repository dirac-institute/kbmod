import numpy as np

from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.wcs import WCS

from kbmod.reprojection_utils import correct_parallax_geometrically_vectorized

from shapely.geometry import Polygon, Point


class Ephems:
    """
    A tiny helper class to store and reflex-correct ephemeris data from an astropy table
    """

    def __init__(self, ephems_table, ra_col, dec_col, mjd_col, guess_dists, earth_loc):
        self.ephems_data = ephems_table.copy()

        self.ra_col = ra_col
        self.dec_col = dec_col
        self.mjd_col = mjd_col
        self.guess_dists = guess_dists
        self.earth_loc = earth_loc

        # Sort table by time
        self.ephems_data.sort(mjd_col)

        # Reflex-correct the target observation data
        for guess_dist in self.guess_dists:
            # Calculate the parallax correction for each RA, Dec in the target observations
            corrected_ra_dec, _ = correct_parallax_geometrically_vectorized(
                self.ephems_data[self.ra_col],
                self.ephems_data[self.dec_col],
                self.ephems_data[self.mjd_col],
                guess_dist,
                self.earth_loc,
            )
            self.ephems_data[self.reflex_corrected_col(self.ra_col, guess_dist)] = corrected_ra_dec.ra.deg
            self.ephems_data[self.reflex_corrected_col(self.dec_col, guess_dist)] = corrected_ra_dec.dec.deg

        self.patches = []

    def get_mjds(self):
        return self.ephems_data[self.mjd_col]

    def get_ras(self, guess_dist=None):
        if guess_dist is None:
            return self.ephems_data[self.ra_col]
        return self.ephems_data[self.reflex_corrected_col(self.ra_col, guess_dist)]

    def get_decs(self, guess_dist=None):
        if guess_dist is None:
            return self.ephems_data[self.dec_col]
        return self.ephems_data[self.reflex_corrected_col(self.dec_col, guess_dist)]

    def reflex_corrected_col(self, col_name, guess_dist):
        """
        Returns the name of the reflex-corrected column for a given column name and guess distance.
        """
        # Fail for column names containing whitespace or underscores
        if " " in col_name or "_" in col_name:
            raise ValueError("Reflex-corrected column names cannot contain whitespace or underscores")
        if not isinstance(guess_dist, float):
            raise ValueError("Reflex-corrected guess distance must be a float")
        if guess_dist == 0.0:
            return col_name
        return f"{col_name}_{guess_dist}"


class RegionSearch:
    """
    A class to filter an ImageCollection by various search criteria (e.g. a cone search,
    matching against an ephemeris, or procedurally dividing a sky into patches and filtering for which
    chips are in each patch.

    Note that underlying ImageCollection is an astropy Table object.
    """

    def __init__(self, ic, guess_dists=[], earth_loc=None, enforce_unique_visit_detector=True):
        self.ic = ic

        if enforce_unique_visit_detector:
            # Check that for each combination of visit and detector, there is only one Image
            # This is a requirement for the reflex correction code
            visit_detectors = set(zip(ic.data["visit"], ic.data["detector"]))
            if len(visit_detectors) != len(ic.data):
                raise ValueError("Multiple images found for the same visit and detector")

        self.guess_dists = guess_dists
        if self.guess_dists:
            if earth_loc is None:
                raise ValueError(
                    "Must provide an EarthLocation if we are taking into account reflex correction."
                )
            self.earth_loc = earth_loc
            self.ic.reflex_correct(self.guess_dists, self.earth_loc)

        self.patches = None

        self.shapes = self._generate_chip_shapes()

    def _generate_chip_shapes(self):
        """
        Generates a dictionary of Polygons for each chip in the ImageCollection. At
        both in the original coordinates and across each reflex-corrected guess distance.

        Returns
        -------
        dict
            A dictionary of Polygons for each chip in the ImageCollection. The keys are the
            [visit][detector][guess_dist] (with 0.0 used for the original coordinates).
        """
        # For each row in the ImageCollection, create a Polygon from the corners of the chip at each guess distance
        shapes = {}
        for row in self.ic.data:
            visit = row["visit"]
            if visit not in shapes:
                shapes[visit] = {}
            detector = row["detector"]
            if detector not in shapes[visit]:
                shapes[visit][detector] = {}
            # We include 0.0 to represent the original coordinates
            # TODO is there a better way to do this?
            for guess_dist in [0.0] + self.guess_dists:
                if guess_dist not in shapes[visit][detector]:
                    shapes[visit][detector][guess_dist] = {}
                ra_corners = [
                    row[self.ic.reflex_corrected_col(f"ra_{corner}", guess_dist)]
                    for corner in ["tl", "tr", "br", "bl"]
                ]
                dec_corners = [
                    row[self.ic.reflex_corrected_col(f"dec_{corner}", guess_dist)]
                    for corner in ["tl", "tr", "br", "bl"]
                ]
                shapes[visit][detector][guess_dist] = Polygon(list(zip(ra_corners, dec_corners)))
        return shapes

    def filter_by_time_range(self, start_mjd, end_mjd):
        """
        Filter the ImageCollection by the given time range. Is performed in-place.

        Note that it uses the "mjd_mid" column to filter.

        Parameters
        ----------
        start_mjd : float
            The start of the time range in MJD.
        end_mjd : float
            The end of the time range in MJD.
        """
        self.ic.filter_by_time_range(start_mjd, end_mjd)

    def filter_by_mjds(self, mjds, time_sep_s=0.001):
        """
        Filter the visits in the ImageCollection by the given MJDs. Is performed in-place.

        Note that the comparison is made against "mjd_mid"

        Parameters
        ----------
        ic : ImageCollection
            The ImageCollection to filter.
        timestamps : list of floats
            List of timestamps to keep.
        time_sep_s : float, optional
            The maximum separation in seconds between the timestamps in the ImageCollection and the timestamps to keep.

        Returns
        -------
        None
        """
        if len(self.ic) < 1:
            return
        mask = np.zeros(len(self.ic), dtype=bool)
        for mjd in mjds:
            mjd_diff = abs(self.ic.data["mjd_mid"] - mjd)
            mask = mask | (mjd_diff <= time_sep_s / (24 * 60 * 60))
        self.ic.data = self.ic.data[mask]
        self.ic.filter_by_mjds(mjds, time_sep_s)

    def generate_patches(
        self,
        arcminutes,
        overlap_percentage,
        image_width,
        image_height,
        pixel_scale,
        dec_range=[-90, 90],
    ):
        self.patches = []
        arcdegrees = arcminutes / 60.0
        overlap = arcdegrees * (overlap_percentage / 100.0)
        num_patches_ra = int(360 / (arcdegrees - overlap))
        num_patches_dec = int(180 / (arcdegrees - overlap))

        for ra_index in range(num_patches_ra):
            ra_start = ra_index * (arcdegrees - overlap)
            center_ra = ra_start + arcdegrees / 2

            for dec_index in range(num_patches_dec):
                dec_start = dec_index * (arcdegrees - overlap) - 90
                center_dec = dec_start + arcdegrees / 2

                if dec_range[0] <= center_dec <= dec_range[1]:
                    patch_id = len(self.patches)
                    self.patches.append(
                        Patch(
                            center_ra,
                            center_dec,
                            arcdegrees,
                            arcdegrees,
                            image_width,
                            image_height,
                            pixel_scale,
                            patch_id,
                        )
                    )

    def get_patches(self):
        return self.patches

    def get_patch(self, patch_id):
        if not self.patches:
            raise ValueError("No patches have been generated yet.")
        if patch_id < 0 or patch_id >= len(self.patches):
            raise ValueError(f"Patch ID {patch_id} is out of range.")
        return self.patches[patch_id]

    def export_image_collection(self, ic_to_export=None, guess_dist=None, patch=None):
        if ic_to_export is None:
            ic_to_export = self.ic
        if len(ic_to_export) < 1:
            raise ValueError(f"ImageCollection is empty, cannot export {ic_to_export}")

        new_ic = ic_to_export.copy()
        if guess_dist is not None:
            new_ic.data["helio_guess_dist"] = guess_dist

        if patch is not None:
            if not isinstance(patch, Patch):
                if not isinstance(patch, int):
                    raise ValueError("Patch must be an integer or a Patch object")
                patch = self.get_patches()[patch]
            patch_wcs = patch.to_wcs()
            new_ic.data["global_wcs"] = patch_wcs.to_header_string()
            new_ic.data["global_wcs_pixel_shape_0"] = patch_wcs.pixel_shape[0]
            new_ic.data["global_wcs_pixel_shape_1"] = patch_wcs.pixel_shape[1]

        new_ic.meta["n_stds"] = len(new_ic)
        new_ic.data["std_idx"] = range(len(new_ic))

        return new_ic

    def search_patches(self, ephems, guess_dist=None, max_overlapping_patches=4):
        """
        Returns all patch indices where the ephemeris entries are found.

        Parameters
        ----------
        ephems : Ephems
            The ephemeris data to search for.
        guess_dist : float, optional
            The guess distance to use for reflex correction. If None, the original coordinates are used.

        Returns
        -------
        set of int
            The indices of the patches that contain the ephemeris entries.
        """
        if guess_dist is not None and guess_dist not in self.guess_dists:
            raise ValueError(f"Guess distance {guess_dist} not specified for RegionSearch")
        if guess_dist is None:
            guess_dist = 0.0

        # Iterate over all items of the ephemeris and check
        # if they are in any of the patches
        ephems_ras = ephems.get_ras(guess_dist)
        ephems_decs = ephems.get_decs(guess_dist)

        # For each ephemeris entry, check if it is in any of the patches
        patch_size_deg = np.sqrt((self.patches[0].width / 2) ** 2 + (self.patches[0].height / 2) ** 2)
        # We need to check if the ephemeris entry is in any of the patches with search around sky
        # coordinates. We do this by checking if the ephemeris entry is in any of the patches
        # with a search radius of half the patch size
        ephems_coords = SkyCoord(
            ephems_ras,
            ephems_decs,
            unit=(u.deg, u.deg),
            frame="icrs",
        )
        # Get the patch center coordinates
        patch_centers = SkyCoord(
            [patch.ra for patch in self.patches],
            [patch.dec for patch in self.patches],
            unit=(u.deg, u.deg),
            frame="icrs",
        )

        # Use search_around_sky
        ephems_idx, patch_idx, _, _ = search_around_sky(ephems_coords, patch_centers, patch_size_deg * u.deg)

        patch_indices = set([])
        for ephem_idx, patch_idx in zip(ephems_idx, patch_idx):
            if patch_idx not in patch_indices:
                curr_ra, curr_dec = ephems_coords[ephem_idx].ra.deg, ephems_coords[ephem_idx].dec.deg
                if self.patches[patch_idx].contains(curr_ra, curr_dec):
                    patch_indices.add(patch_idx)
        return patch_indices

        """
        patch_indices = set([])
        for curr_ra, curr_dec in zip(ephems_ras, ephems_decs):
            found_patches = 0
            for i in patch_indices:
                if self.patches[i].contains(curr_ra, curr_dec):
                    found_patches += 1
            if found_patches < max_overlapping_patches:
                for i, patch in enumerate(self.patches):
                    if i not in patch_indices and patch.contains(curr_ra, curr_dec):
                        patch_indices.add(i)
                        found_patches += 1
                    if found_patches >= max_overlapping_patches:
                        break

        return patch_indices
        """

    def get_image_collection_from_patch(self, patch, guess_dist=None):
        """
        Filters down an ImageCollection to all images that overlap with a given patch.

        A patch may be specified by a Patch object or by its index in the PatchGrid.

        Parameters
        ----------
        patch : Patch or int
            The patch to filter by.
        guess_dist : float, optional
            The guess distance to use for reflex correction. If None, the original coordinates are used.
        """
        if guess_dist is not None and guess_dist not in self.guess_dists:
            raise ValueError(f"Guess distance {guess_dist} not specified for Region Search")
        if guess_dist is None:
            guess_dist = 0.0

        if not isinstance(patch, Patch):
            if not (isinstance(patch, int) or isinstance(patch, np.integer)):
                raise ValueError(f"Patch must be an integer or a Patch object, was: {type(patch)}")
            # Get the patch object from the index
            patch = self.get_patches()[patch]

        # Iterate over all images and check if they overlap with the patch
        mask = np.zeros(len(self.ic), dtype=bool)
        new_ic = self.ic.copy()
        overlap_deg = np.zeros(len(new_ic))
        for i in range(len(new_ic)):
            row = new_ic[i]
            # Get our polygon from the visit and detector and our guess distance
            poly = self.shapes[row["visit"]][row["detector"]][guess_dist]
            overlap_deg = patch.measure_overlap(poly)
            mask[i] = overlap_deg > 0
        new_ic.data = new_ic.data[mask]
        return self.export_image_collection(new_ic, guess_dist=guess_dist, patch=patch)


class Patch:
    """
    A class to represent a patch of the sky.
    The patch is defined by its center coordinates, width, height, and the image size.
    The patch is a square with the given width and height, centered at the given coordinates.
    """

    def __init__(
        self,
        center_ra,
        center_dec,
        width,
        height,
        image_width,
        image_height,
        pixel_scale,
        id,
    ):
        self.ra = center_ra
        self.dec = center_dec
        self.width = width
        self.height = height
        self.image_width = image_width
        self.image_height = image_height
        self.pixel_scale = pixel_scale
        self.id = id

        # Compute the (RA, Dec) corners of the patch
        self.tl_ra = center_ra - width / 2
        self.tl_dec = center_dec + height / 2
        self.tr_ra = center_ra + width / 2
        self.tr_dec = center_dec + height / 2
        self.bl_ra = center_ra - width / 2
        self.bl_dec = center_dec - height / 2
        self.br_ra = center_ra + width / 2
        self.br_dec = center_dec - height / 2

        self.corners = [
            (self.tl_ra, self.tl_dec),
            (self.tr_ra, self.tr_dec),
            (self.br_ra, self.br_dec),
            (self.bl_ra, self.bl_dec),
        ]

        self.polygon = Polygon(self.corners)

    def __str__(self):
        """Returns a string representation of the patch."""
        return f"Patch ID: {self.id} RA: {self.ra}, Dec: {self.dec}, Width (pixels): {self.width}, Height (piexels): {self.height}"

    def to_wcs(self):
        # Use astropy WCS utils to convert the (RA, Dec) corners to
        # a WCS object

        pixel_scale_ra = self.pixel_scale / 60 / 60
        pixel_scale_dec = self.pixel_scale / 60 / 60

        # Initialize a WCS object with 2 axes (RA and Dec)
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [self.image_width / 2, self.image_height / 2]
        wcs.wcs.crval = [self.ra, self.dec]
        wcs.wcs.cdelt = [-pixel_scale_ra, pixel_scale_dec]

        # Rotation matrix, assuming no rotation
        wcs.wcs.pc = [[1, 0], [0, 1]]

        # Define coordinate frame and projection
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.array_shape = (self.image_height, self.image_width)

        return wcs

    def contains(self, ra, dec):
        return self.polygon.contains(Point(ra, dec))

    def measure_overlap(self, poly):
        # Get the overlap between the shapely polygon and our patch in square degrees
        overlap = self.polygon.intersection(poly)
        return overlap.area

    def overlaps_polygon(self, poly):
        # True if the patch overlaps at all with the given shapely polygon
        return self.measure_overlap(poly) > 0
