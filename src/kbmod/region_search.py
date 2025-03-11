from datetime import datetime
import numpy as np

from kbmod import ImageCollection
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.time import Time
from astropy.wcs import WCS

from kbmod.reprojection_utils import correct_parallax_geometrically_vectorized

import lsst.sphgeom as sphgeom
from shapely.geometry import Polygon, Point


def reflex_corrected_col(col_name, guess_dist):
    """
    Returns the name of the reflex-corrected column for a given column name and guess distance.
    """
    if not isinstance(guess_dist, float):
        raise ValueError("Reflex-corrected guess distance must be a float")
    if guess_dist == 0.0:
        return col_name
    return f"{col_name}_{guess_dist}"


class Ephems:
    """
    A tiny helper class to store and reflex-correct ephemeris data from an astropy table
    """

    def __init__(self, ephems_table, ra_col, dec_col, mjd_col, guess_dists, earth_loc):
        self.ephems_data = ephems_table

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
            self.ephems_data[reflex_corrected_col(self.ra_col, guess_dist)] = (
                corrected_ra_dec.ra.deg
            )
            self.ephems_data[reflex_corrected_col(self.dec_col, guess_dist)] = (
                corrected_ra_dec.dec.deg
            )

    def get_mjds(self):
        return self.ephems_data[self.mjd_col]

    def get_ras(self, guess_dist):
        return self.ephems_data[reflex_corrected_col(self.ra_col, guess_dist)]

    def get_decs(self, guess_dist):
        return self.ephems_data[reflex_corrected_col(self.dec_col, guess_dist)]


class RegionSearch:
    """
    A class to filter an ImageCollection by various search criteria (e.g. a cone search,
    matching against an ephemeris, or procedurally dividing a sky into patches and filtering for which
    chips are in each patch.

    Note that underlying ImageCollection is an astropy Table object.
    """

    def __init__(
        self, ic, guess_dists=[], earth_loc=None, enforce_unique_visit_detector=True
    ):
        self.ic = ic

        if enforce_unique_visit_detector:
            # Check that for each combination of visit and detector, there is only one Image
            # This is a requirement for the reflex correction code
            visit_detectors = set(zip(ic.data["visit"], ic.data["detector"]))
            if len(visit_detectors) != len(ic.data):
                raise ValueError(
                    "Multiple images found for the same visit and detector"
                )

        self.guess_dists = guess_dists
        if self.guess_dists:
            if earth_loc is None:
                raise ValueError(
                    "Must provide an EarthLocation if we are taking into account reflex correction."
                )
            self.earth_loc = earth_loc
            for guess_dist in self.guess_dists:
                self.ic = self.reflex_correct_ic(guess_dist)

        # Now that we have reflex corrected, we can calculate the cone search radius
        self.patches = None

        self.shapes = self.generate_chip_shapes()

    def reflex_correct_ic(self, guess_dist):
        # Calculate the reflex-corrected coordinates for each image in the ImageCollection.
        # Note that this modifies the ImageColleciton in place.

        # Calculate the parallax correction for each RA, Dec in the ImageCollection
        corrected_ra_dec, _ = correct_parallax_geometrically_vectorized(
            self.ic["ra"],
            self.ic["dec"],
            self.ic["mjd_mid"],
            guess_dist,
            self.earth_loc,
        )
        self.ic.data[reflex_corrected_col("ra", guess_dist)] = corrected_ra_dec.ra.deg
        self.ic.data[reflex_corrected_col("dec", guess_dist)] = corrected_ra_dec.dec.deg

        # Now we want to reflex-correct the corners for each image in the collection.
        for box_corner in ["tl", "tr", "bl", "br"]:
            corrected_ra_dec_corner, _ = correct_parallax_geometrically_vectorized(
                self.ic[f"ra_{box_corner}"],
                self.ic[f"dec_{box_corner}"],
                self.ic["mjd_mid"],
                guess_dist,
                self.earth_loc,
            )
            self.ic.data[reflex_corrected_col(f"ra_{box_corner}", guess_dist)] = (
                corrected_ra_dec_corner.ra.deg
            )
            self.ic.data[reflex_corrected_col(f"dec_{box_corner}", guess_dist)] = (
                corrected_ra_dec_corner.dec.deg
            )
        return self.ic

    def generate_chip_shapes(self):
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
                    row[reflex_corrected_col(f"ra_{corner}", guess_dist)]
                    for corner in ["tl", "tr", "br", "bl"]
                ]
                dec_corners = [
                    row[reflex_corrected_col(f"dec_{corner}", guess_dist)]
                    for corner in ["tl", "tr", "br", "bl"]
                ]
                shapes[visit][detector][guess_dist] = Polygon(
                    list(zip(ra_corners, dec_corners))
                )
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
        if start_mjd is None and end_mjd is None:
            return
        new_data = self.ic.data
        if start_mjd is not None:
            new_data = new_data[new_data["mjd_mid"] >= start_mjd]
        if end_mjd is not None:
            new_data = new_data[new_data["mjd_mid"] <= end_mjd]
        print(
            f"Filtered down to {len(new_data)} images in the time range {start_mjd} to {end_mjd} from original {len(self.ic)} images"
        )
        self.ic.data = new_data

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

    def generate_patches(
        self,
        arcminutes,
        overlap_percentage,
        image_width,
        image_height,
        pixel_scale,
        dec_range=[-90, 90],
    ):
        self.patch_grid = PatchGrid(
            arcminutes,
            overlap_percentage,
            image_width,
            image_height,
            pixel_scale,
            dec_range,
        )

    def get_patches(self):
        return self.patch_grid.patches

    def export_image_collection(self, ic_to_export=None, guess_dist=None, patch=None):
        new_ic = self.ic.copy() if ic_to_export is None else ic_to_export.copy()
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
            raise ValueError(
                f"Guess distance {guess_dist} not specified for RegionSearch"
            )
        if guess_dist is None:
            guess_dist = 0.0

        # Iterate over all items of the ephemeris and check
        # if they are in any of the patches
        ephems_ras = ephems.get_ras(guess_dist)
        ephems_decs = ephems.get_decs(guess_dist)

        # For each ephemeris entry, check if it is in any of the patches
        patch_indices = set([])
        for curr_ra, curr_dec in zip(ephems_ras, ephems_decs):
            """
            in_prev_patch = False
            if patch_indices:
                for prev_idx in patch_indices:
                    prev_patch = self.patch_grid.patches[prev_idx]
                    if prev_patch.contains(curr_ra, curr_dec):
                        in_prev_patch = True
            if not in_prev_patch:
                for i, patch in enumerate(self.patch_grid.patches):
                    if patch.contains(curr_ra, curr_dec):
                        patch_indices.append(i)
                        break
            """
            found_patches = 0
            for i in patch_indices:
                if self.patch_grid.patches[i].contains(curr_ra, curr_dec):
                    found_patches += 1
            if found_patches < max_overlapping_patches:
                for i, patch in enumerate(self.patch_grid.patches):
                    if i not in patch_indices and patch.contains(curr_ra, curr_dec):
                        patch_indices.add(i)
                        found_patches += 1
                    if found_patches >= max_overlapping_patches:
                        break

        return patch_indices

    def cone_search(self, ra, dec, radius, guess_dist=None, full_overlap=False):
        """
        Filters down an ImageCollection to all images within a given radius of a point.

        Parameters
        ----------
        ra : float
            The right ascension of the center of the cone in degrees.
        dec : float
            The declination of the center of the cone in degrees.
        radius : float
            The radius of the cone in degrees.
        guess_dist : float, optional
            The guess distance to use for reflex correction. If None, the original coordinates are used.
        full_overlap : bool, optional
            If True, the cone is considered to overlap with an image if the cone fully contains the image.

        Returns
        -------
        ImageCollection
            The filtered ImageCollection.
        """
        if guess_dist is not None and guess_dist not in self.guess_dists:
            raise ValueError(
                f"Guess distance {guess_dist} not specified for RegionSearch"
            )
        if guess_dist is None:
            guess_dist = 0.0

        # Create a mask for all images that are within the cone, False by default
        mask = np.zeros(len(self.ic), dtype=bool)
        if full_overlap:
            # We set the mask to be true by default to allow for the bitwise AND operation
            # failing if any corners are outside the cone.
            mask = np.ones(len(self.ic), dtype=bool)

        # Get the center coordinate of our cone
        center_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        # For each corner column in the ImageCollection, check if the corner is within the cone
        for corner in ["tl", "tr", "br", "bl"]:
            corner_coord = SkyCoord(
                ra=self.ic[f"{reflex_corrected_col('ra', guess_dist)}_{corner}"]
                * u.deg,
                dec=self.ic[f"{reflex_corrected_col('dec', guess_dist)}_{corner}"]
                * u.deg,
                frame="icrs",
            )
            if full_overlap:
                # We keep the image only if all corners are in the cone
                mask = mask & center_coord.separation(corner_coord).deg <= radius
            else:
                # We keep the image as long as at least one corner is in the cone
                mask = mask | center_coord.separation(corner_coord).deg <= radius

        self.ic.data = self.ic.data[mask]

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
            raise ValueError(
                f"Guess distance {guess_dist} not specified for Region Search"
            )
        if guess_dist is None:
            guess_dist = 0.0

        if not isinstance(patch, Patch):
            if not isinstance(patch, int):
                raise ValueError("Patch must be an integer or a Patch object")
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


def point_in_bbox(ra, dec, img):
    print(f"Checking if chip with ({ra}, {dec}) is in image {img['dataId']}")
    print("Computing lat lon from TL of bbox")
    tl_pt = sphgeom.UnitVector3d(
        sphgeom.LonLat.fromDegrees(img["ra_tl"], img["dec_tl"])
    )

    print("Computing bottom right of bbox")
    br_pt = sphgeom.UnitVector3d(
        sphgeom.LonLat.fromDegrees(img["ra_br"], img["dec_br"])
    )
    bbox = sphgeom.Box(tl_pt, br_pt)

    return bbox.contains(sphgeom.UnitVector3d(sphgeom.LonLat.fromDegrees(ra, dec)))


def img_to_convex_polygon(img):
    # Compute lat lons from the corners of the images
    # Compute the lat lon from the top left corner
    tl_pt = sphgeom.UnitVector3d(
        sphgeom.LonLat.fromDegrees(img["ra_tl"], img["dec_tl"])
    )
    # Compute the lat lon from the top right corner
    tr_pt = sphgeom.UnitVector3d(
        sphgeom.LonLat.fromDegrees(img["ra_tr"], img["dec_tr"])
    )
    # Compute the lat lon from the bottom left corner
    bl_pt = sphgeom.UnitVector3d(
        sphgeom.LonLat.fromDegrees(img["ra_bl"], img["dec_bl"])
    )
    # Compute the lat lon from the bottom right corner
    br_pt = sphgeom.UnitVector3d(
        sphgeom.LonLat.fromDegrees(img["ra_br"], img["dec_br"])
    )

    # Turn each corner into a sphgeom UnitVector3d
    tl_vec = sphgeom.UnitVector3d.fromLonLat(tl_pt)
    tr_vec = sphgeom.UnitVector3d.fromLonLat(tr_pt)
    bl_vec = sphgeom.UnitVector3d.fromLonLat(bl_pt)
    br_vec = sphgeom.UnitVector3d.fromLonLat(br_pt)

    # Create a sphgeom convex polygone from our corners
    poly = sphgeom.ConvexPolygon([tl_vec, tr_vec, br_vec, bl_vec])
    return poly


def point_in_convex_polygon(ra, dec, img):
    print(f"Checking if chip with ({ra}, {dec}) is in image {img['dataId']}")

    poly = img_to_convex_polygon(img)

    # Compute the lat lon from the point we are checking
    return poly.contains(sphgeom.UnitVector3d(sphgeom.LonLat.fromDegrees((ra, dec))))


class PatchGrid:
    def __init__(
        self,
        arcminutes,
        overlap_percentage,
        image_width,
        image_height,
        pixel_scale,
        dec_range=[-90, 90],
    ):
        """

        Parameters
        ----------
        arcminutes : int
            The size of the patches in arcminutes.
        overlap_percentage : float
            The percentage of overlap between patches.
        dec_range : list of int, optional
            The range of declinations to cover. Default is [-90, 90] which is the whole sky.
        """
        self.arcminutes = arcminutes
        self.overlap_percentage = overlap_percentage
        self.dec_range = dec_range

        self.image_width = image_width
        self.image_height = image_height
        self.pixel_scale = pixel_scale

        self.patches = self._create_patches()

    def _create_patches_orig(self):
        # TODO Remove this function? Is it outdated?
        patches = []
        for ra in range(0, 360, self.arcminutes):
            for dec in range(self.dec_range[0], self.dec_range[1], self.arcminutes):
                patches.append(Patch(ra, dec, self.arcminutes, self.arcminutes))
        return patches

    def _create_patches(self):
        patches = []
        arcdegrees = self.arcminutes / 60.0
        overlap = arcdegrees * (self.overlap_percentage / 100.0)
        num_patches_ra = int(360 / (arcdegrees - overlap))
        num_patches_dec = int(180 / (arcdegrees - overlap))

        for ra_index in range(num_patches_ra):
            ra_start = ra_index * (arcdegrees - overlap)
            center_ra = ra_start + arcdegrees / 2

            for dec_index in range(num_patches_dec):
                dec_start = dec_index * (arcdegrees - overlap) - 90
                center_dec = dec_start + arcdegrees / 2

                if self.dec_range[0] <= center_dec <= self.dec_range[1]:
                    patches.append(
                        Patch(
                            center_ra,
                            center_dec,
                            arcdegrees,
                            arcdegrees,
                            self.image_width,
                            self.image_height,
                            self.pixel_scale,
                        )
                    )

        return patches


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
    ):
        self.ra = center_ra
        self.dec = center_dec
        self.width = width
        self.height = height
        self.image_width = image_width
        self.image_height = image_height
        self.pixel_scale = pixel_scale

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

        self.polygon = self.create_polygon()
        self.convex_polygon = self._create_convex_polygon()

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

    def create_polygon(self):
        return Polygon(self.corners)

    def _create_convex_polygon(self):
        # Create a sphgeom polygon from the corners of the patch

        # Compute the lat lons from the corners of the patch
        # Compute the lat lon from the top left corner
        # TODO remove min hack, maybe mod 90 instead?
        tl_pt = sphgeom.UnitVector3d(
            sphgeom.LonLat.fromDegrees(self.tl_ra, min(90, self.tl_dec))
        )
        # Compute the lat lon from the top right corner
        tr_pt = sphgeom.UnitVector3d(
            sphgeom.LonLat.fromDegrees(self.tr_ra, min(90, self.tr_dec))
        )
        # Compute the lat lon from the bottom left corner
        bl_pt = sphgeom.UnitVector3d(
            sphgeom.LonLat.fromDegrees(self.bl_ra, min(self.bl_dec, 90))
        )
        # Compute the lat lon from the bottom right corner
        br_pt = sphgeom.UnitVector3d(
            sphgeom.LonLat.fromDegrees(self.br_ra, min(90, self.br_dec))
        )

        return sphgeom.ConvexPolygon([tl_pt, tr_pt, br_pt, bl_pt])

    def contains(self, ra, dec):
        return self.polygon.contains(Point(ra, dec))

    def measure_overlap(self, poly):
        # Get the overlap between the shapely polygon and our patch in square degrees
        overlap = self.polygon.intersection(poly)
        return overlap.area

    def overlaps_polygon(self, poly):
        # True if the patch overlaps at all with the given shapely polygon
        return self.measure_overlap(poly) > 0

    def contains_convex_poly(self, poly):
        self.convex_polygon.contains(poly)
