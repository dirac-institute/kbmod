import numpy as np

from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
from astropy.wcs import WCS

from kbmod.reprojection_utils import correct_parallax_geometrically_vectorized

from shapely.geometry import Polygon, Point


def patch_arcmin_to_pixels(patch_size, pixel_scale):
    """Helper function to convert a given patch size in arcminutes to pixels based on a pixel scale

    Parameters
    ----------
    patch_size : float
        The size of the patch in arcminutes.
    pixel_scale : float
        The pixel scale in arcseconds per pixel.

    Returns
    -------
    int
        The number of pixels in the patch size.
    """
    # Convert the patch size from arcminutes to arcseconds and then divide by the pixel scale.
    patch_pixels = int(np.ceil(patch_size * 60 / pixel_scale))
    return patch_pixels


class Ephems:
    """
    A tiny helper class to store and reflex-correct ephemeris data from an astropy table
    """

    def __init__(self, ephems_table, ra_col, dec_col, mjd_col, guess_dists, earth_loc):
        """
        Parameters
        ----------
        ephems_table : astropy.table.Table
            The ephemeris data table.
        ra_col : str
            The name of the column containing the RA values in degrees
        dec_col : str
            The name of the column containing the Dec values in degrees
        mjd_col : str
            The name of the column containing the MJD values
        guess_dists : list of floats
            The guess distances in AU for reflex correction
        earth_loc : astropy.coordinates.EarthLocation
            The Earth location for reflex correction
        """
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
            # Skip correction for distance 0.0 - use original coordinates
            if guess_dist == 0.0:
                continue
            # Calculate the parallax correction for each RA, Dec in the target observations
            corrected_ra_dec, _ = correct_parallax_geometrically_vectorized(
                self.ephems_data[self.ra_col],
                self.ephems_data[self.dec_col],
                self.ephems_data[self.mjd_col],
                guess_dist,
                self.earth_loc,
            )
            # Store the corrected RA, Dec in a new column
            self.ephems_data[self._reflex_corrected_col(self.ra_col, guess_dist)] = corrected_ra_dec.ra.deg
            self.ephems_data[self._reflex_corrected_col(self.dec_col, guess_dist)] = corrected_ra_dec.dec.deg

    def get_mjds(self):
        """Returns the MJD column of the ephemeris data"""
        return self.ephems_data[self.mjd_col]

    def get_ras(self, guess_dist=None):
        """
        Returns the RA column of the ephemeris data in degrees

        Parameters
        ----------
        guess_dist : float, optional
            The guess distance to use for reflex correction. If None, the original coordinates are used.
        """
        if guess_dist is None:
            return self.ephems_data[self.ra_col]
        return self.ephems_data[self._reflex_corrected_col(self.ra_col, guess_dist)]

    def get_decs(self, guess_dist=None):
        """
        Returns the Dec column of the ephemeris data in degrees

        Parameters
        ----------
        guess_dist : float, optional
            The guess distance to use for reflex correction. If None, the original coordinates are used.
        """
        if guess_dist is None:
            return self.ephems_data[self.dec_col]
        return self.ephems_data[self._reflex_corrected_col(self.dec_col, guess_dist)]

    def _reflex_corrected_col(self, col_name, guess_dist):
        """
        Returns the name of the reflex-corrected column for a given column name and guess distance.

        Parameters
        ----------
        col_name : str
            The name of the column to be reflex-corrected.
        guess_dist : float
            The guess distance in AU for reflex correction.
        """
        if not isinstance(guess_dist, float):
            raise ValueError("Reflex-corrected guess distance must be a float")
        if guess_dist == 0.0:
            return col_name
        return f"{col_name}_{guess_dist}"


class RegionSearch:
    """
    A class to filter an ImageCollection by various search criteria. Primarily intended to be used
    to divide the night sky into a grid and export ImageCollections which contain images that overlap
    with the patches in the grid. The class also provides methods for searching across the patches
    such as seeing which patches of sky contain entries from a given ephemeris table.

    Note that by convention, the patches are defined to already be in whatever reflex-corrected coordinates
    are being used. So a given object might be in a different patch depending on the guess distance used
    for reflex correction.

    Also allows other methods for filtering down the ImageCollection such as by a time range or by a list of MJDs.

    Note that the underlying ImageCollection is an astropy Table object.
    """

    def __init__(self, ic, guess_dists=[], earth_loc=None, enforce_unique_visit_detector=True):
        """
        Parameters
        ----------
        ic : ImageCollection
            The ImageCollection to filter.
        guess_dists : list of floats
            The guess distances in AU for reflex correction.
        earth_loc : astropy.coordinates.EarthLocation
            The Earth location for reflex correction. Must be provided if guess_dists is not empty.
        enforce_unique_visit_detector : bool
            Whether to enforce that there is only one image per visit and detector in the ImageCollection.
        """
        self.ic = ic

        if enforce_unique_visit_detector:
            # Check that for each combination of visit and detector, there is only one Image
            # This is a requirement for the reflex correction code
            visit_detectors = set(zip(ic.data["visit"], ic.data["detector"]))
            if len(visit_detectors) != len(ic.data):
                raise ValueError("Multiple images found for the same visit and detector")

        self.guess_dists = guess_dists
        if self.guess_dists:
            # We will need to reflex-correct the ImageCollection
            if earth_loc is None:
                raise ValueError(
                    "Must provide an EarthLocation if we are taking into account reflex correction."
                )
            self.earth_loc = earth_loc
            self.ic.reflex_correct(self.guess_dists, self.earth_loc)

        # Generate the polygon objects for each chip in the ImageCollection (as a visit, detector, guess_dist)
        self.chip_shapes = self._generate_chip_shapes()

        # Initially Patches are not defined.
        self.patches = None

    def _generate_chip_shapes(self):
        """
        Generates a dictionary of Polygons for each chip in the ImageCollection, both
        in the original coordinates and across each reflex-corrected guess distance.

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
            shape_dists = [0.0] + self.guess_dists if 0.0 not in self.guess_dists else self.guess_dists
            for guess_dist in shape_dists:
                if guess_dist not in shapes[visit][detector]:
                    shapes[visit][detector][guess_dist] = {}
                # Get the corners of the chip in RA and Dec for this guess distance
                ra_corners = [
                    row[self.ic.reflex_corrected_col(f"ra_{corner}", guess_dist)]
                    for corner in ["tl", "tr", "br", "bl"]
                ]
                dec_corners = [
                    row[self.ic.reflex_corrected_col(f"dec_{corner}", guess_dist)]
                    for corner in ["tl", "tr", "br", "bl"]
                ]
                # Create a shapely polygon from the corners to efficiently check for overlap with this chip
                shapes[visit][detector][guess_dist] = Polygon(list(zip(ra_corners, dec_corners)))
        return shapes

    def _get_patch_radius(self):
        """
        Returns the radius of the patches in degrees, which is the distance from the center of the patch
        to one of its corners (the radius of its circumcribing circle).

        Returns
        -------
        astropy.units.Quantity
            The radius of the patches in degrees.
        """
        if not self.patches:
            raise ValueError("No patches have been generated yet.")

        # All patches are the same size, so we can just return the radius of the first patch
        return self.patches[0].patch_radius()

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
        if len(self.ic) < 1:
            return
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
        self.ic.filter_by_mjds(mjds, time_sep_s)

    def generate_patches(
        self,
        arcminutes,
        overlap_percentage,
        pixel_scale,
        dec_range=[-90, 90],
    ):
        """
        Generate patches of the sky based on the given parameters.
        The patches are generated in a RA-Dec aligned grid, with the given overlap percentage determing
        how much overlap there is between adjacent patches. Note that since we are doing overlap in both
        dimension, an `overlap_percentage` of 50% will produce 4x the patches of a grid with 0% overlap.

        Note that by convention, the patches are defined to already be in whatever reflex-corrected coordinates
        are being used. So a given object might be in a different patch depending on the guess distance used
        for reflex correction.

        The generated list of patches is stored in the `self.patches` attribute, with each `Patch` object
        containing the center coordinates, width, height, and image size as well as an ID corresponding
        to its index in list `self.patches`.

        Parameters
        ----------
        arcminutes : float
            The size of the patches in arcminutes. Patches are square, so this is both the width and height.
        overlap_percentage : float
            The percentage of overlap between adjacent patches, expressed in [0,100]
        pixel_scale : float
            The pixel scale in arcseconds per pixel.
        dec_range : list of float, optional
            The range of declinations to include in the patches. Default is [-90, 90].

        Returns
        -------
        None
        """
        self.patches = []

        # Calculate the patch length in pixels
        patch_len_pixels = patch_arcmin_to_pixels(arcminutes, pixel_scale)

        # Get the patch overlap in degrees
        arcdegrees = arcminutes / 60.0
        overlap = arcdegrees * (overlap_percentage / 100.0)

        # Determine the length (in number of patches) of our patch grid in both the RA and Dec dimensions
        patch_grid_len_ra = int(360 / (arcdegrees - overlap))
        patch_grid_len_dec = int(180 / (arcdegrees - overlap))

        # Generate the patches in a grid
        for ra_index in range(patch_grid_len_ra):
            # The RA coordinate in degrees for the start of the current row of patches
            ra_start = ra_index * (arcdegrees - overlap)
            # The center RA coordinate in degrees for the current row of patches
            center_ra = ra_start + arcdegrees / 2

            # Generate the patches for the current row
            for dec_index in range(patch_grid_len_dec):
                # The Dec coordinate in degrees for the current patch
                dec_start = dec_index * (arcdegrees - overlap) - 90
                # The center Dec coordinate in degrees for the current patch
                center_dec = dec_start + arcdegrees / 2

                if dec_range[0] <= center_dec <= dec_range[1]:
                    # Since the center Dec coordinate is within the specified range
                    # create a new Patch object and add it to the list of patches
                    patch_id = len(self.patches)  # Index of this patch in self.patches
                    self.patches.append(
                        Patch(
                            center_ra,
                            center_dec,
                            arcdegrees,
                            arcdegrees,
                            patch_len_pixels,
                            patch_len_pixels,
                            pixel_scale,
                            patch_id,
                        )
                    )

    def get_patches(self):
        """
        Returns the grid of patches as a single list of Patch objects.

        Note that this is a 1D flattened representation of the grid of patches.
        """
        return self.patches

    def get_patch(self, patch_id):
        """
        Returns the Patch object with the given ID.

        Parameters
        ----------
        patch_id : int
            The ID of the patch to return.
        Returns
        -------
        Patch
            The Patch object with the given ID.
        """
        if not self.patches:
            raise ValueError("No patches have been generated yet.")
        if patch_id < 0 or patch_id >= len(self.patches):
            raise ValueError(f"Patch ID {patch_id} is out of range.")
        return self.patches[patch_id]

    def match_ic_to_patches(self, ic, guess_dist, earth_loc):
        """
        Returns all patch indices where the ImageCollection images are found.

        Note that by convention, the patches are defined to already be in whatever reflex-corrected coordinates
        are being used.

        Parameters
        ----------
        ic : ImageCollection
            The ImageCollection to search for.
        guess_dist : float
            The guess distance to use for reflex correction. If 0.0, the original coordinates are used.
        earth_loc : astropy.coordinates.EarthLocation
            The Earth location for reflex correction.

        Returns
        -------
        set of int
            The indices of the patches that contain the ImageCollection images.
        """
        if guess_dist not in self.guess_dists and guess_dist != 0.0:
            raise ValueError(f"Guess distance {guess_dist} not specified for RegionSearch")
        # Since we already have a method for searching patches by ephemeris entries,
        # we can convert the necessary columns in an ImageCollection to an Ephems object
        # and use that method.
        ic_as_ephem = Ephems(
            ic.data,
            ra_col="ra",
            dec_col="dec",
            mjd_col="mjd_mid",
            guess_dists=[guess_dist],
            earth_loc=earth_loc,
        )
        return self.search_patches_by_ephems(ic_as_ephem, guess_dist=guess_dist)

    def search_patches_by_ephems(self, ephems, guess_dist=None, map_obj_to_patches=False):
        """
        Returns all patch indices where the ephemeris entries are found.

        Note that by convention, the patches are defined to already be in whatever reflex-corrected coordinates
        are being used. So a given object in the ephemeris table might be in a different patch depending on the
        guess distance specified.

        Parameters
        ----------
        ephems : region_search.Ephems
            The ephemeris data to search for.
        guess_dist : float, optional
            The guess distance to use for reflex correction. If None or 0.0, the original coordinates are used.
        map_obj_to_patches : bool, optional
            Whether to return a mapping of list of ephemeris objects to patch indices where they were found.
            If False, only the set of patch indices is returned. Default is False.

        Returns
        -------
        set of int
            The indices of the patches that contain the ephemeris entries.
        """
        if guess_dist is not None and guess_dist != 0.0 and guess_dist not in self.guess_dists:
            raise ValueError(f"Guess distance {guess_dist} not specified for RegionSearch")
        if guess_dist is None:
            guess_dist = 0.0

        # Prepare skycoords for the reflex-corrected ephemeris entries to efficiently search against the patch centers
        ephems_ras = ephems.get_ras(guess_dist)
        ephems_decs = ephems.get_decs(guess_dist)
        ephems_coords = SkyCoord(
            ephems_ras,
            ephems_decs,
            unit=(u.deg, u.deg),
            frame="icrs",
        )
        # Get the center coordinates of all patches. Note that by convention, we already
        # consider the coordinates of the patches to be in our (optionally) reflex-corrected coordinate space.
        patch_centers = SkyCoord(
            [patch.ra for patch in self.patches],
            [patch.dec for patch in self.patches],
            unit=(u.deg, u.deg),
            frame="icrs",
        )

        # Use search_around_sky to find the indices of the patches that may contain the ephemeris entries.
        # This is a fast, coarse search for matching patches since it uses a circular search radius of each
        # patch's circumscribing circle rather than the actual patch boundaries. We will next filter out
        # results that are not actually in the patch boundaries.
        ephems_idx, patch_idx, _, _ = search_around_sky(
            ephems_coords, patch_centers, self._get_patch_radius()
        )

        # Now we need to check if the ephemeris entry is actually in the boundaries of the patch
        # rather than just its circumscribing circle.
        res_patch_indices = set([])
        obj_to_patches = {}
        for ephem_idx, patch_idx in zip(ephems_idx, patch_idx):
            if map_obj_to_patches or patch_idx not in res_patch_indices:
                # Check if the ephemeris entry is in the patch
                curr_ra, curr_dec = ephems_coords[ephem_idx].ra.deg, ephems_coords[ephem_idx].dec.deg
                if self.patches[patch_idx].contains(curr_ra, curr_dec):
                    res_patch_indices.add(patch_idx)
                    if map_obj_to_patches:
                        obj_name = ephems.ephems_data[ephem_idx]["Name"]
                        if obj_name not in obj_to_patches:
                            obj_to_patches[obj_name] = set([])
                        obj_to_patches[obj_name].add(patch_idx)

        if not map_obj_to_patches:
            return res_patch_indices
        return res_patch_indices, obj_to_patches

    def search_patches_within_radius(self, ephems, search_radius, guess_dist=None):
        """
        Returns all patch indices where the ephemeris entries are found within a search radius.

        Parameters
        ----------
        ephems : region_search.Ephems
            The ephemeris data to search for.
        search_radius : float
            The search radius in degrees.
        guess_dist : float, optional
            The guess distance to use for reflex correction. If None or 0.0, the original coordinates are used.

        Returns
        -------
        set of int
            The indices of the patches that are within the search radius of the ephemeris entries.
        """
        if guess_dist is not None and guess_dist != 0.0 and guess_dist not in self.guess_dists:
            raise ValueError(f"Guess distance {guess_dist} not specified for RegionSearch")
        if guess_dist is None:
            guess_dist = 0.0

        # Prepare skycoords for the reflex-corrected ephemeris trajectory
        ephems_ras = ephems.get_ras(guess_dist)
        ephems_decs = ephems.get_decs(guess_dist)
        ephems_coords = SkyCoord(
            ephems_ras,
            ephems_decs,
            unit=(u.deg, u.deg),
            frame="icrs",
        )

        # Get the center coordinates of all patches
        patch_centers = SkyCoord(
            [patch.ra for patch in self.patches],
            [patch.dec for patch in self.patches],
            unit=(u.deg, u.deg),
            frame="icrs",
        )

        # We want any patch that *overlaps* the search cone.
        # So effective search radius = search_radius + patch_radius
        # Note: _get_patch_radius returns radius in degrees
        search_limit = search_radius * u.deg + self._get_patch_radius()

        # search_around_sky returns indices of matches
        # idx1 is index into ephems_coords, idx2 is index into patch_centers
        _, patch_idx, _, _ = search_around_sky(ephems_coords, patch_centers, search_limit)

        return set(patch_idx)

    def plot_patches(
        self,
        patch_ids,
        output_path=None,
        ephems_ra=None,
        ephems_dec=None,
        title=None,
        figsize=(12, 10),
        show_all_patches=False,
    ):
        """
        Plot specified patches on an RA-Dec grid with optional ephemeris overlay.

        Parameters
        ----------
        patch_ids : list or set of int
            The IDs of patches to plot.
        output_path : str, optional
            Path to save the plot. If None, plot is displayed.
        ephems_ra : array-like, optional
            RA coordinates of ephemeris points to overlay.
        ephems_dec : array-like, optional
            Dec coordinates of ephemeris points to overlay.
        title : str, optional
            Title for the plot.
        figsize : tuple, optional
            Figure size (width, height).
        show_all_patches : bool, optional
            If True, show all patches in light gray as background.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection

        fig, ax = plt.subplots(figsize=figsize)

        # Collect patch rectangles
        rectangles = []
        patch_centers_ra = []
        patch_centers_dec = []

        for patch_id in patch_ids:
            if patch_id >= len(self.patches):
                print(f"Warning: patch_id {patch_id} out of range")
                continue
            patch = self.patches[patch_id]
            # Create rectangle from bottom-left corner
            rect = mpatches.Rectangle(
                (patch.bl_ra, patch.bl_dec),
                patch.width,
                patch.height,
                linewidth=1.5,
                edgecolor="blue",
                facecolor="lightblue",
                alpha=0.5,
            )
            rectangles.append(rect)
            patch_centers_ra.append(patch.ra)
            patch_centers_dec.append(patch.dec)

        # Add rectangles to plot
        for rect in rectangles:
            ax.add_patch(rect)

        # Plot patch centers
        if patch_centers_ra:
            ax.scatter(
                patch_centers_ra,
                patch_centers_dec,
                c="blue",
                s=20,
                marker="x",
                label="Patch Centers",
                zorder=3,
            )

        # Overlay ephemeris points if provided
        if ephems_ra is not None and ephems_dec is not None:
            ax.plot(
                ephems_ra, ephems_dec, "ro-", markersize=3, label="Ephemeris Trajectory", alpha=0.8, zorder=4
            )

        # Set axis limits based on data
        if patch_centers_ra:
            ra_margin = max(0.5, (max(patch_centers_ra) - min(patch_centers_ra)) * 0.1)
            dec_margin = max(0.5, (max(patch_centers_dec) - min(patch_centers_dec)) * 0.1)
            ax.set_xlim(
                max(patch_centers_ra) + ra_margin, min(patch_centers_ra) - ra_margin
            )  # Inverted for RA
            ax.set_ylim(min(patch_centers_dec) - dec_margin, max(patch_centers_dec) + dec_margin)

        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(title or f"Patches ({len(patch_ids)} shown)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # Invert RA axis (astronomical convention)
        ax.invert_xaxis()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Plot saved to {output_path}")

        return fig

    def get_patch_info(self, patch_id):
        """
        Get detailed information about a patch.

        Parameters
        ----------
        patch_id : int
            The ID of the patch.

        Returns
        -------
        dict
            Dictionary with patch center, corners, and dimensions.
        """
        if patch_id >= len(self.patches):
            raise ValueError(f"patch_id {patch_id} out of range (max: {len(self.patches)-1})")

        patch = self.patches[patch_id]
        return {
            "id": patch_id,
            "center_ra": patch.ra,
            "center_dec": patch.dec,
            "width_deg": patch.width,
            "height_deg": patch.height,
            "corners": patch.corners,
            "image_width": patch.image_width,
            "image_height": patch.image_height,
        }

    def export_image_collection(self, ic_to_export=None, guess_dist=None, patch=None, in_place=True):
        """
        Exports the ImageCollection to a new ImageCollection with the given guess distance and patch information.

        This populates various columns associated with the guess distance and patch information that
        will be helpful for later processing this ImageCollection as a WorkUnit.

        Parameters
        ----------
        ic_to_export : ImageCollection, optional
            The ImageCollection to export. If None, the self.ic is used
        guess_dist : float, optional
            The guess distance associated with the patch. To be applied to the exported ImageCollection's 'helio_guess_dist' column.
        patch : Patch or int, optional
            The patch to associate with the exported ImageCollection. If None, no patch information is added. May be either a Patch object or its index in self.patches
        in_place : bool, optional
            Whether to modify the ImageCollection in place with additional metadata or make a copy. Default is True.
        """
        if ic_to_export is None:
            # Export the current ImageCollection
            ic_to_export = self.ic

        if len(ic_to_export) < 1:
            raise ValueError(f"ImageCollection is empty, cannot export {ic_to_export}")

        new_ic = ic_to_export if in_place else ic_to_export.copy()

        # Add the metadata about the guess distance used when choosing this ImageCollection
        if guess_dist is not None:
            new_ic.data["helio_guess_dist"] = guess_dist

        # Add the metadata about the patch this ImageCollection represents.
        if patch is not None:
            if not isinstance(patch, Patch):
                if not isinstance(patch, int):
                    raise ValueError("Patch must be an integer or a Patch object")
                patch = self.get_patches()[patch]

            # Populate the patch information to construct the patch WCS
            patch_wcs = patch.to_wcs()
            new_ic.data["global_wcs"] = patch_wcs.to_header_string()
            new_ic.data["global_wcs_pixel_shape_0"] = patch_wcs.pixel_shape[0]
            new_ic.data["global_wcs_pixel_shape_1"] = patch_wcs.pixel_shape[1]

        # Reset standardizer-related metadata in our ImageCollection.
        new_ic.meta["n_stds"] = len(new_ic)
        new_ic.data.meta["std_idx"] = list(range(len(new_ic)))

        return new_ic

    def get_image_collection_from_patch(self, patch, guess_dist=0.0, min_overlap=0, max_images=None):
        """
        Filters down an ImageCollection to all images that overlap with a given patch.

        A patch may be specified by a Patch object or by its index in the PatchGrid.

        Note that by convention, the patches are defined to already be in whatever reflex-corrected coordinates
        are being used. So a given object might be in a different patch depending on the guess distance used
        for reflex correction.

        Adds an 'overlap_deg' column to the ImageCollection, which is the area of overlap
        between the patch and each image in square degrees.

        Parameters
        ----------
        patch : Patch or int
            The patch to filter by.
        guess_dist : float, optional
            The guess distance to use for reflex correction. If 0.0, the original coordinates are used.
        min_overlap : float, optional
            The minimum overlap area in square degrees to include an image in the filtered ImageCollection.
        max_images : int, optional
            The maximum number of images to return. If None, all images are returned.
        """
        if guess_dist is not None and guess_dist != 0.0 and guess_dist not in self.guess_dists:
            raise ValueError(f"Guess distance {guess_dist} not specified for Region Search")

        if not isinstance(patch, Patch):
            if not (isinstance(patch, int) or isinstance(patch, np.integer)):
                raise ValueError(f"Patch must be an integer or a Patch object, was: {type(patch)}")
            # Get the patch object from the index
            patch = self.get_patches()[patch]

        # To check if any of the images in the ImageCollection overlap with the patch,
        # first create a skycoord of the reflex-corrected image centers.
        ic_coords = SkyCoord(
            ra=self.ic.data[self.ic.reflex_corrected_col("ra", guess_dist)],
            dec=self.ic.data[self.ic.reflex_corrected_col("dec", guess_dist)],
            unit=(u.deg, u.deg),
            frame="icrs",
        )
        # Get the patch center coordinates
        patch_center = SkyCoord(
            ra=patch.ra,
            dec=patch.dec,
            unit=(u.deg, u.deg),
            frame="icrs",
        )

        # We want to get the maximum separation between the patch center and any corner of the chip
        # so that we can pre-filter out images that are too far away to overlap with the patch (since
        # checking overlap with polygons is expensive).
        chip_distance = 0 * u.deg
        for corner in ["tl", "tr", "br", "bl"]:
            curr_corner = SkyCoord(
                ra=self.ic.data[self.ic.reflex_corrected_col(f"ra_{corner}", guess_dist)][0],
                dec=self.ic.data[self.ic.reflex_corrected_col(f"dec_{corner}", guess_dist)][0],
                unit=(u.deg, u.deg),
                frame="icrs",
            )
            chip_distance = max(chip_distance, curr_corner.separation(ic_coords[0]))
        # The maximum separation between the patch center and the image center for there to be any overlap
        # of the image on the patch.
        max_sep = patch.patch_radius() + chip_distance

        # Get the indices of ic_coords that are within the patch size of the patch center
        seps = ic_coords.separation(patch_center)
        candidate_indices = np.where(seps <= max_sep)[0]

        # Iterate over all candidates and check if they actually overlap with the patch
        overlap_deg = np.zeros(len(self.ic), dtype=float)
        for ic_idx in candidate_indices:
            # Get our polygon from the visit and detector and our guess distance
            poly = self.chip_shapes[self.ic.data["visit"][ic_idx]][self.ic.data["detector"][ic_idx]][
                guess_dist
            ]
            overlap_deg[ic_idx] = patch.measure_overlap(poly)

        # Slice the ImageCollection to the subset of images that overlap with the patch
        overlap_mask = overlap_deg > min_overlap
        new_ic = self.ic[overlap_mask]
        new_ic["overlap_deg"] = overlap_deg[overlap_mask]
        if len(new_ic.data) < 1:
            # No images overlap with the patch
            return new_ic

        if max_images is not None and len(new_ic.data) > max_images:
            # Limit the number of images to the maximum number of images requested,
            # prioritizing the images with the highest overlap by sorting
            new_ic.data.sort(["overlap_deg"], reverse=True)
            new_ic.data = new_ic.data[:max_images]

        return self.export_image_collection(ic_to_export=new_ic, guess_dist=guess_dist, patch=patch)


class Patch:
    """
    A class to represent a RA-Dec aligned patch of the sky.
    Note that by convention, patches are defined as already existing in the reflex-corrected coordinate space
    of interest (or not reflex-corrected if guess_dist is 0.0).
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
        """
        Parameters
        ----------
        center_ra : float
            The center RA coordinate of the patch in degrees.
        center_dec : float
            The center Dec coordinate of the patch in degrees.
        width : float
            The width of the patch in degrees.
        height : float
            The height of the patch in degrees.
        image_width : int
            The width of the image in pixels.
        image_height : int
            The height of the image in pixels.
        pixel_scale : float
            The pixel scale in arcseconds per pixel.
        id : int
            The ID of the patch.
        """
        # Initialize the patch with the given parameters
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

        # Create a polygon representation of the patch
        self.polygon = Polygon(self.corners)

    def __str__(self):
        """Returns a string representation of the patch."""
        return f"Patch ID: {self.id} RA: {self.ra}, Dec: {self.dec}, Width (pixels): {self.width}, Height (piexels): {self.height}"

    def to_wcs(self):
        """Creates a WCS object from the patch parameters."""
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
        """Returns if an (RA, Dec) coordinate is within the patch.

        Note that the coordinates should be in the same coordinate system as the patch (i.e., reflex-corrected if applicable).

        Parameters
        ----------
        ra : float
            The RA coordinate in degrees.
        dec : float
            The Dec coordinate in degrees.
        """
        return self.polygon.contains(Point(ra, dec))

    def measure_overlap(self, poly):
        """
        Measures the overlap between the patch and a given shapely polygon.

        Parameters
        ----------
        poly : shapely.geometry.Polygon
            The polygon to measure the overlap with.
        Returns
        -------
        float
            The area of overlap in square degrees.
        """
        # Get the overlap between the shapely polygon and our patch in square degrees
        overlap = self.polygon.intersection(poly)
        return overlap.area

    def overlaps_polygon(self, poly):
        """
        Checks if the patch overlaps with a given shapely polygon.

        Note that the polygon's coordinates should be in the same coordinate system as the patch (i.e., reflex-corrected if applicable).

        Parameters
        ----------
        poly : shapely.geometry.Polygon
            The polygon to check for overlap with.

        Returns
        -------
        bool
            True if the patch overlaps with the polygon, False otherwise.
        """
        # True if the patch overlaps at all with the given shapely polygon
        return self.measure_overlap(poly) > 0

    def patch_radius(self):
        """
        Returns the radius of the patch in degrees, which is the distance from the center of the patch
        to one of its corners.

        Returns
        -------
        asropy.units.Quantity
            The radius of the patch in degrees.
        """
        return np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2) * u.deg
