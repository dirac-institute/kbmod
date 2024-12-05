"""A collection of utility functions for generating stamps."""

import jax.numpy as jnp
from numba import jit
from numba.typed import List
import numpy as np

from kbmod.configuration import SearchConfiguration
from kbmod.results import Results
from kbmod.search import (
    HAS_GPU,
    DebugTimer,
    ImageStack,
    RawImage,
    StampCreator,
    StampParameters,
    StampType,
    Logging,
    Trajectory,
)
from kbmod.trajectory_utils import predict_pixel_locations


logger = Logging.getLogger(__name__)


@jit(nopython=True)
def extract_stamp_np(img, x_val, y_val, radius):
    """Generate a single stamp as an numpy array from a given numpy array
    representation of the image.

    Parameters
    ----------
    img : `numpy.ndarray`
        The image data.
    x_val : `int`
        The x value corresponding to the center of the stamp.
    y_val : `int`
        The y value corresponding to the center of the stamp..
    radius : `int`
        The radius of the stamp. Must be >= 1.

    Returns
    -------
    stamp : `numpy.ndarray`
        A square matrix representing the stamp with NaNs anywhere
        there is no data.
    """
    (img_height, img_width) = img.shape

    # Compute the start and end x locations in the full image [x_img_s, x_img_e] and the
    # corresponding bounds in the stamp [x_stp_s, x_stp_e].
    x_img_s = 0 if x_val - radius < 0 else x_val - radius
    x_img_e = img_width if x_val + radius + 1 >= img_width else x_val + radius + 1
    x_width = x_img_e - x_img_s
    x_stp_s = x_img_s - (x_val - radius)
    x_stp_e = x_stp_s + x_width

    # Compute the start and end y locations in the full image [y_img_s, y_img_e] and the
    # corresponding bounds in the stamp [y_stp_s, y_stp_e].
    y_img_s = 0 if y_val - radius < 0 else y_val - radius
    y_img_e = img_height if y_val + radius + 1 >= img_height else y_val + radius + 1
    y_width = y_img_e - y_img_s
    y_stp_s = y_img_s - (y_val - radius)
    y_stp_e = y_stp_s + y_width

    # Create the stamp. Start with an array of NaN and then fill in whatever we cut
    # out of the image. Don't fill in anything if the stamp is completely off the image.
    stamp = np.full((2 * radius + 1, 2 * radius + 1), np.nan)
    if y_img_s <= y_img_e and x_img_s <= x_img_e:
        stamp[y_stp_s:y_stp_e, x_stp_s:x_stp_e] = img[y_img_s:y_img_e, x_img_s:x_img_e]
    return stamp


# Note that batching this over trajectories using a double loop (trjs and times)
# and writing into a single array does not help much. This is surprising because
# I would have expected compiling the outer loop to make a huge difference.
@jit(nopython=True)
def _extract_stamp_stack(imgs, x_vals, y_vals, radius, mask=None):
    """Generate a T x S x S sized array of stamps where T is the number
    of times to use and S is the stamp width (2 * radius + 1).

    Parameters
    ----------
    imgs : `numpy.ndarray` or `list` of `numpy.ndarray`
        All of the image data. This can either be a single T x H x W array, where T is the number
        of times, H is the image height, and W is the image width, or a list of T arrays each
        of which is H x W.
    x_vals : `np.array` of `int`
        The x values at the center of the stamp. Must be length T.
    y_vals : `np.array` of `int`
        The y values at the center of the stamp. Must be length T.
    radius : `int`
        The radius of the stamp. Must be >= 1.
    mask : `numpy.ndarray`, optional
        A numpy array of bools indicating which images to use. If None,
        uses all of the images.

    Returns
    -------
    stamp_stack : `numpy.ndarray`
        A T x S x S sized array where T is the number of times to use
        and S is the stamp width (2 * radius + 1).
    """
    num_times = len(imgs)

    # Fill in the stamp list.
    stamp_stack = np.full((num_times, 2 * radius + 1, 2 * radius + 1), np.nan)
    for idx in range(num_times):
        if mask is None or mask[idx]:
            stamp_stack[idx, :, :] = extract_stamp_np(imgs[idx], x_vals[idx], y_vals[idx], radius)
    return stamp_stack


class StampMaker:
    """A class to make stamps from an image stack.

    Attributes
    ----------
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.
    img_data : `list` of `numpy.ndarray` or `numpy.ndarray`
        The data from all science images from which to construct the stamps.
    var_data : `list` of `numpy.ndarray` or `numpy.ndarray`
        The data from all variance images from which to construct the stamps.
    radius : `int`
        The default stamp radius.
    num_times : `int`
        The number of time stamps.
    zeroed_times : `numpy.ndarray`
        The zeroed timestamps from the image stack computed once and cached.

    Parameters
    ----------
    im_stack : `ImageStack`
        The images from which to build the co-added stamps.
    radius : `int`
        The default stamp radius.
    mk_copy : `bool`
        Make a local copy of the data. This accelerates stamp generation
        at the cost of a lot more memory.
    """

    def __init__(self, im_stack, radius, mk_copy=True):
        self.im_stack = im_stack
        self.zeroed_times = np.array(im_stack.build_zeroed_times())
        self.num_times = len(self.zeroed_times)

        # Extract the image data. If mk_copy == True this copies it all into a
        # local numpy array for speed.
        if mk_copy:
            self.img_data = np.empty((self.num_times, im_stack.get_height(), im_stack.get_width()))
            for idx in range(self.num_times):
                self.img_data[idx, :, :] = im_stack.get_single_image(idx).get_science().image
        else:
            self.img_data = List()
            for idx in range(self.num_times):
                self.img_data.append(im_stack.get_single_image(idx).get_science().image)

        # Extract the variance data. If mk_copy == True this copies it all into a
        # local numpy array for speed.
        if mk_copy:
            self.var_data = np.empty((self.num_times, im_stack.get_height(), im_stack.get_width()))
            for idx in range(self.num_times):
                self.var_data[idx, :, :] = im_stack.get_single_image(idx).get_variance().image
        else:
            self.var_data = List()
            for idx in range(self.num_times):
                self.var_data.append(im_stack.get_single_image(idx).get_variance().image)

        if radius < 1:
            raise ValueError(f"Invalid stamp radius {radius}")
        self.radius = radius
        self.stamp_width = 2 * radius + 1

    def create_stamp_stack(self, trj, mask=None, use_var=False):
        """Generate the stack of all stamps for a given trajectory.

        Parameters
        ----------
        trj : `Trajectory`
            The trajectory to evaluate.
        mask : `numpy.ndarray`, optional
            A numpy array of bools indicating which images to use. If None,
            uses all of them.
        use_var : `bool`
            Create a stack from the variance layer instead of the science layer.

        Returns
        -------
        stamp_stack : `numpy.ndarray`
            A T x (2*R+1) x (2*R+1) sized array where T is the number of times and R is
            the stamp radius.
        """
        # Compute the predicted x and y positions as integers (vectorized).
        # We do not use an explicit floor in order to stay consistent with the
        # existing implementation.
        x_vals = (self.zeroed_times * trj.vx + trj.x + 0.5).astype(int)
        y_vals = (self.zeroed_times * trj.vy + trj.y + 0.5).astype(int)

        # Use the numba compiled function to compute the actual stamps.
        if use_var:
            return _extract_stamp_stack(self.var_data, x_vals, y_vals, self.radius, mask=mask)
        else:
            return _extract_stamp_stack(self.img_data, x_vals, y_vals, self.radius, mask=mask)

    def make_coadds(self, trj, coadd_types, mask=None):
        """The different coadded stamps: sum, mean, median, variance weighted.

        Parameters
        ----------
        trj : `Trajectory`
            The trajectory to evaluate.
        coadd_types : `list` or `set`
            A list of coadd types to generate. Can include:
            "sum", "mean", "median", and "weighted".
        mask : `numpy.ndarray`, optional
            A numpy array of bools indicating which images to use. If None,
            uses all of them.

        Returns
        -------
        coadds : `dict`
            A dictionary mapping the coadd name to a numpy array for that coadded stamp.
        """
        sci_stack = self.create_stamp_stack(trj, mask=mask, use_var=False)
        (n_times, height, width) = sci_stack.shape

        # We need to mask out any columns with all NaNs.  We do this by setting ALL the science
        # values to 0.0 and the variance values to a very large number.
        some_pixel_valid = np.all(np.isnan(sci_stack), axis=0)
        if np.any(some_pixel_valid):
            pixel_mask = np.tile(some_pixel_valid.flatten(), n_times).reshape(n_times, height, width)
            sci_stack[pixel_mask] = 0.0

        coadds = {}
        if "sum" in coadd_types:
            coadds["sum"] = np.nansum(sci_stack, axis=0)
        if "mean" in coadd_types:
            coadds["mean"] = np.nanmean(sci_stack, axis=0)
        if "median" in coadd_types:
            coadds["median"] = np.nanmedian(sci_stack, axis=0)
        if "jax_median" in coadd_types:
            coadds["jax_median"] = np.asarray(jnp.nanmedian(jnp.asarray(sci_stack), axis=0))

        if "weighted" in coadd_types:
            # The variance weighted computation requires a second stack of stamps.
            var_stack = self.create_stamp_stack(trj, mask=mask, use_var=True)

            # Compute the pixels that are valid to use in the variance weighted computation.
            pix_valid = ~(np.isnan(sci_stack) | np.isnan(var_stack) | (var_stack == 0.0))

            # Compute the weighted science values and the weights.
            weights = np.zeros((n_times, height, width))
            weights[pix_valid] = 1.0 / var_stack[pix_valid]

            # Compute the variance weighted values of the science pixels.
            weighted_sci = np.zeros((n_times, height, width))
            weighted_sci[pix_valid] = sci_stack[pix_valid] * weights[pix_valid]
            weighted_sum = np.sum(weighted_sci, axis=0)

            # Compute the scaling factor (sum of the weights) for each pixel.
            # If a pixel has no data, then use a large scaling factor to avoid divide by zero.
            sum_of_weights = np.sum(weights, axis=0)
            sum_of_weights[sum_of_weights == 0.0] = 1e24

            coadds["weighted"] = weighted_sum / sum_of_weights

        return coadds

    # A JAX + Vectorized approach for generating all the coadd stamps at once.
    def append_coadds(self, result_data, coadd_types, use_masks=False):
        """Append one or more stamp coadds to the results data without filtering.

        result_data : `Results`
            The current set of results. Modified directly.
        coadd_types : `list` or `set`
            A list of coadd types to generate. Can include:
            "sum", "mean", "median", and "weighted".
        use_masks : `bool`
            Use the 'obs_valid' column to filter which images go into the coadds. Otherwise
            (if False), use all images.
            Default: False

        Returns
        -------
        coadds : `dict`
            A dictionary mapping the coadd name to a numpy array for that coadded stamp.
        """
        # Set up the information for per-image masking. We can only mask if we have the
        # 'obs_valid' column to use as a mask.
        use_masks = use_masks and "obs_valid" in result_data.colnames
        current_mask = None

        # Compute the position of each result at each time.
        num_trjs = len(result_data)
        x_pos = predict_pixel_locations(
            self.zeroed_times,
            result_data["x"].data,
            result_data["vx"].data,
            centered=True,
            as_int=True,
        )
        y_pos = predict_pixel_locations(
            self.zeroed_times,
            result_data["y"].data,
            result_data["vy"].data,
            centered=True,
            as_int=True,
        )

        # Collect science stamps for each result using the predicted positions.
        # TODO: See if we can vectorize this.
        all_stamps = np.empty((num_trjs, self.num_times, self.stamp_width, self.stamp_width))
        for idx in range(num_trjs):
            if use_masks:
                current_mask = result_data["obs_valid"][idx]
            all_stamps[idx, :, :, :] = _extract_stamp_stack(
                self.img_data,
                x_pos[idx],
                y_pos[idx],
                self.radius,
                mask=current_mask,
            )

        # Compute each of the 'basic' coadds.
        if "sum" in coadd_types:
            coadds = jnp.nan_to_num(jnp.nansum(all_stamps, axis=1), nan=0.0)
            result_data.table["coadd_sum"] = coadds
        if "mean" in coadd_types:
            coadds = jnp.nan_to_num(jnp.nanmean(all_stamps, axis=1), nan=0.0)
            result_data.table["coadd_mean"] = coadds
        if "median" in coadd_types:
            coadds = jnp.nan_to_num(jnp.nanmedian(all_stamps, axis=1), nan=0.0)
            result_data.table["coadd_median"] = coadds

        # Compute the weighted coadd.
        if "weighted" in coadd_types:
            # Create a stack of the variance stamps. TODO: See if we can vectorize this.
            var_stack = np.empty((num_trjs, self.num_times, self.stamp_width, self.stamp_width))
            for idx in range(num_trjs):
                if use_masks:
                    current_mask = result_data["obs_valid"][idx]
                var_stack[idx, :, :, :] = _extract_stamp_stack(
                    self.var_data,
                    x_pos[idx],
                    y_pos[idx],
                    self.radius,
                    mask=current_mask,
                )
            var_stack = jnp.asarray(var_stack)

            # Compute the pixels that are valid to use in the variance weighted computation.
            pix_valid = ~(jnp.isnan(all_stamps) | jnp.isnan(var_stack) | (var_stack == 0.0))

            # Compute the weights and the weighted science values.
            weights = jnp.where(pix_valid, 1.0 / var_stack, 0.0)
            sum_of_weights = jnp.sum(weights, axis=1)
            sum_of_weighted_science = jnp.sum(jnp.where(pix_valid, all_stamps * weights, 0.0), axis=1)

            # Compute the actual coadds.
            coadds = jnp.where(sum_of_weights == 0.0, 0.0, sum_of_weighted_science / sum_of_weights)
            result_data.table["coadd_weighted"] = coadds
