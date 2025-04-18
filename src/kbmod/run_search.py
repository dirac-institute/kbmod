import logging

import numpy as np
import sys

import kbmod.search as kb

from .filters.clustering_filters import apply_clustering
from .filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from .filters.stamp_filters import append_all_stamps, append_coadds

from .results import Results
from .trajectory_generator import create_trajectory_generator
from .trajectory_utils import predict_pixel_locations


logger = kb.Logging.getLogger(__name__)


def configure_kb_search_stack(search, config):
    """Configure the kbmod SearchStack object from a search configuration.

    Parameters
    ----------
    search : `kb.StackSearch`
        The SearchStack object.
    config : `SearchConfiguration`
        The configuration parameters
    """
    width = search.get_image_width()
    height = search.get_image_height()

    # Set the search bounds.
    if config["x_pixel_bounds"] and len(config["x_pixel_bounds"]) == 2:
        search.set_start_bounds_x(config["x_pixel_bounds"][0], config["x_pixel_bounds"][1])
    elif config["x_pixel_buffer"] and config["x_pixel_buffer"] > 0:
        search.set_start_bounds_x(-config["x_pixel_buffer"], width + config["x_pixel_buffer"])

    if config["y_pixel_bounds"] and len(config["y_pixel_bounds"]) == 2:
        search.set_start_bounds_y(config["y_pixel_bounds"][0], config["y_pixel_bounds"][1])
    elif config["y_pixel_buffer"] and config["y_pixel_buffer"] > 0:
        search.set_start_bounds_y(-config["y_pixel_buffer"], height + config["y_pixel_buffer"])

    # Set the results per pixel.
    search.set_results_per_pixel(config["results_per_pixel"])

    # If we are using gpu_filtering, enable it and set the parameters.
    if config["sigmaG_filter"] and config["gpu_filter"]:
        logger.debug("Using in-line GPU sigmaG filtering methods")
        coeff = SigmaGClipping.find_sigma_g_coeff(
            config["sigmaG_lims"][0],
            config["sigmaG_lims"][1],
        )
        search.enable_gpu_sigmag_filter(
            np.array(config["sigmaG_lims"]) / 100.0,
            coeff,
            config["lh_level"],
        )
    else:
        search.disable_gpu_sigmag_filter()

    # If we are using an encoded image representation on GPU, enable it and
    # set the parameters.
    if config["encode_num_bytes"] > 0:
        search.enable_gpu_encoding(config["encode_num_bytes"])


def check_gpu_memory(config, stack, trj_generator=None):
    """Check whether we can run this search on the GPU.

    Parameters
    ----------
    config : `SearchConfiguration`
        The configuration parameters
    stack : `ImageStack`
        The stack before the masks have been applied. Modified in-place.
    trj_generator : `TrajectoryGenerator`, optional
        The object to generate the candidate trajectories for each pixel.

    Returns
    -------
    valid : `bool`
        Returns True if the search will fit on GPU and False otherwise.
    """
    bytes_free = kb.get_gpu_free_memory()
    logger.debug(f"Checking GPU memory needs (Free memory = {bytes_free} bytes):")

    # Compute the size of the PSI/PHI images and the full image stack (for stamp creation).
    gpu_float_size = sys.getsizeof(np.single(10.0))
    img_stack_size = stack.get_total_pixels() * gpu_float_size
    logger.debug(f"  PSI = {img_stack_size} bytes\n  PHI = {img_stack_size} bytes")

    # Compute the size of the candidates
    trj_size = sys.getsizeof(kb.Trajectory())
    if trj_generator is not None:
        candidate_memory = trj_size * len(trj_generator)
    else:
        candidate_memory = 0
    logger.debug(f"  Candidates = {candidate_memory} bytes.")

    # Compute the size of the results.
    result_memory = (stack.get_width() * stack.get_height() * config["results_per_pixel"]) * trj_size
    logger.debug(f"  Results = {result_memory} bytes.")

    return bytes_free > (2 * img_stack_size + result_memory + candidate_memory)


class SearchRunner:
    """A class to run the KBMOD grid search."""

    def __init__(self):
        pass

    def load_and_filter_results(self, search, config):
        """This function loads results that are output by the gpu grid search.
        Results are loaded in chunks and evaluated to see if the minimum
        likelihood level has been reached. If not, another chunk of results is
        fetched. The results are filtered using a clipped-sigmaG filter as they
        are loaded and only the passing results are kept.

        Parameters
        ----------
        search : `kbmod.search`
            The search function object.
        config : `SearchConfiguration`
            The configuration parameters
        chunk_size : int
            The number of results to load at a given time from search.

        Returns
        -------
        keep : `Results`
            A Results object containing values from trajectories.
        """
        # Parse and check the configuration parameters.
        num_obs = config["num_obs"]
        sigmaG_lims = config["sigmaG_lims"]
        clip_negative = config["clip_negative"]
        lh_level = config["lh_level"]
        max_lh = config["max_lh"]
        chunk_size = config["chunk_size"]
        if chunk_size <= 0:
            raise ValueError(f"Invalid chunk size {chunk_size}")

        # Set up the list of results.
        do_tracking = config["track_filtered"]
        img_stack = search.get_imagestack()
        keep = Results(track_filtered=do_tracking)

        # Set up the clipped sigmaG filter.
        if sigmaG_lims is not None:
            bnds = sigmaG_lims
        else:
            bnds = [25, 75]
        clipper = SigmaGClipping(bnds[0], bnds[1], 2, clip_negative)

        total_found = search.get_number_total_results()
        logger.info(f"Retrieving Results (total={total_found})")
        likelihood_limit = False
        res_num = 0
        total_count = 0

        # Keep retrieving results until they fall below the threshold or we run out of results.
        while likelihood_limit is False and res_num < total_found:
            logger.info(f"Chunk Start = {res_num} (size={chunk_size})")
            results = search.get_results(res_num, chunk_size)
            logger.info(f"Chunk Max Likelihood = {results[0].lh}")
            logger.info(f"Chunk Min. Likelihood = {results[-1].lh}")

            trj_batch = []
            for i, trj in enumerate(results):
                # Stop as soon as we hit a result below our limit, because anything after
                # that is not guarrenteed to be valid due to potential on-GPU filtering.
                if trj.lh < lh_level:
                    likelihood_limit = True
                    break

                if trj.lh < max_lh:
                    trj_batch.append(trj)
                    total_count += 1

            batch_size = len(trj_batch)
            logger.info(f"Extracted batch of {batch_size} results for total of {total_count}")

            if batch_size > 0:
                psi_batch = search.get_psi_curves(trj_batch)
                phi_batch = search.get_phi_curves(trj_batch)

                result_batch = Results.from_trajectories(trj_batch, track_filtered=do_tracking)
                result_batch.add_psi_phi_data(psi_batch, phi_batch)

                # Do the sigma-G filtering and subsequent stats filtering.
                if config["sigmaG_filter"]:
                    apply_clipped_sigma_g(clipper, result_batch)
                obs_row_mask = result_batch["obs_count"] >= num_obs
                result_batch.filter_rows(obs_row_mask, "obs_count")
                logger.debug(f"After obs_count >= {num_obs}. Batch size = {len(result_batch)}")

                if lh_level > 0.0:
                    lh_row_mask = result_batch["likelihood"] >= lh_level
                    result_batch.filter_rows(lh_row_mask, "likelihood")
                    logger.debug(f"After likelihood >= {lh_level}. Batch size = {len(result_batch)}")

                # Add the results to the final set.
                keep.extend(result_batch)
            res_num += chunk_size
        return keep

    def do_gpu_search(self, config, stack, trj_generator):
        """Performs search on the GPU.

        Parameters
        ----------
        config : `SearchConfiguration`
            The configuration parameters
        stack : `ImageStack`
            The stack before the masks have been applied. Modified in-place.
        trj_generator : `TrajectoryGenerator`
            The object to generate the candidate trajectories for each pixel.

        Returns
        -------
        keep : `Results`
            The results.
        """
        if not check_gpu_memory(config, stack, trj_generator):
            raise ValueError("Insufficient GPU memory to conduct the search.")

        # Do some very basic checking of the configuration parameters.
        min_num_obs = int(config["num_obs"])
        if min_num_obs > stack.img_count():
            raise ValueError(
                f"num_obs ({min_num_obs}) is greater than the number of images ({stack.img_count()})."
            )

        # Create the search object which will hold intermediate data and results.
        search = kb.StackSearch(stack)
        configure_kb_search_stack(search, config)

        search_timer = kb.DebugTimer("grid search", logger)
        logger.debug(f"{trj_generator}")

        # Do the actual search.
        candidates = [trj for trj in trj_generator]
        try:
            search.search_all(candidates, int(config["num_obs"]))
        except:
            # Delete the search object to force the GPU memory cleanup.
            del search
            raise

        search_timer.stop()

        # Load the results.
        keep = self.load_and_filter_results(search, config)

        # Force the deletion of the on-GPU data.
        search.clear_psi_phi()

        return keep

    def run_search(
        self,
        config,
        stack,
        trj_generator=None,
        workunit=None,
        extra_meta=None,
    ):
        """This function serves as the highest-level python interface for starting
        a KBMOD search given an ImageStack and SearchConfiguration.

        Parameters
        ----------
        config : `SearchConfiguration`
            The configuration parameters
        stack : `ImageStack`
            The stack before the masks have been applied. Modified in-place.
        trj_generator : `TrajectoryGenerator`, optional
            The object to generate the candidate trajectories for each pixel.
            If None uses the default EclipticCenteredSearch
        workunit : `WorkUnit`, optional
            An optional WorkUnit with additional meta-data, including the per-image WCS.
        extra_meta : `dict`, optional
            Any additional metadata to save as part of the results file.

        Returns
        -------
        keep : `Results`
            The results.
        """
        if config["debug"]:
            logging.basicConfig(level=logging.DEBUG)
            logger.debug("Starting Search")
            logger.debug(kb.stat_gpu_memory_mb())

        if not kb.HAS_GPU:
            logger.warning("Code was compiled without GPU.")

        full_timer = kb.DebugTimer("KBMOD", logger)

        # Apply the mask to the images.
        if config["do_mask"]:
            for i in range(stack.img_count()):
                stack.get_single_image(i).apply_mask(0xFFFFFF)

        # Perform the actual search.
        if trj_generator is None:
            trj_generator = create_trajectory_generator(config, work_unit=None)
        keep = self.do_gpu_search(config, stack, trj_generator)

        if config["do_clustering"]:
            cluster_timer = kb.DebugTimer("clustering", logger)
            mjds = [stack.get_obstime(t) for t in range(stack.img_count())]
            cluster_params = {
                "cluster_type": config["cluster_type"],
                "cluster_eps": config["cluster_eps"],
                "cluster_v_scale": config["cluster_v_scale"],
                "times": np.asarray(mjds),
            }
            apply_clustering(keep, cluster_params)
            cluster_timer.stop()

        # Generate coadded stamps without filtering -- both the "stamp" column
        # as well as any additional coadds.
        stamp_radius = config["stamp_radius"]
        stamp_type = config["stamp_type"]
        coadds = set(config["coadds"])
        coadds.add(stamp_type)

        # Add all the "coadd_*" columns and a "stamp" column. This is only
        # short term until we stop using the "stamp" column.
        append_coadds(keep, stack, coadds, stamp_radius)
        if f"coadd_{stamp_type}" in keep.colnames:
            keep.table["stamp"] = keep.table[f"coadd_{stamp_type}"]

        # Extract all the stamps for all time steps and append them onto the result rows.
        if config["save_all_stamps"]:
            append_all_stamps(keep, stack, stamp_radius)

        # Append additional information derived from the WorkUnit if one is provided,
        # including a global WCS and per-time (RA, dec) predictions for each image.
        if workunit is not None:
            keep.table.wcs = workunit.wcs
            append_positions_to_results(workunit, keep)

        # Create and save any additional meta data that should be saved with the results.
        num_img = stack.img_count()

        if extra_meta is not None:
            meta_to_save = extra_meta.copy()
        else:
            meta_to_save = {}
        meta_to_save["num_img"] = num_img
        meta_to_save["dims"] = stack.get_width(), stack.get_height()
        keep.set_mjd_utc_mid(np.array([stack.get_obstime(i) for i in range(num_img)]))

        if config["result_filename"] is not None:
            logger.info(f"Saving results table to {config['result_filename']}")
            if not config["save_all_stamps"]:
                keep.write_table(
                    config["result_filename"], cols_to_drop=["all_stamps"], extra_meta=meta_to_save
                )
            else:
                keep.write_table(config["result_filename"], extra_meta=meta_to_save)
        full_timer.stop()

        return keep

    def run_search_from_work_unit(self, work):
        """Run a KBMOD search from a WorkUnit object.

        Parameters
        ----------
        work : `WorkUnit`
            The input data and configuration.

        Returns
        -------
        keep : `Results`
            The results.
        """
        trj_generator = create_trajectory_generator(work.config, work_unit=work)

        # Extract extra metadata. We do not use the full org_image_meta table from the WorkUnit
        # because this can be very large and varies with the source. Instead we only save a
        # few pre-defined fields to the results data.
        extra_meta = work.get_constituent_meta(["visit", "filter"])

        # Run the search.
        return self.run_search(
            work.config,
            work.im_stack,
            trj_generator=trj_generator,
            workunit=work,
            extra_meta=extra_meta,
        )


def append_positions_to_results(workunit, results):
    """Append predicted (x, y) and (RA, dec) positions in the original images. If
    the images were reprojected, also appends the (RA, dec) in the common frame.

    Parameters
    ----------
    workunit : `WorkUnit`
        The WorkUnit with all the WCS information.
    results : `Results`
        The current table of results including the per-pixel trajectories.
        This is modified in-place.
    """
    num_results = len(results)
    if num_results == 0:
        return  # Nothing to do

    num_times = workunit.im_stack.img_count()
    times = workunit.im_stack.build_zeroed_times()

    # Predict where each candidate trajectory will be at each time step in the
    # common WCS frame. These are the pixel locations used to assess the trajectory.
    xp = predict_pixel_locations(times, results["x"], results["vx"], as_int=False)
    yp = predict_pixel_locations(times, results["y"], results["vy"], as_int=False)
    results.table["pred_x"] = xp
    results.table["pred_y"] = yp

    # Compute the predicted (RA, dec) positions for each trajectory the common WCS
    # frame and original image WCS frames.
    all_inds = np.arange(num_times)
    all_ra = np.zeros((len(results), num_times))
    all_dec = np.zeros((len(results), num_times))
    if workunit.wcs is not None:
        logger.info("Found common WCS. Adding global_ra and global_dec columns.")

        # Compute the (RA, dec) for each result x time in the common WCS frame.
        skypos = workunit.wcs.pixel_to_world(xp, yp)
        results.table["global_ra"] = skypos.ra.degree
        results.table["global_dec"] = skypos.dec.degree

        # Loop over the trajectories to build the (RA, dec) positions in each image's WCS frame.
        for idx in range(num_results):
            # Build a list of this trajectory's RA, dec position at each time.
            pos_list = [skypos[idx, j] for j in range(num_times)]
            img_skypos = workunit.image_positions_to_original_icrs(
                image_indices=all_inds,  # Compute for all times.
                positions=pos_list,
                input_format="radec",
                output_format="radec",
                filter_in_frame=False,
            )

            # We get back a list of SkyCoord, because we gave a list.
            # So we flatten it and extract the coordinate values.
            for time_idx in range(num_times):
                all_ra[idx, time_idx] = img_skypos[time_idx].ra.degree
                all_dec[idx, time_idx] = img_skypos[time_idx].dec.degree

    else:
        logger.info("No common WCS found. Skipping global_ra and global_dec columns.")

        # If there are no global WCS, we just predict per image.
        for time_idx in range(num_times):
            wcs = workunit.get_wcs(time_idx)
            if wcs is not None:
                skypos = wcs.pixel_to_world(xp[:, time_idx], yp[:, time_idx])
                all_ra[:, time_idx] = skypos.ra.degree
                all_dec[:, time_idx] = skypos.dec.degree

    # Add the per-image coordinates to the results table.
    results.table["img_ra"] = all_ra
    results.table["img_dec"] = all_dec

    # If we have have per-image WCSes, compute the pixel location in the original image.
    if "per_image_wcs" in workunit.org_img_meta.colnames:
        img_x = np.zeros((len(results), num_times))
        img_y = np.zeros((len(results), num_times))
        for time_idx in range(num_times):
            wcs = workunit.org_img_meta["per_image_wcs"][time_idx]
            if wcs is not None:
                xy_pos = wcs.world_to_pixel_values(all_ra[:, time_idx], all_dec[:, time_idx])
                img_x[:, time_idx] = xy_pos[0]
                img_y[:, time_idx] = xy_pos[1]

        results.table["img_x"] = img_x
        results.table["img_y"] = img_y
