import logging
import numpy as np
import psutil
import os
import time

import kbmod.search as kb

from .filters.clustering_filters import apply_clustering
from .filters.clustering_grid import apply_trajectory_grid_filter
from .filters.sigma_g_filter import apply_clipped_sigma_g, SigmaGClipping
from .filters.stamp_filters import append_all_stamps, append_coadds, filter_stamps_by_cnn
from .filters.sns_filters import no_op_filter

from .results import Results, write_results_to_files_destructive
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

    # Set the filtering parameters.
    search.set_min_obs(int(config["num_obs"]))
    search.set_min_lh(config["lh_level"])

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

    # Clear the cached results.
    search.clear_results()


def check_gpu_memory(config, stack, trj_generator=None):
    """Check whether we can run this search on the GPU.

    Parameters
    ----------
    config : `SearchConfiguration`
        The configuration parameters
    stack : `ImageStackPy`
        The stack of image data.
    trj_generator : `TrajectoryGenerator`, optional
        The object to generate the candidate trajectories for each pixel.

    Returns
    -------
    valid : `bool`
        Returns True if the search will fit on GPU and False otherwise.
    """
    bytes_free = kb.get_gpu_free_memory()
    logger.debug(f"Checking GPU memory needs (Free memory = {bytes_free} bytes):")

    # Compute the size of the PSI/PHI images using the encoded size (-1 means 4 bytes).
    gpu_float_size = config["encode_num_bytes"] if config["encode_num_bytes"] > 0 else 4
    img_stack_size = stack.get_total_pixels() * gpu_float_size
    logger.debug(
        f"  PSI/PHI encoding at {gpu_float_size} bytes per pixel.\n"
        f"  PSI = {img_stack_size} bytes\n  PHI = {img_stack_size} bytes"
    )

    # Compute the size of the candidates
    num_candidates = 0 if trj_generator is None else len(trj_generator)
    candidate_memory = kb.TrajectoryList.estimate_memory(num_candidates)
    logger.debug(f"  Candidates ({num_candidates}) = {candidate_memory} bytes.")

    # Compute the size of the results.  We use the bounds from the search dimensions
    # (not the raw image dimensions).
    search_width = stack.width
    if config["x_pixel_bounds"] and len(config["x_pixel_bounds"]) == 2:
        search_width = config["x_pixel_bounds"][1] - config["x_pixel_bounds"][0]
    elif config["x_pixel_buffer"] and config["x_pixel_buffer"] > 0:
        search_width += 2 * config["x_pixel_buffer"]

    search_height = stack.height
    if config["y_pixel_bounds"] and len(config["y_pixel_bounds"]) == 2:
        search_height = config["y_pixel_bounds"][1] - config["y_pixel_bounds"][0]
    elif config["y_pixel_buffer"] and config["y_pixel_buffer"] > 0:
        search_height += 2 * config["y_pixel_buffer"]

    num_results = search_width * search_height * config["results_per_pixel"]
    result_memory = kb.TrajectoryList.estimate_memory(num_results)
    logger.debug(f"  Results ({num_results}) = {result_memory} bytes.")

    return bytes_free > (2 * img_stack_size + result_memory + candidate_memory)


class SearchRunner:
    """A class to run the KBMOD grid search.

    Attributes
    ----------
    phase_times : `dict`
        A dictionary mapping the search phase to the timing information,
        a list of [starting time, ending time] in seconds.
    phase_memory : `dict`
        A dictionary mapping the search phase the memory information,
        a list of [starting memory, ending memory] in bytes.
    """

    def __init__(self, config=None):
        self.phase_times = {}
        self.phase_memory = {}

    def _start_phase(self, phase_name):
        """Start recording stats for the current phase.

        Parameters
        ----------
        phase_name : `str`
            The current phase.
        """
        # Record the start time.
        self.phase_times[phase_name] = [time.time(), None]

        # Record the starting memory.
        memory_info = psutil.Process().memory_info()
        self.phase_memory[phase_name] = [memory_info.rss, None]

    def _end_phase(self, phase_name):
        """Finish recording stats for the current phase.

        Parameters
        ----------
        phase_name : `str`
            The current phase.
        """
        if phase_name not in self.phase_times:
            raise KeyError(f"Phase {phase_name} has not been started.")

        # Record the end time.
        self.phase_times[phase_name][1] = time.time()
        delta_t = self.phase_times[phase_name][1] - self.phase_times[phase_name][0]
        logger.debug(f"Finished {phase_name} in {delta_t} seconds.")

        # Record the starting memory.
        memory_info = psutil.Process().memory_info()
        self.phase_memory[phase_name][1] = memory_info.rss

    def display_phase_stats(self):
        """Output the statistics for each phase."""
        for phase in self.phase_times:
            print(f"{phase}:")

            if self.phase_times[phase][1] is not None:
                delta_t = self.phase_times[phase][1] - self.phase_times[phase][0]
                print(f"    Time (sec) = {delta_t}")
            else:
                print(f"    Time (sec) = Unfinished")

            print(f"    Memory Start (mb) = {self.phase_memory[phase][0] / (1024.0 * 1024.0)}")
            if self.phase_memory[phase][1] is not None:
                print(f"    Memory End (mb) = {self.phase_memory[phase][1] / (1024.0 * 1024.0)}")
            else:
                print(f"    Memory End (mb) = Unfinished")

    def load_and_filter_results(self, search, config, batch_size=100_000):
        """This function loads results that are output by the grid search.
        It can then generate psi + phi curves and perform sigma-G filtering
        (depending on the parameter settings).

        Parameters
        ----------
        search : `kbmod.search`
            The search function object.
        config : `SearchConfiguration`
            The configuration parameters
        batch_size : `int`
            The number of results to load at once. This is used to limit the
            memory usage when loading results.
            Default is 100000.

        Returns
        -------
        keep : `Results`
            A Results object containing values from trajectories.
        """
        self._start_phase("load_and_filter_results")
        num_times = search.get_num_images()

        # Set up the clipped sigmaG filter.
        if config["sigmaG_lims"] is not None:
            bnds = config["sigmaG_lims"]
        else:
            bnds = [25, 75]
        clipper = SigmaGClipping(bnds[0], bnds[1], 2, config["clip_negative"])

        keep = Results(track_filtered=config["track_filtered"])

        # Retrieve a reference to all the results and compile the results table.
        result_trjs = search.get_all_results()
        logger.info(f"Retrieving Results (total={len(result_trjs)})")
        if len(result_trjs) < 1:
            logger.info(f"No results found.")
            return keep
        logger.info(f"Max Likelihood = {result_trjs[0].lh}")
        logger.info(f"Min. Likelihood = {result_trjs[-1].lh}")

        # Perform near duplicate filtering.
        if config["near_dup_thresh"] is not None and config["near_dup_thresh"] > 0:
            self._start_phase("near duplicate removal")
            bin_width = config["near_dup_thresh"]
            max_dt = np.max(search.zeroed_times) - np.min(search.zeroed_times)
            logger.info(f"Prefiltering Near Duplicates (bin_width={bin_width}, max_dt={max_dt})")
            result_trjs, _ = apply_trajectory_grid_filter(result_trjs, bin_width, max_dt)
            logger.info(f"After prefiltering {len(result_trjs)} remaining.")
            self._end_phase("near duplicate removal")

        # Transform the results into a Result table in batches while doing sigma-G filtering.
        batch_start = 0
        while batch_start < len(result_trjs):
            batch_end = min(batch_start + batch_size, len(result_trjs))
            batch = result_trjs[batch_start:batch_end]
            batch_results = Results.from_trajectories(batch, track_filtered=config["track_filtered"])

            if config["generate_psi_phi"]:
                psi_phi_batch = search.get_all_psi_phi_curves(batch)
                batch_results.add_psi_phi_data(psi_phi_batch[:, :num_times], psi_phi_batch[:, num_times:])

            # Do the sigma-G filtering and subsequent stats filtering.
            if config["sigmaG_filter"]:
                if not config["generate_psi_phi"]:
                    raise ValueError("Unable to do sigma-G filtering without psi and phi curves.")
                apply_clipped_sigma_g(clipper, batch_results)

                # Re-test the obs_count and likelihood after sigma-G has removed points.
                row_mask = batch_results["obs_count"] >= config["num_obs"]
                if config["lh_level"] > 0.0:
                    row_mask = row_mask & (batch_results["likelihood"] >= config["lh_level"])
                batch_results.filter_rows(row_mask, "sigma-g")
                logger.debug(f"After sigma-G filtering, batch size = {len(batch_results)}")

            # Append the unfiltered results to the final table.
            logger.debug(f"Added {len(batch_results)} results from batch [{batch_start}, {batch_end}).")
            keep.extend(batch_results)
            batch_start += batch_size

        # Save the timing information.
        self._end_phase("load_and_filter_results")

        # Return the extracted and unfiltered results.
        return keep

    def do_core_search(self, config, stack, trj_generator):
        """Performs search on the GPU.

        Parameters
        ----------
        config : `SearchConfiguration`
            The configuration parameters
        stack : `ImageStackPy`
            The stack of image data.
        trj_generator : `TrajectoryGenerator`
            The object to generate the candidate trajectories for each pixel.

        Returns
        -------
        keep : `Results`
            The results.
        """
        self._start_phase("do_core_search")

        use_gpu = not config["cpu_only"]
        if use_gpu and not check_gpu_memory(config, stack, trj_generator):
            raise ValueError("Insufficient GPU memory to conduct the search.")

        # Create the search object which will hold intermediate data and results.
        search = kb.StackSearch(
            stack.sci,
            stack.var,
            stack.psfs,
            stack.zeroed_times,
            config["encode_num_bytes"],
        )
        configure_kb_search_stack(search, config)

        # Do the actual search.
        self._start_phase("grid search")
        logger.debug(f"{trj_generator}")
        candidates = [trj for trj in trj_generator]
        try:
            search.search_all(candidates, use_gpu)
        except:
            # Delete the search object to force the memory cleanup.
            del search
            raise
        self._end_phase("grid search")

        # Load the results.
        keep = self.load_and_filter_results(search, config)

        # Delete the search object to force the memory cleanup.
        # Of the psi/phi images on CPU and GPU.
        del search

        self._end_phase("do_core_search")
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
        stack : `ImageStackPy`
            The stack of image data.
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

            # Output basic binary information.
            logger.debug(f"GPU Code Enabled: {kb.HAS_CUDA}")
            logger.debug(f"OpenMP Enabled: {kb.HAS_OMP}")
            logger.debug(kb.stat_gpu_memory_mb())
            logger.debug("Config:")
            logger.debug(str(config))

        # Determine how many images have at least 10% valid pixels.  Make sure
        # num_obs is no larger than 80% of the valid images.
        img_count = np.count_nonzero(stack.get_masked_fractions() < 0.9)
        if img_count == 0:
            raise ValueError("No valid images in input.")
        if config["num_obs"] == -1 or config["num_obs"] >= img_count:
            logger.info(f"Automatically setting num_obs = {img_count} (from {config['num_obs']}).")
            config.set("num_obs", img_count)

        self._start_phase("KBMOD")

        if not config.validate():
            raise ValueError("Invalid configuration")

        if not kb.HAS_CUDA:
            logger.warning("Code was compiled without GPU using CPU only.")
            config.set("cpu_only", True)

        # Perform the actual search.
        if trj_generator is None:
            trj_generator = create_trajectory_generator(config, work_unit=None)
        keep = self.do_core_search(config, stack, trj_generator)

        if config["do_clustering"] and len(keep) > 1:
            self._start_phase("clustering")
            cluster_params = {
                "cluster_type": config["cluster_type"],
                "cluster_eps": config["cluster_eps"],
                "cluster_v_scale": config["cluster_v_scale"],
                "times": np.asarray(stack.times),
            }
            apply_clustering(keep, cluster_params)
            self._end_phase("clustering")

        # Filter by max_results, keeping only the results with the highest likelihoods.
        # This should be the last step of the filtering phase, but before we add auxiliary
        # information like stamps.
        if config["max_results"] > -1 and config["max_results"] < len(keep):
            self._start_phase("max_results")
            logger.info(f"Filtering {len(keep)}results to max_results={config['max_results']}")
            keep.sort("likelihood", descending=True)
            keep.filter_rows(np.arange(config["max_results"]), "max_results")
            self._end_phase("max_results")

        # Generate coadded stamps without filtering -- both the "stamp" column
        # as well as any additional coadds.
        self._start_phase("stamp generation")
        stamp_radius = config["stamp_radius"]
        stamp_type = config["stamp_type"]
        coadds = set(config["coadds"])
        coadds.add(stamp_type)

        # Add all the "coadd_*" columns and a "stamp" column. This is only
        # short term until we stop using the "stamp" column.
        append_coadds(keep, stack, coadds, stamp_radius, nightly=config["nightly_coadds"])
        if f"coadd_{stamp_type}" in keep.colnames:
            keep.table["stamp"] = keep.table[f"coadd_{stamp_type}"]

        # sns_filter
        if config["sns_filter"]:
            no_op_filter(keep)

        # if CNN is enabled, add the classification and probabilities to the results.
        if config["cnn_filter"]:
            if config["cnn_model"] is None:
                raise ValueError("cnn_model must be set to use cnn_filter.")
            self._start_phase("cnn filtering")
            filter_stamps_by_cnn(
                keep,
                config["cnn_model"],
                coadd_type=config["cnn_coadd_type"],
                stamp_radius=config["cnn_stamp_radius"],
                coadd_radius=config["stamp_radius"],
            )
            self._end_phase("cnn filtering")

        # Extract all the stamps for all time steps and append them onto the result rows.
        if config["save_all_stamps"]:
            append_all_stamps(keep, stack, stamp_radius)
        self._end_phase("stamp generation")

        # Append additional information derived from the WorkUnit if one is provided,
        # including a global WCS and per-time (RA, dec) predictions for each image.
        if workunit is not None:
            self._start_phase("append_positions_to_results")
            keep.table.wcs = workunit.wcs
            if config["compute_ra_dec"]:
                append_positions_to_results(workunit, keep)
            self._end_phase("append_positions_to_results")

        # Create and save any additional meta data that should be saved with the results.
        num_img = stack.num_times

        self._start_phase("write results")
        if extra_meta is not None:
            meta_to_save = extra_meta.copy()
        else:
            meta_to_save = {}
        meta_to_save["num_img"] = num_img
        meta_to_save["dims"] = stack.width, stack.height
        keep.set_mjd_utc_mid(np.array(stack.times))

        if config["result_filename"] is not None:
            write_results_to_files_destructive(
                config["result_filename"],
                keep,
                extra_meta=meta_to_save,
                separate_col_files=config["separate_col_files"],
                drop_columns=config["drop_columns"],
                overwrite=True,
            )

            if config["save_config"]:
                # create provenance directory write out the config file
                result_dir = os.path.dirname(config["result_filename"])
                base_file = os.path.basename(config["result_filename"])
                for ext in keep._supported_formats:
                    if base_file.endswith(ext):
                        base_file = base_file[: -len(ext)]
                        break
                provenance_dir = os.path.join(result_dir, base_file + "_provenance")
                os.makedirs(provenance_dir, exist_ok=True)
                config.to_file(os.path.join(provenance_dir, base_file + "_config.yaml"))
        self._end_phase("write results")

        # Display the stats for each of the search phases.
        self._end_phase("KBMOD")
        if config["debug"]:
            self.display_phase_stats()

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
        # few pre-defined fields to the results data.  If these columns are not present in the
        # WorkUnit, they are skipped in the meta data.
        extra_meta = work.get_constituent_meta(
            [
                "visit",  # The visit number of the original images.
                "filter",  # The filter used for the original images.
                "data_loc",  # The location of the original image data.
                "dataId",  # The Butler data set ID for the original images.
            ]
        )

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

    num_times = workunit.im_stack.num_times
    times = workunit.im_stack.zeroed_times

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
