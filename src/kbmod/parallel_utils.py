# Helper function for parallel processing (must be at module level for pickling)
def _process_results_chunk_parallel(args):
    """
    Process a chunk of results in parallel.
    args is a tuple: (chunk_data, static_context)

    chunk_data: (start, end, chunk_ra, chunk_dec)
    static_context: (num_times, global_wcs, original_wcses, all_obstimes, reprojected,
                     reprojection_frame, barycentric_distance, geocentric_distances,
                     per_image_indices, image_locations)
    """
    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord

    from kbmod.reprojection_utils import image_positions_to_original_icrs

    chunk_data, context = args
    start, end, chunk_ra, chunk_dec = chunk_data
    (
        num_times,
        global_wcs,
        original_wcses,
        all_obstimes,
        reprojected,
        reprojection_frame,
        barycentric_distance,
        geocentric_distances,
        per_image_indices,
        image_locations,
    ) = context

    chunk_all_ra = np.zeros((end - start, num_times))
    chunk_all_dec = np.zeros((end - start, num_times))
    all_inds = np.arange(num_times)

    for local_idx in range(end - start):
        # Reconstruct SkyCoord for this result at each time
        pos_list = [
            SkyCoord(ra=chunk_ra[local_idx, j] * u.degree, dec=chunk_dec[local_idx, j] * u.degree)
            for j in range(num_times)
        ]
        img_skypos = image_positions_to_original_icrs(
            image_indices=all_inds,
            positions=pos_list,
            reprojected_wcs=global_wcs,
            original_wcses=original_wcses,
            all_times=all_obstimes,
            input_format="radec",
            output_format="radec",
            filter_in_frame=False,
            reprojection_frame=reprojection_frame if reprojected else "original",
            barycentric_distance=barycentric_distance,
            geocentric_distances=geocentric_distances,
            per_image_indices=per_image_indices,
            image_locations=image_locations,
        )
        for time_idx in range(num_times):
            chunk_all_ra[local_idx, time_idx] = img_skypos[time_idx].ra.degree
            chunk_all_dec[local_idx, time_idx] = img_skypos[time_idx].dec.degree

    return start, chunk_all_ra, chunk_all_dec
