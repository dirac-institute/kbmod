import logging
import numpy as np
import concurrent.futures
import reproject
from astropy.nddata import CCDData
from tqdm.asyncio import tqdm

from kbmod import is_interactive
from kbmod.core.image_stack_py import ImageStackPy
from kbmod.reprojection_config import LEGACY_CONFIG
from kbmod.search import KB_NO_DATA
from kbmod.work_unit import (
    add_image_data_to_hdul,
    read_image_data_from_hdul,
    WorkUnit,
)
from astropy.io import fits
import os
from copy import copy

# The number of executors to use in the parallel reprojecting function.
MAX_PROCESSES = 8
_DEFAULT_TQDM_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"

logger = logging.getLogger(__name__)


def reproject_image(image, original_wcs, common_wcs, config=LEGACY_CONFIG):
    """Given an ndarray representing image data (either science or variance,
    when used with `reproject_work_unit`), as well as a common wcs, return the reprojected
    image and footprint as a numpy.ndarray.

    Attributes
    ----------
    image : `numpy.ndarray`
        The image data to be reprojected.
    original_wcs : `astropy.wcs.WCS`
        The WCS of the original image.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    config : `AdaptiveReprojectionConfig`, optional
        The explicit adaptive reprojection options. Defaults to
        `LEGACY_CONFIG`, which reproduces KBMOD's historical behavior. The
        effective PSF must be generated with the same configuration used here,
        so callers that reproject science should pass the same object to the
        PSF path.

    Returns
    -------
    new_image : `numpy.ndarray`
        The image data reprojected with a common `astropy.wcs.WCS`.
    footprint : `numpy.ndarray`
        An array containing the footprint of pixels that have data.
        for footprint[i][j], it's 1 if there is a corresponding reprojected
        pixel and 0 if there is no data.
    """
    image_data = CCDData(image, unit="adu")
    image_data.wcs = original_wcs

    footprint = np.zeros(common_wcs.array_shape, dtype=np.ubyte)

    # if the input image is actually a stack of images, we need to duplicate the
    # footprint to match the total number of images.
    if type(image) is list:
        footprint = np.repeat(footprint[np.newaxis, :, :], len(image), axis=0)

    new_image, _ = reproject.reproject_adaptive(
        image_data,
        common_wcs,
        shape_out=common_wcs.array_shape,
        output_footprint=footprint,
        **config.as_kwargs(),
    )

    # if we passed in a stack of ndarrays (i.e. science, varianace, mask), we only
    # need to return the first footprint, as they should all be the same.
    if footprint.ndim == 3:
        footprint = footprint[0]

    return new_image.astype(np.float32), footprint


# Threshold on the normalized cross-correlation between two constituent
# effective PSFs before they are considered interchangeable within one output
# frame. Below this, a single kernel per time is not scientifically defensible.
PSF_UNIFORMITY_NCC = 0.995


def _reprojection_provenance(config):
    """Describe the operator that produced a set of effective PSFs.

    Recorded on the reprojected WorkUnit so a stored kernel can be traced to the
    configuration that made it. Its absence in older files is meaningful: those
    kernels are native or legacy and are not effective common-frame PSFs.
    """
    return {
        "psf_source": "effective",
        "preset": config.preset_name,
        "config_hash": config.hexdigest,
        "reproject_version": config.provenance["reproject_version"],
    }


def _choose_native_eval_position(original_wcs, common_wcs, margin):
    """Pick a native position that is usable for effective-PSF generation.

    The naive choice -- the native frame's center -- fails for tiled mosaics,
    where a constituent's center can map outside the common frame entirely. The
    position must instead lie in the overlap of the two footprints, and far
    enough inside the native frame that the stamp plus the resampler's sample
    region still fit.

    Returns
    -------
    `tuple` or `None`
        ``(x, y)`` in native pixels, or `None` when no usable position exists.
    """
    native_height, native_width = original_wcs.array_shape
    common_height, common_width = common_wcs.array_shape

    # Corners and center of the common frame, expressed in native pixels.
    xs = [0.0, common_width - 1.0, 0.0, common_width - 1.0, (common_width - 1) / 2.0]
    ys = [0.0, 0.0, common_height - 1.0, common_height - 1.0, (common_height - 1) / 2.0]
    sky = common_wcs.pixel_to_world_values(xs, ys)
    px, py = original_wcs.world_to_pixel_values(*sky)
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    if not np.any(np.isfinite(px)) or not np.any(np.isfinite(py)):
        return None

    low_x = max(margin, float(np.nanmin(px)))
    high_x = min(native_width - 1.0 - margin, float(np.nanmax(px)))
    low_y = max(margin, float(np.nanmin(py)))
    high_y = min(native_height - 1.0 - margin, float(np.nanmax(py)))
    if high_x < low_x or high_y < low_y:
        return None

    candidate = (0.5 * (low_x + high_x), 0.5 * (low_y + high_y))

    # The candidate must also land inside the common frame.
    candidate_sky = original_wcs.pixel_to_world_values(*candidate)
    out_x, out_y = common_wcs.world_to_pixel_values(*candidate_sky)
    if not (0 <= float(out_x) < common_width and 0 <= float(out_y) < common_height):
        return None
    return candidate


def _effective_psf_for_image(psf_kernel, original_wcs, common_wcs, config, native_xy=None):
    """Resample one image's native PSF into the common frame.

    The kernel stored per image is centered, so it is placed at a chosen native
    position and pushed through the same configured operator as the science
    image. By default the position is chosen inside the overlap of the native
    and common footprints -- the native frame center is not safe, since a
    mosaic tile's center can map outside the common frame entirely.

    Returns
    -------
    `kbmod.psf_reprojection.EffectivePsf`
    """
    from kbmod.psf_reprojection import make_effective_psf

    kernel = np.asarray(psf_kernel)
    radius = kernel.shape[0] // 2

    if native_xy is None:
        from kbmod.psf_reprojection import _required_padding

        margin = radius + _required_padding(config, 1.0)
        native_xy = _choose_native_eval_position(original_wcs, common_wcs, margin)
        if native_xy is None:
            # Fall back to the native center so the failure surfaces from
            # make_effective_psf with its more specific diagnostics.
            shape = original_wcs.array_shape
            native_xy = (shape[1] / 2.0, shape[0] / 2.0)
    origin = (int(round(native_xy[0])) - radius, int(round(native_xy[1])) - radius)

    return make_effective_psf(kernel, origin, original_wcs, common_wcs, config=config)


def _normalized_cross_correlation(first, second):
    """Zero-lag normalized cross-correlation of two kernels, zero-padded."""
    width = max(first.shape[0], second.shape[0])
    if width % 2 == 0:
        width += 1

    def centered(kernel):
        out = np.zeros((width, width), dtype=np.float64)
        radius = kernel.shape[0] // 2
        offset = width // 2 - radius
        out[offset : offset + kernel.shape[0], offset : offset + kernel.shape[1]] = kernel
        return out

    a, b = centered(first), centered(second)
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else 0.0


def _effective_psfs_for_indices(psf_kernels, wcs_list, common_wcs, config, obstime, indices):
    """Resample each constituent PSF, tolerating ones that cannot be measured.

    A mosaic tile that only clips the common frame may have no position that is
    inside its own footprint, inside the overlap, and far enough from the edge
    for the resampler all at once. Such a constituent contributes few pixels but
    must not abort the whole reprojection, so it is skipped and named in a
    warning rather than silently dropped or fatally raised on.

    Returns
    -------
    `tuple`
        ``(effective_psfs, kept_indices, skipped)``.

    Raises
    ------
    ValueError
        If no constituent at this time can produce an effective PSF.
    """
    from kbmod.psf_reprojection import EffectivePsfError

    effective, kept, skipped = [], [], []
    for kernel, wcs, index in zip(psf_kernels, wcs_list, indices):
        try:
            effective.append(_effective_psf_for_image(kernel, wcs, common_wcs, config))
            kept.append(index)
        except EffectivePsfError as err:
            skipped.append((index, str(err)))

    if not effective:
        details = "; ".join(f"index {i}: {reason}" for i, reason in skipped)
        raise ValueError(f"No image at obstime {obstime} could produce an effective PSF. {details}")

    if skipped:
        logger.warning(
            "Obstime %s: %d of %d constituent image(s) could not produce an effective PSF and were "
            "excluded from the PSF for this time (%s). Their pixels are still reprojected, so that "
            "part of the mosaic is searched with another constituent's PSF.",
            obstime,
            len(skipped),
            len(psf_kernels),
            ", ".join(f"index {i}" for i, _ in skipped),
        )

    return effective, kept, skipped


def _combine_constituent_psfs(effective_psfs, obstime, indices):
    """Pick one kernel for a time, or refuse when they disagree.

    KBMOD stores one kernel per time. When several detector images share an
    observation time, that is only defensible if their resampled PSFs are
    interchangeable. Historically the code took ``indices[0]`` silently, which
    is wrong whenever the constituents differ.

    Raises
    ------
    ValueError
        If the constituent PSFs are not equivalent. Searching part of a mosaic
        with another detector's PSF is not something to do quietly, and tiled
        multi-PSF search is deliberately out of scope here.
    """
    if len(effective_psfs) == 1:
        return effective_psfs[0]

    reference = effective_psfs[0]
    failures = []
    for offset, candidate in enumerate(effective_psfs[1:], start=1):
        ncc = _normalized_cross_correlation(reference.kernel, candidate.kernel)
        if ncc < PSF_UNIFORMITY_NCC:
            failures.append(
                f"image index {indices[offset]} (normalized cross-correlation {ncc:.4f} vs "
                f"index {indices[0]})"
            )

    if failures:
        raise ValueError(
            f"Images sharing obstime {obstime} have effective PSFs that are not interchangeable: "
            + "; ".join(failures)
            + f". The threshold is {PSF_UNIFORMITY_NCC}. KBMOD stores one PSF per time, so searching "
            "this mosaic would apply one detector's PSF to another's pixels. Split the mosaic by "
            "detector, or reproject the constituents separately."
        )

    return reference


def reproject_work_unit(
    work_unit,
    common_wcs,
    frame="original",
    parallelize=True,
    max_parallel_processes=MAX_PROCESSES,
    write_output=False,
    directory=None,
    filename=None,
    show_progress=None,
    reprojection_config=LEGACY_CONFIG,
):
    """Given a WorkUnit and a WCS, reproject all of the images in the ImageStackPy
    into a common WCS.

    Attributes
    ----------
    work_unit : `kbmod.WorkUnit`
        The WorkUnit to be reprojected.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    frame : `str`
        The WCS frame of reference to use when reprojecting.
        Can either be 'original' or 'ebd' to specify which WCS to access
        from the WorkUnit.
    parallelize : `bool`
        If True, use multiprocessing to reproject the images in parallel.
        Default is True.
    max_parallel_processes : `int`
        The maximum number of parallel processes to use when reprojecting. Only
        used when parallelize is True. Default is 8. For more see
        `concurrent.futures.ProcessPoolExecutor` in the Python docs.
    write_output : `bool`
        Whether or not to write the reprojection results out as a sharded `WorkUnit`.
    directory : `str`
        The directory where output will be written if `write_output` is set to True.
    filename : `str`
        The base filename where output will be written if `write_output` is set to True.
    show_progress : `bool` or `None`, optional
        If `None` use default settings, when a boolean forces the progress bar to be
        displayed or hidden.

    Returns
    -------
    A `kbmod.WorkUnit` reprojected with a common `astropy.wcs.WCS`, or `None` in the case
    where `write_output` is set to True.
    """
    if work_unit.reprojected:
        raise ValueError("Unable to reproject a reprojected WorkUnit.")

    show_progress = is_interactive() if show_progress is None else show_progress
    if (work_unit.lazy or write_output) and (directory is None or filename is None):
        raise ValueError("can't write output to sharded fits without directory and filename provided.")
    if work_unit.lazy:
        return reproject_lazy_work_unit(
            work_unit,
            common_wcs,
            frame=frame,
            max_parallel_processes=max_parallel_processes,
            directory=directory,
            filename=filename,
            show_progress=show_progress,
            reprojection_config=reprojection_config,
        )
    if parallelize:
        return _reproject_work_unit_in_parallel(
            work_unit,
            common_wcs,
            frame,
            max_parallel_processes,
            write_output=write_output,
            directory=directory,
            filename=filename,
            show_progress=show_progress,
            reprojection_config=reprojection_config,
        )
    else:
        return _reproject_work_unit(
            work_unit,
            common_wcs,
            frame,
            write_output=write_output,
            directory=directory,
            filename=filename,
            show_progress=show_progress,
            reprojection_config=reprojection_config,
        )


def _reproject_work_unit(
    work_unit,
    common_wcs,
    frame="original",
    write_output=False,
    directory=None,
    filename=None,
    show_progress=False,
    reprojection_config=LEGACY_CONFIG,
):
    """Given a WorkUnit and a WCS, reproject all of the images in the ImageStackPy
    into a common WCS.

    Attributes
    ----------
    work_unit : `kbmod.WorkUnit`
        The WorkUnit to be reprojected.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    frame : `str`
        The WCS frame of reference to use when reprojecting.
        Can either be 'original' or 'ebd' to specify which WCS to access
        from the WorkUnit.
    write_output : `bool`
        Whether or not to write the reprojection results out as a sharded `WorkUnit`.
    directory : `str`
        The directory where output will be written if `write_output` is set to True.
    filename : `str`
        The base filename where output will be written if `write_output` is set to True.
    disable_show_progress : `bool`
            Whether or not to disable the `tqdm` show_progress bar.

    Returns
    -------
    A `kbmod.WorkUnit` reprojected with a common `astropy.wcs.WCS`, or `None` in the case
    where `write_output` is set to True.
    """
    unique_obstimes, unique_obstime_indices = work_unit.get_unique_obstimes_and_indices()

    # Create a list of the correct WCS. We do this extraction once and reuse for all images.
    if frame == "original":
        wcs_list = work_unit.get_constituent_meta("per_image_wcs")
    elif frame == "ebd":
        wcs_list = work_unit.get_constituent_meta("ebd_wcs")
    else:
        raise ValueError("Invalid projection frame provided.")

    stack = ImageStackPy()
    for obstime_index, o_i in tqdm(
        enumerate(zip(unique_obstimes, unique_obstime_indices)),
        bar_format=_DEFAULT_TQDM_BAR,
        desc="Reprojecting",
        disable=not show_progress,
    ):
        time, indices = o_i
        science_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
        variance_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
        mask_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
        footprint_add = np.zeros(common_wcs.array_shape, dtype=np.ubyte)

        for index in indices:
            science = work_unit.im_stack.sci[index]
            variance = work_unit.im_stack.var[index]
            mask = work_unit.im_stack.get_mask(index)

            original_wcs = wcs_list[index]
            if original_wcs is None:
                raise ValueError(f"No WCS provided for index {index}")

            reprojected_science, footprint = reproject_image(science, original_wcs, common_wcs)

            footprint_add += footprint
            # we'll enforce that there be no overlapping images at the same time,
            # for now. We might be able to add some ability co-add in the future.
            if np.any(footprint_add > 1):
                raise ValueError("Images with the same obstime are overlapping.")

            reprojected_variance, _ = reproject_image(variance, original_wcs, common_wcs)

            reprojected_mask, _ = reproject_image(mask, original_wcs, common_wcs)

            # change all the NaNs to zeroes so that the matrix addition works properly.
            # `footprint_add` will maintain the information about what areas of the frame
            # don't have any data so that we can change it back after we combine.
            reprojected_science[np.isnan(reprojected_science)] = 0.0
            reprojected_variance[np.isnan(reprojected_variance)] = 0.0
            reprojected_mask[np.isnan(reprojected_mask)] = 0.0

            science_add += reprojected_science
            variance_add += reprojected_variance
            mask_add += reprojected_mask

        # change all the values where there are is no corresponding data to `KB_NO_DATA`.
        gaps = footprint_add == 0
        science_add[gaps] = KB_NO_DATA
        variance_add[gaps] = KB_NO_DATA
        mask_add[gaps] = 1

        # transforms the mask back into a bitmask. Note that we need to be explicit
        # about the dtypes for 0.0 and 1.0, otherwise mask_add will be cast to float64.
        mask_add = np.where(np.isclose(mask_add, 0.0, atol=0.2), np.float32(0.0), np.float32(1.0))

        # Resample each constituent's PSF through the same operator as the
        # science image, then require them to be interchangeable before
        # collapsing to the single kernel KBMOD stores per time.
        effective, kept, _ = _effective_psfs_for_indices(
            [work_unit.im_stack.psfs[i] for i in indices],
            [wcs_list[i] for i in indices],
            common_wcs,
            reprojection_config,
            time,
            list(indices),
        )
        psf = _combine_constituent_psfs(effective, time, kept).kernel

        if write_output:
            _write_images_to_shard(
                science_add=science_add,
                variance_add=variance_add,
                mask_add=mask_add,
                psf=psf,
                wcs=common_wcs,
                obstime=time,
                obstime_index=obstime_index,
                indices=indices,
                directory=directory,
                filename=filename,
            )
        else:
            stack.append_image(
                time,
                science_add,
                variance_add,
                mask=mask_add,
                psf=psf,
            )

    if write_output:
        # Create a copy of the WorkUnit to write the global metadata.
        # We preserve the metgadata for the consituent images.
        new_work_unit = copy(work_unit)
        new_work_unit._per_image_indices = unique_obstime_indices
        new_work_unit.wcs = common_wcs
        new_work_unit.reprojected = True
        new_work_unit.reprojection_frame = frame
        new_work_unit.reprojection_provenance = _reprojection_provenance(reprojection_config)

        hdul = new_work_unit.metadata_to_hdul()
        hdul.writeto(os.path.join(directory, filename))
    else:
        # Create a new WorkUnit with the new image stack and global WCS.
        # We preserve the metgadata for the consituent images.
        new_wunit = WorkUnit(
            im_stack=stack,
            config=work_unit.config,
            wcs=common_wcs,
            per_image_indices=unique_obstime_indices,
            reprojected=True,
            reprojection_frame=frame,
            reprojection_provenance=_reprojection_provenance(reprojection_config),
            barycentric_distance=work_unit.barycentric_distance,
            org_image_meta=work_unit.org_img_meta,
        )

        return new_wunit


def _reproject_work_unit_in_parallel(
    work_unit,
    common_wcs,
    frame="original",
    max_parallel_processes=MAX_PROCESSES,
    write_output=False,
    directory=None,
    filename=None,
    show_progress=False,
    reprojection_config=LEGACY_CONFIG,
):
    """Given a WorkUnit and a WCS, reproject all of the images in the image stack
    into a common WCS. This function uses multiprocessing to reproject the images
    in parallel.

    Attributes
    ----------
    work_unit : `kbmod.WorkUnit`
        The WorkUnit to be reprojected.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    frame : `str`
        The WCS frame of reference to use when reprojecting.
        Can either be 'original' or 'ebd' to specify which WCS to access
        from the WorkUnit.
    max_parallel_processes : `int`
        The maximum number of parallel processes to use when reprojecting.
        Default is 8. For more see `concurrent.futures.ProcessPoolExecutor` in
        the Python docs.
    write_output : `bool`
        Whether or not to write the reprojection results out as a sharded `WorkUnit`.
    directory : `str`
        The directory where output will be written if `write_output` is set to True.
    filename : `str`
        The base filename where output will be written if `write_output` is set to True.
    show_progress : `bool`
            Whether or not to enable the `tqdm` show_progress bar.

    Returns
    -------
    A `kbmod.WorkUnit` reprojected with a common `astropy.wcs.WCS`, or `None` in the case
    where `write_output` is set to True.
    """

    # get all the unique obstimes
    unique_obstimes, unique_obstimes_indices = work_unit.get_unique_obstimes_and_indices()

    future_reprojections = []
    with concurrent.futures.ProcessPoolExecutor(max_parallel_processes) as executor:
        # for a given list of obstime indices, collect all the science, variance, and mask images.
        for obstime_index, o_i in enumerate(zip(unique_obstimes, unique_obstimes_indices)):
            obstime, indices = o_i
            original_wcs = _validate_original_wcs(work_unit, indices, frame)

            # Get the list of science, variance, or mask images for each unique obstime.
            # We create a mask since we implicitly store it in the
            science_images_at_obstime = []
            variance_images_at_obstime = []
            mask_images_at_obstime = []
            for i in indices:
                sci = work_unit.im_stack.sci[i]
                var = work_unit.im_stack.var[i]
                mask = work_unit.im_stack.get_mask(i)

                science_images_at_obstime.append(sci)
                variance_images_at_obstime.append(var)
                mask_images_at_obstime.append(mask)

            if write_output:
                effective, kept, _ = _effective_psfs_for_indices(
                    [work_unit.im_stack.psfs[i] for i in indices],
                    original_wcs,
                    common_wcs,
                    reprojection_config,
                    obstime,
                    list(indices),
                )
                psf_array = _combine_constituent_psfs(effective, obstime, kept).kernel
                future_reprojections.append(
                    executor.submit(
                        _reproject_and_write,
                        science_images=science_images_at_obstime,
                        variance_images=variance_images_at_obstime,
                        mask_images=mask_images_at_obstime,
                        psf=psf_array,
                        obstime=obstime,
                        obstime_index=obstime_index,
                        indices=indices,
                        common_wcs=common_wcs,
                        original_wcs=original_wcs,
                        directory=directory,
                        filename=filename,
                    )
                )
            else:
                # call `_reproject_images` in parallel.
                future_reprojections.append(
                    executor.submit(
                        _reproject_images,
                        science_images=science_images_at_obstime,
                        variance_images=variance_images_at_obstime,
                        mask_images=mask_images_at_obstime,
                        obstime=obstime,
                        common_wcs=common_wcs,
                        original_wcs=original_wcs,
                        psfs=[work_unit.im_stack.psfs[i] for i in indices],
                        indices=list(indices),
                        reprojection_config=reprojection_config,
                    )
                )
        # Need to consume the generator producted by tqdm to update the show_progress bar so we instantiate a list
        list(
            tqdm(
                concurrent.futures.as_completed(future_reprojections),
                total=len(future_reprojections),
                bar_format=_DEFAULT_TQDM_BAR,
                desc="Reprojecting",
                disable=not show_progress,
            )
        )

    # Wait for all the multiprocessing to finish
    concurrent.futures.wait(future_reprojections, return_when=concurrent.futures.ALL_COMPLETED)

    if write_output:
        for result in future_reprojections:
            if not result.result():
                raise RuntimeError("one or more jobs failed.")

        new_work_unit = copy(work_unit)
        new_work_unit._per_image_indices = unique_obstimes_indices
        new_work_unit.wcs = common_wcs
        new_work_unit.reprojected = True
        new_work_unit.reprojection_frame = frame
        new_work_unit.reprojection_provenance = _reprojection_provenance(reprojection_config)

        hdul = new_work_unit.metadata_to_hdul()
        hdul.writeto(os.path.join(directory, filename))
    else:
        stack = ImageStackPy()
        for result in future_reprojections:
            science_add, variance_add, mask_add, time, psf = result.result()

            stack.append_image(
                time,
                science_add,
                variance_add,
                mask=mask_add,
                psf=psf,
            )

        # sort by the time_stamp
        stack.sort_by_time()

        # Add the image stack to a new WorkUnit and return it.  We preserve the metgadata
        # for the consituent images.
        new_wunit = WorkUnit(
            im_stack=stack,
            config=work_unit.config,
            wcs=common_wcs,
            per_image_indices=unique_obstimes_indices,
            reprojected=True,
            reprojection_frame=frame,
            reprojection_provenance=_reprojection_provenance(reprojection_config),
            barycentric_distance=work_unit.barycentric_distance,
            org_image_meta=work_unit.org_img_meta,
        )

        return new_wunit


def reproject_lazy_work_unit(
    work_unit,
    common_wcs,
    directory,
    filename,
    frame="original",
    max_parallel_processes=MAX_PROCESSES,
    show_progress=None,
    reprojection_config=LEGACY_CONFIG,
):
    """Given a WorkUnit and a WCS, reproject all of the images in the image stack
    into a common WCS. This function is used with lazily evaluated `WorkUnit`s and
    multiprocessing to reproject the images in parallel, and only loads the individual
    image frames at runtime. Currently only works for sharded `WorkUnit`s loaded with
    the `lazy` option.

    Attributes
    ----------
    work_unit : `kbmod.WorkUnit`
        The WorkUnit to be reprojected.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    directory : `str`
        The directory where the `WorkUnit` fits shards will be output.
    filename : `str`
        The base filename (will be the actual name of the primary/metadata
        fits file and included with the index number in the filename of the
        shards).
    frame : `str`
        The WCS frame of reference to use when reprojecting.
        Can either be 'original' or 'ebd' to specify which WCS to access
        from the WorkUnit.
    max_parallel_processes : `int`
        The maximum number of parallel processes to use when reprojecting.
        Default is 8. For more see `concurrent.futures.ProcessPoolExecutor` in
        the Python docs.
    show_progress : `bool` or `None`, optional
        If `None` use default settings, when a boolean forces the progress bar to be
        displayed or hidden.
    """
    show_progress = is_interactive() if show_progress is None else show_progress
    if not work_unit.lazy:
        raise ValueError("WorkUnit must be lazily loaded.")

    # get all the unique obstimes
    unique_obstimes, unique_obstimes_indices = work_unit.get_unique_obstimes_and_indices()

    future_reprojections = []
    with concurrent.futures.ProcessPoolExecutor(max_parallel_processes) as executor:
        # for a given list of obstime indices, collect all the science, variance, and mask images.
        for obstime_index, o_i in enumerate(zip(unique_obstimes, unique_obstimes_indices)):
            obstime, indices = o_i
            original_wcs = _validate_original_wcs(work_unit, indices, frame)
            # get the list of images for each unique obstime
            file_paths_at_obstime = [work_unit.file_paths[i] for i in indices]

            # call `_reproject_images` in parallel.
            future_reprojections.append(
                executor.submit(
                    _load_images_and_reproject,
                    file_paths=file_paths_at_obstime,
                    indices=indices,
                    obstime=obstime,
                    obstime_index=obstime_index,
                    common_wcs=common_wcs,
                    original_wcs=original_wcs,
                    directory=directory,
                    filename=filename,
                    reprojection_config=reprojection_config,
                )
            )

        # Need to consume the generator producted by tqdm to update the show_progress bar so we instantiate a list
        list(
            tqdm(
                concurrent.futures.as_completed(future_reprojections),
                total=len(future_reprojections),
                bar_format=_DEFAULT_TQDM_BAR,
                desc="Reprojecting",
                disable=not show_progress,
            )
        )

    concurrent.futures.wait(future_reprojections, return_when=concurrent.futures.ALL_COMPLETED)

    for result in future_reprojections:
        if not result.result():
            raise RuntimeError("one or more jobs failed.")

    # We use new metadata for the new images and the same metadata for the original images.
    new_work_unit = copy(work_unit)
    new_work_unit._per_image_indices = unique_obstimes_indices
    new_work_unit.wcs = common_wcs
    new_work_unit.reprojected = True
    new_work_unit.reprojecton = frame

    hdul = new_work_unit.metadata_to_hdul()
    hdul.writeto(os.path.join(directory, filename))


def _validate_original_wcs(work_unit, indices, frame="original"):
    """Given a work unit and a set of indices, verify that the WCS is not None for
    any of the indices. If it is, raise a ValueError.

    Parameters
    ----------
    work_unit : `kbmod.WorkUnit`
        The WorkUnit with WCS to be validated.
    indices : list[int]
        The indices to be validated in work_unit.
    frame : `str`
        The WCS frame of reference to use when reprojecting.
        Can either be 'original' or 'ebd' to specify which WCS to access
        from the WorkUnit.

    Returns
    -------
    list[`astropy.wcs.WCS`]
        The list of validated WCS objects for these indices

    Raises
    ------
    ValueError
        If any WCS objects are None, raise an error.
    """

    if frame == "original":
        original_wcs = [work_unit.get_wcs(i) for i in indices]
    elif frame == "ebd":
        original_wcs = [work_unit.get_constituent_meta("ebd_wcs")[i] for i in indices]
    else:
        raise ValueError("Invalid projection frame provided.")

    if len(original_wcs) == 0:
        raise ValueError(f"No WCS found for frame {frame}")
    if np.any(original_wcs) is None:
        # find indices where the wcs is None
        bad_indices = np.where(original_wcs == None)
        # get values from `indices` where original_wcs is None
        work_unit_indices = [indices[i] for i in bad_indices]
        raise ValueError(f"No WCS provided for work_unit index(s) {work_unit_indices}")

    return original_wcs


def _load_images_and_reproject(
    file_paths,
    indices,
    obstime,
    obstime_index,
    common_wcs,
    original_wcs,
    directory,
    filename,
    reprojection_config=LEGACY_CONFIG,
):
    """Load image data from `WorkUnit` shards. Intermediary step
    for when the `WorkUnit` is loaded lazily.

    Parameters
    ----------
    file_paths : `list[str]`
        List of strings comtaining the images to be reprojected and stitched.
    inidces : `list[int]`
        List of `WorkUnit` indices corresponding to the original positions
        of the images within the `ImageStackPy`.
    obstime : `float`
        observation times for set of images.
    obstime_index : `int`
        the index of the unique obstime.
        i.e. the new index of the mosaicked image in the `ImageStackPy`.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    original_wcs : `list[astropy.wcs.WCS]`
        The list of WCS objects for these images.
    directory : `str`
        The directory to output the new sharded and reprojected `WorkUnit`.
    filename : `str`
        The base filename for the sharded and reprojected `WorkUnit`.
    """
    science_images = []
    variance_images = []
    mask_images = []
    psfs = []

    for file_path, index in zip(file_paths, indices):
        with fits.open(file_path) as hdul:
            sci, var, mask, _, psf, _ = read_image_data_from_hdul(hdul, index)
            science_images.append(sci.astype(np.single))
            variance_images.append(var.astype(np.single))
            mask_images.append(mask.astype(bool))
            psfs.append(psf.astype(np.single))

    # Resample the loaded native PSFs, rather than passing the first one
    # through unchanged as this path used to.
    effective, kept, _ = _effective_psfs_for_indices(
        psfs, original_wcs, common_wcs, reprojection_config, obstime, list(indices)
    )
    psf = _combine_constituent_psfs(effective, obstime, kept).kernel

    return _reproject_and_write(
        science_images=science_images,
        variance_images=variance_images,
        mask_images=mask_images,
        psf=psf,
        obstime=obstime,
        obstime_index=obstime_index,
        common_wcs=common_wcs,
        original_wcs=original_wcs,
        indices=indices,
        directory=directory,
        filename=filename,
    )


def _reproject_and_write(
    science_images,
    variance_images,
    mask_images,
    psf,
    obstime,
    obstime_index,
    indices,
    common_wcs,
    original_wcs,
    directory,
    filename,
):
    """Reproject a set of images and write out the output to a sharded `WorkUnit.

    Parameters
    ----------
    science_images : `list[numpy.ndarray]`
        List of ndarrays that represent the science images to be reprojected.
    variance_images : `list[numpy.ndarray]`
        List of ndarrays that represent the variance images to be reprojected.
    mask_images : `list[numpy.ndarray]`
        List of ndarrays that represent the mask images to be reprojected.
    psf : `numpy.ndarray`
        The PSF kernel.
    obstime : `float`
        observation times for set of images.
    obstime_index : `int`
        the index of the unique obstime.
        i.e. the new index of the mosaicked image in the `ImageStackPy`.
    inidces : `list[int]`
        List of `WorkUnit` indices corresponding to the original positions
        of the images within the `ImageStacPy`.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    original_wcs : `list[astropy.wcs.WCS]`
        The list of WCS objects for these images.
    directory : `str`
        The directory to output the new sharded and reprojected `WorkUnit`.
    filename : `str`
        The base filename for the sharded
    """
    science_add, variance_add, mask_add, obstime, _ = _reproject_images(
        science_images,
        variance_images,
        mask_images,
        obstime,
        common_wcs,
        original_wcs,
    )

    _write_images_to_shard(
        science_add=science_add,
        variance_add=variance_add,
        mask_add=mask_add,
        psf=psf,
        wcs=common_wcs,
        obstime=obstime,
        obstime_index=obstime_index,
        indices=indices,
        directory=directory,
        filename=filename,
    )

    return True


def _reproject_images(
    science_images,
    variance_images,
    mask_images,
    obstime,
    common_wcs,
    original_wcs,
    psfs=None,
    indices=None,
    reprojection_config=None,
):
    """This is the worker function that will be parallelized across multiple processes.
    Given a set of science, variance, and mask images, use astropy's reproject
    function to reproject them into a common WCS.

    Parameters
    ----------
    science_images : `list[numpy.ndarray]`
        List of ndarrays that represent the science images to be reprojected.
    variance_images : `list[numpy.ndarray]`
        List of ndarrays that represent the variance images to be reprojected.
    mask_images : `list[numpy.ndarray]`
        List of ndarrays that represent the mask images to be reprojected.
    obstime : `float`
        observation time for each image.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    original_wcs : `list[astropy.wcs.WCS]`
        The list of WCS objects for these images.

    Returns
    -------
    science_add : `numpy.ndarray`
        The reprojected science image.
    variance_add : `numpy.ndarray`
        The reprojected variance image.
    mask_add : `numpy.ndarray`
        The reprojected mask image.
    time : `float`
        The observation time of the original images.
    psf : `numpy.ndarray` or `None`
        The effective PSF in the common frame, resampled through this same
        operator. `None` when no native PSFs were supplied.

    Raises
    ------
    ValueError
        If any images overlap, raise an error.
    """
    science_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
    variance_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
    mask_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
    footprint_add = np.zeros(common_wcs.array_shape, dtype=np.ubyte)

    for science, variance, mask, this_original_wcs in zip(
        science_images, variance_images, mask_images, original_wcs
    ):
        # reproject science, variance, and mask images simulataneously.
        reprojected_images, footprints = reproject_image(
            [science, variance, mask], this_original_wcs, common_wcs
        )

        footprint_add += footprints
        # we'll enforce that there be no overlapping images at the same time,
        # for now. We might be able to add some ability co-add in the future.
        if np.any(footprint_add > 1):
            raise ValueError("Images with the same obstime are overlapping.")

        # change all the NaNs to zeroes so that the matrix addition works properly.
        # `footprint_add` will maintain the information about what areas of the frame
        # don't have any data so that we can change it back after we combine.
        reprojected_images[np.isnan(reprojected_images)] = 0.0

        science_add += reprojected_images[0]
        variance_add += reprojected_images[1]
        mask_add += reprojected_images[2]

    # change all the values where there are is no corresponding data to `KB_NO_DATA`.
    gaps = footprint_add == 0
    science_add[gaps] = KB_NO_DATA
    variance_add[gaps] = KB_NO_DATA
    mask_add[gaps] = 1

    # transforms the mask back into a bitmask.
    mask_add = np.where(np.isclose(mask_add, 0.0, atol=0.2), np.float32(0.0), np.float32(1.0))

    effective_psf = None
    if psfs is not None:
        # Resample each constituent PSF through the operator just applied to
        # the science planes, then require the constituents to agree before
        # collapsing to the one kernel KBMOD stores per time.
        label = list(indices) if indices is not None else list(range(len(psfs)))
        effective, kept, _ = _effective_psfs_for_indices(
            psfs, original_wcs, common_wcs, reprojection_config, obstime, label
        )
        effective_psf = _combine_constituent_psfs(effective, obstime, kept).kernel

    return science_add, variance_add, mask_add, obstime, effective_psf


def _write_images_to_shard(
    science_add, variance_add, mask_add, psf, wcs, obstime, obstime_index, indices, directory, filename
):
    """Takes in a set of post-reprojection image adds and
    writes them to a fits file..

    Parameters
    ----------
    science_add : `numpy.ndarray`
        ndarry containing the reprojected science image add.
    variance_add : `numpy.ndarray`
        ndarry containing the reprojected variance image add.
    mask_add : `numpy.ndarray`
        ndarry containing the reprojected mask image add.
    psf : `numpy.ndarray`
        the kernel of the PSF.
    wcs : `astropy.wcs.WCS`
        the common_wcs used in reprojection.
    obstime : `float`
        observation time for each image.
    obstime_index : `int`
        the obstime index in the original `ImageStackPy`.
    indices : `list[int]`
        the per image indices.
    directory : `str`
        the directory to output the `WorkUnit` shard to.
    filename : `str`
        the base filename to use for the shard.
    """
    n_indices = len(indices)
    sub_hdul = fits.HDUList()

    # Append all of the image data to the sub_hdul.
    add_image_data_to_hdul(
        sub_hdul,
        obstime_index,
        science_add,
        variance_add,
        mask_add,
        obstime,
        psf_kernel=psf,
        wcs=wcs,
    )

    # Add the indexing information.
    sci_hdu = sub_hdul[f"SCI_{obstime_index}"]
    sci_hdu.header["NIND"] = n_indices
    for j in range(n_indices):
        sci_hdu.header[f"IND_{j}"] = indices[j]
    sub_hdul.append(sci_hdu)

    # Write out the file.
    sub_hdul.writeto(os.path.join(directory, f"{obstime_index}_{filename}"))
