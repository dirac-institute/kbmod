import numpy as np
import concurrent.futures
import reproject
from astropy.nddata import CCDData
from astropy.wcs import WCS
from tqdm.asyncio import tqdm

from kbmod import is_interactive
from kbmod.search import KB_NO_DATA, ImageStack, LayeredImage
from kbmod.work_unit import WorkUnit
from kbmod.wcs_utils import append_wcs_to_hdu_header
from astropy.io import fits
import os
from copy import copy


# The number of executors to use in the parallel reprojecting function.
MAX_PROCESSES = 8
_DEFAULT_TQDM_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"


def reproject_image(image, original_wcs, common_wcs):
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
        bad_value_mode="ignore",
        output_footprint=footprint,
        roundtrip_coords=False,
    )

    # if we passed in a stack of ndarrays (i.e. science, varianace, mask), we only
    # need to return the first footprint, as they should all be the same.
    if footprint.ndim == 3:
        footprint = footprint[0]

    return new_image.astype(np.float32), footprint


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
):
    """Given a WorkUnit and a WCS, reproject all of the images in the ImageStack
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
        )


def _reproject_work_unit(
    work_unit,
    common_wcs,
    frame="original",
    write_output=False,
    directory=None,
    filename=None,
    show_progress=False,
):
    """Given a WorkUnit and a WCS, reproject all of the images in the ImageStack
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
    images = work_unit.im_stack.get_images()
    unique_obstimes, unique_obstime_indices = work_unit.get_unique_obstimes_and_indices()

    # Create a list of the correct WCS. We do this extraction once and reuse for all images.
    if frame == "original":
        wcs_list = work_unit.get_constituent_meta("per_image_wcs")
    elif frame == "ebd":
        wcs_list = work_unit.get_constituent_meta("ebd_wcs")
    else:
        raise ValueError("Invalid projection frame provided.")

    stack = ImageStack()
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
            image = images[index]
            science = image.get_science_array()
            variance = image.get_variance_array()
            mask = image.get_mask_array()

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

        psf = images[indices[0]].get_psf()

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
            new_layered_image = LayeredImage(
                science_add,
                variance_add,
                mask_add,
                psf,
                time,
            )
            stack.append_image(new_layered_image, force_move=True)

    if write_output:
        # Create a copy of the WorkUnit to write the global metadata.
        # We preserve the metgadata for the consituent images.
        new_work_unit = copy(work_unit)
        new_work_unit._per_image_indices = unique_obstime_indices
        new_work_unit.wcs = common_wcs
        new_work_unit.reprojected = True
        new_work_unit.reprojection_frame = frame

        hdul = new_work_unit.metadata_to_hdul()
        hdul.writeto(os.path.join(directory, filename))
    else:
        # Create a new WorkUnit with the new ImageStack and global WCS.
        # We preserve the metgadata for the consituent images.
        new_wunit = WorkUnit(
            im_stack=stack,
            config=work_unit.config,
            wcs=common_wcs,
            per_image_indices=unique_obstime_indices,
            reprojected=True,
            reprojection_frame=frame,
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
):
    """Given a WorkUnit and a WCS, reproject all of the images in the ImageStack
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

    # get the list of images from the work_unit outside the for-loop
    images = work_unit.im_stack.get_images()

    future_reprojections = []
    with concurrent.futures.ProcessPoolExecutor(max_parallel_processes) as executor:
        # for a given list of obstime indices, collect all the science, variance, and mask images.
        for obstime_index, o_i in enumerate(zip(unique_obstimes, unique_obstimes_indices)):
            obstime, indices = o_i
            original_wcs = _validate_original_wcs(work_unit, indices, frame)
            # get the list of images for each unique obstime
            images_at_obstime = [images[i] for i in indices]

            # convert each image into a science, variance, or mask "image", i.e. a list of numpy arrays.
            science_images_at_obstime = [this_image.get_science_array() for this_image in images_at_obstime]
            variance_images_at_obstime = [this_image.get_variance_array() for this_image in images_at_obstime]
            mask_images_at_obstime = [this_image.get_mask_array() for this_image in images_at_obstime]

            if write_output:
                psf_array = _get_first_psf_at_time(work_unit, obstime)
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

        hdul = new_work_unit.metadata_to_hdul()
        hdul.writeto(os.path.join(directory, filename))
    else:
        stack = ImageStack([])
        for result in future_reprojections:
            science_add, variance_add, mask_add, time = result.result()
            psf = _get_first_psf_at_time(work_unit, obstime)

            # And then stack the numpy arrays into a LayeredImage.
            new_layered_image = LayeredImage(
                science_add,
                variance_add,
                mask_add,
                psf,
                time,
            )
            stack.append_image(new_layered_image, force_move=True)

        # sort by the time_stamp
        stack.sort_by_time()

        # Add the imageStack to a new WorkUnit and return it.  We preserve the metgadata
        # for the consituent images.
        new_wunit = WorkUnit(
            im_stack=stack,
            config=work_unit.config,
            wcs=common_wcs,
            per_image_indices=unique_obstimes_indices,
            reprojected=True,
            reprojection_frame=frame,
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
):
    """Given a WorkUnit and a WCS, reproject all of the images in the ImageStack
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


def _get_first_psf_at_time(work_unit, time):
    """Given a work_unit, find the first psf object at a given time

    Parameters
    ----------
    work_unit : `kbmod.WorkUnit`
        The WorkUnit to be searched
    time : float
        The MJD of the observation(s) to search for in the work_unit.

    Returns
    -------
    `numpy.ndarray`
        The kernel of the first PSF found at the given time.

    Raises
    ------
    ValueError
        If the time is not found in list of observation times in the work_unit,
        raise an error.
    """
    obstimes = np.asarray(work_unit.get_all_obstimes())

    # if the time isn't in the list of times, raise an error.
    if time not in obstimes:
        raise ValueError(f"Observation time {time} not found in work unit.")

    images = work_unit.im_stack.get_images()
    index = np.where(obstimes == time)[0][0]
    return images[index].get_psf()


def _load_images_and_reproject(
    file_paths, indices, obstime, obstime_index, common_wcs, original_wcs, directory, filename
):
    """Load image data from `WorkUnit` shards. Intermediary step
    for when the `WorkUnit` is loaded lazily.

    Parameters
    ----------
    file_paths : `list[str]`
        List of strings comtaining the images to be reprojected and stitched.
    inidces : `list[int]`
        List of `WorkUnit` indices corresponding to the original positions
        of the images within the `ImageStack`.
    obstime : `float`
        observation times for set of images.
    obstime_index : `int`
        the index of the unique obstime.
        i.e. the new index of the mosaicked image in
        the `ImageStack`.
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
            science_images.append(hdul[f"SCI_{index}"].data.astype(np.single))
            variance_images.append(hdul[f"VAR_{index}"].data.astype(np.single))
            mask_images.append(hdul[f"MSK_{index}"].data.astype(bool))
            psfs.append(hdul[f"PSF_{index}"].data.astype(np.single))

    return _reproject_and_write(
        science_images=science_images,
        variance_images=variance_images,
        mask_images=mask_images,
        psf=psfs[0],
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
        i.e. the new index of the mosaicked image in
        the `ImageStack`.
    inidces : `list[int]`
        List of `WorkUnit` indices corresponding to the original positions
        of the images within the `ImageStack`.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    original_wcs : `list[astropy.wcs.WCS]`
        The list of WCS objects for these images.
    directory : `str`
        The directory to output the new sharded and reprojected `WorkUnit`.
    filename : `str`
        The base filename for the sharded
    """
    science_add, variance_add, mask_add, obstime = _reproject_images(
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


def _reproject_images(science_images, variance_images, mask_images, obstime, common_wcs, original_wcs):
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

    return science_add, variance_add, mask_add, obstime


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
        the obstime index in the original `ImageStack`.
    indices : `list[int]`
        the per image indices.
    directory : `str`
        the directory to output the `WorkUnit` shard to.
    filename : `str`
        the base filename to use for the shard.
    """
    n_indices = len(indices)
    sub_hdul = fits.HDUList()

    sci_hdu = image_add_to_hdu(science_add, f"SCI_{obstime_index}", obstime, wcs)
    sci_hdu.header["NIND"] = n_indices
    for j in range(n_indices):
        sci_hdu.header[f"IND_{j}"] = indices[j]
    sub_hdul.append(sci_hdu)

    var_hdu = image_add_to_hdu(variance_add, f"VAR_{obstime_index}", obstime)
    sub_hdul.append(var_hdu)

    msk_hdu = image_add_to_hdu(mask_add, f"MSK_{obstime_index}", obstime)
    sub_hdul.append(msk_hdu)

    psf_hdu = fits.hdu.image.ImageHDU(psf)
    psf_hdu.name = f"PSF_{obstime_index}"
    sub_hdul.append(psf_hdu)
    sub_hdul.writeto(os.path.join(directory, f"{obstime_index}_{filename}"))


def image_add_to_hdu(add, name, obstime, wcs=None):
    """Helper function that creates a HDU out of post reproject added image.

    Parameters
    ----------
    add : `np.ndarray`
        The image to convert.
    name : `str`
        The name of the image (type + index).
    obstime : `float`
        The observation time.
    wcs : `astropy.wcs.WCS`
        An optional WCS to include in the header.

    Returns
    -------
    hdu : `astropy.io.fits.hdu.image.ImageHDU`
        The image extension.
    """
    hdu = fits.hdu.image.ImageHDU(add)

    # If the WCS is given, copy each entry into the header.
    if wcs is not None:
        append_wcs_to_hdu_header(wcs, hdu.header)

    # Set the time stamp.
    hdu.header["MJD"] = obstime
    hdu.name = name
    return hdu
