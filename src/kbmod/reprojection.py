import numpy as np
import concurrent.futures
import reproject
from astropy.nddata import CCDData
from astropy.wcs import WCS

from kbmod.search import KB_NO_DATA, PSF, ImageStack, LayeredImage, RawImage
from kbmod.work_unit import WorkUnit

# The number of executors to use in the parallel reprojecting function.
MAX_PROCESSES = 8


def reproject_image(image, original_wcs, common_wcs):
    """Given an ndarray representing image data (either science or variance,
    when used with `reproject_work_unit`), as well as a common wcs, return the reprojected
    image and footprint as a numpy.ndarray.

    Attributes
    ----------
    image : `kbmod.search.RawImage` or `numpy.ndarray`
        The image data to be reprojected.
    original_wcs : `astropy.wcs.WCS`
        The WCS of the original image.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.

    Returns
    ----------
    new_image : `numpy.ndarray`
        The image data reprojected with a common `astropy.wcs.WCS`.
    footprint : `numpy.ndarray`
        An array containing the footprint of pixels that have data.
        for footprint[i][j], it's 1 if there is a corresponding reprojected
        pixel and 0 if there is no data.
    """
    if type(image) is RawImage:
        image = image.image

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
    work_unit, common_wcs, frame="original", parallelize=True, max_parallel_processes=MAX_PROCESSES
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
        Can either be 'original' or 'ebd' to specify whether to
        use the WorkUnit._per_image_wcs or ._per_image_ebd_wcs
        respectively.
    parallelize : `bool`
        If True, use multiprocessing to reproject the images in parallel.
        Default is True.
    max_parallel_processes : `int`
        The maximum number of parallel processes to use when reprojecting. Only
        used when parallelize is True. Default is 8. For more see
        `concurrent.futures.ProcessPoolExecutor` in the Python docs.

    Returns
    ----------
    A `kbmod.WorkUnit` reprojected with a common `astropy.wcs.WCS`.
    """
    if parallelize:
        return _reproject_work_unit_in_parallel(work_unit, common_wcs, frame, max_parallel_processes)
    else:
        return _reproject_work_unit(work_unit, common_wcs, frame)


def _reproject_work_unit(work_unit, common_wcs, frame="original"):
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
        Can either be 'original' or 'ebd' to specify whether to
        use the WorkUnit._per_image_wcs or ._per_image_ebd_wcs
        respectively.
    Returns
    ----------
    A `kbmod.WorkUnit` reprojected with a common `astropy.wcs.WCS`.
    """
    height, width = common_wcs.array_shape
    images = work_unit.im_stack.get_images()
    obstimes = np.array(work_unit.get_all_obstimes())

    unique_obstimes = np.unique(obstimes)
    per_image_indices = []

    stack = ImageStack()
    for time in unique_obstimes:
        indices = list(np.where(obstimes == time)[0])
        per_image_indices.append(indices)

        science_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
        variance_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
        mask_add = np.zeros(common_wcs.array_shape, dtype=np.float32)
        footprint_add = np.zeros(common_wcs.array_shape, dtype=np.ubyte)

        for index in indices:
            image = images[index]
            science = image.get_science()
            variance = image.get_variance()
            mask = image.get_mask()

            if frame == "original":
                original_wcs = work_unit.get_wcs(index)
            elif frame == "ebd":
                original_wcs = work_unit._per_image_ebd_wcs[index]
            else:
                raise ValueError("Invalid projection frame provided.")

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

        new_layered_image = LayeredImage(
            science_add,
            variance_add,
            mask_add,
            psf,
            time,
        )
        stack.append_image(new_layered_image, force_move=True)

    per_image_wcs = work_unit._per_image_wcs
    per_image_ebd_wcs = work_unit._per_image_ebd_wcs

    new_wunit = WorkUnit(
        im_stack=stack,
        config=work_unit.config,
        wcs=common_wcs,
        constituent_images=work_unit.constituent_images,
        per_image_wcs=per_image_wcs,
        per_image_ebd_wcs=per_image_ebd_wcs,
        per_image_indices=per_image_indices,
        reprojected=True,
    )

    return new_wunit


def _reproject_work_unit_in_parallel(
    work_unit, common_wcs, frame="original", max_parallel_processes=MAX_PROCESSES
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
        Can either be 'original' or 'ebd' to specify whether to
        use the WorkUnit._per_image_wcs or ._per_image_ebd_wcs
        respectively.
    max_parallel_processes : `int`
        The maximum number of parallel processes to use when reprojecting.
        Default is 8. For more see `concurrent.futures.ProcessPoolExecutor` in
        the Python docs.

    Returns
    ----------
    A `kbmod.WorkUnit` reprojected with a common `astropy.wcs.WCS`.
    """

    # get all the unique obstimes
    all_obstimes = np.array(work_unit.get_all_obstimes())
    unique_obstimes = np.unique(all_obstimes)

    # Create the list of lists of indicies for each unique obstimes. i.e. [[0], [1], [2,3]]
    unique_obstimes_indices = [list(np.where(all_obstimes == time)[0]) for time in unique_obstimes]

    # get the list of images from the work_unit outside the for-loop
    images = work_unit.im_stack.get_images()

    future_reprojections = []
    with concurrent.futures.ProcessPoolExecutor(max_parallel_processes) as executor:
        # for a given list of obstime indices, collect all the science, variance, and mask images.
        for indices in unique_obstimes_indices:
            original_wcs = _validate_original_wcs(work_unit, indices, frame)
            # get the list of images for each unique obstime
            images_at_obstime = [images[i] for i in indices]

            # convert each image into a science, variance, or mask "image", i.e. a list of numpy arrays.
            science_images_at_obstime = [this_image.get_science().image for this_image in images_at_obstime]
            variance_images_at_obstime = [this_image.get_variance().image for this_image in images_at_obstime]
            mask_images_at_obstime = [this_image.get_mask().image for this_image in images_at_obstime]

            obstimes = [all_obstimes[i] for i in indices]

            # call `_reproject_images` in parallel.
            future_reprojections.append(
                executor.submit(
                    _reproject_images,
                    science_images=science_images_at_obstime,
                    variance_images=variance_images_at_obstime,
                    mask_images=mask_images_at_obstime,
                    obstimes=obstimes,
                    common_wcs=common_wcs,
                    original_wcs=original_wcs,
                )
            )

    # when all the multiprocessing has finished, convert the returned numpy arrays to RawImages.
    concurrent.futures.wait(future_reprojections, return_when=concurrent.futures.ALL_COMPLETED)
    stack = ImageStack([])
    for result in future_reprojections:
        science_add, variance_add, mask_add, time = result.result()
        psf = _get_first_psf_at_time(work_unit, time)

        # And then stack the RawImages into a LayeredImage.
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

    # Add the imageStack to a new WorkUnit and return it.
    new_wunit = WorkUnit(
        im_stack=stack,
        config=work_unit.config,
        wcs=common_wcs,
        constituent_images=work_unit.constituent_images,
        per_image_wcs=work_unit._per_image_wcs,
        per_image_ebd_wcs=work_unit._per_image_ebd_wcs,
        per_image_indices=unique_obstimes_indices,
        reprojected=True,
    )

    return new_wunit


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
        Can either be 'original' or 'ebd' to specify whether to
        use the WorkUnit._per_image_wcs or ._per_image_ebd_wcs
        respectively.

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
        original_wcs = [work_unit._per_image_ebd_wcs[i] for i in indices]
    else:
        raise ValueError("Invalid projection frame provided.")

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
    `kbmod.serach.PSF`
        The first PSF object found at the given time.

    Raises
    ------
    ValueError
        If the time is not found in list of observation times in the work_unit,
        raise an error.
    """
    obstimes = np.array(work_unit.get_all_obstimes())

    # if the time isn't in the list of times, raise an error.
    if time not in obstimes:
        raise ValueError(f"Observation time {time} not found in work unit.")

    images = work_unit.im_stack.get_images()
    index = np.where(obstimes == time)[0][0]
    return images[index].get_psf()


def _reproject_images(science_images, variance_images, mask_images, obstimes, common_wcs, original_wcs):
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
    obstimes : `list[float]`
        List of observation times for each image.
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

    # all the obstimes should be identical, so we can just use the first one.
    time = obstimes[0]

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

    return science_add, variance_add, mask_add, time
