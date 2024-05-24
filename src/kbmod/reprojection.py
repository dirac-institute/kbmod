import numpy as np
import concurrent.futures
import reproject
from astropy.nddata import CCDData
from astropy.wcs import WCS

from kbmod.search import KB_NO_DATA, PSF, ImageStack, LayeredImage, RawImage
from kbmod.work_unit import WorkUnit

# The number of executors to use in the parallel reprojecting function.
NUM_EXECUTORS = 8

def reproject_raw_image(image, original_wcs, common_wcs, obs_time):
    """Given an ndarray representing image data (either science or variance,
    when used with `reproject_work_unit`), as well as a common wcs, return the reprojected
    image and footprint as a numpy.ndarray.

    Attributes
    ----------
    image : `kbmod.search.RawImage`
        The image data to be reprojected.
    original_wcs : `astropy.wcs.WCS`
        The WCS of the original image.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.
    obs_time : float
        The MJD of the observation.
    Returns
    ----------
    new_image : `numpy.ndarray`
        The image data reprojected with a common `astropy.wcs.WCS`.
    footprint : `numpy.ndarray`
        An array containing the footprint of pixels that have data.
        for footprint[i][j], it's 1 if there is a corresponding reprojected
        pixel and 0 if there is no data.
    """
    return _reproject_image(image.image, original_wcs, common_wcs)


def reproject_ndarray_image(image, original_wcs, common_wcs):
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
    ----------
    new_image : `numpy.ndarray`
        The image data reprojected with a common `astropy.wcs.WCS`.
    footprint : `numpy.ndarray`
        An array containing the footprint of pixels that have data.
        for footprint[i][j], it's 1 if there is a corresponding reprojected
        pixel and 0 if there is no data.
    """
    return _reproject_image(image, original_wcs, common_wcs)


def _reproject_image(image, original_wcs, common_wcs):
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
    ----------
    new_image : `numpy.ndarray`
        The image data reprojected with a common `astropy.wcs.WCS`.
    footprint : `numpy.ndarray`
        An array containing the footprint of pixels that have data.
        for footprint[i][j], it's 1 if there is a corresponding reprojected
        pixel and 0 if there is no data.
    """
    image_data = CCDData(image, unit="adu")
    image_data.wcs = original_wcs

    new_image, footprint = reproject.reproject_adaptive(
        image_data, common_wcs, shape_out=common_wcs.array_shape, bad_value_mode="ignore"
    )

    return new_image, footprint


def reproject_work_unit(work_unit, common_wcs, frame="original", parallelize=True):
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
    parallelize : bool
        If True, use multiprocessing to reproject the images in parallel.
        Default is True.

    Returns
    ----------
    A `kbmod.WorkUnit` reprojected with a common `astropy.wcs.WCS`.
    """
    if parallelize:
        return _reproject_work_unit_in_parallel(work_unit, common_wcs, frame)
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

    image_list = []

    unique_obstimes = np.unique(obstimes)
    per_image_indices = []

    for time in unique_obstimes:
        indices = list(np.where(obstimes == time)[0])
        per_image_indices.append(indices)

        science_add = np.zeros(common_wcs.array_shape)
        variance_add = np.zeros(common_wcs.array_shape)
        mask_add = np.zeros(common_wcs.array_shape)
        footprint_add = np.zeros(common_wcs.array_shape)

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

            reprojected_science, footprint = reproject_raw_image(science, original_wcs, common_wcs, time)

            footprint_add += footprint
            # we'll enforce that there be no overlapping images at the same time,
            # for now. We might be able to add some ability co-add in the future.
            if np.any(footprint_add > 1.0):
                raise ValueError("Images with the same obstime are overlapping.")

            reprojected_variance, _ = reproject_raw_image(variance, original_wcs, common_wcs, time)

            reprojected_mask, _ = reproject_raw_image(mask, original_wcs, common_wcs, time)

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
        gaps = footprint_add == 0.0
        science_add[gaps] = KB_NO_DATA
        variance_add[gaps] = KB_NO_DATA
        mask_add[gaps] = 1

        # transforms the mask back into a bitmask.
        mask_add = np.where(np.isclose(mask_add, 0.0, atol=0.2), 0.0, 1.0)

        science_raw_image = RawImage(img=science_add.astype("float32"), obs_time=time)
        variance_raw_image = RawImage(img=variance_add.astype("float32"), obs_time=time)
        mask_raw_image = RawImage(img=mask_add.astype("float32"), obs_time=time)

        psf = images[indices[0]].get_psf()

        new_layered_image = LayeredImage(
            science_raw_image,
            variance_raw_image,
            mask_raw_image,
            psf,
        )

        image_list.append(new_layered_image)

    per_image_wcs = work_unit._per_image_wcs
    per_image_ebd_wcs = work_unit._per_image_ebd_wcs

    stack = ImageStack(image_list)
    new_wunit = WorkUnit(
        im_stack=stack,
        config=work_unit.config,
        wcs=common_wcs,
        constituent_images=work_unit.constituent_images,
        per_image_wcs=per_image_wcs,
        per_image_ebd_wcs=per_image_ebd_wcs,
        per_image_indices=per_image_indices,
    )

    return new_wunit


def _reproject_work_unit_in_parallel(work_unit, common_wcs, frame="original"):
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
    with concurrent.futures.ProcessPoolExecutor(NUM_EXECUTORS) as executor:
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
                executor.submit(_reproject_images,
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
    time_stamps = []
    image_list = []
    for result in future_reprojections:
        science_add, variance_add, mask_add, time = result.result()
        science_raw_image = RawImage(img=science_add.astype("float32"), obs_time=time)
        variance_raw_image = RawImage(img=variance_add.astype("float32"), obs_time=time)
        mask_raw_image = RawImage(img=mask_add.astype("float32"), obs_time=time)

        psf = _get_first_psf_at_time(work_unit, time)
        # And then stack the RawImages into a LayeredImage.
        new_layered_image = LayeredImage(
            science_raw_image,
            variance_raw_image,
            mask_raw_image,
            psf,
        )

        # append timestamps and layeredImages to lists
        time_stamps.append(time)
        image_list.append(new_layered_image)

    # sort the time_stamps, use the ordering to sort image_list
    image_list = [image_list[i] for i in np.argsort(time_stamps)]
    stack = ImageStack(image_list)

    # Add the imageStack to a new WorkUnit and return it.
    new_wunit = WorkUnit(
        im_stack=stack,
        config=work_unit.config,
        wcs=common_wcs,
        constituent_images=work_unit.constituent_images,
        per_image_wcs=work_unit._per_image_wcs,
        per_image_ebd_wcs=work_unit._per_image_ebd_wcs,
        per_image_indices=unique_obstimes_indices,
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
    science_add = np.zeros(common_wcs.array_shape)
    variance_add = np.zeros(common_wcs.array_shape)
    mask_add = np.zeros(common_wcs.array_shape)
    footprint_add = np.zeros(common_wcs.array_shape)

    # all the obstimes should be identical, so we can just use the first one.
    time = obstimes[0]

    for science, variance, mask, this_original_wcs in zip(science_images, variance_images, mask_images, original_wcs):

        # reproject science, variance, and mask images simulataneously.
        reprojected_images, footprints = reproject_ndarray_image([science, variance, mask], this_original_wcs, common_wcs)

        footprint_add += footprints[0]
        # we'll enforce that there be no overlapping images at the same time,
        # for now. We might be able to add some ability co-add in the future.
        if np.any(footprint_add > 1.0):
            raise ValueError("Images with the same obstime are overlapping.")

        # change all the NaNs to zeroes so that the matrix addition works properly.
        # `footprint_add` will maintain the information about what areas of the frame
        # don't have any data so that we can change it back after we combine.
        reprojected_images[np.isnan(reprojected_images)] = 0.0

        science_add += reprojected_images[0]
        variance_add += reprojected_images[1]
        mask_add += reprojected_images[2]

    # change all the values where there are is no corresponding data to `KB_NO_DATA`.
    gaps = footprint_add == 0.0
    science_add[gaps] = KB_NO_DATA
    variance_add[gaps] = KB_NO_DATA
    mask_add[gaps] = 1

    # transforms the mask back into a bitmask.
    mask_add = np.where(np.isclose(mask_add, 0.0, atol=0.2), 0.0, 1.0)

    return science_add, variance_add, mask_add, time
