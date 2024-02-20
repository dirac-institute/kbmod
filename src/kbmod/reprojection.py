import numpy as np
import reproject
from astropy.nddata import CCDData
from astropy.wcs import WCS

from kbmod.search import KB_NO_DATA, PSF, ImageStack, LayeredImage, RawImage
from kbmod.work_unit import WorkUnit


def reproject_raw_image(image, original_wcs, common_wcs, obs_time):
    """Given an ndarray representing image data (either science or variance,
    when used with `reproject_work_unit`), as well as a common wcs, return the reprojected
    RawImage.

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
    image_data = CCDData(image.image, unit="adu")
    image_data.wcs = original_wcs

    new_image, footprint = reproject.reproject_interp(
        image_data, common_wcs, shape_out=common_wcs.array_shape, order="bicubic"
    )

    return new_image, footprint


def reproject_work_unit(work_unit, common_wcs):
    """Given a WorkUnit and a WCS, reproject all of the images in the ImageStack
    into a common WCS.

    Attributes
    ----------
    work_unit : `kbmod.WorkUnit`
        The WorkUnit to be reprojected.
    common_wcs : `astropy.wcs.WCS`
        The WCS to reproject all the images into.

    Returns
    ----------
    A `kbmod.WorkUnit` reprojected with a common `astropy.wcs.WCS`.
    """
    height, width = common_wcs.array_shape
    images = work_unit.im_stack.get_images()
    obstimes = np.array(work_unit.get_all_obstimes())

    image_list = []

    unique_obstimes = np.unique(obstimes)

    for time in unique_obstimes:
        indices = list(np.where(obstimes == time)[0])

        science_add = np.zeros(common_wcs.array_shape)
        variance_add = np.zeros(common_wcs.array_shape)
        mask_add = np.zeros(common_wcs.array_shape)
        footprint_add = np.zeros(common_wcs.array_shape)

        for index in indices:
            image = images[index]
            science = image.get_science()
            variance = image.get_variance()
            mask = image.get_mask()
            original_wcs = work_unit.get_wcs(index)
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
        mask_add = np.where(np.isclose(mask_add, 0.0, atol=1e-01), 0.0, 1.0)

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

    stack = ImageStack(image_list)
    new_wunit = WorkUnit(im_stack=stack, config=work_unit.config, wcs=common_wcs)

    return new_wunit
