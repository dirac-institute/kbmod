from astropy.wcs import WCS
from astropy.nddata import CCDData
import reproject
import numpy as np

from kbmod.work_unit import WorkUnit
from kbmod.search import RawImage, LayeredImage, ImageStack, KB_NO_DATA, PSF

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
    A `numpy.ndarray` of the image data reprojected with a common `astropy.wcs.WCS`,
    as well as the footprint of the reprojection (also an `numpy.ndarray`).
    """
    image_data = CCDData(image.image, unit="adu")
    image_data.wcs = original_wcs

    new_image, footprint  = reproject.reproject_interp(
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

    if len(work_unit.per_image_wcs) != len(images):
        raise ValueError("no per_image_wcs provided for WorkUnit")

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
            original_wcs = work_unit.per_image_wcs[index]

            reprojected_science, footprint = reproject_raw_image(
                science, original_wcs, common_wcs, time
            )

            footprint_add += footprint
            # we'll enforce that there be no overlapping images at the same time,
            # for now. We might be able to add some ability co-add in the future.
            if np.any(footprint_add > 1.):
                raise ValueError("Images with the same obstime are overlapping.")

            reprojected_variance, _ = reproject_raw_image(
                variance, original_wcs, common_wcs, time
            )

            reprojected_mask, _ = reproject_raw_image(
                mask, original_wcs, common_wcs, time
            )

            # change all the NaNs to zeroes so that the matrix addition works properly.
            # `footprint_add` will maintain the information about what areas of the frame
            # don't have any data so that we can change it back after we combine.
            reprojected_science[np.isnan(reprojected_science)] = 0.
            reprojected_variance[np.isnan(reprojected_variance)] = 0.
            reprojected_mask[np.isnan(reprojected_mask)] = 0.

            science_add += reprojected_science
            variance_add += reprojected_variance
            mask_add += reprojected_mask

        # change all the values where there are is no corresponding data to `KB_NO_DATA.`
        gaps = footprint_add == 0.
        science_add[gaps] = KB_NO_DATA
        variance_add[gaps] = KB_NO_DATA
        mask_add[gaps] = KB_NO_DATA

        science_raw_image = RawImage(img=science_add.astype("float32"), obs_time=time)
        variance_raw_image = RawImage(img=variance_add.astype("float32"), obs_time=time)
        mask_raw_image = RawImage(img=variance_add.astype("float32"), obs_time=time)

        psf = images[indices[0]].get_psf()

        new_layered_image = LayeredImage(
            science_raw_image,
            variance_raw_image,
            mask_raw_image,
            psf,
        )

        image_list.append(new_layered_image)
    
    stack = ImageStack(image_list)
    new_wunit = WorkUnit(im_stack=stack, config=work_unit.config)
    new_wunit.wcs = common_wcs

    return new_wunit


