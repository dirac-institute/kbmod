from astropy.wcs import WCS
from astropy.nddata import CCDData
import reproject
import numpy as np

from kbmod.work_unit import WorkUnit
from kbmod.search import RawImage, LayeredImage, ImageStack

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
    A `kbmod.search.RawImage` reprojected with a common `astropy.wcs.WCS`.
    """
    image_data = CCDData(image.image, unit="adu")
    image_data.wcs = original_wcs

    new_image, _ = reproject.reproject_interp(
        image_data, common_wcs, shape_out=common_wcs.array_shape, order="bicubic"
    )

    return RawImage(img=new_image.astype("float32"), obs_time=obs_time)

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

    if len(work_unit.per_image_wcs) != len(images):
        raise ValueError("no per_image_wcs provided for WorkUnit")

    image_list = []

    for index, image in enumerate(images):
        science = image.get_science()
        variance = image.get_variance()
        obs_time = image.get_obstime()
        original_wcs = work_unit.per_image_wcs[index]

        reprojected_science = reproject_raw_image(
            science, original_wcs, common_wcs, obs_time
        )

        reprojected_variance = reproject_raw_image(
            variance, original_wcs, common_wcs, obs_time
        )

        mask = image.get_mask()
        psf = image.get_psf()

        new_layered_image = LayeredImage(
            reprojected_science,
            reprojected_science,
            mask,
            psf
        )

        image_list.append(new_layered_image)
    
    stack = ImageStack(image_list)
    new_wunit = WorkUnit(im_stack=stack, config=work_unit.config)
    new_wunit.wcs = common_wcs

    return new_wunit


