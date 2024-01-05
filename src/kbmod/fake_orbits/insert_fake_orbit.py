from astropy.wcs import WCS

from kbmod.fake_orbits.pyoorb_helper import PyoorbOrbit
from kbmod.search import ImageStack, LayeredImage, PSF, RawImage
from kbmod.work_unit import WorkUnit

def safe_add_fake_detection(img, x, y, flux):
    """Add a fake detection to a LayeredImage.

    Parameters
    ----------
    img : `RawImage` or `LayeredImage`
        The image to modify.
    x : `float`
        The x pixel location of the fake object.
    y : `float`
        The y pixel location of the fake object.
    flux : `float`
        The flux value.

    Returns
    -------
    result : `bool`
        A result indicating whether the detection was inserted within
        the image and on an unmasked pixel.
    """
    # Check the pixels are in bounds
    if x < 0 or x >= img.get_width():
        return False
    if y < 0 or y >= img.get_height():
        return False

    # Check that no mask flags are set.
    if img.get_mask().get_pixel(int(y), int(x)) != 0:
        return False

    sci = img.get_science()
    psf = img.get_psf()
    dim = psf.get_dim()

    initial_x = x - psf.get_radius()
    initial_y = y - psf.get_radius()
    for i in range(dim):
        for j in range(dim):
            sci.interpolated_add(float(initial_x + i), float(initial_y + j), flux * psf.get_value(i, j))

    return True
 

def insert_fake_orbit_into_work_unit(worku, orbit, flux, obscode):
    """Insert the predicted positions for an oorb orbit into the WorkUnit images.

    Parameters
    ----------
    worku : `WorkUnit`
        The WorkUnit to modify.
    orbit : `PyoorbOrbit`
        The orbit to use.
    flux : `float`
        The object brightness in the image.
    obscode : `str`
        The observator code to use for predictions.
    """
    mjds = worku.get_all_obstimes()
    predicted_pos = orbit.get_ephemerides(mjds, obscode)

    results = []
    for i in range(len(mjds)):
        current_wcs = worku.get_wcs(i)
        pixel_loc = current_wcs.world_to_pixel(predicted_pos[i])
        x = pixel_loc[0].item()
        y = pixel_loc[1].item()

        img = worku.im_stack.get_single_image(i)
        if safe_add_fake_detection(img, x, y, flux):
            results.append((x, y))
        else:
            results.append(None)

    return results
