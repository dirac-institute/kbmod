Reprojection
============

A KBMOD search assumes that each pixel of each image in the :py:class:`~~kbmod.search.ImageStack` are aligned in the same RA and dec and space. To ensure that is the case in searches where the observations weren't pixel oriented deliberately, we use our reprojection utilities to transform the data.

Simple Reproject
----------------

This the first and simplest type of reprojection that we do. Given a :py:class:`~~kbmod.search.WorkUnit` with images of differing WCSs, choose a common WCS and reproject all the images to it.

The only requirement for this reprojection is that images that have the same observation time in the stack aren't overlapping (so any images taken an instrument that has multiple detectors and tiling, such as LSST, are fine!). Please note that we don't currently check and filter based on how much of the original image ends up in the new reprojected stack, so if the images aren't overlapping in radec space you might have some mostly or even totally empty images.

To run a simple reprojection, take a :py:class:`~~kbmod.search.WorkUnit` and a common wcs and pass those into `reproject_work_unit`. For more info on the various configuration parameters for the job, see the :ref:`kbmod.reprojection` reference.

Barycentric Correction
----------------------

Because KBMOD mostly searches for objects that are past Neptune and are therefore quite far away from us and center of the solar system, most of their apparent motion on the sky is caused by the parallax from the Earth moving around the sun, and not the actual orbit of the object. This unfortunately adds a lot of non-linearity to the trajectory and makes the objects much harder find. To remedy this problem, we take our images and correct our observations to simulate what they would look like if the observation was taken from the solar system barycenter.

To accomplish this, we do the following:
 * take a guess distance from the barycenter and assume that there is a virtual object at that distance
 * find the distance from this virtual point to the actual observation point on earth.
 * "correct the parallax" by reprojecting the point into the ICRS coordinates that it would be at if it was actually observed from the barycenter.
 * randomly sample N points from a given image's WCS and repeat the process
 * fit a new WCS from these points
 * replace the image's old WCS with this new "explicit barycentric distance" (EBD) WCS.
 * run the reprojection code on this image and others with a common WCS in EBD space.
 * each pixel will be resampled into the new EBD space, and the parallax motion will be corrected for!

Here's a diagram describing this process:

.. image:: ../_static/brute_force_wcs_fitting.png

