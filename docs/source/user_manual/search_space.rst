Search Algorithm and Search Space
=================================

The KBMOD algorithm algorithm performs shift and stack using a grid of potential trajectories. The granularity of this grid is critical to trading off completeness and computational cost. If the grid is too fine, the algorithm will spend a lot of time checking similar trajectories. If the grid is too coarse, the algorithm might miss true objects. Below we discuss the sampling method in more detail.

Search Overview
---------------

KBMOD operates by considering a candidate set of velocities and computing the likelihood of an object for the cross product of **each** velocity and **each** starting pixel. This can be viewed as a pair of nested loops (although it implemented in a parallelized GPU search)::

    For each starting pixel:
        For each velocity in the grid:
            Compute the likelihood of this trajectory

Thus the code checks all of the velocities starting starting out of each pixel.

Due to memory limitations a set numner of trajectories are kept for **each** starting pixel (configured by the ``results_per_pixel`` configuration
parameter with a default of 8). At the end of the search the code will return ``num_pixels * results_per_pixel`` results ranked by their likelihood.

Starting Pixels
---------------

By default KBMOD will attempt to shift and stack using each pixel in the initial image as a potential starting location. If we run KBMOD on an image with width ``w`` and height ``h``, we get ``w * h`` possible starting locations. See :ref:`Data Model` for more information on how the images are stored.

KBMOD provides the ability to expand or contract the range of starting pixels using two different methods:

1. Adding a fixed sized buffer around the edge of the images. This is done using the ``x_pixel_buffer`` and ``y_pixel_buffer`` parameters to set separate buffers for the x and y dimensions. If ``x_pixel_buffer = 5`` the code will add 5 pixels to both sides of the image, using starting pixels from [-5, w + 5).
2. Specifying absolute pixel boundaries. This is done using the ``x_pixel_bounds`` and ``y_pixel_bounds`` parameters. For example you can *reduce* the size of the search by using ``x_pixel_bounds = [10, 100]``, which will search [10, 100) regardless of the image's width. Similarly, you can increase the region of the search by setting the bounds outside the image's area.

Velocity Grid
-------------

Perhaps the most complex aspect of the KBMOD algorithm is how it defines the grid of search velocities. The grid is defined by two sets of parameters: a sampling of absolute velocities (``v_arr``) in pixels per day and a sampling of the velocities' angles (``ang_arr``) in radians. Each sampling consists of values defining the range and number of sampling steps. 

The velocity array ``v_arr`` uses the format [minimum velocity, maximum velocity, number of steps]. The setting ``v_arr = [92.0, 526.0, 256]`` samples velocities from 92 pixels per day to 526 pixels per day with 256 equally spaced samples.

The complexity of the velocity grid comes from the fact that the angles specified by ``ang_arr`` are **not** absolute angles in pixel space, but rather offsets from a given suggested angle. The user can specify this suggested angle directly with the parameter ``average_angle``. If no such parameter is given the code computes a suggested angle based on the ecliptic angle for the images (as defined by their WCS). This allows KBMOD to focus on trajectories around where the most objects are expected to be.

Another important factor is that ``ang_arr`` is defined as [offset for min angle, offset for max_angle, number of steps]. So the settings::

    average_angle = 1.0
    ang_arr = [0.5, 0.5, 100]

produce a search grid from angle 0.5 (``average_angle - ang_arr[0]``) to 1.5 (``average_angle + ang_arr[1]``) using 100 steps. Note that the first element of ``ang_arr`` is **subtracted** from ``average_angle`` to provide the lower bound and the second element is **added** to ``average_angle`` to provide the upper bound.

Given the linear sampling for both velocities and angles, the full set of candidate trajectories is computed as::


    for (int a = 0; a < angleSteps; ++a) {
        for (int v = 0; v < velocitySteps; ++v) {
            searchList[a * velocitySteps + v].xVel = cos(angles[a]) * velocities[v];
            searchList[a * velocitySteps + v].yVel = sin(angles[a]) * velocities[v];
        }
    }

where ``angles`` contains the list of angles to test and ``velocities`` contains the list of velocities.
