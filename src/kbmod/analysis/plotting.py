import math
import numpy as np
import warnings

import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import simple_norm, ZScaleInterval, AsinhStretch, ImageNormalize

import matplotlib.pyplot as plt

from kbmod.search import ImageStack, LayeredImage, RawImage
from kbmod.results import Results

__all__ = [
    "iter_over_obj",
    "transform_rect",
    "plot_field",
    "plot_bbox",
    "plot_bboxes",
    "plot_footprint",
    "plot_footprints",
    "plot_all_objs",
    "plot_focal_plane",
    "plot_cutouts",
    "plot_image",
    "plot_multiple_images",
    "plot_time_series",
    "plot_result_row",
]


def iter_over_obj(objects):
    """Folds the given list of objects on their ``Name`` column and
    iterates over them sorted by date-time stamp.

    Parameters
    -----------
    objects : `astropy.table.Table`
        Table of objects.

    Returns
    --------
    obj : `iterator`
        Iterator over individual object observations.
    """
    names = set(objects["Name"])
    for name in names:
        obj = objects[objects["Name"] == name]
        obj.sort("epoch")
        yield obj


def transform_rect(points):
    """Given a rectangle defined by 4 points (clockwise convention)
    returns top-left point, width, height, and angle of rectangle.

    Parameters
    ----------
    points : `list`
        List of 4 tuples representing (x, y) coordinates of the
        corners of a rectanlge, in clockwise convention.

    Returns
    -------
    xy : `tuple`
        Top left corner (x, y) coordinates.
    width : `float`
        Width
    height : `float`
        Height
    angle : `float`
        Angle of rotation, in radians.
    """
    calc_dist = lambda p1, p2: np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    calc_angle = lambda p1, p2: np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    # flip height so the xy becomes top left, then we don't have to guess
    # which point we need to return
    width = calc_dist(points[0], points[1])
    height = calc_dist(points[1], points[2])
    xy = points[0]

    angle = calc_angle(points[-1], points[0])

    return xy, width, -height, angle


def plot_field(ax, center, radius):
    """Adds a circle of the given radius at the given
    center coordinates to the given axis.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    center : `tuple` or `list`
        An `(x, y)` pair of coordiantes
    radius : `float`
        Radius of the circle around the center.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Modified Axis.
    """
    ax.scatter(*center, color="black", label="Pointing area")
    circ = plt.Circle(center, radius, fill=False, color="black")
    ax.add_artist(circ)
    return ax


def plot_bbox(ax, bbox):
    """Adds the footprint defined by the given BBOX to the axis.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    bbox : `list`
        List of 4 tuples representing (x, y) coordinates of the
        corners of a rectanlge, in clockwise convention.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Modified Axis.
    """
    xy, width, height, angle = transform_rect(bbox)
    rect = plt.Rectangle(xy, width, height, angle=angle, fill=None, color="black")
    ax.add_artist(rect)
    return ax


def plot_bboxes(ax, bboxes):
    """Adds the footprints defined by each given BBOX to the axis.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    bbox : `list[list]`
        List of bboxes. Each bbox is a list of 4 tuples representing
        (x, y) coordinates of the corners of a rectanlge, in clockwise
        convention.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Modified Axis.
    """
    for bbox in bboxes:
        ax = plot_bbox(ax, bbox, figure)
    return ax


def plot_footprint(ax, wcs):
    """Adds the footprint defined by the given WCS to the axis.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    wcs : `astropy.wcs.WCS`
        World Coordinate System instance from which the footprint will
        be calculated from.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Modified Axis.
    """
    xy, width, height, angle = transform_rect(wcs.calc_footprint())
    rect = plt.Rectangle(xy, width, height, angle=angle, fill=None, color="black")
    ax.add_artist(rect)
    return ax


def plot_footprints(ax, wcs_list):
    """Adds the footprints defined by each given WCS to the axis.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    wcs_list : `list[astropy.wcs.WCS]`
        List of WCS objects whose footprints are being plotted.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Modified Axis.
    """
    for wcs in wcs_list:
        ax = plot_footprint(ax, wcs, figure)
    return ax


def plot_all_objs(ax, objects, count=-1, show_field=False, center=None, radius=1.1, lw=0.9, ms=1):
    """Plots object tracks on the given axis.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    objects : `astropy.table.Table`
        Table of objects. Must contain columns ``Name``, ``RA`` and ``DEC``
        identifying unique objects, their right ascension and declination
        coordinates.
    count : `int`
        Number of tracks to plot. Default -1, plots all objects.
    show_field : `bool`
        `False` by default, when `True` requires ``center`` and
        ``radius``to be specified. Adds an circle around the center
        coordinates with the given radius.
    center : `None` or `tuple`
        A pair of ``(x, y)`` coordinates of the center of the field.
    radius : `float`
        Radius of the field, in decimal degrees. Useful to provide
        context as to which objects might have landed into the field
        of view. Default: 1.1 degrees, to match the DECam FOV.
    lw : `float`
        Line width of the object's trajectory.
    ms : `float`
        Marker size of the object's ephemerides.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    """
    if show_field:
        plot_field(ax, center, radius)

    if count < 0:
        return ax

    for i, obj in enumerate(iter_over_obj(objects)):
        if count > 0 and i == count:
            break
        ax.plot(obj["RA"], obj["DEC"], label=obj["Name"][0], marker="o", lw=lw, ms=ms)

    return ax


def plot_focal_plane(ax, hdulist, showExtName=True, txt_x_offset=20 * u.arcsec, txt_y_offset=-120 * u.arcsec):
    """Plots the footprint of given HDUList on the axis.

    Iterates over each HDU in the HDUList and attempts to plot its
    footprint if it can determine a valid WCS. Otherwise skips it.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    hdulist : `list[astropy.io.fits.HDUList]`
        An `HDUList` object.
    showExtName : `bool`
        Display the value of keycard ``EXTNAME`` as a label at the
        center of the plotted footprint, if the header contains a
        keycard ``EXTNAME``.
    txt_x_offset : `astropy.units.arcsec`
        X-axis offset of the EXTNAME label, if one exists. Default: 20
    txt_y_offset : `astropy.units.arcsec`
        Y-axis offset of the EXTNAME label, if one exists. Default: -120

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Axis.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        wcss = [WCS(hdu.header) for hdu in hdulist]

    # I really wish that WCS would pop an error when unable to
    # init from a header instead of returning a default.
    default_wcs = WCS().to_header_string()
    for hdu in hdulist:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            wcs = WCS(hdu.header)

        if default_wcs != wcs.to_header_string():
            ax = plot_footprint(ax, wcs)
            pt = wcs.pixel_to_world(0, 0)
            # units vs quantities
            xoffset = txt_x_offset.to(u.deg).value
            yoffset = txt_y_offset.to(u.deg).value
            # we need to move diagonally to the right and down to center the text
            x, y = pt.ra.deg + xoffset, pt.dec.deg + yoffset
            ax.text(x, y, hdu.header.get("EXTNAME", None), clip_on=True)

    return ax


def plot_cutouts(axes, cutouts, remove_extra_axes=True):
    """Plots cutouts (images) onto given axes.

    The number of axes must be equal to or greater
    than the number of cutouts.

    Parameters
    ----------
    ax : `list[matplotlib.pyplot.Axes]`
        Axes.
    cutouts : `list`, `np.array` or `astropy.ndutils.Cutout2D`
        Collection of numpy arrays or ``Cutout2D`` objects
        to plot.
    remove_extra_axes : `bool`, optional
         When `True` (default), the axes that would be
         left empty are removed from the plot.

    Raises
    -------
    ValueError - When number of given axes is less than
    the number of given cutouts.
    """
    nplots = len(cutouts)

    axs = axes.ravel()
    naxes = len(axs)

    if naxes < nplots:
        raise ValueError(f"N axes ({len(axs)}) doesn't match N plots ({nplots}).")

    for ax, cutout in zip(axs, cutouts):
        img = cutout.data if isinstance(cutout, Cutout2D) else cutout
        norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
        im = ax.imshow(cutout.data, norm=norm)
        ax.set_aspect("equal")
        ax.axvline(cutout.shape[0] / 2, c="red", lw=0.25)
        ax.axhline(cutout.shape[1] / 2, c="red", lw=0.25)

    if remove_extra_axes and naxes > nplots:
        for ax in axs[nplots - naxes :]:
            ax.remove()

    return axes


def plot_image(img, ax=None, figure=None, norm=True, title=None, show_counts=True):
    """Plots an image on an axis and figure.

    If no axis is given creates a new figure. Draws a crosshair at the
    center of the image. Must provide both axis and figure; figure is
    required so a colorbar could be attached to it.

    If a one-dimensional array is given (as for a stamp), the function
    will try to reshape as a square image.

    Parameters
    ----------
    img : `np.ndarray`, `RawImage`, or `LayeredImage`
        The image data.
    ax : `matplotlib.pyplot.Axes` or `None`
        Axes, `None` by default.
    figure : `matplotlib.pyplot.Figure` or `None`
        Figure, `None` by default.
    norm: `bool`, optional
        Normalize the image using Astropy's `ImageNormalize`
        using `ZScaleInterval` and `AsinhStretch`. `True` by
        default.
    title : `str` or None, optional
        Title of the plot.
    show_counts : `bool`
        Show the counts color bar. ``True`` by default.

    Returns
    -------
    figure : `matplotlib.pyplot.Figure`
        Modified Figure.
    ax : `matplotlib.pyplot.Axes`
        Modified Axes.
    """
    if ax is None and figure is None:
        figure, ax = plt.subplots(figsize=(15, 10))
    elif ax is not None and figure is None:
        raise ValueError("Provide both figure and axis, or provide none.")
    elif ax is None and figure is not None:
        raise ValueError("Provide both figure and axis, or provide none.")
    else:
        pass

    # Check the image's type and convert to an numpy array.
    if type(img) is RawImage:
        img = img.image
    elif type(img) is LayeredImage:
        img = img.get_science().image

    # If the image array is 1-dimensional, see if it can be unpacked into a square
    # image (used to unpack stamps).
    if len(img.shape) == 1:
        stamp_width = int(math.sqrt(img.shape[0]))
        if img.size != stamp_width * stamp_width:
            raise ValueError("Unable to reshape array of shape = {img.shape}")
        img = img.reshape(stamp_width, stamp_width)

    # Normalize the image if requested.
    if norm:
        norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
        im = ax.imshow(img, norm=norm)
    else:
        im = ax.imshow(img)

    ax.axhline(img.shape[0] / 2, c="red", lw=0.5)
    ax.axvline(img.shape[1] / 2, c="red", lw=0.5)
    ax.set_title(title)
    if show_counts:
        figure.colorbar(im, label="Counts")

    return figure, ax


def plot_multiple_images(images, figure=None, columns=3, labels=None, norm=False):
    """Plot multiple images in a grid.

    Parameters
    ----------
    images : a `list`, `numpy.ndarray`, or `ImageStack` of images.
        The series of images to plot.
    figure : `matplotlib.pyplot.Figure` or `None`
        Figure, ``None`` by default.
    columns : `int`
        The number of columns to use. 3 by default.
    labels : `list` of `str`
        The labels to use for each image. ``None`` by default.
    norm: `bool`
        Normalize the image using Astropy's `ImageNormalize`
        using `ZScaleInterval` and `AsinhStretch`. ``False`` by
        default.
    """
    # Automatically unpack an ImageStack.
    if type(images) is ImageStack:
        num_imgs = images.img_count()
        if labels is None:
            labels = [f"Time={images.get_obstime(i)}" for i in range(num_imgs)]
        images = [images.get_single_image(i).get_science().image for i in range(num_imgs)]

    num_imgs = len(images)
    num_rows = math.ceil(num_imgs / columns)

    # Create a new figure if needed.
    if figure is None:
        figure = plt.figure(figsize=(15, 10))

    for idx, img in enumerate(images):
        ax = figure.add_subplot(num_rows, columns, idx + 1)
        if labels is None:
            title = f"Number {idx}"
        else:
            title = labels[idx]
        plot_image(img, ax=ax, figure=figure, norm=norm, title=title, show_counts=False)

    plt.subplots_adjust(wspace=0.25, hspace=0.4)


def plot_time_series(values, times=None, indices=None, ax=None, figure=None, title=None):
    """Plot a time series on the graph.

    Parameters
    ----------
    values : a `list` or `numpy.ndarray` of floats
        The array of the values at each time.
    times : a `list` or `numpy.ndarray` of floats
        The array of the time stamps. If ``None`` then uses equally
        spaced points. `None` by default.
    indices : a `list` or `numpy.ndarray` of bools
        The array of which indices are valid. If ``None`` then
        all indices are considered valid. `None` by default.
    ax : `matplotlib.pyplot.Axes` or `None`
        Axes, `None` by default.
    figure : `matplotlib.pyplot.Figure` or `None`
        Figure, `None` by default.
    title : `str` or None, optional
        Title of the plot. `None` by default.
    """
    y_values = np.array(values)

    # If no axes were given, create a new figure.
    if ax is None:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

    # If no valid indices are given, use them all.
    if indices is None:
        indices = np.array([True] * len(values), dtype=bool)
    else:
        indices = np.array(indices, dtype=bool)

    # If the times are not given, then use linear spacing.
    if times is None:
        x_values = np.linspace(0, len(values) - 1, len(values), dtype=int)
    else:
        x_values = np.array(times)

    # Plot the data with the curve in blue, the valid points as blue dots,
    # and the invalid indices as smaller red dots.
    ax.plot(x_values, y_values, "b")
    ax.plot(x_values[indices], y_values[indices], "b.", ms=25)
    ax.plot(x_values[~indices], y_values[~indices], "r.", ms=10)

    if title is not None:
        ax.set_title(title)


def plot_result_row(row, times=None, figure=None):
    """Plot a single row of the results table.

    Parameters
    ----------
    row : `astropy.table.row.Row`
        The information from the results to plot.
    times : a `list` or `numpy.ndarray` of floats
        The array of the time stamps. If ``None`` then uses equally
        spaced points. `None` by default.
    figure : `matplotlib.pyplot.Figure` or `None`
        Figure, `None` by default.
    """
    if figure is None:
        figure = plt.figure()

    # Create subfigures on the top and bottom.
    (fig_top, fig_bot) = figure.subfigures(2, 1)

    # In the top subfigure plot the coadded stamp on the left and
    # the light curve on the right.
    (ax_stamp, ax_lc) = fig_top.subplots(1, 2)
    if row["stamp"] is not None:
        plot_image(row["stamp"], ax=ax_stamp, figure=fig_top, norm=True, title="Coadded Stamp")
    else:
        ax_stamp.text(0.5, 0.5, "No Stamp")

    if "psi_curve" in row.colnames and "psi_curve" in row.colnames:
        psi = row["psi_curve"]
        phi = row["phi_curve"]
        lc = np.full(psi.shape, 0.0)

        valid = (phi != 0) & np.isfinite(psi) & np.isfinite(phi)
        if "obs_valid" in row.colnames:
            valid = valid & row["obs_valid"]

        lc[valid] = psi[valid] / phi[valid]
        plot_time_series(lc, times, indices=valid, ax=ax_lc, figure=fig_top, title="Light curve")
    else:
        ax_lc.text(0.5, 0.5, "No Lightcurve")

    # If there are all_stamps, plot those.
    if "all_stamps" in row.colnames:
        if times is not None:
            labels = [f"T={times[i]}" for i in range(len(times))]
        else:
            labels = None
        plot_multiple_images(row["all_stamps"], figure=fig_bot, columns=5, labels=labels)
    else:
        ax = fig_bot.add_axes([0, 0, 1, 1])
        ax.text(0.5, 0.5, "No Individual Stamps")
