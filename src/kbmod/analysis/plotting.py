import warnings

import numpy as np

import astropy.units as u
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import simple_norm, ZScaleInterval, AsinhStretch, ImageNormalize

import matplotlib.pyplot as plt


__all__ = [
    "iter_over_obj",
    "transform_rect",
    "plot_field",
    "plot_bbox",
    "plot_bboxes",
    "plot_footprint",
    "plot_footprints",
    "plot_all_objs",
    "plot_focalplane",
    "plot_cutouts",
    "plot_img",
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


def plot_img(img, ax=None, figure=None, norm=True, title=None):
    """Plots an image on an axis and figure.

    If no axis is given creates a new figure. Draws a crosshair at the
    center of the image. Must provide both axis and figure; figure is
    required so a colorbar could be attached to it.

    Parameters
    ----------
    img : `np.array`
        Image array
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

    Returns
    -------
    figure : `matplotlib.pyplot.Figure`
        Modified Figure.
    ax : `matplotlib.pyplot.Axes`
        Modified Axes.
    """
    if ax is None and figure is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    elif ax is not None or figure is not None:
        raise ValueError("Provide both figure and axis, or provide none.")
    else:
        pass

    if norm:
        norm = ImageNormalize(img, interval=ZScaleInterval(), stretch=AsinhStretch())
        im = ax.imshow(img, norm=norm)
    else:
        im = ax.imshow(img)

    ax.axvline(img.shape[0] / 2, c="red", lw=0.5)
    ax.axhline(img.shape[1] / 2, c="red", lw=0.5)
    ax.set_title(title)
    fig.colorbar(im, label="Counts")

    return fig, ax
