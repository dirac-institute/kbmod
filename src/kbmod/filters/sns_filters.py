import numpy as np

"""KBMOD Implementation of Wes Fraser's SNS filtering.
Done as part of @SamSandwich07's summer project"""


def peak_offset_filter(res, peak_offset_max=6):
    """Remove rows in the results objects whose peak offset eclipses peak_offset_max pixels

    Parameters
    ----------
    res : `Results`
        The search results containing trajectories.
        This object is modified by filtering out rows.
    peak_offset_max : `int`
        The max allowable distance between stamp peak and centre of stamp.
        The default value is 6.

    Raises
    ------
    RuntimeError :
      Input results do not contain "coadd_mean" column.
    """
    if "coadd_mean" not in res.colnames:
        raise RuntimeError("coadd_mean column not present in results")

    stamps = res["coadd_mean"]
    (N, a, b) = stamps.shape
    (gx, gy) = np.meshgrid(np.arange(b), np.arange(a))
    gx = gx.reshape(a * b)
    gy = gy.reshape(a * b)
    rs_stamps = stamps.reshape(N, a * b)
    args = np.argmax(rs_stamps, axis=1)
    X = gx[args]
    Y = gy[args]
    radial_d = ((X - b / 2) ** 2 + (Y - a / 2) ** 2) ** 0.5
    w = np.where(radial_d < peak_offset_max)
    res.table = res.table[w]
