import numpy as np
from kbmod.results import Results

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


def predictive_line_cluster(res, dmjds, dist_lim=4.0, min_samp=2, init_select_proc_distance=60):
    """Determine clustering based on the linearity of the best centroid.

    Parameters
    ----------
    res : `Results`
        The search results containing trajectories.
        This object is modified by filtering out rows.
    dmjds : `list`
        List of dates in dmjd format.
    dist_lim : `double`
    min_samp : `int`
    init_select_proc_distance : `int`
    """

    snr = res["psi_curve"] / np.sqrt(res["phi_curve"])
    max_snr = np.max(snr, axis=1)

    x_col = res["x"].data
    y_col = res["y"].data
    vx_col = res["vx"].data
    vy_col = res["vy"].data
    res_copy = res

    proc_inds = np.arange(len(x_col))
    clust_inds = []

    while len(max_snr) > 0:
        arg_max = np.argmax(max_snr)
        x_o = x_col[arg_max]
        y_o = y_col[arg_max]
        rx_o = vx_col[arg_max]
        ry_o = vy_col[arg_max]

        # this secondary where command is necessary because of memory overflows in large detection lists
        w = np.where(
            (x_col > x_o - init_select_proc_distance)
            & (x_col < x_o + init_select_proc_distance)
            & (y_col > y_o - init_select_proc_distance)
            & (y_col < y_o + init_select_proc_distance)
        )

        W = np.where(
            ((x_col[w[0]] - x_col[arg_max]) ** 2 + (y_col[w[0]] - y_col[arg_max]) ** 2)
            < init_select_proc_distance**2
        )

        w = w[0][W[0]]

        x_col_subset = x_col[w]
        y_col_subset = y_col[w]
        vx_col_subset = vx_col[w]
        vy_col_subset = vy_col[w]

        drx = vx_col_subset - rx_o
        dry = vy_col_subset - ry_o
        dt = dmjds  # just for clarity

        x_n, y_n = (
            x_o - drx * dt[-1],
            y_o - dry * dt[-1],
        )  # predicted centroid  position of secondary detection shifted at the differential wrong rate.

        dx, dy = (x_n - x_o), (
            y_n - y_o
        )  # predicted centroid shifted such that best detection is now at origin
        dxp = dx * y_col_subset
        dyp = dy * x_col_subset
        xm = x_n * y_o
        ym = y_n * x_o
        dx2 = dx**2
        dy2 = dy**2
        top = np.abs(dyp - dxp + xm - ym)
        bottom = np.sqrt(dx2 + dy2)
        dist = top / bottom

        clust = np.where((dist < dist_lim) | (np.isnan(dist)) | ((dist < dist_lim) & (drx == 0) & (dry == 0)))

        if len(clust[0]) >= min_samp:
            clust_inds.append(proc_inds[arg_max])

        mask = np.ones(len(res_copy), dtype="bool")
        mask[w[clust]] = False
        x_col = x_col[mask]
        y_col = y_col[mask]
        vx_col = vx_col[mask]
        vy_col = vy_col[mask]
        res_copy = res_copy[mask]
        proc_inds = proc_inds[mask]
        max_snr = max_snr[mask]

    clust_inds.sort()
    res.table = res.table[clust_inds]
