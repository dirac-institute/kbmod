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


def predictive_line_cluster(res, dmjds, dist_lim=4.0, min_samp=2, init_select_proc_distance=60):
    """???

    Parameters
    ----------
    res : `Results`
        The search results containing trajectories.
        This object is modified by filtering out rows.


    Raises
    ------
    RuntimeError :
      Input results do not contain "coadd_mean" column.
    """
    pass
    # proc_filt_detections = np.copy(filt_detections)

    # proc_inds = np.arange(len(proc_filt_detections))
    # clust_detections, clust_inds = [], []

    # #for i in range(0,10):
    # while len(proc_filt_detections)>0:

    #     arg_max = np.argmax(proc_filt_detections[:,5]) # 5 - max on SNR, 4 is flux
    #     #pyl.imshow(normer(stamps[arg_max]))
    #     #pyl.show()
    #     #exit()
    #     x_o, y_o, rx_o, ry_o, f_o, snr_o = proc_filt_detections[arg_max, :6]

    #     #this secondary where command is necessary because of memory overflows in large detection lists
    #     w = np.where( (proc_filt_detections[:,0] > proc_filt_detections[arg_max,0]-55) & (proc_filt_detections[:,0] < proc_filt_detections[arg_max,0] +init_select_proc_distance)
    #                  & (proc_filt_detections[:,1] > proc_filt_detections[arg_max,1]-55) & (proc_filt_detections[:,1] < proc_filt_detections[arg_max,1] +init_select_proc_distance))

    #     W = np.where( ((proc_filt_detections[w[0],0]-proc_filt_detections[arg_max,0])**2 + (proc_filt_detections[w[0],1]-proc_filt_detections[arg_max,1])**2) < init_select_proc_distance**2)
    #     w = w[0][W[0]]

    #     fd_subset = proc_filt_detections[w]

    #     drx = fd_subset[:,2] - rx_o
    #     dry = fd_subset[:,3] - ry_o
    #     dt = dmjds # just for clarity

    #     x_n, y_n = x_o - drx*dt[-1], y_o - dry*dt[-1] # predicted centroid  position of secondary detection shifted at the differential wrong rate.

    #     dx, dy = (x_n - x_o), (y_n - y_o) # predicted centroid shifted such that best detection is now at origin
    #     dxp = dx*fd_subset[:,1]
    #     dyp = dy*fd_subset[:,0]
    #     xm = x_n*y_o
    #     ym = y_n*x_o
    #     dx2 = dx**2
    #     dy2 = dy**2
    #     top = np.abs(dyp - dxp + xm - ym)
    #     bottom = np.sqrt( dx2 + dy2 )
    #     dist = top/bottom
    #     #dist = np.abs( (y_n-y_o)*fd_subset[:, 0] - (x_n-x_o)*fd_subset[:,1] + x_n*y_o - y_n*x_o    ) / np.sqrt( (x_n-x_o)**2 + (y_n-y_o)**2)

    #     vert_distance = np.abs(y_n - fd_subset[:,1])
    #     hor_distance = np.abs(x_n - fd_subset[:,0])

    #     clust = np.where( (dist<dist_lim) | (np.isnan(dist)) | ((dist<dist_lim) & (drx==0) & (dry==0)))
    #     not_clust = np.where(~( (dist<dist_lim) | (np.isnan(dist)) | ((dist<dist_lim) & (drx==0) & (dry==0))) )

    #     #clust = np.where( ( (hor_distance < dist_lim_x)&(vert_distance<dist_lim_y) ) | (np.isnan(dist)) | ((dist<dist_lim)&(dx==0)&(dy==0) ))
    #     #not_clust = np.where( ~( ( (hor_distance < dist_lim_x)&(vert_distance<dist_lim_y) ) | (np.isnan(dist)) | ((dist<dist_lim)&(dx==0)&(dy==0) ) ))

    #     if len(clust[0])>=min_samp:
    #         clust_detections.append(proc_filt_detections[arg_max])
    #         clust_inds.append(proc_inds[arg_max])

    #     mask = np.ones(len(proc_filt_detections), dtype='bool')
    #     mask[w[clust]] = False
    #     proc_filt_detections = proc_filt_detections[mask]
    #     proc_inds = proc_inds[mask]

    # clust_detections = np.array(clust_detections)
    # clust_stamps = stamps[np.array(clust_inds)]

    # return clust_detections, clust_stamps
