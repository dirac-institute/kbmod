from sklearn.cluster import DBSCAN
import numpy as np
import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom

def maximum_expected_detections(im_count, min_obs, 
mask_amount, actual_expected):
    max_expected_fraction = 0
    for masked_count in range(im_count-min_obs+1):
        max_expected_fraction += ncr(im_count, masked_count) * \
        (mask_amount)**masked_count * \
        (1-mask_amount)**(im_count-masked_count)
    return max_expected_fraction*actual_expected

def add_trajectory(image_list, tr, psf, times):
    init_time = times[0]
    for i,t_on in zip(image_list, times):
        t = t_on - init_time
        i.add_object( tr.x+tr.x_v*t, tr.y+tr.y_v*t, tr.flux, psf )

def compare_trajectory(a, b, v_thresh, pix_thresh):
    # compare flux too?
    if (b.obs_count == 0 and 
    abs(a.x-b.x)<=pix_thresh and 
    abs(a.y-b.y)<=pix_thresh and 
    abs(a.x_v/b.x_v-1)<v_thresh and 
    abs(a.y_v/b.y_v-1)<v_thresh):
        b.obs_count += 1
        return True
    else:
        return False

def compare_trajectory_once(a, b, v_thresh, pix_thresh):
    # compare flux too?
    if ( 
    abs(a.x-b.x)<=pix_thresh and 
    abs(a.y-b.y)<=pix_thresh and 
    abs(a.x_v/b.x_v-1)<v_thresh and 
    abs(a.y_v/b.y_v-1)<v_thresh):
        return True
    else:
        return False

def match_trajectories(results_list, test_list, v_thresh, pix_thresh):
    matches = []
    unmatched = []
    for r in results_list:
        if any(compare_trajectory(r, test, v_thresh, pix_thresh)
        for test in test_list):
            matches.append(r)
    for t in test_list:
        if (t.obs_count == 0):
            unmatched.append(t)
        t.obs_count = 0
    return matches, unmatched

# adapted from analyzeImage.py
def cluster_trajectories( results, dbscan_args=None):

        """
        Use scikit-learn algorithm of density-based spatial clustering of
        applications with noise (DBSCAN)
        (http://scikit-learn.org/stable/modules/generated/
            sklearn.cluster.DBSCAN.html)
        to cluster the results of the likelihood image search using starting
        location, total velocity and slope of trajectory.

        Parameters
        ----------

        results: numpy recarray, required
        The results output from findObjects in searchImage.

        dbscan_args: dict, optional
        Additional arguments for the DBSCAN instance. See options in link
        above.

        Returns
        -------

        db_cluster: DBSCAN instance
        DBSCAN instance with clustering completed. To get cluster labels use
        db_cluster.labels_

        top_vals: list of integers
        The indices in the results array where the most likely object in each
        cluster is located.
        """

        default_dbscan_args = dict(eps=0.1, min_samples=1)

        if dbscan_args is not None:
            default_dbscan_args.update(dbscan_args)
        dbscan_args = default_dbscan_args

        slope_arr = []
        intercept_arr = []
        t0x_arr = []
        t0y_arr = []
        vel_total_arr = []
        vx_arr = []
        vel_x_arr = []
        vel_y_arr = []
        for r in results:
            t0x_arr.append(r.x)
            t0y_arr.append(r.y)
            vel_x_arr.append(r.x_v)
            vel_y_arr.append(r.y_v)

        db_cluster = DBSCAN(**dbscan_args)

        scaled_t0x = t0x_arr - np.min(t0x_arr)
        if np.max(scaled_t0x) > 0.:
            scaled_t0x = scaled_t0x/np.max(scaled_t0x)
        scaled_t0y = t0y_arr - np.min(t0y_arr)
        if np.max(scaled_t0y) > 0.:
            scaled_t0y = scaled_t0y/np.max(scaled_t0y)
        scaled_vx = vel_x_arr - np.min(vel_x_arr)
        if np.max(scaled_vx) > 0.:
            scaled_vx /= np.max(scaled_vx)
        scaled_vy = vel_y_arr - np.min(vel_y_arr)
        if np.max(scaled_vy) > 0.:
            scaled_vy /= np.max(scaled_vy)

        db_cluster.fit(np.array([scaled_t0x, scaled_t0y,
                                 scaled_vx, scaled_vy
                                ], dtype=np.float).T)

        top_vals = []
        for cluster_num in np.unique(db_cluster.labels_):
            cluster_vals = np.where(db_cluster.labels_ == cluster_num)[0]
            top_vals.append(cluster_vals[0])

        return db_cluster, top_vals

def calc_centers(t, timeArr):
    #ix, iy = zip(*[(t.x,t.y) for t in trajectories] )
    #xv, yv = zip(*[(t.x_v, t.y_v) for t in trajectories] )
    #startLocArr = np.array( [np.array(ix), np.array(iy)] )
    #velArr = np.array( [np.array(xv), np.array(yv)] )
    startLocArr = np.array( [t.x, t.y] )
    #print(startLocArr)
    velArr = np.array( [t.x_v, t.y_v] )
    centerArr = []
    for time in timeArr:
        centerArr.append(startLocArr + (velArr*time))
    return np.array(centerArr)

def create_postage_stamp(imageArray, traj,
                       timeArr, stamp_width):

    """
    Create postage stamp image coadds of potential objects traveling along
    a trajectory.
    Parameters
    ----------
    imageArray: numpy array, required
    The masked input images.
    objectStartArr: numpy array, required
    An array with the starting location of the object in pixels.
    velArr: numpy array, required
    The x,y velocity in pixels/hr. of the object trajectory.
    timeArr: numpy array, required
    The time in hours of each image starting from 0 at the first image.
    stamp_width: numpy array or list, [2], required
    The row, column dimensions of the desired output image.
    Returns
    -------
    stampImage: numpy array
    The coadded postage stamp.
    singleImagesArray: numpy array
    The postage stamps that were added together to create the coadd.
    """

    singleImagesArray = []
    stampWidth = np.array(stamp_width, dtype=int)
    #print stampWidth
    stampImage = np.zeros(stampWidth)

    #if len(np.shape(imageArray)) < 3:
    #    imageArray = [imageArray]

    measureCoords = calc_centers(traj, timeArr)
    #print(measureCoords)
    if len(np.shape(measureCoords)) < 2:
        measureCoords = [measureCoords]
    off_edge = []
    for centerCoords in measureCoords:
        #print((centerCoords[0] + stampWidth[0]/2 + 1) )
        #print( np.shape(imageArray[0])[1])
        if (centerCoords[0] + stampWidth[0]/2 + 1) > np.shape(imageArray[0])[1]:
            #raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            off_edge.append(True)
        elif (centerCoords[0] - stampWidth[0]/2) < 0:
            #raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            off_edge.append(True)
        elif (centerCoords[1] + stampWidth[1]/2 + 1) > np.shape(imageArray[0])[0]:
            #raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            off_edge.append(True)
        elif (centerCoords[1] - stampWidth[1]/2) < 0:
            #raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            off_edge.append(True)
        else:
            off_edge.append(False)

    i=0
    for image in imageArray:
        if off_edge[i] is False:
            xmin = int(np.rint(measureCoords[i,1]-stampWidth[0]/2))
            xmax = int(xmin + stampWidth[0])
            ymin = int(np.rint(measureCoords[i,0]-stampWidth[1]/2))
            ymax = int(ymin + stampWidth[1])
            #print xmin, xmax, ymin, ymax
            single_stamp = image[xmin:xmax, ymin:ymax]
            single_stamp[np.isnan(single_stamp)] = 0.
            single_stamp[np.isinf(single_stamp)] = 0.
            single_stamp[single_stamp < -9000.] = 0.
            #if len(np.where(single_stamp == 0.)[0]) > 221.:
            #    singleImagesArray.append(single_stamp)
            #    continue
            stampImage += single_stamp
            singleImagesArray.append(single_stamp)
        else:
            single_stamp = np.zeros((stampWidth))
            singleImagesArray.append(single_stamp)

        i+=1
    return stampImage, singleImagesArray


