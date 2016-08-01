import astropy.wcs
import astropy.units as u
import astropy.coordinates as astroCoords
import numpy as np
import matplotlib.mlab as mlab
import astropy.convolution as conv
import matplotlib.pyplot as plt
from createImage import createImage as ci
from astropy.io import fits
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN


class analyzeImage(object):

    def calcArrayLimits(self, imShape, centerX, centerY, scaleFactor, sigmaArr):

        xmin = int(centerX-(scaleFactor*sigmaArr[0]))
        xmax = int(1+centerX+(scaleFactor*sigmaArr[0]))
        ymin = int(centerY-(scaleFactor*sigmaArr[1]))
        ymax = int(1+centerY+(scaleFactor*sigmaArr[1]))
        if ((xmin < 0) | (ymin < 0) | (xmax >= imShape[0]) | (ymax >= imShape[1])):
            maxXOff = xmax-imShape[0]+1
            maxYOff = ymax-imShape[1]+1
            minXOff = xmin*(-1.)
            minYOff = ymin*(-1.)
            offset = np.max([maxXOff, maxYOff, minXOff, minYOff])
            xmin += offset
            xmax -= offset
            ymin += offset
            ymax -= offset
        else:
            offset = None

        return xmin, xmax, ymin, ymax, offset

    def createAperture(self, imShape, locationArray, radius, mask=False):

        """
        Create a circular aperture for an image. Aperture area will be 1's 
        and all area outside will be 0's. Just multiply aperture by image to get
        everything outside aperture masked out.

        Parameters
        ----------

        imShape: list, [2], required
        The row, column dimensions of the image.

        locationArray: list, [Nx2], required
        The locations in the image where apertures should be centered.

        radius: float, required
        The radius of the circular aperture in pixels.

        mask: boolean, optional, default=False
        If true, then aperture area inside is set to 0's and outside to 1's
        making this a mask of the area instead.

        Returns
        -------

        apertureArray: numpy array
        Array of the same size as imShape but with 1's inside the aperture and
        0's outside unless mask is set to True then it is the opposite.
        """
        apertureArray = np.zeros((imShape))

        if len(np.shape(locationArray)) < 2:
            locationArray = [locationArray]

        for center in locationArray:
            centerX = center[0]
            centerY = center[1]
            for ix in range(0, int(imShape[0])):
                for iy in range(0, int(imShape[1])):
                    distX = centerX - ix
                    distY = centerY - iy
                    if np.sqrt((distX**2)+(distY**2)) <= radius:
                        apertureArray[ix, iy] = 1.

        if mask==True:
            apertureArray -= 1
            apertureArray = np.abs(apertureArray)

        return apertureArray

    def trackSingleObject(self, imageArray, gaussSigma):

        objectCoords = []
        for image in imageArray:
            newImage = createImage().convolveGaussian(image, gaussSigma)
            maxIdx = np.argmax(newImage)
            objectCoords.append(np.unravel_index(maxIdx, np.shape(newImage)))
        return objectCoords

    def plotSingleTrajectory(self, imageArray, gaussSigma):

        objCoords = self.trackSingleObject(imageArray, gaussSigma)
        fig = plt.figure(figsize=(12,12))
        plt.plot(np.array(objCoords)[:,0], np.array(objCoords)[:,1], '-ko')
        plt.xlim((0, np.shape(imageArray[0])[0]))
        plt.ylim((0, np.shape(imageArray[0])[1]))

        return fig

    def calcSNR(self, image, centerArr, gaussSigma, background, imSize, apertureScale=1.6):

        if isinstance(background, np.ndarray):
            backgroundArray = background
        else:
            backgroundArray = np.ones((imSize))*background

        apertureScale = 1.6 #See derivation here: http://wise2.ipac.caltech.edu/staff/fmasci/GaussApRadius.pdf
        aperture = self.createAperture(imSize, centerArr, apertureScale, gaussSigma[0])
        sourceCounts = np.sum(image*aperture)
        if sourceCounts < 0:
            sourceCounts = 0.0
        noiseCounts = np.sum(backgroundArray*aperture)

        snr = sourceCounts/np.sqrt(sourceCounts+noiseCounts)
        return snr

    def calcTheorySNR(self, sourceFlux, centerArr, gaussSigma, background, imSize, apertureScale=1.6):

        if isinstance(background, np.ndarray):
            backgroundArray = background
        else:
            backgroundArray = np.ones((imSize))*background

        sourceTemplate = createImage().createGaussianSource(centerArr, gaussSigma, imSize, sourceFlux)

        aperture = self.createAperture(imSize, centerArr, apertureScale, gaussSigma[0])
        sourceCounts = np.sum(sourceTemplate*aperture)
        noiseCounts = np.sum(backgroundArray*aperture)

        snr = sourceCounts/np.sqrt(sourceCounts+noiseCounts)
        return snr

    def addMask(self, imageArray, locations, gaussSigma):

        maskedArray = np.zeros((np.shape(imageArray)))
        scaleFactor = 4.
        i = 0
        for image in imageArray:
            maskedArray[i] = image * self.createAperture(np.shape(image), locations, scaleFactor, gaussSigma, mask=True)
            i+=1

        return maskedArray

    def return_ra_dec(self, t0_pos, t0_vel, image_times, t0_mjd, wcs,
                      position_error, telescope_code):

        """
        Return a set of ra and dec coordinates for a trajectory.
        Used as input into Bernstein and Khushalani (2000) orbit fitting
        code found here: http://www.physics.upenn.edu/~garyb/#software


        Parameters
        ----------

        t0_pos: numpy array, [2], required
        The starting x,y pixel location

        t0_vel: numpy array, [2], required
        The x,y velocity of the object in pixels/hr.

        image_times: numpy array, required
        An array containing the image times in hours with the first image at
        time 0.

        t0_mjd: numpy array, required
        The MJD times of each image.

        wcs: astropy.wcs.wcs instance, required
        The astropy.wcs instance of the first image.

        position_error: numpy array, required
        The position error in the observations in arcsec.

        telescope_code: int, required
        The telescope code for Bernstein and Khushalani (2000)
        orbit fitting software. (Subaru is 568).

        Returns
        -------

        ra_dec_coords: numpy array
        Array of strings with the (mjd, ra, dec, 
        position_error, telescope_code) for each image in the trajectory.
        """

        pixel_vals = []
        for time_pt in image_times:
            pixel_vals.append(t0_pos + t0_vel*time_pt)
        pixel_vals = np.array(pixel_vals)
        coord_vals = astroCoords.SkyCoord.from_pixel(pixel_vals[:,0], pixel_vals[:,1], wcs)
        coord_list = coord_vals.to_string('hmsdms')
        output_list = []
        for coord_val, mjd, err_val in zip(coord_list, t0_mjd, position_error):
            coord_ra, coord_dec = coord_val.split(' ')
            ra_h = coord_ra.split('h')[0]
            ra_m = coord_ra.split('m')[0].split('h')[1]
            ra_s = str('%.4f') % float(coord_ra.split('s')[0].split('m')[1])
            dec_d = coord_dec.split('d')[0]
            dec_m = coord_dec.split('m')[0].split('d')[1]
            dec_s = str('%.4f') % float(coord_dec.split('s')[0].split('m')[1])
            output_list.append(str('%.4f' + '  ' + '%s:%s:%s' + '  ' + '%s:%s:%s' +
                                   '  ' + '%.2f  %i') % (mjd+2400000.5, ra_h, ra_m,
                                                         ra_s, dec_d, dec_m, dec_s,
                                                         err_val, telescope_code))
        ra_dec_coords = np.array(output_list, dtype=str)

        return ra_dec_coords

    def createPostageStamp(self, imageArray, objectStartArr, velArr,
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
        stampWidth = np.array(stamp_width)
        stampImage = np.zeros(stampWidth)
        if len(np.shape(imageArray)) < 3:
            imageArray = [imageArray]

        measureCoords = ci().calcCenters(np.array(objectStartArr), np.array(velArr), timeArr)

        if len(np.shape(measureCoords)) < 2:
            measureCoords = [measureCoords]
        for centerCoords in measureCoords:
            if (centerCoords[0] + stampWidth[0]/2 + 1) > np.shape(imageArray[0])[1]:
                raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            elif (centerCoords[0] - stampWidth[0]/2) < 0:
                raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            elif (centerCoords[1] + stampWidth[1]/2 + 1) > np.shape(imageArray[0])[0]:
                raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            elif (centerCoords[1] - stampWidth[1]/2) < 0:
                raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')

        i=0
        for image in imageArray:
            xmin = np.rint(measureCoords[i,1]-stampWidth[0]/2)
            xmax = xmin + stampWidth[0]
            ymin = np.rint(measureCoords[i,0]-stampWidth[1]/2)
            ymax = ymin + stampWidth[1]
            stampImage += image[xmin:xmax, ymin:ymax]
            singleImagesArray.append(image[xmin:xmax, ymin:ymax])

            i+=1
        return stampImage, singleImagesArray

    def plotTrajectory(self, results_arr, image_times, raw_im,
                       im_plot_args=None, traj_plot_args=None):

        """
        Plot an object's trajectory along a section of one of the 
        original masked images.

        Parameters
        ----------

        results_arr: numpy recarray, required
        The results output from findObjects in searchImage.

        image_times: numpy array, required
        An array containing the image times in hours with the first image at
        time 0.

        raw_im: numpy array, required
        One of the masked original images. See loadMaskedImages
        in searchImage.py.

        im_plot_args: dict, optional
        Plotting arguments for the masked image.

        traj_plot_args: dict, optional
        Scatter plot arguments for the trajectory on top of masked image.

        Returns
        -------

        ax: matplotlib axes instance
        Returns instance after plt.imshow and plt.plot
        """

        t0_pos = [results_arr['t0_x'], results_arr['t0_y']]
        pixel_vel = [results_arr['v_x'], results_arr['v_y']]
        coords = [np.array(t0_pos) +
                  np.array([pixel_vel[0]*it, pixel_vel[1]*it])
                  for it in image_times]
        coords = np.array(coords)

        default_im_plot_args = {'cmap': 'Greys_r', 'origin': 'lower'}
        default_traj_plot_args = {'marker': 'o', 'c': 'r'}

        if im_plot_args is not None:
            default_im_plot_args.update(im_plot_args)
        im_plot_args = default_im_plot_args

        if traj_plot_args is not None:
            default_traj_plot_args.update(traj_plot_args)
        traj_plot_args = default_traj_plot_args

        ax = plt.gca()
        plt.imshow(raw_im, **im_plot_args)
        plt.plot(coords[:, 0], coords[:, 1], **traj_plot_args)
        plt.xlim((t0_pos[0]-25, t0_pos[0]+75))
        plt.ylim((t0_pos[1]-25, t0_pos[1]+75))
        return ax

    def plotLightCurves(self, im_array, results_arr, image_times):

        """
        Plots light curve of trajectory using array of masked images.

        Parameters
        ----------
        im_array: numpy array, required
        The masked original images. See loadMaskedImages
        in searchImage.py.

        results_arr: numpy recarray, required
        The results output from findObjects in searchImage.

        image_times: numpy array, required
        An array containing the image times in hours with the first image at
        time 0.

        Returns
        -------
        ax: matplotlib axes instance
        The axes instance where the plt.plot of the lightcurve was drawn.
        """


        t0_pos = [results_arr['t0_x'], results_arr['t0_y']]
        pixel_vel = [results_arr['v_x'], results_arr['v_y']]
        coords = [np.array(t0_pos) +
                  np.array([pixel_vel[0]*it, pixel_vel[1]*it])
                  for it in image_times]
        coords = np.array(coords)
        aperture = self.createAperture([11,11], [5., 5.],
                                       1., mask=False)

        ax = plt.gca()
        plt.plot(image_times, [np.sum(im_array[x][coords[x,1]-5:coords[x,1]+6,
                                                  coords[x,0]-5:coords[x,0]+6]*aperture)
                               for x in range(0, len(image_times))])
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Flux')
        return ax

    def clusterResults(self, results, dbscan_args=None):

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
        for target_num in range(len(results)):

            slope = results['v_x'][target_num]/results['v_y'][target_num]
            vel_total = np.sqrt(results['v_x'][target_num]**2 +
                                results['v_y'][target_num]**2)
            intercept = results['t0_x'][target_num] - results['t0_y'][target_num]*slope
            slope_arr.append(slope)
            intercept_arr.append(intercept)
            vel_total_arr.append(vel_total)
            t0x = results['t0_x'][target_num]
            t0x_arr.append(t0x)
            t0y = results['t0_y'][target_num]
            t0y_arr.append(t0y)
            vx = np.arctan(results['v_x'][target_num]/results['v_y'][target_num])
            vx_arr.append(vx)

        db_cluster = DBSCAN(**dbscan_args)

        scaled_t0x = t0x_arr - np.min(t0x_arr)
        scaled_t0x = scaled_t0x/np.max(scaled_t0x)
        scaled_t0y = t0y_arr - np.min(t0y_arr)
        scaled_t0y = scaled_t0y/np.max(scaled_t0y)
        scaled_vel = np.array(vel_total_arr) - np.min(vel_total_arr)
        scaled_vel = scaled_vel/np.max(scaled_vel)
        scaled_slope = np.array(slope_arr) - np.min(slope_arr)
        scaled_slope = scaled_slope/np.max(scaled_slope)

        db_cluster.fit(np.array([scaled_t0x, scaled_t0y,
                                 scaled_vel, scaled_slope]).T)

        return db_cluster

    def sortCluster(self, results, db, masked_array, image_times):

        """
        Takes the most likely results in each cluster and creates postage
        stamps for each object that are then used to just whether the result
        is a real object or not. This is determined by taking an aperture
        in the center of the postage stamps and comparing the maximum flux
        inside to that outside. Bright, unmasked objects that are stationary
        will produce streaks and this ratio will be close to 1. Moving objects
        will have brighter centers and higher values of this ratio. The results
        are then sorted using this ratio.

        Parameters
        ----------

        results: numpy recarray, required
        The results output from findObjects in searchImage.

        db: DBSCAN instance
        DBSCAN instance with clustering completed. Could be output from
        clusterResults above.

        masked_array: numpy array, required
        An array with the input images multiplied by the mask. See
        loadMaskedImages in searchImage.py

        image_times: numpy array, required
        An array containing the image times in hours with the first image at
        time 0.

        Returns
        -------

        best_targets: numpy array
        The indices in the results array of a sorted list of the best targets
        using the criteria described above. Will have the length of the number
        of clusters labeled in db.
        """

        top_val = []
        for cluster_num in np.unique(db.labels_):
            cluster_vals = np.where(db.labels_ == cluster_num)[0]
            top_val.append(cluster_vals[0])

        full_set = []
        set_vals = []
        for val in np.unique(top_val):
            try:
                ps = self.createPostageStamp(masked_array,
                                             list(results[['t0_x',
                                                           't0_y']][val]),
                                             list(results[['v_x',
                                                           'v_y']][val]),
                                             image_times, [25.0, 25.0])
                full_set.append(ps[0])
                set_vals.append(val)
            except ValueError:
                continue
        print 'Done with Postage Stamps'

        set_vals=np.array(set_vals)

        aperture = self.createAperture(np.shape(full_set[0]), [12., 12.],
                                       2., mask=False)
        aperture_mask = self.createAperture(np.shape(full_set[0]), 
                                            [12., 12.], 2., mask=True)

        #maxes = [np.max(full_set[exp_num]*aperture)/
        maxes = [np.mean(full_set[exp_num][np.where(aperture>0.)])/
                 np.max(full_set[exp_num]*aperture_mask) 
                 for exp_num in range(len(full_set))]

        for max_val in range(len(maxes)):
            if np.isinf(maxes[max_val]):
                maxes[max_val] = -1.0
        best_targets = set_vals[np.array(np.argsort(maxes)[::-1])]

        return best_targets
