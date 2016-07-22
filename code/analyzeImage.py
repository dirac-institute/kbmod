import astropy.wcs
import astropy.units as u
import astropy.coordinates as astroCoords
import numpy as np
import matplotlib.mlab as mlab
import astropy.convolution as conv
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.spatial.distance import euclidean


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

    def createAperture(self, imShape, locationArray, sigma, scaleFactor, mask=False):

        apertureArray = np.zeros((imShape))

        if len(np.shape(sigma)) < 1:
            radius=scaleFactor*sigma
        else:
            radius = scaleFactor*sigma[0]

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

    def return_ra_dec(self, t0_pos, t0_vel, image_times, t0_mjd, wcs):

        pixel_vals = []
        for time_pt in image_times:
            pixel_vals.append(t0_pos + t0_vel*time_pt)
        pixel_vals = np.array(pixel_vals)
        coord_vals = astroCoords.SkyCoord.from_pixel(pixel_vals[:,1], pixel_vals[:,0], wcs)
        coord_list = coord_vals.to_string('hmsdms')
        output_list = []
        for coord_val, mjd in zip(coord_list, t0_mjd):
            coord_ra, coord_dec = coord_val.split(' ')
            ra_h = coord_ra.split('h')[0]
            ra_m = coord_ra.split('m')[0].split('h')[1]
            ra_s = str('%.4f') % float(coord_ra.split('s')[0].split('m')[1])
            dec_d = coord_dec.split('d')[0]
            dec_m = coord_dec.split('m')[0].split('d')[1]
            dec_s = str('%.4f') % float(coord_dec.split('s')[0].split('m')[1])
            output_list.append(str('%.4f' + '  ' + '%s:%s:%s' + '  ' + '%s:%s:%s' +
                                   '  ' + '0.1  568') % (mjd+2400000.5, ra_h, ra_m,
                                                         ra_s, dec_d, dec_m, dec_s))
        return np.array(output_list, dtype=str)

    def createPostageStamp(self, imageArray, objectStartArr, velArr,
                           timeArr, gaussSigma, scaleFactor, wcs_list=None,
                           starLocs = None):

        singleImagesArray = []
        stampWidth = np.array(np.array(gaussSigma)*scaleFactor, dtype=int)
        stampImage = np.zeros(((2*stampWidth)+1))
        if len(np.shape(imageArray)) < 3:
            imageArray = [imageArray]

        measureCoords = ci().calcCenters(np.array(objectStartArr), np.array(velArr), timeArr)
#        measureCoords = self.calcPixelLocationsFromEcliptic([objectStartArr], velArr[0],
#                                                            velArr[1], timeArr, wcs_list)
        if len(np.shape(measureCoords)) < 2:
            measureCoords = [measureCoords]
        for centerCoords in measureCoords:
            if (centerCoords[0] + stampWidth[0] + 1) > np.shape(imageArray[0])[0]:
                raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            elif (centerCoords[0] - stampWidth[0]) < 0:
                raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            elif (centerCoords[1] + stampWidth[1] + 1) > np.shape(imageArray[0])[1]:
                raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')
            elif (centerCoords[1] - stampWidth[1]) < 0:
                raise ValueError('The boundaries of your postage stamp for one of the images go off the edge')

        i=0
        for image in imageArray:
            xmin = np.rint(measureCoords[i,0]-stampWidth[0])
            xmax = xmin + stampWidth[0]*2 + 1
            ymin = np.rint(measureCoords[i,1]-stampWidth[1])
            ymax = ymin + stampWidth[1]*2 + 1
            if starLocs is None:
                stampImage += image[xmin:xmax, ymin:ymax]
                singleImagesArray.append(image[xmin:xmax, ymin:ymax])
            else:
                starInField = False
                for star in starLocs:
                    distX = star[0] - measureCoords[i,0]
                    distY = star[1] - measureCoords[i,1]
                    if np.sqrt((distX**2)+(distY**2)) <= scaleFactor*gaussSigma[0]:
                        print star
                        starInField = True
                if starInField == False:
                    stampImage += image[xmin:xmax, ymin:ymax]
                    singleImagesArray.append(image[xmin:xmax, ymin:ymax])
                else:
                    print 'Star in Field for Image ', str(i+1)

            i+=1
        return stampImage, singleImagesArray

    def plotTrajectory(self, results_arr, image_times, raw_im,
                       im_plot_args=None, traj_plot_args=None):

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
        plt.plot(coords[:, 1], coords[:, 0], **traj_plot_args)
        plt.xlim((t0_pos[1]-25, t0_pos[1]+75))
        plt.ylim((t0_pos[0]-25, t0_pos[0]+75))
        return ax

    def plotLightCurves(self, im_array, results_arr, image_times):

        t0_pos = [results_arr['t0_x'], results_arr['t0_y']]
        pixel_vel = [results_arr['v_x'], results_arr['v_y']]
        coords = [np.array(t0_pos) +
                  np.array([pixel_vel[0]*it, pixel_vel[1]*it])
                  for it in image_times]
        coords = np.array(coords)

        ax = plt.gca()
#        plt.plot(image_times, [im_array[x][coords[x, 0], coords[x, 1]]
#                               for x in range(0, len(image_times))])
        plt.plot(image_times, [np.sum(im_array[x][coords[x, 0]-2:coords[x, 0]+3,
                                           coords[x, 1]-2:coords[x, 1]+3])
                               for x in range(0, len(image_times))])
        return ax

    def clusterResults(self, results, dbscan_args):

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

        from sklearn.cluster import DBSCAN

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
        db_cluster.fit(np.array([t0y_arr, t0x_arr, np.array(vel_total_arr), slope_arr]).T)

        return db_cluster
