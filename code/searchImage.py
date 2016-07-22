import os
import numpy as np
import lsst.afw.image as afwImage
import astropy.coordinates as astroCoords
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import convolve
from scipy.spatial.distance import euclidean
from createImage import createImage as ci


class searchImage(object):

    """
    A class of methods used to create likelihood images and search for objects.
    """

    def __init__(self):
        self.search_coords_x = None
        self.search_coords_y = None
        self.base_x = None
        self.base_y = None

    def createMask(self, image_folder, threshold):

        """Create the mask to use on every image. Since some moving objects
        are bright enough to get masked in some image we create a master mask
        using only objects that appear in a percentage of single image masks
        defined by the threshold parameter.

        Parameters
        ----------

        image_folder: str, required
        The path to where the images are stored.

        threshold: float, required
        The lowest threshold on the fraction of single epoch images a pixel
        needs to be masked in before it is added to the master mask. Should
        be between 0 and 1.

        Returns
        -------

        mask_image: numpy array
        The master mask to use as input in other methods.
        """

        maskImArray = None

        image_count = 0
        for filename in os.listdir(image_folder):
            image_file = os.path.join(image_folder, filename)
            exposure = afwImage.ExposureF(image_file)

            mask_image = exposure.getMaskedImage()
            get_mask_image = mask_image.getMask()
            mask = get_mask_image.getArray()
            if maskImArray is None:
                maskImArray = np.zeros(np.shape(mask))
            maskImArray[np.where(mask == 0)] += 1.0
            image_count += 1

        maskImArray = maskImArray/np.float(image_count)
        mask_image = (maskImArray > threshold)*1.

        return mask_image

    def calcPsi(self, image_folder, mask_array):

        """
        Calculate the Psi Images for each of the original images.

        Parameters
        ----------

        image_folder: str, required
        The path to where the images are stored.

        mask_array: numpy array, required
        The mask to use for the images. Could be output from createMask method.

        Returns
        -------

        psi_array: numpy array
        The psi images of the input images with psf used in convolution
        coming from the included psf from LSST DM processing.
        """

        psi_array = None

        for filename in sorted(os.listdir(image_folder)):

            print str('On Image ' + filename)

            image_file = os.path.join(image_folder, filename)
            exposure = afwImage.ExposureF(image_file)

            psf_image = exposure.getPsf()
            psf_array = psf_image.computeKernelImage().getArray()

            exp_image = exposure.getMaskedImage()

            image_array = exp_image.getImage().getArray()
            image_array = image_array * mask_array

            variance_array = exp_image.getVariance().getArray()

            psi_image = convolve((1/variance_array) * image_array, psf_array)

            if psi_array is None:
                psi_array = np.copy(psi_image)
                psi_array = [psi_array]
            else:
                psi_array = np.append(psi_array, [psi_image], axis=0)

        return psi_array

    def calcPhi(self, image_folder, mask_array):

        """
        Calculate the Phi Images for each of the original images.

        Parameters
        ----------

        image_folder: str, required
        The path to where the images are stored.

        mask_array: numpy array, required
        The mask to use for the images. Could be output from createMask method.

        Returns
        -------

        phi_array: numpy array
        The phi images of the input images with psf used in convolution
        coming from the included psf from LSST DM processing.
        """

        phi_array = None

        for filename in sorted(os.listdir(image_folder)):

            print str('On Image ' + filename)

            image_file = os.path.join(image_folder, filename)
            exposure = afwImage.ExposureF(image_file)

            psf_image = exposure.getPsf()
            psf_array = psf_image.computeKernelImage().getArray()

            exp_image = exposure.getMaskedImage()

            variance_array = exp_image.getVariance().getArray()

            phi_image = convolve((1/variance_array)*mask_array, psf_array)

            if phi_array is None:
                phi_array = np.copy(phi_image)
                phi_array = [phi_array]
            else:
                phi_array = np.append(phi_array, [phi_image], axis=0)

        return phi_array

    def loadPSF(self, image_folder):

        """
        Return an array of the psf arrays for each image.

        Parameters
        ----------

        image_folder: str, required
        The path to where the images are stored

        Returns
        -------

        psf_array: numpy array
        An array holding the psf arrays of each image.
        """

        psf_array = None

        for filename in sorted(os.listdir(image_folder)):

            image_file = os.path.join(image_folder, filename)
            exposure = afwImage.ExposureF(image_file)

            psf_exp = exposure.getPsf()
            psf_image = psf_exp.computeKernelImage().getArray()

            if psf_array is None:
                psf_array = np.copy(psf_image)
                psf_array = [psf_array]
            else:
                psf_array = np.append(psf_array, [psf_image], axis=0)

        return psf_array

    def loadImageTimes(self, image_folder):

        """
        This method loads the timestamp of each image and returns an
        array of the time each image was taken in hours where the
        first image time is set at 0. Will also return the MJD values
        of each image.

        Parameters
        ----------
        image_folder: str, required
        Folder where the images are stored.

        Returns
        -------
        image_times: numpy array
        Numpy array holding the time in hours after the first image
        was taken. First image is set at time 0.0.

        image_mjd: numpy array
        MJD values of each image.
        """

        image_mjd = []

        for filename in sorted(os.listdir(image_folder)):
            hdulist = fits.open(os.path.join(image_folder, filename))
            image_mjd.append(hdulist[0].header['MJD'])

        image_mjd = np.array(image_mjd)
        image_times = image_mjd - image_mjd[0]
        image_times*=24.

        return image_times, image_mjd

    def loadWCSList(self, image_folder):

        """
        This method loads the WCS information for every image.

        Parameters
        ----------
        image_folder: str, required
        Folder where the images are stored.

        Returns
        -------
        wcs_list: list
        List containing WCS info for each image.
        """

        wcs_list = []

        for filename in sorted(os.listdir(image_folder)):
            hdulist = fits.open(os.path.join(image_folder, filename))
            wcs_list.append(WCS(hdulist[1].header))

        return wcs_list

    def loadMaskedImages(self, image_folder, mask):

        """
        Return an array with the raw images multiplied by the mask.

        Parameters
        ----------

        image_folder: str, required
        The path to where the images are stored.

        mask_array: numpy array, required
        The mask to use for the images. Could be output from createMask method.

        Returns
        -------

        im_array: numpy array
        The input images multiplied by the mask.
        """

        im_array = None

        for filename in sorted(os.listdir(image_folder)):

            print str('On Image ' + filename)

            image_file = os.path.join(image_folder, filename)
            hdulist = fits.open(image_file)

            if im_array is None:
                im_array = np.array(hdulist[1].data*mask)
                im_array = [im_array]
            else:
                im_array = np.append(im_array, [hdulist[1].data*mask], axis=0)

        return im_array

    def calcAlphaNuEcliptic(self, psiArray, phiArray,
                            objectStartArr, vel_array, timeArr, wcs):

        if len(np.shape(psiArray)) == 2:
            psiArray = [psiArray]
            phiArray = [phiArray]

        objectStartArr = np.array(objectStartArr)
        if self.search_coords_x is None:
            pixel_coords = self.calcPixelLocationsFromEcliptic(objectStartArr,
                                                               vel_array[:,0],
                                                               vel_array[:,1],
                                                               timeArr, wcs)
            search_coords_x = np.reshape(np.array(pixel_coords[0]), (len(vel_array), len(timeArr)))
            search_coords_y = np.reshape(np.array(pixel_coords[1]), (len(vel_array), len(timeArr)))
            self.search_coords_x = search_coords_x
            self.search_coords_y = search_coords_y
            self.search_coords_dict = {}
            for vel_vals, s_x, s_y in zip(vel_array, search_coords_x, search_coords_y):
                vel_str = '%s_%s' % (vel_vals[0], vel_vals[1])
                self.search_coords_dict[vel_str] = np.array([s_x[-1] - self.base_x,
                                                             s_y[-1] - self.base_y])
        else:
            search_coords_x = self.search_coords_x - self.base_x + objectStartArr[0][0]
            search_coords_y = self.search_coords_y - self.base_y + objectStartArr[0][1]

        #Don't want array values to go negative or will wrap around.
        #Since convolution leaves border psi/phi equal to 0 anyway we can just set negative values to 0.
        search_coords_x[np.where(search_coords_x<0)] = 0.
        search_coords_y[np.where(search_coords_y<0)] = 0.

        psiTotal = np.zeros(len(objectStartArr))
        phiTotal = np.zeros(len(objectStartArr))
        pixel_locs_x = np.array(search_coords_x, dtype=np.int)
        pixel_locs_y = np.array(search_coords_y, dtype=np.int)

        for imNum in range(0, len(psiArray)):
            try:
                psiTotal += psiArray[imNum][pixel_locs_x[:,imNum], pixel_locs_y[:,imNum]]
                phiTotal += phiArray[imNum][pixel_locs_x[:,imNum], pixel_locs_y[:,imNum]]
            except KeyboardInterrupt:
                break
            except:
                continue

        phi_not0 = np.where(phiTotal != 0.)[0]
        alpha_measurements = np.zeros(len(objectStartArr))
        nu_measurements = np.zeros(len(objectStartArr))

        alpha_measurements[np.where(phiTotal == 0.)] = np.nan
        alpha_measurements[phi_not0] = psiTotal[phi_not0]/phiTotal[phi_not0]

        nu_measurements[np.where(phiTotal == 0.)] = np.nan
        nu_measurements[phi_not0] = psiTotal[phi_not0]/np.sqrt(phiTotal)[phi_not0]

        return alpha_measurements, nu_measurements

    def calcPixelLocationsFromEcliptic(self, pixel_start, vel_par_arr,
                                       vel_perp_arr, time_array, wcs):

        pixel_coords = [[],[]]

        if type(vel_par_arr) is not np.ndarray:
            vel_par_arr = [vel_par_arr]
        if type(vel_perp_arr) is not np.ndarray:
            vel_perp_arr = [vel_perp_arr]
        for start_loc, vel_par, vel_perp in zip(pixel_start, vel_par_arr, vel_perp_arr):
            print vel_par

            start_coord = astroCoords.SkyCoord.from_pixel(start_loc[1],
                                                          start_loc[0],
                                                          wcs[0])
            eclip_coord = start_coord.geocentrictrueecliptic
            eclip_l = []
            eclip_b = []
            for time_step in time_array:
                eclip_l.append(eclip_coord.lon + vel_par*time_step*24.*u.arcsec)
                eclip_b.append(eclip_coord.lat + vel_perp*time_step*24.*u.arcsec)
            eclip_vector = astroCoords.SkyCoord(eclip_l, eclip_b,
                                                frame='geocentrictrueecliptic')
            pixel_coords_set = astroCoords.SkyCoord.to_pixel(eclip_vector, wcs[0])
            pixel_coords[0].append(pixel_coords_set[1])
            pixel_coords[1].append(pixel_coords_set[0])
        pixel_coords = np.array(pixel_coords)

        return pixel_coords

    def findObjectsEcliptic(self, psiArray, phiArray,
                            velocity_grid, likelihood_cutoff, timeArr, wcs,
                            xRange=None, yRange=None, out_file=None):

        """
        The main method used to search the likelihood images for objects.

        Parameters
        ----------

        psiArray: numpy array, required
        An array containing all the psi images from calcPsi

        phiArray: numpy array, required
        An array containing all the phi images from calcPhi

        velocity_grid: numpy array [N x 2], required
        The velocity values with N pairs of velocity values, [m, n], where m is
        the velocity parallel to the ecliptic and the n is the velocity
        perpendicular to the ecliptic in pixels/hr.

        likelihood_cutoff: float, required
        The likelihood signal to noise value below which we will ignore
        potential objects.

        timeArr: numpy array, required
        An array containing the image times in hours with the first image at
        time 0.

        wcs: list, required
        The list of wcs instances for each image.

        xRange: list, optional, default=None
        A list containing the [start, stop] pixel values in the x-direction.
        If None then it will cover the length of the images in the x-direction.

        yRange: list, optional, default=None
        A list containing the [start, stop] pixel values in the y-direction.
        If None then it will cover the length of the images in the y-direction.

        out_file: str, optional, default=None
        A string indicating the filename in which to save results if desired.
        """

        parallel_steps = velocity_grid[:, 0]
        perp_steps = velocity_grid[:, 1]

        if self.base_x is None:
            if xRange is None:
                self.base_x = 0
            else:
                self.base_x = xRange[0]
        if self.base_y is None:
            if yRange is None:
                self.base_y = 0
            else:
                self.base_y = yRange[0]

        vel_array = []
        for para_vel, perp_vel in zip(parallel_steps, perp_steps):
            vel_array.append([para_vel, perp_vel])
        vel_array = np.array(vel_array)
        print vel_array

        topVel = []
        topT0 = []
        topScores = []
        topAlpha = []
        if xRange is None:
            x_min = 0
            x_max = np.shape(psiArray[0])[0]
        else:
            x_min = xRange[0]
            x_max = xRange[1]
        if yRange is None:
            y_min = 0
            y_max = np.shape(psiArray[0])[1]
        else:
            y_min = yRange[0]
            y_max = yRange[1]

        for rowPos in xrange(x_min, x_max):
            print rowPos
            for colPos in xrange(y_min, y_max):
                objectStartArr = np.zeros((len(vel_array),2))
                objectStartArr[:,0] += rowPos
                objectStartArr[:,1] += colPos
                alphaArray, nuArray = self.calcAlphaNuEcliptic(psiArray,
                                                               phiArray,
                                                               objectStartArr,
                                                               vel_array,
                                                               timeArr,
                                                               wcs)
                for objNu, objAlpha, objVel in zip(nuArray, alphaArray, vel_array):
                    if objNu > likelihood_cutoff:
                        topScores.append(objNu)
                        topT0.append([rowPos, colPos])
                        topVel.append(objVel)
                        topAlpha.append(objAlpha)

        topScores = np.array(topScores)
        topT0 = np.array(topT0)
        topVel = np.array(topVel)
        topAlpha = np.array(topAlpha)
        print len(topScores)

        rankings = np.argsort(topScores)[-1::-1]
        keepVel = []
        keepT0 = []
        keepScores = []
        keepAlpha = []
        keepPixelVel = []

        for objNum in rankings:
            if objNum % 1000 == 0:
                print objNum
            testT0 = topT0[objNum]
            testVel = topVel[objNum]
            test_vel_str = '%s_%s' % (testVel[0], testVel[1])
            testEclFinalPos = testT0 + self.search_coords_dict[test_vel_str]
            keepT0.append(testT0)
            keepVel.append(testVel)
            keepPixelVel.append([(testEclFinalPos[0]-testT0[0])/timeArr[-1]/24.,
                                     (testEclFinalPos[1]-testT0[1])/timeArr[-1]/24.])
            keepScores.append(topScores[objNum])
            keepAlpha.append(topAlpha[objNum])
        keepT0 = np.array(keepT0)
        keepVel = np.array(keepVel)
        keepPixelVel = np.array(keepPixelVel)
        keepScores = np.array(keepScores)
        keepAlpha = np.array(keepAlpha)

        print "Starting Positions: \n", keepT0
        print "Velocity Vectors: \n", keepVel
        print "Pixel Velocity Vectors: \n", keepPixelVel
        print "Likelihood: \n", keepScores
        print "Best estimated flux: \n", keepAlpha

        results_array = np.rec.fromarrays([keepT0[:,0], keepT0[:,1],
                                           keepVel[:,0], keepVel[:,1],
                                           keepPixelVel[:,0], keepPixelVel[:,1],
                                           keepScores, keepAlpha],
                                          names = str('t0_x,' +
                                                      't0_y,' +
                                                      'theta_par,' +
                                                      'theta_perp,' +
                                                      'v_x,' +
                                                      'v_y,' +
                                                      'likelihood,' +
                                                      'est_flux'))

        if out_file is not None:
            np.savetxt(out_file, results_array.T, fmt = '%.4f',
                       header='%s %s %s %s %s %s %s %s' % results_array.dtype.names)

        return results_array
