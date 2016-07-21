import os
import numpy as np
import lsst.afw.image as afwImage
from scipy.ndimage import convolve
from scipy.spatial.distance import euclidean
from createImage import createImage as ci


class searchImage(object):

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
        needs to be masked in before it is added to the master mask.

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
        mask_image = (maskImArray > 0.75)*1.

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

        for filename in os.listdir(image_folder):

            print str('On Image ' + filename)

            image_file = os.path.join(image_folder, filename)
            exposure = afwImage.ExposureF(image_file)

            psf_image = exposure.getPsf()
            psf_array = psf_image.computeImage()

            exp_image = exposure.getMaskedImage()

            image_array = exp_image.getImage().getArray()
            image_array *= mask_array

            variance_array = exp_image.getVariance().getArray()

            psi_image = convolve((1/variance_array) *
                                 (image_array), psf_array)

            if psi_array is None:
                psi_array = np.copy(psi_image)
            else:
                psi_array = np.append(psi_array, psi_image)

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

        for filename in os.listdir(image_folder):

            print str('On Image ' + filename)

            image_file = os.path.join(image_folder, filename)
            exposure = afwImage.ExposureF(image_file)

            psf_image = exposure.getPsf()
            psf_array = psf_image.computeImage()

            exp_image = exposure.getMaskedImage()

            variance_array = exp_image.getVariance().getArray()

            phi_image = convolve((1/variance_array)*mask_array, psf_array)

            if phi_array is None:
                phi_array = np.copy(phi_image)
            else:
                phi_array = np.append(phi_array, phi_image)

        return phi_array

    def calcPixelShifts(self, vel_array, time_arr):

        # Velocity in pixel/hr. Time in hrs.

        pixel_shifts_x = np.zeros((len(vel_array), len(time_arr)))
        pixel_shifts_y = np.zeros((len(vel_array), len(time_arr)))
        for vel_num in xrange(0, len(vel_array)):
            pixel_shifts_x[vel_num] = vel_array[vel_num, 0]*time_arr
            pixel_shifts_y[vel_num] = vel_array[vel_num, 1]*time_arr

        return pixel_shifts_x, pixel_shifts_y

    def findLikelyObjects(self, psi_array, phi_array, psf_width, vel_array,
                          time_arr, x_range=None, y_range=None,
                          num_results=10, out_file=None):

        psf_pixel_area = np.pi*(psf_width**2)
        temp_results = int(np.ceil(psf_pixel_area)*num_results)*15

        for psi_image, phi_image in zip(psi_array, phi_array):
            # Discount any points where dividing by phiArray would blow up
            psi_image[phi_image == 0.] = 0.

        top_vel = np.zeros((temp_results, 2))
        top_t0 = np.zeros((temp_results, 2))
        top_scores = np.zeros(temp_results)
        top_alpha = np.zeros(temp_results)

        if x_range is None:
            x_min = 0
            x_max = np.shape(psi_array[0])[0]
        else:
            x_min = x_range[0]
            x_max = x_range[1]
        if y_range is None:
            y_min = 0
            y_max = np.shape(psi_array[0])[1]
        else:
            y_min = y_range[0]
            y_max = y_range[1]

        self.pix_shifts_x, self.pix_shifts_y = self.calcPixelShifts(vel_array,
                                                                    time_arr)

        for rowPos in xrange(x_min, x_max):
            print rowPos
            for colPos in xrange(y_min, y_max):
                objectStartArr = np.zeros((len(vel_array), 2))
                objectStartArr[:, 0] += rowPos
                objectStartArr[:, 1] += colPos
                alphaArray, nuArray = self.calcAlphaNu(psi_array,
                                                       phi_array,
                                                       objectStartArr,
                                                       vel_array,
                                                       time_arr)
                for objNu, objAlpha, objVel in zip(nuArray, alphaArray, vel_array):
                    if objNu > np.min(top_scores):
                        idx = np.argmin(top_scores)
                        top_scores[idx] = objNu
                        top_t0[idx] = [rowPos, colPos]
                        top_vel[idx] = objVel
                        top_alpha[idx] = objAlpha

        rankings = np.argsort(top_scores)[-1::-1]
        keepVel = np.ones((num_results, 2)) * (999.) # To tell if it has been changed or not
        keepT0 = np.zeros((num_results, 2))
        keepScores = np.zeros(num_results)
        keepAlpha = np.zeros(num_results)

        resultsSet = 0
        for objNum in range(0,temp_results):
            testT0 = top_t0[rankings][objNum]
            testVel = top_vel[rankings][objNum]
            keepVal = True
            for t0, vel in zip(keepT0, keepVel):
                if ((euclidean(testT0, t0) <= psf_width*2) or ((euclidean(testT0+(testVel*time_arr[-1]),
                                                                       t0+(vel*time_arr[-1])) <= psf_width*2))):
                    keepVal=False
            if keepVal == True:
                keepT0[resultsSet] = testT0
                keepVel[resultsSet] = testVel
                keepScores[resultsSet] = top_scores[rankings][objNum]
                keepAlpha[resultsSet] = top_alpha[rankings][objNum]
                resultsSet += 1
            if resultsSet == num_results:
                break
        print "\nTop %i results" %num_results
        print "Starting Positions: \n", keepT0
        print "Velocity Vectors: \n", keepVel
        print "Likelihood: \n", keepScores
        print "Best estimated flux: \n", keepAlpha

        results_array = np.rec.fromarrays([keepT0[:,0], keepT0[:,1],
                                           keepVel[:,0], keepVel[:,1],
                                           keepScores, keepAlpha],
                                          names = str('t0_x,' +
                                                      't0_y,' +
                                                      'v_x,' +
                                                      'v_y,' +
                                                      'likelihood,' +
                                                      'est_flux'))

        if out_file is not None:
            np.savetxt(out_file, results_array.T, fmt = '%.4f',
                       header='%s %s %s %s %s %s' % results_array.dtype.names)

        return results_array

    def calcAlphaNu(self, psiArray, phiArray, objectStartArr, velArr, timeArr):

        if len(np.shape(psiArray)) == 2:
            psiArray = [psiArray]
            phiArray = [phiArray]

        search_coords_x = self.pix_shifts_x + objectStartArr[0][0]
        search_coords_y = self.pix_shifts_y + objectStartArr[0][1]

        # Counts on psi, phi images have 0 along borders due to convolution
        # Should do something safer in the future
        search_coords_x[np.where(search_coords_x < 0)] = 0.
        search_coords_y[np.where(search_coords_y < 0)] = 0.

        psiTotal = np.zeros(len(objectStartArr))
        phiTotal = np.zeros(len(objectStartArr))
        pixel_locs_x = np.array(search_coords_x, dtype=np.int)
        pixel_locs_y = np.array(search_coords_y, dtype=np.int)

        for imNum in range(0, len(psiArray)):
            try:
                psiTotal += psiArray[imNum][pixel_locs_x[:, imNum],
                                            pixel_locs_y[:, imNum]]
                phiTotal += phiArray[imNum][pixel_locs_x[:, imNum],
                                            pixel_locs_y[:, imNum]]
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
        nu_measurements[phi_not0] = (psiTotal[phi_not0] /
                                     np.sqrt(phiTotal)[phi_not0])

        return alpha_measurements, nu_measurements

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
