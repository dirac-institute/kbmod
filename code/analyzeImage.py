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

    def createPostageStamp(self, imageArray, objectStartArr, velArr,
                           timeArr, gaussSigma, scaleFactor, wcs_list,
                           starLocs = None):

        singleImagesArray = []
        stampWidth = np.array(np.array(gaussSigma)*scaleFactor, dtype=int)
        stampImage = np.zeros(((2*stampWidth)+1))
        if len(np.shape(imageArray)) < 3:
            imageArray = [imageArray]

#        measureCoords = createImage().calcCenters(objectStartArr, velArr, timeArr)
        measureCoords = self.calcPixelLocationsFromEcliptic([objectStartArr], velArr[0],
                                                            velArr[1], timeArr, wcs_list)
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
            xmin = np.rint(measureCoords[0,i]-stampWidth[0])
            xmax = xmin + stampWidth[0]*2 + 1
            ymin = np.rint(measureCoords[1,i]-stampWidth[1])
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

    def addMask(self, imageArray, locations, gaussSigma):

        maskedArray = np.zeros((np.shape(imageArray)))
        scaleFactor = 4.
        i = 0
        for image in imageArray:
            maskedArray[i] = image * self.createAperture(np.shape(image), locations, scaleFactor, gaussSigma, mask=True)
            i+=1

        return maskedArray

    def definePossibleTrajectories(self, psfSigma, vmin, vmax, maxTime):
        maxRadius = vmax*maxTime
        maxSep = psfSigma*2
        minRadius = vmin*maxTime
        numSteps = int(np.ceil(maxRadius/maxSep))*2
        theta = maxSep/maxRadius
        vRowStart = maxRadius
        vColStart = 0.
        numTraj = int(np.ceil(np.pi*2./theta))
        vRow = np.zeros(numTraj)
        vCol = np.zeros(numTraj)
        vRow[0] = vRowStart
        vCol[0] = vColStart
        for traj in range(1,numTraj):
            vRow[traj] = vRow[traj-1]*np.cos(theta) - vCol[traj-1]*np.sin(theta)
            vCol[traj] = vRow[traj-1]*np.sin(theta) + vCol[traj-1]*np.cos(theta)
        totVRow = np.zeros(numTraj*numSteps)
        totVCol = np.zeros(numTraj*numSteps)
        for stepNum in range(0, numSteps):
            totVRow[numTraj*stepNum:numTraj*(stepNum+1)] = (vRow/numSteps)*(stepNum+1)
            totVCol[numTraj*stepNum:numTraj*(stepNum+1)] = (vCol/numSteps)*(stepNum+1)

        totVRow/=maxTime
        totVCol/=maxTime

        final_positions = np.zeros((len(totVRow), 2))
        for vel_num in xrange(len(totVRow)):
            final_positions[vel_num, 0] = totVRow[vel_num]*maxTime
            final_positions[vel_num, 1] = totVCol[vel_num]*maxTime
        print 'here', len(totVRow)
        keep_idx = [0]
        for pos_idx in xrange(0, len(final_positions)):
            if pos_idx % 100 == 0:
                print pos_idx
            keep_val = True
            for prev_idx in keep_idx:
                if keep_val is False:
                    break
                elif euclidean(final_positions[pos_idx], [0,0]) < maxSep:
                    keep_val=False
                elif euclidean(final_positions[pos_idx], [0,0]) < minRadius:
                    keep_val=False
                elif euclidean(final_positions[pos_idx], final_positions[prev_idx]) < maxSep:
                    keep_val=False
            if keep_val is True:
                keep_idx.append(pos_idx)
        if euclidean(final_positions[0], [0,0]) < maxSep:
            keep_idx.pop(0)
#        print keep_idx
#        print final_positions[keep_idx]

#        totVRow = np.append(totVRow, 0.)
#        totVCol = np.append(totVCol, 0.)
        print len(keep_idx)
        return totVRow[keep_idx], totVCol[keep_idx], numSteps

    def findLikelyTrajectories(self, psiArray, phiArray,
                               psfSigma, v_arr, maxTimeStep, timeArr,
                               xRange=None, yRange=None, numResults=10):

        vRow, vCol, numSteps = self.definePossibleTrajectories(psfSigma, v_arr[0], v_arr[1], maxTimeStep)
        velArr = np.array([vRow, vCol]).T

        psfPixelArea = np.pi*(psfSigma**2)
        tempResults = int(np.ceil(psfPixelArea)*numResults)*5

        topVel = np.zeros((tempResults, 2))
        topT0 = np.zeros((tempResults,2))
        topScores = np.zeros(tempResults)
        topAlpha = np.zeros(tempResults)
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
                objectStartArr = np.zeros((len(vRow),2))
                objectStartArr[:,0] += rowPos
                objectStartArr[:,1] += colPos
                alphaArray, nuArray = self.calcAlphaNu(psiArray, phiArray, objectStartArr, velArr, timeArr)
                for objNu, objAlpha, objVel in zip(nuArray, alphaArray, velArr):
                    if objNu > np.min(topScores):
                        idx = np.argmin(topScores)
                        topScores[idx] = objNu
                        topT0[idx] = [rowPos, colPos]
                        topVel[idx] = objVel
                        topAlpha[idx] = objAlpha

        rankings = np.argsort(topScores)[-1::-1]
        keepVel = np.ones((numResults, 2)) * (999.) # To tell if it has been changed or not
        keepT0 = np.zeros((numResults, 2))
        keepScores = np.zeros(numResults)
        keepAlpha = np.zeros(numResults)

        resultsSet = 0
        for objNum in range(0,tempResults):
            testT0 = topT0[rankings][objNum]
            testVel = topVel[rankings][objNum]
            keepVal = True
            for t0, vel in zip(keepT0, keepVel):
                if ((euclidean(testT0, t0) <= psfSigma) and ((euclidean(testT0+(testVel*timeArr[-1]),
                                                                       t0+(vel*timeArr[-1])) <= psfSigma))):
                    keepVal=False
            if keepVal == True:
                keepT0[resultsSet] = testT0
                keepVel[resultsSet] = testVel
                keepScores[resultsSet] = topScores[rankings][objNum]
                keepAlpha[resultsSet] = topAlpha[rankings][objNum]
                resultsSet += 1
            if resultsSet == numResults:
                break
        print "\nTop %i results" %numResults
        print "Starting Positions: \n", keepT0
        print "Velocity Vectors: \n", keepVel
        print "Likelihood: \n", keepScores
        print "Best estimated flux: \n", keepAlpha

        return keepT0, keepVel, keepScores, keepAlpha

    def calcAlphaNu(self, psiArray, phiArray, objectStartArr, velArr, timeArr):

        if len(np.shape(psiArray)) == 2:
            psiArray = [psiArray]
            phiArray = [phiArray]

        measureCoords = []
        multObjects = False
        if len(np.shape(objectStartArr)) > 1:
            multObjects = True
            for objNum in range(0, len(objectStartArr)):
                measureCoords.append(createImage().calcCenters(objectStartArr[objNum], velArr[objNum], timeArr))
        else:
            measureCoords.append(createImage().calcCenters(objectStartArr, velArr, timeArr))
            measureCoords = np.array(measureCoords)
            objectStartArr = [objectStartArr]

        alphaMeasurements = []
        nuMeasurements = []
        for objNum in range(0, len(objectStartArr)):
            psiTotal = 0
            phiTotal = 0
            for imNum in range(0, len(psiArray)):
                try:
                    psiTotal += psiArray[imNum][measureCoords[objNum][imNum][0], measureCoords[objNum][imNum][1]]
                    phiTotal += phiArray[imNum][measureCoords[objNum][imNum][0], measureCoords[objNum][imNum][1]]
                except:
                    continue
            if (phiTotal != 0):
                alphaMeasurements.append(psiTotal/phiTotal)
                nuMeasurements.append(psiTotal/np.sqrt(phiTotal))
            else:
                alphaMeasurements.append(np.nan)
                nuMeasurements.append(np.nan)

        return alphaMeasurements, nuMeasurements


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

    def findLikelyTrajectoriesParallel(self, psiArray, phiArray, psfSigma,
                                       vMinMax, maxTimeStep, timeArr,
                                       numResults=10, xRange=None,
                                       yRange=None, processes=1):

        import pathos.multiprocessing as mp

        pool = mp.ProcessingPool(processes)

        psiList = [psiArray]*processes
        phiList = [phiArray]*processes
        psfSigmaList = [psfSigma]*processes
        vMaxList = [vmax]*processes
        maxTimeStepList = [maxTimeStep]*processes
        timeArrList = [timeArr]*processes
        numResultsList = [numResults]*processes
        xRangeList = []
        yRangeList = []

        max_overlap = vmax*timeArr[-1]
        x0 = 0
        y0 = 0
        if xRange is not None:
            x_min = xRange[0]
            max_x = xRange[1] - xRange[0]
        else:
            x_min = 0
            max_x = np.shape(psiArray[0])[0]
        if yRange is not None:
            y_min = yRange[0]
            max_y = yRange[1] - yRange[0]
        else:
            y_min = 0
            max_y = np.shape(psiArray[0])[1]
        for proc_num in xrange(processes):
#            x_min -= max_overlap
#            if x_min < 0:
#               x_min = 0
            x_max = x_min + (max_x/2)# 2*max_overlap + (max_x/2)
            xRangeProc = [x_min, x_max]
            yRangeProc = yRange
            xRangeList.append(xRangeProc)
            yRangeList.append(yRangeProc)
            x_min = x_max


        result = pool.map(self.findLikelyTrajectories, psiList,
                                                       phiList,
                                                       psfSigmaList,
                                                       vMaxList,
                                                       maxTimeStepList,
                                                       timeArrList,
                                                       xRangeList,
                                                       yRangeList,
                                                       numResultsList)
        # result = pool.map(self.testIt, [10])

        # keepT0, keepVel, keepScores, keepAlpha = result.get()
        total_result = [[], [], [], []]
        for entry in result:
            for col_num in range(len(entry)):
                total_result[col_num].append(entry[col_num])
        return total_result
        # return keepT0, keepVel, keepScores, keepAlpha
