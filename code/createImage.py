import numpy as np
import matplotlib.mlab as mlab
import astropy.convolution as conv
import matplotlib.pyplot as plt
from astropy.io import fits

class createImage(object):

    def createSimpleBackground(self, xPixels, yPixels, backgroundLevel):
        "Creates 2-d array with given number of pixels."
        backgroundArray = np.ones((xPixels, yPixels))*backgroundLevel
        return backgroundArray

    def createStarSet(self, numImages, imSize, meanFlux, invDensity, psfSigma):
        starArray = np.zeros((imSize))
        numStars = int(imSize[0]*imSize[1]/invDensity)
        xCenters = np.random.randint(0, imSize[0], size=numStars)
        yCenters = np.random.randint(0, imSize[1], size=numStars)
        fluxArray = np.array(np.ones(numStars)*meanFlux +
                                  np.random.uniform(-0.5*meanFlux, 0.5*meanFlux, size=numStars))
        for star in range(0, numStars):
            starArray += self.createGaussianSource([xCenters[star], yCenters[star]], psfSigma,
                                                    imSize, fluxArray[star])

        starImagesArray = np.zeros((numImages, imSize[0], imSize[1]))
        for imNum in range(0, numImages):
            starImagesArray[imNum] = np.copy(starArray)

        return starImagesArray, np.transpose([xCenters, yCenters]), fluxArray

    def applyNoise(self, imageArray):
        noise_added = np.random.poisson(imageArray)
        return noise_added

    def convolveGaussian(self, image, gaussSigma, **kwargs):

        if (type(gaussSigma) is int) or (type(gaussSigma) is float):
            gaussSigma = np.array([gaussSigma, gaussSigma])

        gRow = conv.Gaussian1DKernel(gaussSigma[0])
        gCol = conv.Gaussian1DKernel(gaussSigma[1])
        convImage = np.copy(image)

        for rowNum in range(0, len(image)):
            convImage[rowNum] = conv.convolve(convImage[rowNum], gRow, **kwargs)
        for col in range(0, len(image.T)):
            convImage[:,col] = conv.convolve(convImage[:,col], gCol, **kwargs)

        return convImage

    def convolveSquaredGaussian(self, image, gaussSigma):

        if (type(gaussSigma) is int) or (type(gaussSigma) is float):
            gaussSigma = np.array([gaussSigma, gaussSigma])

        gRow = conv.Gaussian1DKernel(gaussSigma[0])
        gSqRow = conv.CustomKernel(np.power(gRow.array, 2))
        gCol = conv.Gaussian1DKernel(gaussSigma[1])
        gSqCol = conv.CustomKernel(np.power(gCol.array, 2))
        convImage = np.copy(image)

        for rowNum in range(0, len(image)):
            convImage[rowNum] = conv.convolve(convImage[rowNum], gSqRow, boundary=None)
        for col in range(0, len(image.T)):
            convImage[:,col] = conv.convolve(convImage[:,col], gSqCol, boundary=None)

        return convImage

    def createGaussianSource(self, centerArr, sigmaArr, imSize, fluxVal):
        """Creates 2-D Gaussian Point Source

        centerArr: [xCenter, yCenter] in pixels
        sigmaArr: [xSigma, ySigma] in pixels
        imSize: [xPixels, yPixels]
        fluxVal: Flux value of point source"""

        sourceIm = np.zeros((imSize))
        sourceIm[centerArr[0], centerArr[1]] = fluxVal
        newSource = self.convolveGaussian(sourceIm, sigmaArr)

        return newSource

    def calcCenters(self, startLocArr, velArr, timeArr):

        startLocArr = np.array(startLocArr)
        velArr = np.array(velArr)
        centerArr = []
        for time in timeArr:
            centerArr.append(startLocArr + (velArr*time))
        return np.array(centerArr)

    def sumImage(self, imagePieces):

        shape = np.shape(imagePieces[0])
        totalImage = np.zeros((shape))
        for imagePart in imagePieces:
            totalImage += imagePart
        return totalImage

    def createSingleSet(self, outputName, startLocArr, velArr, timeArr, imSize,
                        bkgrdLevel, sourceLevel, sigmaArr, sourceNoise = True, bkgrdNoise = True,
                        addStars = True, starNoise = True, meanIntensity = None, invDensity = 30**2):

        "Create a set of images with a single gaussian psf moving over time."

        objCenters = self.calcCenters(startLocArr, velArr, timeArr)
        imageArray = np.zeros((len(timeArr), imSize[0], imSize[1]))
        varianceArray = np.zeros((len(timeArr), imSize[0], imSize[1]))
        for imNum in xrange(0, len(timeArr)):
            background = self.createSimpleBackground(imSize[0], imSize[1], bkgrdLevel)
            if bkgrdNoise == True:
                noisy_background = self.applyNoise(background)
            else:
                noisy_background = background

            if sourceNoise == True:
                source = self.createGaussianSource(objCenters[imNum], sigmaArr, imSize, np.random.poisson(sourceLevel))
            else:
                source = self.createGaussianSource(objCenters[imNum], sigmaArr, imSize, sourceLevel)

            imageArray[imNum] = self.sumImage([source, noisy_background])
            varianceArray[imNum] = noisy_background - background

        if addStars == True:
            if meanIntensity == None:
                meanIntensity = 40.*bkgrdLevel
            stars, starLocs, starFlux = self.createStarSet(len(timeArr), imSize, meanIntensity, invDensity, sigmaArr)
            if starNoise == True:
                noisy_stars = self.applyNoise(stars)
            else:
                noisy_stars = stars
            imageArray += noisy_stars
            np.savetxt(str(outputName + '_stars.dat'), starLocs)
            np.savetxt(str(outputName + '_starsFlux.dat'), starFlux)

        hdu = fits.PrimaryHDU(imageArray)
        hdu2 = fits.PrimaryHDU(varianceArray)
        hdu.writeto(str(outputName + '.fits'))
        hdu2.writeto(str(outputName + '_var.fits'))

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

    def createPostageStamp(self, imageArray, objectStartArr, velArr, timeArr, gaussSigma, scaleFactor,
                           starLocs = None):

        singleImagesArray = []
        stampWidth = np.array(np.array(gaussSigma)*scaleFactor, dtype=int)
        stampImage = np.zeros(((2*stampWidth)+1))
        if len(np.shape(imageArray)) < 3:
            imageArray = [imageArray]

        measureCoords = createImage().calcCenters(objectStartArr, velArr, timeArr)
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

    def addMask(self, imageArray, locations, gaussSigma):

        maskedArray = np.zeros((np.shape(imageArray)))
        scaleFactor = 4.
        i = 0
        for image in imageArray:
            maskedArray[i] = image * self.createAperture(np.shape(image), locations, scaleFactor, gaussSigma, mask=True)
            i+=1

        return maskedArray

    def definePossibleTrajectories(self, psfSigma, vmax, maxTimeStep):
        maxRadius = vmax*maxTimeStep
        maxSep = psfSigma*2
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
        totVRow = np.append(totVRow, 0.)
        totVCol = np.append(totVCol, 0.)
        return totVRow, totVCol, numSteps

    def findLikelyTrajectories(self, psiArray, phiArray,
                               psfSigma, vmax, maxTimeStep, timeArr,
                               xRange=None, yRange=None, numResults=10):

        vRow, vCol, numSteps = self.definePossibleTrajectories(psfSigma, vmax, maxTimeStep)
        velArr = np.array([vRow, vCol]).T

        psfPixelArea = np.pi*(psfSigma**2)
        tempResults = int(np.ceil(psfPixelArea)*numResults)

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
        keepVel = np.ones((numResults, 2)) * (vmax+1)
        keepT0 = np.zeros((numResults, 2))
        keepScores = np.zeros(numResults)
        keepAlpha = np.zeros(numResults)

        resultsSet = 0
        for objNum in range(0,tempResults):
            testT0 = topT0[rankings][objNum]
            testVel = topVel[rankings][objNum]
            keepVal = True
            for t0, vel in zip(keepT0, keepVel):
                if ((np.sqrt(np.sum(np.power(testT0-t0,2))) < psfSigma) and (testVel[0] == vel[0]) and (testVel[1] == vel[1])):
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

    def calcPsi(self, imageArray, psfSigma, verbose=False, starLocs=None, background=0., mask=None):

        if len(np.shape(imageArray)) == 2:
            imageArray = [imageArray]

        if starLocs is not None:
            scaleFactor = 4.
            mask = self.createAperture(np.shape(imageArray[0]), starLocs,
                                       scaleFactor, psfSigma, mask=True)
        elif mask is None:
            mask = np.ones(np.shape(imageArray[0]))

        if isinstance(background, np.ndarray):
            backgroundArray = background
        else:
            backgroundArray = np.ones((np.shape(imageArray[0])))*background

        i=0
        likeImageArray = []
        for image, backgroundImage in zip(imageArray, backgroundArray):
            print str('On Image ' + str(i+1) + ' of ' + str(len(imageArray)))
            newImage = np.copy(image)

            if background != 0.:
                likelihoodImage = createImage().convolveGaussian((1/backgroundImage)*((newImage*mask)-(backgroundImage*mask)), psfSigma)
            else:
                likelihoodImage = createImage().convolveGaussian(((newImage*mask)-(backgroundImage*mask)), psfSigma)
            #if starLocs is not None:
                #likelihoodImage = mask*likelihoodImage

            likeImageArray.append(likelihoodImage)
            i+=1

        return likeImageArray

    def calcPhi(self, imArrayShape, psfSigma, verbose=False, starLocs=None, background=0.):

        if isinstance(background, np.ndarray):
            backgroundArray = background
        else:
            backgroundArray = np.ones(imArrayShape)*background

        i=0
        likeImageArray = np.zeros(imArrayShape)
        if len(imArrayShape) == 2:
            likeImageArray = [likeImageArray]
            backgroundArray = [backgroundArray]
            maskShape = imArrayShape
        else:
            maskShape = imArrayShape[1:]

        if starLocs is not None:
            scaleFactor = 4.
            mask = self.createAperture(maskShape, starLocs, scaleFactor, psfSigma, mask=True)

        for backgroundImage in backgroundArray:
            print str('On Image ' + str(i+1) + ' of ' + str(len(likeImageArray)))
            # for rowPos in range(0, np.shape(likeImageArray[i])[0]):
            #     print rowPos
            #     for colPos in range(0, np.shape(likeImageArray[i])[1]):
            #         psfImage = createImage().createGaussianSource([rowPos, colPos], [psfSigma, psfSigma], np.shape(likeImageArray[i]), 1.)
            #         if background != 0.:
            #             psfImage /= backgroundImage
            #         psfSquared = createImage().convolveGaussian(psfImage, [psfSigma, psfSigma])
            #         likeImageArray[i][rowPos, colPos] = psfSquared[rowPos, colPos]
            if background != 0.:
                likeImageArray[i] = createImage().convolveSquaredGaussian((1/backgroundImage), [psfSigma, psfSigma])
            else:
                likeImageArray[i] = createImage().convolveSquaredGaussian(np.ones((imArrayShape)), [psfSigma, psfSigma])


            if starLocs is not None:
                likeImageArray[i] = mask*likeImageArray[i]
            i+=1

        return likeImageArray

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

    def findLikelyTrajectoriesParallel(self, psiArray, phiArray, psfSigma,
                                       vmax, maxTimeStep, timeArr,
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

    def testIt(self, x):
        return x*x, x*x*x, x*x*x*x, x*x*x*x*x
