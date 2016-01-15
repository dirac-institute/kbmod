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

    def applyNoise(self, imageArray):
        noise_added = np.random.poisson(imageArray)
        return noise_added

    def convolveGaussian(self, image, gaussSigma):

        if (type(gaussSigma) is int) or (type(gaussSigma) is float):
            gaussSigma = np.array([gaussSigma, gaussSigma])

        gHorizontal = conv.Gaussian1DKernel(gaussSigma[0])
        gVertical = conv.Gaussian1DKernel(gaussSigma[1])
        convImage = np.copy(image)

        for rowNum in range(0, len(image)):
            convImage[rowNum] = conv.convolve(convImage[rowNum], gHorizontal, boundary='extend')
        for col in range(0, len(image.T)):
            convImage[:,col] = conv.convolve(convImage[:,col], gVertical, boundary='extend')

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
                        bkgrdLevel, sourceLevel, sigmaArr):

        "Create a set of images with a single gaussian psf moving over time."

        objCenters = self.calcCenters(startLocArr, velArr, timeArr)
        imageArray = np.zeros((len(timeArr), imSize[0], imSize[1]))
        varianceArray = np.zeros((len(timeArr), imSize[0], imSize[1]))
        for imNum in xrange(0, len(timeArr)):
            background = self.createSimpleBackground(imSize[0], imSize[1], bkgrdLevel)
            noisy_background = self.applyNoise(background)
            source = self.createGaussianSource(objCenters[imNum], sigmaArr, imSize, sourceLevel)
            noisy_source = self.applyNoise(source)
            imageArray[imNum] = self.sumImage([noisy_source, noisy_background])
            varianceArray[imNum] = noisy_background - background
        hdu = fits.PrimaryHDU(np.transpose(imageArray, (0,2,1))) #FITS x/y axis are switched
        hdu2 = fits.PrimaryHDU(np.transpose(varianceArray, (0,2,1)))
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

    def measureLikelihood(self, imageArray, objectStartArr, velArr, timeArr, psfSigma, verbose=False):

        measureCoords = []
        multObjects = False
        if len(np.shape(objectStartArr)) > 1:
            multObjects = True
            for objNum in range(0, len(objectStartArr)):
                measureCoords.append(createImage().calcCenters(objectStartArr[objNum], velArr[objNum], timeArr))
        else:
            measureCoords.append(createImage().calcCenters(objectStartArr, velArr, timeArr))
            measureCoords = np.array(measureCoords[0])

        if len(np.shape(imageArray)) == 2:
            imageArray = [imageArray]

        i=0
        likeArray = []
        for image in imageArray:
            newImage = np.copy(image)
            #Normalize Likelihood images so minimum value is 0.
            if np.min(newImage) < 0:
                newImage += np.abs(np.min(newImage))
            else:
                newImage -= np.min(newImage)
            likeMeasurements = []
            if multObjects == True:
                likeMeasurements.append([])
            else:
                likeMeasurements.append(newImage[measureCoords[i][1], measureCoords[i][0]])
            likeArray.append(likeMeasurements)
            i+=1
        if verbose == True:
            print "Trajectory Coordinates: (x,y)\n", measureCoords
            print "Likelihood values at coordinates: ", likeArray
        return likeArray

    def measureFlux(self, fitsArray, background, objectStartArr, velArr, timeArr, psfSigma, verbose=False):

        measureCoords = []
        multObjects = False
        if len(np.shape(objectStartArr)) > 1:
            multObjects = True
            for objNum in range(0, len(objectStartArr)):
                measureCoords.append(createImage().calcCenters(objectStartArr[objNum], velArr[objNum], timeArr))
        else:
            measureCoords.append(createImage().calcCenters(objectStartArr, velArr, timeArr))
            measureCoords = np.array(measureCoords[0])

        if len(np.shape(fitsArray)) == 2:
            fitsArray = [fitsArray]

        if isinstance(background, np.ndarray):
            backgroundArray = background
        else:
            backgroundArray = np.ones((np.shape(fitsArray[0])))*background

        scaleFactor=4.
        gaussSize = [2*scaleFactor*psfSigma[0]+1, 2*scaleFactor*psfSigma[1]+1]
        gaussCenter = [scaleFactor*psfSigma[0], scaleFactor*psfSigma[1]]
        gaussKernel = createImage().createGaussianSource(gaussCenter, psfSigma, gaussSize, 1.)
        gaussSquared = conv.convolve(gaussKernel, gaussKernel)

        i=0
        fluxArray = []

        for image in fitsArray:

            centerX = measureCoords[i][0]
            centerY = measureCoords[i][1]
            gaussSquared = conv.convolve(gaussKernel, gaussKernel)

            xmin, xmax, ymin, ymax, offset = self.calcArrayLimits(np.shape(image), centerX, centerY,
                                                                  scaleFactor, psfSigma)
            if offset is not None:
                gaussSquared = gaussSquared[offset:-offset, offset:-offset]

            gaussImage = np.zeros((np.shape(image)))
            gaussImage[centerX, centerY] = 1.
            gaussImage = conv.convolve(gaussImage, gaussKernel, boundary='extend')

            top = np.sum(conv.convolve(image.T[xmin:xmax, ymin:ymax]-backgroundArray[xmin:xmax, ymin:ymax],
                                            gaussKernel, boundary='extend')/backgroundArray[xmin:xmax, ymin:ymax])
            bottom = np.sum(gaussSquared/backgroundArray[xmin:xmax, ymin:ymax])
            fluxMeasured = top/bottom
            fluxArray.append(fluxMeasured)
            i+=1
        if verbose == True:
            print "Trajectory Coordinates: (x,y)\n", measureCoords
            print "Flux values at coordinates: ", fluxArray
        return fluxArray

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

    def calcSNR(self, sourceFlux, centerArr, gaussSigma, background, imSize, scaleFactor = 4.):

        if isinstance(background, np.ndarray):
            backgroundArray = background
        else:
            backgroundArray = np.ones((imSize))*background

        sourceTemplate = createImage().createGaussianSource(centerArr, gaussSigma, imSize, sourceFlux)

        #scaleFactor = 2.
        xmin, xmax, ymin, ymax, offset = self.calcArrayLimits(imSize, centerArr[0], centerArr[1],
                                                              scaleFactor, gaussSigma)
        sourceCounts = np.sum(sourceTemplate[xmin:xmax, ymin:ymax])
        noiseCounts = np.sum(backgroundArray[xmin:xmax, ymin:ymax])

        snr = sourceCounts/np.sqrt(sourceCounts+noiseCounts)
        return snr
