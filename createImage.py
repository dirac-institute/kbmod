import numpy as np
import matplotlib.mlab as mlab
import astropy.convolution as conv
import matplotlib.pyplot as plt
from astropy.io import fits

class createImage(object):

    def createSimpleBackground(self, xPixels, yPixels, backgroundLevel):
        "Creates 2-d array with given number of pixels."
        backgroundArray = np.ones((xPixels, yPixels))*backgroundLevel
        noise_mask = np.random.poisson(backgroundArray)
        backgroundArray += noise_mask
        return backgroundArray

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
        for imNum in xrange(0, len(timeArr)):
            background = self.createSimpleBackground(imSize[0], imSize[1], bkgrdLevel)
            source = self.createGaussianSource(objCenters[imNum], sigmaArr, imSize, sourceLevel)
            imageArray[imNum] = self.sumImage([source, background])
        hdu = fits.PrimaryHDU(np.transpose(imageArray, (0,2,1))) #FITS x/y axis are switched
        hdu.writeto(str(outputName + '.fits'))

class analyzeImage(object):

    def measureFlux(self, imageArray, objectStartArr, velArr, timeArr, psfSigma):

        from skimage import restoration

        measureCoords = []
        multObjects = False
        if len(np.shape(objectStartArr)) > 1:
            multObjects = True
            for objNum in range(0, len(objectStartArr)):
                measureCoords.append(createImage().calcCenters(objectStartArr[objNum], velArr[objNum], timeArr))
        else:
            measureCoords.append(createImage().calcCenters(objectStartArr, velArr, timeArr))
            measureCoords = np.array(measureCoords[0])
        print "Trajectory Coordinates: (x,y)\n", measureCoords

        i=0
        fluxArray = []
        for image in imageArray:
            newImage = np.copy(image)
            fluxMeasurements = []
            if multObjects == True:
                fluxMeasurements.append([])
            else:
                fluxMeasurements.append(newImage[measureCoords[i][1], measureCoords[i][0]])
            fluxArray.append(fluxMeasurements)
            i+=1
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
