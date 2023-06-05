import os
import math

from astropy.stats import sigma_clipped_stats

from .multi_extension_fits import MultiExtensionFits
from astro_metadata_translator import ObservationInfo


__all__ = ["DECamCPFits",]


# Here we focus on translating DECam community pipeline data products
# even though astro_metadata_translator supports more than just DECam.
# We could probably split this whole clas in two - one that provides
# the baseline astro_metadata_translator standardizeHeader function
# accross all instruments it supports, and then individual exts, wcs,
# mask and variance builders (not handled by the metadata translator).
# But I figure that makes the example less readable as an example of
# mechanisms at play.
# When we do probably rename this AstroMetadataTranslator class
class DECamCPFits(MultiExtensionFits):
    
    name = "DECamCommunityPipelineFits"
    priority = 2

    @classmethod
    def canStandardize(cls, location):
        parentCanProcess, hdulist = super().canStandardize(location)

        if not parentCanProcess:
            return False, []

        # this is kind of slow and duplicates work
        # but is "the right way" I guess? I'm sure
        # we can do better than this if we think about it
        # a bit
        fname = os.path.basename(location)
        try:
            oi = ObservationInfo(hdulist["PRIMARY"].header, filename=fname)
        except ValueError:
            canStandardize = False
        else:
            # check all required values exist and are not falsy
            canStandardize = all([
                oi.visit_id,
                oi.datetime_begin,
                oi.datetime_begin,
                oi.telescope,
                oi.location.lat,
                oi.location.lon,
                oi.location.height,
            ])

        canStandardize = parentCanProcess and canStandardize
        return canStandardize, hdulist

    def __init__(self, location):
        super().__init__(location)
        #super().__init__(location, set_exts_wcs_bbox=False)
        # Override the default getimgs to filter only science images
        # without focus and guider chips.
        #self.exts = self._getScienceImages(self.hdulist)
        #self.wcs = [WCS(hdu.header) for hdu in self.exts]

    @classmethod
    def _getScienceImages(cls, hdulist):
        """If the given HDU contains a science image, returns True;
        otherwise False.

        The given HDU is presumed made by the DECam imager and
        processed with the DECam Community Pipelines.

        Parameters
        ----------
        hdu : `astropy.fits.HDU`
            Header unit to inspect.

        Returns
        -------
        image_like : `bool`
            True if HDU is image-like, False otherwise.
        """
        exts = []
        for hdu in hdulist:
            exttype = hdu.header.get("DETPOS", False)
            if exttype:
                if "G" not in exttype and "F" not in exttype:
                    exts.append(hdu)

        return exts

    def standardizeHeader(self):
        # this is compatible with URIs, we just need the final stub
        # of the URI, but I'm not doing that now
        fname = os.path.basename(self.location)
        oi = ObservationInfo(self.primary,
                             filename=fname)

        # now I'm stuck having to build the same silly dict as in
        # rubin_scipipe_std even though it doesn't make sense as the
        # width and height of images can vary if let's say we include
        # the focus and tracking detectors....
        standardizedHeader = {}
        standardizedHeader["width"] = self.exts[0].header["NAXIS1"]
        standardizedHeader["height"] = self.exts[0].header["NAXIS2"]

        # here's the real reason why we save this value too - it's
        # a part of the Rubin abstraction. No idea which timestamp he's
        # saving though, start, mid or end exposure?
        standardizedHeader["visit_id"] = oi.visit_id
        standardizedHeader["obs_datetime"] = oi.datetime_begin
        standardizedHeader["mjd"] = oi.datetime_begin.mjd

        # See rubin_scipipe_std for comments on obs_code
        # probably better to store x,y,z of the location?
        standardizedHeader["obs_code"] = oi.telescope
        standardizedHeader["obs_lat"] = oi.location.lat.deg
        standardizedHeader["obs_lon"] = oi.location.lon.deg
        standardizedHeader["obs_elev"] = oi.location.height.value # m

        return standardizedHeader

    def _calcImgVariance(self, hdu):
        variance = hdu.data.copy()
        
        gaina = hdu.header["GAINA"]
        gainb = hdu.header["GAINB"]

        if math.isnan(gaina) or gaina <= 0:
            gaina = 1.0 # Poisson variance
        if math.isnan(gainb) or gainb <= 0:
            gainb = 1.0

        # these numbers don't make no sense?
        read_noisea = hdu.header["RDNOISEA"]
        read_noiseb = hdu.header["RDNOISEB"]

        # Values recorded in DATASEC[A,B] and BIASEC[A,B] mark the
        # good science data, overscan, prescan and then postscan parts
        # I'm just hardcoding them here because these are actual raw
        # files - no way anyone is running kbmod on them and I'm not
        # parsing that for this. 
        ampa_exposure = hdu.data[1:2048,1:4096]
        ampa_overscan = hdu.data[2105:2154,51:4146]
        ampb_exposure = hdu.data[1:2048,1:4096]
        ampb_overscan = hdu.data[2105:2154,51:4146]

        # A future technote, estimate from RAW data by measuring
        # values in the image overscan regions. Overscan image should
        # be masked, we can pass masks in, but because I made them up,
        # it'll just be worse for stats than not masking at all.
        #meana, mediana, stddeva = sigma_clipped_stats(ampa_overscan)
        #meanb, medianb, stddevb = sigma_clipped_stats(ampb_overscan)

        ampa_variance = ampa_exposure/gaina + read_noisea**2
        ampb_variance = ampb_exposure/gainb + read_noiseb**2

        # Technically I guess I should set variance to NO_DATA for
        # portions outside of the science image, but again this is RAW
        # data.
        variance[1:2048,1:4096] = ampa_variance
        variance[2105:2154,51:4146] = ampb_variance

        return hdu.data

    def standardizeVariance(self):           
        return [self._calcImageVariance(hdu) for hdu in self.exts]

    def _maskSingleImg(self, hdu):
        # we'll just randomly mark some pixels as masked
        # corners
        img = hdu.data
        corner_mask = np.zeros(img.shape)
        corner_mask[:20] = 1
        corner_mask[:, :20] = 1
        corner_mask[-20:,] = 1
        corner_mask[:, -20:] = 1

        threshold = img.mean() - img.std()
        bright_mask = img > threshold

        net_mask = mask & bright_mask

        # this should grow the mask for 5 pixels each side
        grow_kernel = np.ones((11, 11))
        grown_mask = convolve2d(net_mask, grow_kernel, mode="same")

        return grown_mask
        
    def standardizeMask(self):
        return (self._maskSingleImg(hdu) for hdu in self.exts)

    def standardizeBBox(self):
        width, height = self.primary["NAXIS1"], self.primary["NAXIS2"]
        bbox = [self.computeBBox(hdu.header, width, height) for
                bbox in self.exts]
        return bbox



    

    
        

        
