import os
import math

from astropy.stats import sigma_clipped_stats
from astropy.io import fits

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
        #hdulist = fits.open(location)
        #return False, hdulist
        parentCanProcess, hdulist = super().canStandardize(location)

        if not parentCanProcess:
            return False, []

        # duplicates work and is kind of slow but is "the right way"? I'm sure
        # we can do better than this if we think about it a bit
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

    def __init__(self, location):
        super().__init__(location)

        # self.exts is filled in super() but that could include tracking and
        # focus chips too (as they fall under image-like conditions). We want
        # only the science data. So let's overwrite it.
        self.exts = self._getScienceImages(self.hdulist)

    def _calcImgVariance(self, hdu):
        """Given an HDU containing the science image, with gain and read noise
        recorded in the header, calculates the variance image.

        The gain is recorded in per-amplified format in ``GAINA`` and ``GAINB``
        keywords. Similarly, the read noise is recorded in ``RDNOISA`` and
        ``RDNOISEB`` keywords.

        The exposure and overscan parts of the chip were determined from a
        header of a DECam Community Pipeline FITS file and hardcoded.

        Masks are not applied before calculating variance.

        Parameters
        ----------
        hdu : `~astropy.io.fits.HDU`
            Image HDU.

        Returns
        -------
        variance : `np.array`
            Variance image
        """
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
        # parsing that for an example.
        ampa_exposure = hdu.data[1:2048,1:4096]
        ampa_overscan = hdu.data[2105:2154,51:4146]
        ampb_exposure = hdu.data[1:2048,1:4096]
        ampb_overscan = hdu.data[2105:2154,51:4146]

        # A future technote: overscan image should be masked, and we can pass
        # masks in. Because I made the masks up however, doing that would be
        # worse for statistics than not doing it at all.
        #meana, mediana, stddeva = sigma_clipped_stats(ampa_overscan)
        #meanb, medianb, stddevb = sigma_clipped_stats(ampb_overscan)

        ampa_variance = ampa_exposure/gaina + read_noisea**2
        ampb_variance = ampb_exposure/gainb + read_noiseb**2

        variance[1:2048,1:4096] = ampa_variance
        variance[2105:2154,51:4146] = ampb_variance

        return hdu.data

    def _maskSingleImg(self, hdu):
        """Create a mask for the given HDU.

        Mask is a simple edge of detector and 1 sigma treshold mask; grown by
        5 pixels each side.

        Parameters
        ----------
        hdu : `~astropy.io.fits.HDU`
            Image HDU.

        Returns
        -------
        mask : `np.array`
            Mask image
        """
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

    def translateHeader(self):
        fname = os.path.basename(self.location)
        oi = ObservationInfo(self.primary, filename=fname)

        # this is the one piece of metadata that is required
        standardizedHeader["mjd"] = oi.datetime_begin.mjd

        # these are all optional
        standardizedHeader["filter"] = oi.filter
        standardizedHeader["visit_id"] = oi.visit_id
        standardizedHeader["obs_code"] = oi.telescope
        standardizedHeader["obs_lat"] = oi.location.lat.deg
        standardizedHeader["obs_lon"] = oi.location.lon.deg
        standardizedHeader["obs_elev"] = oi.location.height.value # m

        return standardizedHeader

    def standardizeVariance(self):
        return (self._calcImageVariance(hdu) for hdu in self.exts)

    def standardizeMask(self):
        return (self._maskSingleImg(hdu) for hdu in self.exts)
