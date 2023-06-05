"""
Class for standardizing FITS files produced by the Vera C. Rubin
Science Pipelines.
"""
from astropy.wcs import WCS
from astropy.time import Time
from scipy.signal import convolve2d

from kbmod.standardizers import MultiExtensionFits


__all__ = ["RubinSciPipeFits", ]


class RubinSciPipeFits(MultiExtensionFits):

    name = "RubinSciencePipelineFits"
    priority = 2

    bit_flag_map = {
            "BAD": 0,
            "CLIPPED": 9,
            "CR": 3,
            "CROSSTALK": 10,
            "DETECTED": 5,
            "DETECTED_NEGATIVE": 6,
            "EDGE": 4,
            "INEXACT_PSF": 11,
            "INTRP": 2,
            "NOT_DEBLENDED": 12,
            "NO_DATA": 8,
            "REJECTED": 13,
            "SAT": 1,
            "SENSOR_EDGE": 14,
            "SUSPECT": 7,
            "UNMASKEDNAN": 15,
        }
    
    mask_flags = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]
   
    @classmethod
    def canStandardize(cls, uploadedFile):
        parentCanProcess, hdulist = super().canStandardize(uploadedFile)

        if not parentCanProcess:
            return False

        # A rough best guess here, I'm certain we can find a Rubin
        # signature somewhere in the header that's a clearer signal
        # that this is a Rubin Sci Pipe product
        primary = hdulist["PRIMARY"].header
        isRubinProcessed = all(("ZTENSION" in primary,
                                "ZPCOUNT" in primary,
                                "ZGCOUNT" in primary,
                                "CCDNUM" in primary))
        
        canStandardize = parentCanProcess and isRubinProcessed
        return canStandardize, hdulist

    def __init__(self, location):
        super().__init__(location, set_exts_wcs_bbox=False)
        # this is the only science-image header for Rubin
        self.exts = [self.hdulist[1], ]
        self.wcs = [WCS(self.hdulist[1].header), ]
        
    def standardizeHeader(self):
        # these are all the values extracted in the old image_info
        # but I'm not sure we need all of them? I'm pretty sure we
        # don't, because IDNUM, visit_id, things like that are not
        # "the standard" FITS things but purely Rubin DECam oriented.
        # We need to abstract that out. The flip-side is that it's
        # nearly guaranteed that the scripts will be using this to
        # figure out what they're working with, because in original
        # the visitid was extracted from filenames in order (I guess)
        # to sort/group them together so I can't just outright get rid
        # of them that easily
        # This whole thing is probably better off not using hard-coded
        # header keywords because the stack does not standardize on
        # those, they standardize on a common api and then translate
        # those to these and vice-versa. I don't know if it's a good
        # idea to expect the stack to be installed though, so we're
        # not left with a lot of recourse.
        # Maybe we can have a general implementation depend on the
        # stack (here) and then instruct users to inherit and provide
        # their own stack-independent maps on case-by-case basis by
        # overwriting this method?
        standardizedHeader = {}

        standardizedHeader["width"] = self.exts[0].header["NAXIS1"]
        standardizedHeader["height"] = self.exts[0].header["NAXIS2"]
        
        standardizedHeader["visit_id"] = self.primary["IDNUM"]
        
        # this used to be called epoch, but I'd rather not because
        # epoch usually means "time-related reference point" not a
        # timestamp, also what timestamp? Start, end, middle? And why
        # both Time object and mjd?
        # this is how it used to be done, but doesn't work - different
        # dataset doesn't contain MJD?
        # Time(self.primary["MJD"], format="mjd", scale="utc")
        standardizedHeader["obs_datetime"] = Time(self.primary["DATE-AVG"], format="isot")
        standardizedHeader["mjd"] = standardizedHeader["obs_datetime"].mjd

        # I guess this actually probably shouldn't be a part of this
        # functionality but an additional user-provided thing to the
        # image_info or perhaps at best an optional value returned
        # from this method. Not all data will have these. Specifically
        # OBSERVAT key? The DECam RAW data does but that contains
        # OBSERVAT= 'CTIO    '           / Observatory name
        # which isn't obs code at all and so do calexps
        standardizedHeader["obs_code"] = self.primary["OBSERVAT"]
        standardizedHeader["obs_lat"] = self.primary["OBS-LAT"]
        standardizedHeader["obs_lon"] = self.primary["OBS-LONG"]
        standardizedHeader["obs_elev"] = self.primary["OBS-ELEV"]

        # For now we can just say that a dictionary is the expected
        # return but we can build out dataclasses for this instead
        return standardizedHeader

    def standardizeBBox(self):
        # calexp is per detector so only 1 bbox, we return a list
        # to be uniform with regards to multiext instruments
        header = self.exts[0].header
        bbox = self.computeBBox(header, header["NAXIS1"],
                                header["NAXIS2"])
        return [bbox, ]

    def standardizeMask(self):
        # Ideally these flag values would not be the given defaults in
        # KBMODConfig - since there is no given default instrument
        # here. The instrument default map would be the thing of the
        # standardizer class itself.
        
        # On the good side, since the map is a product of the
        # instrument and processing pipeline reasonable defaults could
        # be provided by us and not neccessarily by user everytime.
        
        # On the bad side, because KBMODConfig can not be composed,
        # the defaults have to exist in it. But the usage is broken?
        # The defaults are defined in the __init__ - which means
        # overwriting them doesn't change anything
        # conf = KBMODConfig()
        # conf.set_from_dict({
        #    'default_mask_bits_dict' = self.default_mask_bits_dict,
        #    'default_flag_keys' = self.default_flag_keys,
        #    })

        # The solution is to overwrite the actual parameter that goes
        # in the processing - the naming of which isn't easy to make
        # out really, but, and more importantly, why would I even want
        # a config involved here I guess? Rather I'd just like to have
        # a default map as a class attribute? I do need the config for
        # other stuff, and it'd be nice to record these defaults in it
        # but I'm not feeling the interface.
        # Basically what I'm saying here is that I tried dogfooding
        # and I didn't like it. This abstraction doesn't work that
        # well with it, some base functionality doesn't behave as
        # expected, and is hard to understand
        # conf.set_from_dict({
        #    "mask_bits_dict": self.default_mask_bits_dict
        #    "flag_keys": self.default_flag_keys
        #})
        
        # The other gripe I have here is that the Masking classes are
        # actually kind of super complicated to use because they
        # work on ImageStack and push their functionality down to C++
        # and onwards to GPU. We should push some of that out to
        # Python (not neccessarily yet another implementation but a
        # way to run them without having to build RawImages,
        # LayeredImages or ImageStacks and incur a copy of underlying
        # data) so people can experiment on an example images

        # I think I like this way of getting different planes because
        # then nobody needs to remember what index they are under and
        # it's likely they'll remain named consistently
        idx = self.hdulist.index_of("IMAGE")
        threshold_mask = self.hdulist[idx].data > 100

        net_flag = sum([2**self.bit_flag_map[f] for f in self.mask_flags])
        idx = self.hdulist.index_of("MASK")
        mask_data = self.hdulist[idx].data
        flag_mask = net_flag & mask_data
        
        net_mask = threshold_mask & flag_mask

        # this should grow the mask for 5 pixels each side
        grow_kernel = np.ones((11, 11))
        grown_mask = convolve2d(net_mask, grow_kernel, mode="same")
        
        return grown_mask

    def standardizeVariance(self):
        idx = self.hdulist.index_of("VARIANCE")
        return self.hdulist[idx].data
        
