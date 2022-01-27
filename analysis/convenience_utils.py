from astropy.io import fits
from astropy import wcs
from analysis_utils import Interface

def calc_ecliptic_angle(fn):
    #fn = '/media/fraserw/rocketdata/Projects/kbmod/DATA/rerun/diff_warpCompare/deepDiff/03093/HSC-R2/warps/000/DIFFEXP-0220262-0220380-000.fits'

    with fits.open(fn) as han:
        WCS = wcs.WCS(han[1].header)


    interface = Interface()
    return(interface._calc_ecliptic_angle(WCS))
