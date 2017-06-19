import numpy as np
import astropy.coordinates as astroCoords
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

def calc_ecliptic_angle(test_wcs):

    wcs = [test_wcs]
    pixel_coords = [[],[]]
    pixel_start = [[1000, 2000]]
    angle = 0.
    vel_array = np.array([[6.*np.cos(angle), 6.*np.sin(angle)]])
    time_array = [0.0, 1.0, 2.0]

    vel_par_arr = vel_array[:, 0]
    vel_perp_arr = vel_array[:, 1]

    if type(vel_par_arr) is not np.ndarray:
        vel_par_arr = [vel_par_arr]
    if type(vel_perp_arr) is not np.ndarray:
        vel_perp_arr = [vel_perp_arr]
    for start_loc, vel_par, vel_perp in zip(pixel_start, vel_par_arr, vel_perp_arr):

        start_coord = astroCoords.SkyCoord.from_pixel(start_loc[0],
                                                      start_loc[1],
                                                      wcs[0])
        eclip_coord = start_coord.geocentrictrueecliptic
        eclip_l = []
        eclip_b = []
        for time_step in time_array:
            eclip_l.append(eclip_coord.lon + vel_par*time_step*u.arcsec)
            eclip_b.append(eclip_coord.lat + vel_perp*time_step*u.arcsec)
        eclip_vector = astroCoords.SkyCoord(eclip_l, eclip_b,
                                            frame='geocentrictrueecliptic')
        pixel_coords_set = astroCoords.SkyCoord.to_pixel(eclip_vector, wcs[0])
        pixel_coords[0].append(pixel_coords_set[0])
        pixel_coords[1].append(pixel_coords_set[1])
    pixel_coords = np.array(pixel_coords)

    x_dist = pixel_coords[0,0,-1] - pixel_coords[0,0,0]
    y_dist = pixel_coords[1,0,-1] - pixel_coords[1,0,0]

    eclip_angle = np.degrees(np.arctan(y_dist/x_dist))

    return eclip_angle
