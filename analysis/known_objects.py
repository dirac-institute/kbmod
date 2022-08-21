from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from astroquery.imcce import Skybot
from image_info import *

import json
import urllib.request as libreq

def skybot_query_known_objects(stats):
    """
    Finds all known objects that should appear in an image
    given meta data from a FITS file in the form of a 
    ImageInfo.

    Arguments:
       stats - An ImageInfo object holding the 
               metadata for the current image.

    Returns:
       A dictionary of name to SkyCoord for each known object
       within the cone search.
    """
    results = {}

    # Use SkyBoT to look up the known objects with a conesearch.
    # The function returns a QTable.
    results_table = Skybot.cone_search(stats.center,
                                       stats.approximate_radius(),
                                       stats.epoch)

    # Extract the name and sky coordinates for each object.
    num_results = len(results_table["Name"])
    for row in range(num_results):
        name = results_table["Name"][row]
        ra = results_table["RA"][row]
        dec = results_table["DEC"][row]
        results[name] = SkyCoord(ra, dec)
    return results

def create_jpl_query_string(stats):
    """
    Create JPL query string out of the component
    information.

    Argument:
        stats : An ImageInfo object holding the 
                metadata for the current image.

    Returns:
        The query string for JPL conesearch queries or None
        if the ImageInfo object does not have sufficient
        information.
    """
    if not stats.obs_loc_set or stats.center is None:
        return None
    
    base_url = ('https://ssd-api.jpl.nasa.gov/sb_ident.api?sb-kind=a'
                '&mag-required=true&req-elem=false')

    # Format the time query and MPC string.
    t_str = ('obs-time=%f' % stats.epoch.jd)

    # Create a string of data for the observatory.
    if stats.obs_code:
        obs_str = ('mpc-code=%s' % self.obs_code)
    else:
        obs_str = ('lat=%f&lon=%f&alt=%f' %
                   (stats.obs_lat, stats.obs_long, stats.obs_alt))

    # Format the RA query including half width.
    if stats.center.ra.degree < 0:
        stats.center.ra.degree += 360.0
    ra_hms_L = Angle(stats.center.ra - stats.ra_radius()).hms
    ra_hms_H = Angle(stats.center.ra + stats.ra_radius()).hms
    ra_str = ('fov-ra-lim=%02i-%02i-%05.2f,%02i-%02i-%05.2f' %
                (ra_hms_L[0], ra_hms_L[1], ra_hms_L[2],
                 ra_hms_H[0], ra_hms_H[1], ra_hms_H[2]))

    # Format the Dec query including half width.
    dec_str = ''
    dec_dms_L = Angle(stats.center.dec - stats.dec_radius()).dms
    if dec_dms_L[0] >= 0:
        dec_str = ('fov-dec-lim=%02i-%02i-%05.2f' %
                   (dec_dms_L[0], dec_dms_L[1], dec_dms_L[2]))
    else:
        dec_str = ('fov-dec-lim=M%02i-%02i-%05.2f' %
                   (-dec_dms_L[0], -dec_dms_L[1], -dec_dms_L[2]))
    dec_dms_H = Angle(stats.center.dec + stats.dec_radius()).dms
    if dec_dms_H[0] >= 0:
        dec_str = ('%s,02i-%02i-%05.2f' %
                   (dec_str, dec_dms_H[0], dec_dms_H[1], dec_dms_H[2]))
    else:
        dec_str = ('%s,M%02i-%02i-%05.2f' %
                   (dec_str, -dec_dms_H[0], -dec_dms_H[1], -dec_dms_H[2]))

    # Only do the second (more accurate) pass.
    pass_str = 'two-pass=true&suppress-first-pass=true'

    # Complete the full query.
    query = ('%s&%s&%s&%s&%s&%s' %
             (base_url, obs_str, t_str, pass_str, ra_str, dec_str))

    return query

def jpl_query_known_objects(stats):
    """
    Finds all known objects that should appear in an image
    given meta data from a FITS file in the form of a 
    ImageInfo.

    Arguments:
       stats - An ImageInfo object holding the 
               metadata for the current image.

    Returns:
       A dictionary of name to SkyCoord for each known object
       within the cone search.
    """
    results = {}

    query_string = create_jpl_query_string(stats)
    if not query_string:
        print('WARNING: Insufficient data in ImageInfo.')
        return results
    
    print('Querying: %s' % query_string)

    with libreq.urlopen(query_string) as url:
        feed = url.read().decode('utf-8')
        results = json.loads(feed)

        num_results = results["n_second_pass"]
        for item in results["data_second_pass"]:
            name = item[0]
            ra_str = item[1]
            dec_str = item[2].replace('\'', ' ').replace('"', '')            
            results[name] = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    return results


def query_known_objects_mult(all_stats, num_obs, use_jpl=False):
    """
    Finds all known objects that should appear in a series of
    images given the meta data from the corresponding FITS files.

    Arguments:
       all_stats - An ImageInfoSet object holding the 
                   for the current set of images.
       num_obs - The minimum number of images the known object
                 must appear in to be counted.
       use_jpl - Use the JPL small body identification API.

    Returns:
       A dictionary of name to a list of SkyCoords (one SkyCoord for
       each time step) for each known object within the cone search.
       The result lists have None if the object was not within the
       cone search.
    """
    full_results = {}

    num_time_steps = all_stats.num_images
    for t in range(num_time_steps):
        # Look up all known objects for the current file.
        if use_jpl:
            results_t = jpl_query_known_objects(all_stats.stats[t])
        else:
            results_t = skybot_query_known_objects(all_stats.stats[t])

        for name in results_t.keys():
            # If we haven't seen this object before add an array
            # to store the observations.
            if name not in full_results:
                full_results[name] = []

            # Fill in any missing observations with None.
            while len(full_results[name]) < t - 1:
                full_results[name].append(None)

            # Add the new observation.
            full_results[name].append(results_t[name])

    # Filter out results that were not in enough images.
    filtered_results = {}
    for name in full_results.keys():
        observations = full_results[name]

        # Count the times it has occurred.
        count = 0
        for val in observations:
            if val is not None:
                count += 1

        if count >= num_obs:
            # Add None to fill in any missing data at the end.
            while len(observations) < num_time_steps:
                observations.append(None)
            filtered_results[name] = observations

    # Return the results.
    return filtered_results


def count_known_objects_found(known_objects, found_objects, 
                              threshold, num_matches):
    """
    Counts

    Arguments:
       known_objects : dictionary
          A dictionary mapping object name to a list of SkyCoords.
       found_objects : list
          A list of lists with one row for every object.
       thresh : float
          The distance threshold for a match (in arcseconds).
       num_matches : integer
          The minimum number of matching points.

    Returns:
       count : integer
          The number of unique objects found.
    """
    num_found = len(found_objects)
    num_times = len(found_objects[0])
    used = [False] * num_found
    match_count = 0

    for name in known_objects.keys():
        for i in range(num_found):
            if used[i]:
                continue

            # Count the number of "close" observations.
            count = 0
            for t in range(num_times):
                pos1 = known_objects[name][t]
                pos2 = found_objects[i][t]
                if pos1 is not None and pos2 is not None:
                    if pos1.separation(pos2).arcsec <= threshold:
                        count += 1

            if count >= num_matches:
                used[i] = True
                match_count += 1
    return match_count
