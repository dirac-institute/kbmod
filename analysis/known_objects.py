from astropy.coordinates import SkyCoord
from astroquery.imcce import Skybot
from image_info import *

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


def skybot_query_all_times(all_stats, num_obs):
    """
    Finds all known objects that should appear in a series of
    images given the meta data from the corresponding FITS files.

    Arguments:
       all_stats - An ImageInfoSet object holding the 
                   for the current set of images.
       num_obs - The minimum number of images the known object
                 must appear in to be counted.

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
