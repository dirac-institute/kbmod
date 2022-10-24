"""Known object matching is included for debugging purposes to provide signals
into whether there could be known objects in the images and kbmod’s ability to
extract them. Matching is approximate and results might not be comprehensive.
All scientific studies should conduct their own matching analysis.

Acknowledgements: The known object matching uses the IMCCE's SkyBoT VO tool 
(Berthier et. al. 2006) and JPL’s SSD (Solar System Dynamics) API service.
"""

import json
import urllib.request as libreq

import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astroquery.imcce import Skybot
from astroquery.jplhorizons import Horizons
from image_info import *


class KnownObjects:
    def __init__(self):
        self.objects = {}
        self.max_time_index = -1
        self.times = []

    def get_num_times(self):
        return self.max_time_index + 1

    def get_num_results(self):
        return len(self.objects.keys())

    def set_timestamp(self, time_index, t):
        while len(self.times) <= time_index:
            self.times.append(0.0)
        self.times[time_index] = t

    def add_observation(self, name, time_index, sc):
        """
        Add an observation to the set seen known objects.

        Arguments:
            name : string
                Object name from the databse.
            time_index : integer
                The index of the timestep for this observation.
            sc : a SkyCoordinate object
                Indicates the position of the object.
        """
        if self.max_time_index < time_index:
            self.max_time_index = time_index

        # If we haven't seen this object before add an array
        # to store the observations.
        if name not in self.objects:
            self.objects[name] = []

        # Fill in any missing observations with None.
        while len(self.objects[name]) < time_index - 1:
            self.objects[name].append(None)

        # Add the new observation if there isn't one already.
        if len(self.objects[name]) <= time_index:
            self.objects[name].append(sc)

    def get_num_times_seen(self, name):
        """
        Count the number of times we saw an object.

        Arguments:
            name : string
                Object name from the databse.

        Returns:
            integer : the count of times we should have
                      seen this object.
        """
        if not name in self.objects:
            return 0

        count = 0
        for sc in self.objects[name]:
            if sc is not None:
                count = count + 1
        return count

    def pad_results(self):
        """
        Pad the result vectors with None so that they
        are all the same length.
        """
        for name in self.objects.keys():
            # Fill in any missing observations with None.
            while len(self.objects[name]) < self.max_time_index:
                self.objects[name].append(None)

    def filter_observations(self, num_obs):
        """
        Remove all objects from the result list that appear
        fewer than the given number of times.

        Arguments:
            num_obs - The minimum number of images the known object
                      must appear in to be counted.
        """
        all_names = [x for x in self.objects.keys()]
        for name in all_names:
            count = 0
            for sc in self.objects[name]:
                if sc is not None:
                    count = count + 1

            if count < num_obs:
                del self.objects[name]

    def skybot_query_known_objects(self, stats, time_step=-1):
        """
        Finds all known objects that should appear in an image
        given meta data from a FITS file in the form of a
        ImageInfo and adds them to the known objects list.

        Arguments:
           stats : ImageInfo object
               The metadata for the current image.
           time_step : integer
               The time step to use.
        """
        if time_step == -1:
            time_step = self.max_time_step + 1
        self.set_timestamp(time_step, stats.get_epoch())

        # Use SkyBoT to look up the known objects with a conesearch.
        # The function returns a QTable.
        results_table = Skybot.cone_search(stats.center, stats.approximate_radius(), stats.get_epoch())

        # Extract the name and sky coordinates for each object.
        num_results = len(results_table["Name"])
        for row in range(num_results):
            name = results_table["Name"][row]
            ra = results_table["RA"][row]
            dec = results_table["DEC"][row]
            self.add_observation(name, time_step, SkyCoord(ra, dec))

    def skybot_query_known_objects_mult(self, all_stats):
        """
        Finds all known objects that should appear in a series of
        images given the meta data from the corresponding FITS files.

        Arguments:
            all_stats - An ImageInfoSet object holding the
                        for the current set of images.
        """
        num_time_steps = all_stats.num_images
        for t in range(num_time_steps):
            self.skybot_query_known_objects(all_stats.stats[t], t)
        self.pad_results()

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

        base_url = "https://ssd-api.jpl.nasa.gov/sb_ident.api?sb-kind=a&mag-required=true&req-elem=false"

        # Format the time query and MPC string.
        t_str = "obs-time=%f" % stats.get_epoch().jd

        # Create a string of data for the observatory.
        if stats.obs_code:
            obs_str = "mpc-code=%s" % self.obs_code
        else:
            obs_str = "lat=%f&lon=%f&alt=%f" % (stats.obs_lat, stats.obs_long, stats.obs_alt)

        # Format the RA query including half width.
        if stats.center.ra.degree < 0:
            stats.center.ra.degree += 360.0
        ra_hms_L = Angle(stats.center.ra - stats.ra_radius()).hms
        ra_hms_H = Angle(stats.center.ra + stats.ra_radius()).hms
        ra_str = "fov-ra-lim=%02i-%02i-%05.2f,%02i-%02i-%05.2f" % (
            ra_hms_L[0],
            ra_hms_L[1],
            ra_hms_L[2],
            ra_hms_H[0],
            ra_hms_H[1],
            ra_hms_H[2],
        )

        # Format the Dec query including half width.
        dec_str = ""
        dec_dms_L = Angle(stats.center.dec - stats.dec_radius()).dms
        if dec_dms_L[0] >= 0:
            dec_str = "fov-dec-lim=%02i-%02i-%05.2f" % (dec_dms_L[0], dec_dms_L[1], dec_dms_L[2])
        else:
            dec_str = "fov-dec-lim=M%02i-%02i-%05.2f" % (-dec_dms_L[0], -dec_dms_L[1], -dec_dms_L[2])
        dec_dms_H = Angle(stats.center.dec + stats.dec_radius()).dms
        if dec_dms_H[0] >= 0:
            dec_str = "%s,02i-%02i-%05.2f" % (dec_str, dec_dms_H[0], dec_dms_H[1], dec_dms_H[2])
        else:
            dec_str = "%s,M%02i-%02i-%05.2f" % (dec_str, -dec_dms_H[0], -dec_dms_H[1], -dec_dms_H[2])

        # Only do the second (more accurate) pass.
        pass_str = "two-pass=true&suppress-first-pass=true"

        # Complete the full query.
        query = "%s&%s&%s&%s&%s&%s" % (base_url, obs_str, t_str, pass_str, ra_str, dec_str)

        return query

    def jpl_query_known_objects(stats, time_step=-1):
        """
        Finds all known objects that should appear in an image
        given meta data from a FITS file in the form of a
        ImageInfo and adds them to the known objects list.

        Arguments:
           stats : ImageInfo object
               The metadata for the current image.
           time_step : integer
               The time step to use.
        """
        if time_step == -1:
            time_step = self.max_time_step + 1
        self.set_timestamp(time_step, stats.get_epoch())

        query_string = create_jpl_query_string(stats)
        if not query_string:
            print("WARNING: Insufficient data in ImageInfo.")
            return results

        print("Querying: %s" % query_string)

        with libreq.urlopen(query_string) as url:
            feed = url.read().decode("utf-8")
            results = json.loads(feed)

            num_results = results["n_second_pass"]
            for item in results["data_second_pass"]:
                name = item[0]
                ra_str = item[1]
                dec_str = item[2].replace("'", " ").replace('"', "")
                sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                self.add_observation(name, time_step, sc)
        return results

    def jpl_query_known_objects_mult(self, all_stats):
        """
        Finds all known objects that should appear in a series of
        images given the meta data from the corresponding FITS files.

        Arguments:
            all_stats - An ImageInfoSet object holding the
                        for the current set of images.
        """
        num_time_steps = all_stats.num_images
        for t in range(num_time_steps):
            self.jpl_query_known_objects(all_stats.stats[t], t)
        self.pad_results()

    def update_known_objects_pos(self, loc=None):
        """
        Given a set of known objects update their predicted positions
        using the JPL horizons api.

        Arguments:
            loc : The location of the object. An MPC code or a dictionary
                  with entries lat, lon, elevation.
                  None will use geocentric coordinates.
        """
        times_jd = [x.jd for x in self.times]

        for name in self.objects.keys():
            obj = Horizons(id=name, location=loc, epochs=times_jd)
            eph = obj.ephemerides()

            # Update all the positions.
            pos_array = self.objects[name]
            for t in range(len(times_jd)):
                pos_array[t] = SkyCoord(eph["RA"][t] * eph["RA"].unit, eph["DEC"][t] * eph["DEC"].unit)

    def count_known_objects_found(self, found_objects, threshold, num_matches):
        """
        Counts the number of found_objects that appear in the
        list of known objects.

        Arguments:
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

        for name in self.objects.keys():
            for i in range(num_found):
                # Skip found objects we have already used to avoid
                # duplicate counting.
                if used[i]:
                    continue

                # Count the number of "close" observations.
                count = 0
                for t in range(num_times):
                    pos1 = self.objects[name][t]
                    pos2 = found_objects[i][t]
                    if pos1 is not None and pos2 is not None:
                        if pos1.separation(pos2).arcsec <= threshold:
                            count += 1

                if count >= num_matches:
                    used[i] = True
                    match_count += 1
        return match_count
