import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from kbmod.util_functions import get_matched_obstimes


def make_stamps_from_emphems(times, ra, dec, workunit, radius=50):
    """Create image stamps from a list of given ephemeris predictions.

    Parameters
    ----------
    times : list-like
        The predicted times.
    ra : list-like
        The predicted right ascensions (in degrees).
    dec : list-like
        The predicted declinations (in degrees).
    workunit : WorkUnit
        The WorkUnit from which to extract the image data.
    radius : int
        The stamp radius (in pixels).

    Returns
    -------
    match_times : list
        A list of the times that match.
    stamps : list
        A list of the stamps around the predicted position.
    """
    ra = np.array(ra)
    dec = np.array(dec)
    times = np.array(times)

    obs_times = workunit.get_all_obstimes()
    matched_inds = get_matched_obstimes(obs_times, times)

    # Generate a stamp for each matching time.
    match_times = []
    stamps = []
    for query_num, match_index in enumerate(matched_inds):
        if match_index == -1:
            # No match. Skip.
            continue

        # Compute the object's pixel coordinates and generate a stamp around that.
        curr_wcs = workunit.get_wcs(match_index)
        sci_image = workunit.im_stack.get_single_image(match_index).get_science()
        px, py = curr_wcs.world_to_pixel(SkyCoord(ra[query_num], dec[query_num], unit="deg"))
        stamp = sci_image.create_stamp(px, py, radius, False)

        stamps.append(stamp)
        match_times.append(obs_times[match_index])

    return match_times, stamps


class ssoisPrecovery:
    """
    This class is designed to use the Solar System Object Image Search (SSOIS) website provided by CADC
    and accessible at this website: https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/ssois/index.html.

    When using this we should make sure to include the attributions from the website:

    `
    For more information about the inner workings of SSOIS, please read the the following paper:
    [Gwyn, Hill and Kavelaars (2012)](http://adsabs.harvard.edu/abs/2012PASP..124..579G) .
    Please cite this paper in your publications.

    If you have used CADC facilities for your research,
    please include the following acknowledgment in your publications:
    *This research used the facilities of the Canadian Astronomy Data Centre operated by the
    National Research Council of Canada with the support of the Canadian Space Agency.*
    `
    """

    def format_search_by_arc_url(
        self, mpc_file, start_year=1990, start_month=1, start_day=1, end_year=2020, end_month=8, end_day=1
    ):
        """
        Create the correct url for SSOIS query by arc

        Inputs
        ------
        mpc_file: str
            Filename for mpc formatted file containing observations of object.

        start_year ... end_day: int
            The dates for the start and end windows of possible precovery imaging dates.
            Note that the first date allowed is Jan. 1 1990.

        Returns
        -------
        base_url: str
            URL for the SSOIS that will return the desired search results.
        """

        mpc_file_string_list = []
        with open(mpc_file, "r") as file:
            for line in file:
                mpc_file_string_list.append(line)

        base_url = "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/ssos/ssosclf.pl?lang=en;obs="
        for line in mpc_file_string_list:
            for char in line:
                if char == " ":
                    base_url += "+"
                elif char == "\n":
                    base_url += "%0D%0A"
                else:
                    base_url += char
        base_url += ";search=bern"
        base_url += ";epoch1={}+{:02}+{:02}".format(start_year, start_month, start_day)
        base_url += ";epoch2={}+{:02}+{:02}".format(end_year, end_month, end_day)
        base_url += ";eunits=bern;extres=no;xyres=no;format=tsv"

        return base_url

    def query_ssois(self, url):
        """
        Gathers results from SSOIS service and returns them in a pandas dataframe.

        Input
        -----
        url: str
            URL for search through SSOIS service

        Returns
        -------
        results_df: pandas dataframe
            Pandas dataframe containing search results
        """

        results_df = pd.read_csv(url, delimiter="\t")
        # Avoid problems querying column in pandas with '/' in name
        results_df.rename(columns={"Telescope/Instrument": "Telescope_or_Instrument"}, inplace=True)

        return results_df
