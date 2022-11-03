import pandas as pd


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
