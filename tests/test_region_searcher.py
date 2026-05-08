import unittest
import argparse
import sys
import os
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from unittest.mock import MagicMock, patch

try:
    from kbmod_cmdline.region_searcher import region_searcher

    print(f"DEBUG: Loaded region_searcher from {sys.modules['kbmod_cmdline.region_searcher'].__file__}")
    print(f"DEBUG: Signature: {inspect.signature(region_searcher)}")
except ImportError as e:
    print(f"DEBUG: Failed to import region_searcher: {e}")
    raise e

from astropy.coordinates import EarthLocation
from astropy.table import Table


class TestRegionSearcher(unittest.TestCase):
    def setUp(self):
        self.ic_filename = "dummy_ic.ecsv"
        self.ephem_filename = "dummy_ephem.csv"

        # Create dummy ImageCollection file with all required columns
        # Build a minimal valid WCS JSON string
        import json
        from astropy.wcs import WCS

        dummy_wcs = WCS(naxis=2)
        dummy_wcs.wcs.crpix = [500, 500]
        dummy_wcs.wcs.crval = [0.0, 0.0]
        dummy_wcs.wcs.cdelt = [-0.2 / 3600, 0.2 / 3600]
        dummy_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        dummy_wcs.array_shape = (1000, 1000)
        wcs_header = dummy_wcs.to_header(relax=True)
        wcs_header["NAXIS1"] = 1000
        wcs_header["NAXIS2"] = 1000
        wcs_str = json.dumps({k: v for k, v in wcs_header.items()}, separators=(",", ":"))

        ic_table = Table(
            {
                "visit": [101, 102, 103, 104],
                "detector": [1, 1, 1, 1],
                "ra": [0.0, 0.1, 0.2, 359.9],
                "dec": [0.0, 0.0, 0.0, 0.0],
                "mjd_mid": [60000.0, 60000.1, 60000.2, 60000.3],
                "ra_tl": [0.0] * 4,
                "dec_tl": [0.0] * 4,
                "ra_tr": [0.0] * 4,
                "dec_tr": [0.0] * 4,
                "ra_br": [0.0] * 4,
                "dec_br": [0.0] * 4,
                "ra_bl": [0.0] * 4,
                "dec_bl": [0.0] * 4,
                "std_idx": [0, 1, 2, 3],
                "ext_idx": [0, 0, 0, 0],
                "std_name": ["ButlerStandardizer"] * 4,
                "config": ["{}"] * 4,
                "filter": ["r", "r", "r", "r"],
                "obs_code": ["I11", "I11", "I11", "I11"],
                "wcs": [wcs_str] * 4,
                "obs_lon": [-70.7494] * 4,
                "obs_lat": [-30.2444] * 4,
                "obs_elev": [2663.0] * 4,
                "wcs_err": [1e-6] * 4,
                "band": ["r", "r", "r", "r"],
            }
        )
        ic_table.meta["n_stds"] = 4
        ic_table.write(self.ic_filename, format="ascii.ecsv", overwrite=True)

        # Create dummy ephem file (CSV is standard for ephems input in region_searcher?)
        # region_searcher: Table.read(known_objects_ephem)
        # It handles CSV usually.
        ephem_table = Table(
            {
                "ra": [0.0, 10.0],
                "dec": [0.0, 10.0],
                "RA": [0.0, 10.0],
                "Dec": [0.0, 10.0],
                "mjd_mid": [60000.0, 60000.0],
                "Clean Name": ["Obj1", "Obj2"],
            }
        )
        ephem_table.write(self.ephem_filename, format="ascii.csv", overwrite=True)

    def tearDown(self):
        if os.path.exists(self.ic_filename):
            os.remove(self.ic_filename)
        if os.path.exists(self.ephem_filename):
            os.remove(self.ephem_filename)

    def test_region_searcher_with_filtering(self):
        import kbmod_cmdline.region_searcher

        # We still mock RegionSearch to avoid heavy computation and patch generation
        mock_rs_cls = MagicMock()
        mock_rs_instance = MagicMock()
        mock_rs_cls.return_value = mock_rs_instance
        mock_rs_instance.match_ic_to_patches.return_value = {0, 1}
        mock_rs_instance.search_patches_within_radius.return_value = {0}
        mock_rs_instance.search_patches_by_ephems.return_value = ({0}, {"Obj1": {0}})

        # Mock ImageCollection class ONLY to intercept reflex_correct if needed?
        # Actually RegionSearch constructor takes 'ic'.
        # region_searcher loads ic -> passes to RegionSearch.
        # So we don't need to mock ImageCollection read anymore!
        # But we might want to mock the 'ic' object to avoid errors in RegionSearch init?
        # No, let's let it be a real Table.

        with patch.object(kbmod_cmdline.region_searcher, "RegionSearch", mock_rs_cls):

            region_searcher(
                ic_path=self.ic_filename,
                guess_distance=0.0,  # Use 0.0 to skip reflex correction logic requirement on columns
                site_name="Rubin",
                patch_size=10,
                patch_overlap_percentage=0.0,
                pixel_scale=0.2,
                bands_to_drop=[],
                max_wcs_err=0.2,
                out_dir="test_out_dir",
                known_objects_ephem=self.ephem_filename,
                search_radius=1.5,
                overwrite=False,
                no_generate=True,
            )

            mock_rs_instance.search_patches_within_radius.assert_called_once()
            args, _ = mock_rs_instance.search_patches_within_radius.call_args
            self.assertAlmostEqual(args[1], 1.5)

    def test_region_searcher_no_filtering(self):
        import kbmod_cmdline.region_searcher

        mock_rs_cls = MagicMock()
        mock_rs_instance = MagicMock()
        mock_rs_cls.return_value = mock_rs_instance
        mock_rs_instance.match_ic_to_patches.return_value = {0, 1}

        with patch.object(kbmod_cmdline.region_searcher, "RegionSearch", mock_rs_cls):

            region_searcher(
                ic_path=self.ic_filename,
                guess_distance=0.0,
                site_name="Rubin",
                patch_size=10,
                patch_overlap_percentage=0.0,
                pixel_scale=0.2,
                bands_to_drop=[],
                max_wcs_err=0.2,
                out_dir="test_out_dir",
                known_objects_ephem=None,
                search_radius=None,
                overwrite=False,
                no_generate=True,
            )
            mock_rs_instance.search_patches_within_radius.assert_not_called()


if __name__ == "__main__":
    unittest.main()
