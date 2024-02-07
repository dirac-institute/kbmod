from astropy.io import fits
from astropy.table import Table
import os
import tempfile
import unittest
from pathlib import Path
from yaml import safe_load

from kbmod.configuration import SearchConfiguration
from kbmod.search import StampParameters, StampType


class test_configuration(unittest.TestCase):
    def test_validate(self):
        config = SearchConfiguration()

        # Add the minimal parameter and check it passes.
        config.set("im_filepath", "Here")
        try:
            config.validate()
        except ValueError:
            self.fail("validate() raised ValueError.")

    def test_set(self):
        config = SearchConfiguration()
        self.assertIsNone(config["im_filepath"])
        self.assertEqual(config["encode_num_bytes"], -1)

        config.set("im_filepath", "Here")
        config.set("encode_num_bytes", 2)
        self.assertEqual(config["im_filepath"], "Here")
        self.assertEqual(config["encode_num_bytes"], 2)

        # The set should fail when using unknown parameters and strict checking.
        self.assertRaises(KeyError, config.set, "My_new_param", 100, strict=True)

    def set_multiple(self):
        config = SearchConfiguration()
        self.assertIsNone(config["im_filepath"])
        self.assertEqual(config["encode_num_bytes"], -1)

        d = {"im_filepath": "Here", "encode_num_bytes": 2}
        config.set_multiple(d)
        self.assertEqual(config["im_filepath"], "Here")
        self.assertEqual(config["encode_num_bytes"], 2)

    def test_from_dict(self):
        d = {"im_filepath": "Here2", "num_obs": 5}
        config = SearchConfiguration.from_dict(d)
        self.assertEqual(config["im_filepath"], "Here2")
        self.assertEqual(config["num_obs"], 5)

    def test_from_hdu(self):
        t = Table(
            [
                ["Here3"],
                ["7"],
                ["null"],
                [
                    "[1, 2]",
                ],
            ],
            names=("im_filepath", "num_obs", "cluster_type", "ang_arr"),
        )
        hdu = fits.table_to_hdu(t)

        config = SearchConfiguration.from_hdu(hdu)
        self.assertEqual(config["im_filepath"], "Here3")
        self.assertEqual(config["num_obs"], 7)
        self.assertEqual(config["ang_arr"], [1, 2])
        self.assertIsNone(config["cluster_type"])

    def test_to_hdu(self):
        d = {
            "im_filepath": "Here2",
            "num_obs": 5,
            "cluster_type": None,
            "mask_bits_dict": {"bit1": 1, "bit2": 2},
            "do_clustering": False,
            "res_filepath": "There",
            "ang_arr": [1.0, 2.0, 3.0],
        }
        config = SearchConfiguration.from_dict(d)
        hdu = config.to_hdu()

        self.assertEqual(hdu.data["im_filepath"][0], "Here2\n...")
        self.assertEqual(hdu.data["num_obs"][0], "5\n...")
        self.assertEqual(hdu.data["cluster_type"][0], "null\n...")
        self.assertEqual(hdu.data["mask_bits_dict"][0], "{bit1: 1, bit2: 2}")
        self.assertEqual(hdu.data["res_filepath"][0], "There\n...")
        self.assertEqual(hdu.data["ang_arr"][0], "[1.0, 2.0, 3.0]")

    def test_to_yaml(self):
        d = {
            "im_filepath": "Here2",
            "num_obs": 5,
            "cluster_type": None,
            "mask_bits_dict": {"bit1": 1, "bit2": 2},
            "do_clustering": False,
            "res_filepath": "There",
            "ang_arr": [1.0, 2.0, 3.0],
        }
        config = SearchConfiguration.from_dict(d)
        yaml_str = config.to_yaml()

        yaml_dict = safe_load(yaml_str)
        self.assertEqual(yaml_dict["im_filepath"], "Here2")
        self.assertEqual(yaml_dict["num_obs"], 5)
        self.assertEqual(yaml_dict["cluster_type"], None)
        self.assertEqual(yaml_dict["mask_bits_dict"]["bit1"], 1)
        self.assertEqual(yaml_dict["mask_bits_dict"]["bit2"], 2)
        self.assertEqual(yaml_dict["res_filepath"], "There")
        self.assertEqual(yaml_dict["ang_arr"][0], 1.0)
        self.assertEqual(yaml_dict["ang_arr"][1], 2.0)
        self.assertEqual(yaml_dict["ang_arr"][2], 3.0)

    def test_save_and_load_yaml(self):
        config = SearchConfiguration()
        num_defaults = len(config._params)

        # Overwrite some defaults.
        config.set("im_filepath", "Here")
        config.set("output_suffix", "txt")
        config.set("mask_grow", 5)

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "tmp_config_data.yaml")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(FileNotFoundError, SearchConfiguration.from_file, file_path)

            # Correctly saves file.
            config.to_file(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Correctly loads file.
            try:
                config2 = SearchConfiguration.from_file(file_path)
            except ValueError:
                self.fail("load_configuration() raised ValueError.")

            self.assertEqual(len(config2._params), num_defaults)
            self.assertEqual(config2["im_filepath"], "Here")
            self.assertEqual(config2["res_filepath"], None)
            self.assertEqual(config2["mask_grow"], 5)
            self.assertEqual(config2["output_suffix"], "txt")

    def test_save_and_load_fits(self):
        config = SearchConfiguration()
        num_defaults = len(config._params)

        # Overwrite some defaults.
        config.set("im_filepath", "Here2")
        config.set("output_suffix", "csv")
        config.set("mask_grow", 7)
        config.set("mask_bits_dict", {"bit1": 1, "bit2": 2})

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = os.path.join(dir_name, "test.fits")
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(FileNotFoundError, SearchConfiguration.from_file, file_path)

            # Generate empty data for the first two tables and config for the third.
            hdu0 = fits.PrimaryHDU()
            hdu1 = fits.ImageHDU()
            hdu_list = fits.HDUList([hdu0, hdu1, config.to_hdu()])
            hdu_list.writeto(file_path)

            # Correctly loads file.
            config2 = SearchConfiguration()
            with fits.open(file_path) as ff:
                config2 = SearchConfiguration.from_hdu(ff[2])

            self.assertEqual(len(config2._params), num_defaults)
            self.assertEqual(config2["im_filepath"], "Here2")
            self.assertEqual(config2["mask_grow"], 7)
            self.assertEqual(config2["output_suffix"], "csv")

            # Check that we correctly parse dictionaries and Nones.
            self.assertEqual(len(config2["mask_bits_dict"]), 2)
            self.assertIsNone(config2["res_filepath"])


if __name__ == "__main__":
    unittest.main()
