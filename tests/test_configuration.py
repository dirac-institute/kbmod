from astropy.io import fits
from astropy.table import Table
import os
import tempfile
import unittest
from pathlib import Path
from yaml import safe_load

from kbmod.configuration import SearchConfiguration


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

    def test_copy(self):
        d = {"im_filepath": "Here2", "encode_num_bytes": -1}
        config = SearchConfiguration.from_dict(d)

        # Create a copy and change values.
        config2 = config.copy()
        config2.set("im_filepath", "who knows?")
        config2.set("encode_num_bytes", 2000)
        self.assertEqual(config2["im_filepath"], "who knows?")
        self.assertEqual(config2["encode_num_bytes"], 2000)

        # Confirm the original configuration is unchanged.
        self.assertEqual(config["im_filepath"], "Here2")
        self.assertEqual(config["encode_num_bytes"], -1)

    def test_from_hdu(self):
        t = Table(
            [
                ["Here3"],
                ["7"],
                ["null"],
                ["[1, 2]"],
                ["{name: test_gen, p1: [1.0, 2.0], p2: 2.0}"],
            ],
            names=("im_filepath", "num_obs", "cluster_type", "ang_arr", "generator_config"),
        )
        hdu = fits.table_to_hdu(t)

        config = SearchConfiguration.from_hdu(hdu)
        self.assertEqual(config["im_filepath"], "Here3")
        self.assertEqual(config["num_obs"], 7)
        self.assertEqual(config["ang_arr"], [1, 2])
        self.assertEqual(config["generator_config"]["name"], "test_gen")
        self.assertEqual(config["generator_config"]["p1"], [1.0, 2.0])
        self.assertEqual(config["generator_config"]["p2"], 2.0)
        self.assertIsNone(config["cluster_type"])

    def test_to_hdu(self):
        d = {
            "im_filepath": "Here2",
            "num_obs": 5,
            "cluster_type": None,
            "do_clustering": False,
            "legacy_filename": "There",
            "res_filepath": "There",
            "generator_config": {"name": "test_gen", "p1": [1.0, 2.0], "p2": 2.0},
            "basic_array": [1.0, 2.0, 3.0],
        }
        config = SearchConfiguration.from_dict(d)
        hdu = config.to_hdu()

        self.assertEqual(hdu.data["im_filepath"][0], "Here2\n...")
        self.assertEqual(hdu.data["num_obs"][0], "5\n...")
        self.assertEqual(hdu.data["cluster_type"][0], "null\n...")
        self.assertEqual(hdu.data["legacy_filename"][0], "There\n...")
        self.assertEqual(hdu.data["res_filepath"][0], "There\n...")
        self.assertEqual(hdu.data["generator_config"][0], "{name: test_gen, p1: [1.0, 2.0], p2: 2.0}")
        self.assertEqual(hdu.data["basic_array"][0], "[1.0, 2.0, 3.0]")

    def test_to_yaml(self):
        d = {
            "im_filepath": "Here2",
            "num_obs": 5,
            "cluster_type": None,
            "do_clustering": False,
            "legacy_filename": "There",
            "generator_config": {"name": "test_gen", "p1": [1.0, 2.0], "p2": 2.0},
        }
        config = SearchConfiguration.from_dict(d)
        yaml_str = config.to_yaml()

        yaml_dict = safe_load(yaml_str)
        self.assertEqual(yaml_dict["im_filepath"], "Here2")
        self.assertEqual(yaml_dict["num_obs"], 5)
        self.assertEqual(yaml_dict["cluster_type"], None)
        self.assertEqual(yaml_dict["legacy_filename"], "There")
        self.assertEqual(yaml_dict["generator_config"]["name"], "test_gen")
        self.assertEqual(yaml_dict["generator_config"]["p1"], [1.0, 2.0])
        self.assertEqual(yaml_dict["generator_config"]["p2"], 2.0)

    def test_save_and_load_yaml(self):
        config = SearchConfiguration()
        num_defaults = len(config._params)

        # Overwrite some defaults.
        config.set("im_filepath", "Here")
        config.set("lh_level", 25.0)

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
            self.assertEqual(config2["lh_level"], 25.0)

    def test_save_and_load_fits(self):
        config = SearchConfiguration()
        num_defaults = len(config._params)

        # Overwrite some defaults.
        config.set("im_filepath", "Here2")
        config.set("lh_level", 25.0)

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
            self.assertEqual(config2["lh_level"], 25.0)

            # Check that we correctly parse Nones.
            self.assertIsNone(config2["legacy_filename"])


if __name__ == "__main__":
    unittest.main()
