from astropy.io import fits
from astropy.table import Table
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(config["encode_psi_bytes"], -1)

        config.set("im_filepath", "Here")
        config.set("encode_psi_bytes", 2)
        self.assertEqual(config["im_filepath"], "Here")
        self.assertEqual(config["encode_psi_bytes"], 2)

        # The set should fail when using unknown parameters and strict checking.
        self.assertRaises(KeyError, config.set, "My_new_param", 100, strict=True)

    def test_set_from_dict(self):
        # Everything starts at its default.
        config = SearchConfiguration()
        self.assertIsNone(config["im_filepath"])
        self.assertEqual(config["num_obs"], 10)

        d = {"im_filepath": "Here2", "num_obs": 5}
        config.set_from_dict(d)
        self.assertEqual(config["im_filepath"], "Here2")
        self.assertEqual(config["num_obs"], 5)

    def test_set_from_table(self):
        # Everything starts at its default.
        config = SearchConfiguration()
        self.assertIsNone(config["im_filepath"])
        self.assertEqual(config["num_obs"], 10)

        t = Table([["Here3"], [7]], names=("im_filepath", "num_obs"))
        config.set_from_table(t)
        self.assertEqual(config["im_filepath"], "Here3")
        self.assertEqual(config["num_obs"], 7)

    def test_to_table(self):
        # Everything starts at its default.
        config = SearchConfiguration()
        d = {"im_filepath": "Here2", "num_obs": 5}
        config.set_from_dict(d)

        t = config.to_table()
        self.assertEqual(len(t), 1)
        self.assertEqual(t["im_filepath"][0], "Here2")
        self.assertEqual(t["num_obs"][0], 5)

    def test_save_and_load_yaml(self):
        config = SearchConfiguration()
        num_defaults = len(config._params)

        # Overwrite some defaults.
        config.set("im_filepath", "Here")
        config.set("output_suffix", "txt")
        config.set("mask_grow", 5)

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = f"{dir_name}/tmp_config_data.cfg"
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            config2 = SearchConfiguration()
            self.assertRaises(ValueError, config2.load_from_yaml_file, file_path)

            # Correctly saves file.
            config.save_to_yaml_file(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Correctly loads file.
            try:
                config2.load_from_yaml_file(file_path)
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
        print(config)

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = f"{dir_name}/test.fits"
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            config2 = SearchConfiguration()
            self.assertRaises(ValueError, config2.load_from_fits_file, file_path)

            # Generate measningless data for table 0 and the configuration for table 1.
            t0 = Table([[1] * 10, [2] * 10, [3] * 10], names=("A", "B", "C"))
            t0.write(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Append the FITS data to extension=1
            config.append_to_fits(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Correctly loads file.
            try:
                config2.load_from_fits_file(file_path, layer=2)
            except ValueError:
                self.fail("load_from_fits_file() raised ValueError.")

            self.assertEqual(len(config2._params), num_defaults)
            self.assertEqual(config2["im_filepath"], "Here2")
            self.assertEqual(config2["mask_grow"], 7)
            self.assertEqual(config2["output_suffix"], "csv")

            # Check that we correctly parse dictionaries and Nones.
            self.assertEqual(len(config2["mask_bits_dict"]), 2)
            self.assertIsNone(config2["res_filepath"])


if __name__ == "__main__":
    unittest.main()
