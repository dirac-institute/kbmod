import tempfile
import unittest

from kbmod.configuration import KBMODConfig
from pathlib import Path


class test_configuration(unittest.TestCase):
    def test_validate(self):
        config = KBMODConfig()

        # Without the im_filepath, validate raises an error.
        self.assertRaises(ValueError, config.validate)

        # Add the minimal parameter and check it passes.
        config.set("im_filepath", "Here")
        try:
            config.validate()
        except ValueError:
            self.fail("validate() raised ValueError.")

    def test_setting(self):
        config = KBMODConfig()
        self.assertIsNone(config["im_filepath"])
        self.assertEqual(config["encode_psi_bytes"], -1)

        config.set("im_filepath", "Here")
        config.set("encode_psi_bytes", 2)
        self.assertEqual(config["im_filepath"], "Here")
        self.assertEqual(config["encode_psi_bytes"], 2)

        # The set should fail when using unknown parameters and strict checking.
        self.assertRaises(KeyError, config.set, "My_new_param", 100, strict=True)

    def test_save_and_load(self):
        config = KBMODConfig()
        num_defaults = len(config._params)

        # Overwrite some defaults.
        config.set("im_filepath", "Here")
        config.set("output_suffix", "txt")
        config.set("mask_grow", 5)

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = f"{dir_name}/tmp_config_data.cfg"
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            config2 = KBMODConfig()
            self.assertRaises(ValueError, config2.load_from_file, file_path)

            # Correctly saves file.
            config.save_configuration(file_path)
            self.assertTrue(Path(file_path).is_file())

            # Correctly loads file.
            try:
                config2.load_from_file(file_path)
            except ValueError:
                self.fail("load_configuration() raised ValueError.")

            self.assertEqual(len(config2._params), num_defaults)
            self.assertEqual(config2["im_filepath"], "Here")
            self.assertEqual(config2["res_filepath"], None)
            self.assertEqual(config2["mask_grow"], 5)
            self.assertEqual(config2["output_suffix"], "txt")

if __name__ == "__main__":
    unittest.main()
