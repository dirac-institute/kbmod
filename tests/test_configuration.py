import tempfile
import unittest

from kbmod.configuration import *
from pathlib import Path


class test_configuration(unittest.TestCase):
    def setUp(self):
        self.min_params = {"im_filepath": "Here", "res_filepath": "There"}

    def test_check_required(self):
        loader = ConfigLoader()

        # Check that is passes without all the required parameters.
        try:
            loader.check_required(self.min_params)
        except ValueError:
            self.fail("check_required() raised ValueError.")

        # Check that it fails with missing parameters.
        self.assertRaises(ValueError, loader.check_required, {"im_filepath": "Somewhere"})
        self.assertRaises(ValueError, loader.check_required, {"res_filepath": "Somewhere"})

    def test_filtering(self):
        loader = ConfigLoader()

        # Check filter doesn't remove required or defaults.
        self.assertEqual(self.min_params, loader.filter_unused(self.min_params))

        good_params = {"do_mask": True, "mask_num_images": 10}
        self.assertEqual(good_params, loader.filter_unused(good_params, verbose=False))

        # Check that it does remove unrecognized params.
        mixed_params = {"mask_num_images": 5, "unused_test_param": 10}
        filtered = loader.filter_unused(mixed_params, verbose=False)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered["mask_num_images"], 5)
        self.assertFalse("unused_test_param" in filtered)

    def test_merge_with_default(self):
        loader = ConfigLoader()
        num_defaults = len(loader._default_params)

        # Merge in an empty dictionary.
        defaults = loader.merge_defaults({}, verbose=False)
        self.assertEqual(len(defaults), num_defaults)

        # Merge in a dictionary with no extra values.
        params = {"mask_num_images": 5, "time_file": "here.txt"}
        merged = loader.merge_defaults(params, verbose=False)
        self.assertEqual(len(merged), num_defaults)
        self.assertEqual(merged["mask_num_images"], 5)
        self.assertEqual(merged["time_file"], "here.txt")

        # Merge in a file with non-default keys.
        merged2 = loader.merge_defaults(self.min_params, verbose=False)
        self.assertEqual(len(merged2), num_defaults + len(self.min_params))
        self.assertEqual(merged2["im_filepath"], self.min_params["im_filepath"])
        self.assertEqual(merged2["res_filepath"], self.min_params["res_filepath"])

    def test_save_and_load(self):
        loader = ConfigLoader()
        num_defaults = len(loader._default_params)

        # Add extra valid parameters and a single invalid parameter.
        self.min_params["mask_num_images"] = 15
        self.min_params["unused_test_param"] = 10
        self.min_params["v_arr"] = [1.0, 2.0, 10]

        with tempfile.TemporaryDirectory() as dir_name:
            file_path = f"{dir_name}/tmp_config_data.cfg"
            self.assertFalse(Path(file_path).is_file())

            # Unable to load non-existent file.
            self.assertRaises(ValueError, loader.load_configuration, file_path)

            # Correctly saves file.
            loader.save_configuration(self.min_params, file_path)
            self.assertTrue(Path(file_path).is_file())

            # Correctly loads file.
            try:
                loaded_params = loader.load_configuration(file_path, verbose=False)
            except ValueError:
                self.fail("load_configuration() raised ValueError.")
            self.assertGreater(len(loaded_params), num_defaults)
            self.assertEqual(loaded_params["im_filepath"], self.min_params["im_filepath"])
            self.assertEqual(loaded_params["res_filepath"], self.min_params["res_filepath"])
            self.assertEqual(loaded_params["mask_num_images"], 15)
            self.assertEqual(loaded_params["v_arr"], [1.0, 2.0, 10])
            self.assertFalse("unused_test_param" in loaded_params)


if __name__ == "__main__":
    unittest.main()
