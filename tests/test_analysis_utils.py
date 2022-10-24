import unittest

from analysis_utils import *
from kbmod import *

class test_analysis_utils(unittest.TestCase):
    def setUp(self):
        # The configuration parameters.
        self.default_mask_bits_dict = {
            "BAD": 0,
            "CLIPPED": 9,
            "CR": 3,
            "CROSSTALK": 10,
            "DETECTED": 5,
            "DETECTED_NEGATIVE": 6,
            "EDGE": 4,
            "INEXACT_PSF": 11,
            "INTRP": 2,
            "NOT_DEBLENDED": 12,
            "NO_DATA": 8,
            "REJECTED": 13,
            "SAT": 1,
            "SENSOR_EDGE": 14,
            "SUSPECT": 7,
            "UNMASKEDNAN": 15,
        }
        self.default_flag_keys = ["BAD", "EDGE", "NO_DATA", "SUSPECT", "UNMASKEDNAN"]
        self.default_repeated_flag_keys = []
        self.config = {  # Mandatory values
            "im_filepath": None,
            "res_filepath": None,
            "time_file": None,
            # Suggested values
            "v_arr": [92.0, 526.0, 256],
            "ang_arr": [3.1415 / 15, 3.1415 / 15, 128],
            # Optional values
            "output_suffix": "search",
            "mjd_lims": None,
            "average_angle": None,
            "do_mask": True,
            "mask_num_images": 2,
            "mask_threshold": None,
            "mask_grow": 10,
            "lh_level": 10.0,
            "psf_val": 1.4,
            "num_obs": 10,
            "num_cores": 1,
            "visit_in_filename": [0, 6],
            "file_format": "{0:06d}.fits",
            "sigmaG_lims": [25, 75],
            "chunk_size": 500000,
            "max_lh": 1000.0,
            "filter_type": "clipped_sigmaG",
            "center_thresh": 0.00,
            "peak_offset": [2.0, 2.0],
            "mom_lims": [35.5, 35.5, 2.0, 0.3, 0.3],
            "stamp_type": "sum",
            "stamp_radius": 10,
            "eps": 0.03,
            "gpu_filter": False,
            "do_clustering": True,
            "do_stamp_filter": True,
            "clip_negative": False,
            "sigmaG_filter_type": "lh",
            "cluster_type": "all",
            "cluster_function": "DBSCAN",
            "mask_bits_dict": self.default_mask_bits_dict,
            "flag_keys": self.default_flag_keys,
            "repeated_flag_keys": self.default_repeated_flag_keys,
            "bary_dist": None,
            "encode_psi_bytes": -1,
            "encode_phi_bytes": -1,
            "known_obj_thresh": None,
            "known_obj_jpl": False,
        }

        # image properties
        self.img_count = 10
        self.dim_x = 15
        self.dim_y = 20
        self.noise_level = 2.0
        self.variance = self.noise_level**2
        self.p = psf(1.0)

        # create image set with single moving object
        self.imlist = []
        for i in range(self.img_count):
            time = i / self.img_count
            im = layered_image(str(i), self.dim_x, self.dim_y, self.noise_level, self.variance, time, self.p)
            self.imlist.append(im)
        self.stack = image_stack(self.imlist)

    def test_per_image_mask(self):
        kb_post_process = PostProcess(self.config)
        
        # Set each mask pixel in a row to one masking reason.
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            msk = img.get_mask()
            for x in range(self.dim_x):
                msk.set_pixel(x, 3, 2**x)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_mask(msk)
            self.stack.set_single_image(i, img)

        # Mask with the default keys.
        kb_post_process.apply_mask(self.stack, mask_num_images=2, mask_threshold=None, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if y == 3 and (x == 0 or x == 4 or x == 7 or x == 8 or x == 15):
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

    def test_mask_threshold(self):
        kb_post_process = PostProcess(self.config)
        
        # Set one science pixel per image above the threshold
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            sci = img.get_science()
            sci.set_pixel(2 + i, 8, 501.0)
            sci.set_pixel(1 + i, 9, 499.0)
            
            # We need to reset the images because of how pybind handles pass by reference.
            img.set_science(sci)
            self.stack.set_single_image(i, img)

        # With default threshold (None) nothing should be masked.  
        kb_post_process.apply_mask(self.stack, mask_num_images=2, mask_threshold=None, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    self.assertTrue(sci.pixel_has_data(x, y))

        # With a threshold of 500 one pixel per image should be masked. 
        kb_post_process.apply_mask(self.stack, mask_num_images=2, mask_threshold=500, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if x == 2 + i and y == 8:
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

    def test_mask_grow(self):
        kb_post_process = PostProcess(self.config)
        
        # Set one science pixel per image above the threshold
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            sci = img.get_science()
            sci.set_pixel(2 + i, 8, 501.0)
            
            # We need to reset the images because of how pybind handles pass by reference.
            img.set_science(sci)
            self.stack.set_single_image(i, img)

        # With default threshold (None) nothing should be masked.  
        kb_post_process.apply_mask(self.stack, mask_num_images=2, mask_threshold=500, mask_grow=2)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    dist = abs(2 + i - x) + abs(y - 8)
                    self.assertEqual(sci.pixel_has_data(x, y), dist > 2)

    def test_global_mask(self):
        self.config["repeated_flag_keys"] = ["CR"]
        kb_post_process = PostProcess(self.config)
        
        # Set each mask pixel in a single row depending on the image number.
        for i in range(self.img_count):
            img = self.stack.get_single_image(i)
            msk = img.get_mask()
            for x in range(self.dim_x):
                if x >= i:
                    msk.set_pixel(x, 1, 8)

            # We need to reset the images because of how pybind handles pass by reference.
            img.set_mask(msk)
            self.stack.set_single_image(i, img)

        # Apply the global mask with mask_num_images=5 and check that we mask the correct pixels.
        kb_post_process.apply_mask(self.stack, mask_num_images=5, mask_threshold=None, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if y == 1 and x >= 4:
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))

        # Apply the global mask with mask_num_images=3 and check that we mask 2 more pixels.
        kb_post_process.apply_mask(self.stack, mask_num_images=3, mask_threshold=None, mask_grow=0)
        for i in range(self.img_count):
            sci = self.stack.get_single_image(i).get_science()
            for x in range(self.dim_x):
                for y in range(self.dim_y):
                    if y == 1 and x >= 2:
                        self.assertFalse(sci.pixel_has_data(x, y))
                    else:
                        self.assertTrue(sci.pixel_has_data(x, y))


if __name__ == "__main__":
    unittest.main()
