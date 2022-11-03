import unittest

from kbmod.analysis_utils import *
from kbmod.search import *

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
        
        # search parameters
        self.angle_steps = 10
        self.velocity_steps = 10
        self.min_angle = 0.0
        self.max_angle = 1.5
        self.min_vel = 5.0
        self.max_vel = 40.0

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
        
        # Set up old_results object for analysis_utils.PostProcess
        self.num_curves = 4
        # First 3 passing indices
        psi_curves = [np.array([1.0 + (x / 100) for x in range(20)]) for _ in range(self.num_curves - 1)]
        phi_curves = [np.array([1.0 + (y / 100) for y in range(20)]) for _ in range(self.num_curves - 1)]
        # Failing index
        # Failing index (generate a list of psi values such that the elements 2 and 14 are filtered
        # by sigmaG filtering.
        failing_psi = [0.0 + (z / 100) for z in range(20)]
        failing_psi[14] = -100.0
        failing_psi[2] = 100.0
        psi_curves.append(np.array(failing_psi))
        phi_curves.append(np.array([1.0 for _ in range(20)]))

        psi_good_indices = [z for z in range(20)]
        psi_good_indices.remove(14)
        psi_good_indices.remove(2)
        self.good_indices = np.array(psi_good_indices)
        # Original likelihood
        results = [1.0 for _ in range(self.num_curves)]

        self.old_results = {}
        self.old_results["psi_curves"] = psi_curves
        self.old_results["phi_curves"] = phi_curves
        self.old_results["results"] = results

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
                        
    def test_apply_clipped_average_single_thread(self):
        # make sure apply_clipped_average works when num_cores == 1
        kb_post_process = PostProcess(self.config)

        res = kb_post_process.apply_clipped_average(self.old_results, {})

        # check to ensure first three indices pass
        self.assertEqual(len(res), self.num_curves)
        for r in res[: self.num_curves - 1]:
            self.assertNotEqual(res[1][0], -1)
        # check to ensure that the last index fails
        self.assertEqual(len(res[self.num_curves - 1][1]), len(self.good_indices))
        for index in range(len(res[self.num_curves - 1][1])):
            self.assertEqual(res[self.num_curves - 1][1][index], self.good_indices[index])

    def test_apply_clipped_average_multi_thread(self):
        # make sure apply_clipped_average works when multithreading is enabled
        self.config["num_cores"] = 2
        kb_post_process = PostProcess(self.config)

        res = kb_post_process.apply_clipped_average(self.old_results, {})

        # check to ensure first three indices pass
        self.assertEqual(len(res), self.num_curves)
        for r in res[: self.num_curves - 1]:
            self.assertNotEqual(res[1][0], -1)
        # check to ensure that the last index fails
        self.assertEqual(len(res[self.num_curves - 1][1]), len(self.good_indices))
        for index in range(len(res[self.num_curves - 1][1])):
            self.assertEqual(res[self.num_curves - 1][1][index], self.good_indices[index])

    def test_apply_clipped_sigmaG_single_thread(self):
        # make sure apply_clipped_sigmaG works when num_cores == 1
        kb_post_process = PostProcess(self.config)

        res = kb_post_process.apply_clipped_sigmaG(self.old_results, {"sigmaG_filter_type": "lh"})

        # check to ensure first three indices pass
        self.assertEqual(len(res), self.num_curves)
        for r in res[: self.num_curves - 1]:
            self.assertNotEqual(res[1][0], -1)
        # check to ensure that the last index fails
        self.assertEqual(len(res[self.num_curves - 1][1]), len(self.good_indices))
        for index in range(len(res[self.num_curves - 1][1])):
            self.assertEqual(res[self.num_curves - 1][1][index], self.good_indices[index])

    def test_apply_clipped_sigmaG_multi_thread(self):
        # make sure apply_clipped_sigmaG works when multithreading is enabled
        self.config["num_cores"] = 2
        kb_post_process = PostProcess(self.config)

        res = kb_post_process.apply_clipped_sigmaG(self.old_results, {"sigmaG_filter_type": "lh"})

        # check to ensure first three indices pass
        self.assertEqual(len(res), self.num_curves)
        for r in res[: self.num_curves - 1]:
            self.assertNotEqual(res[1][0], -1)
        # check to ensure that the last index fails
        self.assertEqual(len(res[self.num_curves - 1][1]), len(self.good_indices))
        for index in range(len(res[self.num_curves - 1][1])):
            self.assertEqual(res[self.num_curves - 1][1][index], self.good_indices[index])
        
    def test_apply_stamp_filter_single_thread(self):
        # make sure apply_stamp_filter works when num_cores == 1
        
        # object properties
        self.object_flux = 250.0
        self.start_x = 4
        self.start_y = 3
        self.x_vel = 2.0
        self.y_vel = 1.0
        
        for i in range(self.img_count):
            time = i / self.img_count
            self.imlist[i].add_object(
                self.start_x + time * self.x_vel + 0.5,
                self.start_y + time * self.y_vel + 0.5,
                self.object_flux,
            )
        
        stack = image_stack(self.imlist)
        search = stack_search(stack)
        search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.img_count / 2),
        )

        kb_post_process = PostProcess(self.config)

        mjds = np.array(stack.get_times())
        keep = kb_post_process.load_results(
            search,
            mjds,
            {},
            self.config["lh_level"],
            chunk_size=self.config["chunk_size"],
            filter_type='kalman',
            max_lh=self.config["max_lh"],
        )

        res = kb_post_process.apply_stamp_filter(keep, search)

        self.assertIsNotNone(res["stamps"])
        self.assertIsNotNone(res["final_results"])

    def test_apply_stamp_filter_multi_thread(self):
        # make sure apply_stamp_filter works when multithreading is enabled
        
        # object properties
        self.object_flux = 250.0
        self.start_x = 4
        self.start_y = 3
        self.x_vel = 2.0
        self.y_vel = 1.0
        
        for i in range(self.img_count):
            time = i / self.img_count
            self.imlist[i].add_object(
                self.start_x + time * self.x_vel + 0.5,
                self.start_y + time * self.y_vel + 0.5,
                self.object_flux,
            )
        
        stack = image_stack(self.imlist)
        search = stack_search(stack)
        search.search(
            self.angle_steps,
            self.velocity_steps,
            self.min_angle,
            self.max_angle,
            self.min_vel,
            self.max_vel,
            int(self.img_count / 2),
        )

        self.config["num_cores"] = 2
        kb_post_process = PostProcess(self.config)

        mjds = np.array(stack.get_times())
        keep = kb_post_process.load_results(
            search,
            mjds,
            {},
            self.config["lh_level"],
            chunk_size=self.config["chunk_size"],
            filter_type="kalman",
            max_lh=self.config["max_lh"],
        )

        res = kb_post_process.apply_stamp_filter(keep, search)

        self.assertIsNotNone(res["stamps"])
        self.assertIsNotNone(res["final_results"])


if __name__ == "__main__":
    unittest.main()
