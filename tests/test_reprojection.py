import os
import unittest
import numpy as np
import numpy.testing as npt
from utils.utils_for_tests import get_absolute_data_path
import tempfile

from kbmod.core.image_stack_py import ImageStackPy
from kbmod.reprojection import (
    reproject_work_unit,
    reproject_lazy_work_unit,
    _effective_psfs_for_indices,
    _normalized_cross_correlation,
    _validate_original_wcs,
)
from kbmod.reprojection_config import LEGACY_CONFIG
from kbmod.search import pixel_value_valid
from kbmod.work_unit import WorkUnit, LEGACY_PSF_SOURCE


class test_reprojection(unittest.TestCase):
    def setUp(self):
        self.data_path = get_absolute_data_path("shifted_wcs_diff_dimms_tiled.fits")
        self.test_wunit = WorkUnit.from_fits(self.data_path, show_progress=False)
        self.common_wcs = self.test_wunit.get_wcs(0)

        # Set the data_loc metadata to make sure it propagates correctly.
        self.num_org_images = len(self.test_wunit.im_stack)
        self.data_locs = [f"test_data_loc_{i}" for i in range(self.num_org_images)]
        self.test_wunit.org_img_meta["data_loc"] = self.data_locs

    def test_reproject(self):
        # test exception conditions
        self.assertRaises(
            ValueError,
            reproject_work_unit,
            work_unit=self.test_wunit,
            common_wcs=self.common_wcs,
            write_output=True,
            show_progress=False,
        )

        self.test_wunit.lazy = True
        self.assertRaises(
            ValueError,
            reproject_work_unit,
            work_unit=self.test_wunit,
            common_wcs=self.common_wcs,
            show_progress=False,
        )
        self.test_wunit.lazy = False

        test_conditions = [
            (True, False, True),
            (True, False, True),
            (True, True, True),
            (False, False, True),
            (False, False, False),
        ]
        for parallelize, lazy, write_out in test_conditions:
            with self.subTest(parallelize=parallelize, lazy=lazy, write_out=write_out):
                if write_out:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        if lazy:
                            self.test_wunit.to_sharded_fits("test_wunit.fits", tmpdir)
                            wunit = WorkUnit.from_sharded_fits("test_wunit.fits", tmpdir, lazy=True)
                        else:
                            wunit = self.test_wunit
                        reproject_work_unit(
                            wunit,
                            self.common_wcs,
                            "original",
                            parallelize=parallelize,
                            write_output=write_out,
                            directory=tmpdir,
                            filename="repr_wu.fits",
                            show_progress=False,
                        )
                        reprojected_wunit = WorkUnit.from_sharded_fits("repr_wu.fits", tmpdir)
                else:
                    reprojected_wunit = reproject_work_unit(
                        self.test_wunit,
                        self.common_wcs,
                        parallelize=parallelize,
                        show_progress=False,
                    )

                assert reprojected_wunit.wcs != None
                assert reprojected_wunit.im_stack.width == 60
                assert reprojected_wunit.im_stack.height == 50

                test_dists = self.test_wunit.get_constituent_meta("geocentric_distance")
                reproject_dists = reprojected_wunit.get_constituent_meta("geocentric_distance")
                assert test_dists == reproject_dists

                # Make sure the data_loc metadata is propagated correctly.
                loaded_data_locs = reprojected_wunit.get_constituent_meta("data_loc")
                for i in range(self.num_org_images):
                    assert loaded_data_locs[i] == self.data_locs[i]

                # will be 3 as opposed to the four in the original `WorkUnit`,
                # as the last two images have the same obstime and therefore
                # get condensed to one image.
                assert len(reprojected_wunit.im_stack) == 3
                data = [
                    [
                        reprojected_wunit.im_stack.sci[i],
                        reprojected_wunit.im_stack.var[i],
                        reprojected_wunit.im_stack.get_mask(i),
                    ]
                    for i in range(3)
                ]

                for img in data:
                    # test that mask values are binary
                    assert np.all(np.array(img[2] == 1.0) | np.array(img[2] == 0.0))

                test_vals = np.array(
                    [
                        115.519264,
                        94.1921,
                        114.12677,
                        4.0,
                        1.0,
                    ]
                ).astype("float32")

                # Make sure the PSF for the object hasn't been warped in the no-op case.
                # We allow a little error in case the result is compressed as it is written
                # to a file.
                self.assertAlmostEqual(data[0][0][5][53], test_vals[0], delta=0.05)

                # test other object locations
                self.assertAlmostEqual(data[1][0][30][36], test_vals[1], delta=0.05)
                self.assertAlmostEqual(data[2][0][4][18], test_vals[2], delta=0.05)

                # test variance
                assert not pixel_value_valid(data[2][1][25][0])
                self.assertAlmostEqual(data[2][1][25][9], test_vals[3], delta=0.05)

                # test that mask values are projected without interpolation/bleeding
                assert len(data[2][2][36][data[2][2][36] == 1.0]) == 9
                assert len(data[2][2][34][data[2][2][34] == 1.0]) == 9

                assert len(reprojected_wunit._per_image_indices) == 3
                assert reprojected_wunit._per_image_indices[2] == [2, 3]

    def test_except_add_overlapping_images(self):
        """Make sure that the reprojection fails when images at the same time
        have overlapping pixels."""
        new_times = np.copy(self.test_wunit.im_stack.times)
        new_times[1] = new_times[0]
        new_stack = ImageStackPy(
            new_times,
            self.test_wunit.im_stack.sci,
            self.test_wunit.im_stack.var,
            psfs=self.test_wunit.im_stack.psfs,
        )
        self.test_wunit.im_stack = new_stack

        for parallelize in [True, False]:
            with self.subTest(parallelize=parallelize):
                try:
                    reproject_work_unit(
                        self.test_wunit,
                        self.common_wcs,
                        parallelize=parallelize,
                        show_progress=False,
                    )
                except ValueError as e:
                    assert str(e) == "Images with the same obstime are overlapping."

    @staticmethod
    def _distinct_psf_kernels():
        """Three deliberately distinct, normalized 3x3 PSF kernels.

        They differ in peak position and asymmetry, not merely in scale, so that
        selecting the wrong one cannot be masked by a loose numerical tolerance.
        The shipped reprojection fixture stores four *identical* PSFs, which makes
        a wrong-PSF selection undetectable; these replace them.
        """
        k0 = np.array([[0.5, 0.2, 0.0], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        k1 = np.array([[0.0, 0.2, 0.0], [0.2, 0.2, 0.2], [0.0, 0.2, 0.0]], dtype=np.float32)
        k2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, 0.2, 0.5]], dtype=np.float32)
        kernels = [k0 / k0.sum(), k1 / k1.sum(), k2 / k2.sum()]

        # Validate that the PSFs are distinct.
        for a in range(len(kernels)):
            for b in range(a + 1, len(kernels)):
                assert not np.allclose(kernels[a], kernels[b])
        return kernels

    def test_reproject_selects_psf_of_matching_time(self):
        """Each reprojected image must carry the PSF derived from *its own* obstime.

        Regression test for the leaked-loop-variable defect in the parallel
        reprojection path, where the PSF was looked up with the final value of
        the submission loop's `obstime` rather than the time of the result being
        processed, giving every output image the last obstime's PSF.

        Since Phase 2 the stored kernel is the *effective* PSF -- the native
        model resampled through the same operator as the science image -- so the
        expectation is computed per time rather than compared against the native
        kernel. The test still discriminates: it first asserts the resampled
        kernels remain mutually distinguishable, so matching the right one is a
        real constraint.

        Images that share an obstime are deliberately given the same kernel so
        that this isolates the wrong-time defect from the separate question of
        which of several same-time constituent PSFs should be chosen.
        """
        unique_times, unique_indices = self.test_wunit.get_unique_obstimes_and_indices()
        kernels = self._distinct_psf_kernels()
        self.assertEqual(len(unique_times), len(kernels))

        wcs_list = self.test_wunit.get_constituent_meta("per_image_wcs")

        # Assign one distinct kernel per unique time; images sharing a time share a kernel.
        for kernel, indices in zip(kernels, unique_indices):
            for i in indices:
                self.test_wunit.im_stack.psfs[i] = kernel.copy()

        # The expected effective PSF for each time, generated the same way the
        # reprojection does, from that time's first usable constituent.
        expected_by_time = {}
        for kernel, time, indices in zip(kernels, unique_times, unique_indices):
            effective, _, _ = _effective_psfs_for_indices(
                [self.test_wunit.im_stack.psfs[i] for i in indices],
                [wcs_list[i] for i in indices],
                self.common_wcs,
                LEGACY_CONFIG,
                time,
                list(indices),
            )
            expected_by_time[float(time)] = effective[0].kernel

        # Guard the guard: resampling must not have collapsed the kernels into
        # lookalikes, or picking the wrong one would go undetected.
        resampled = list(expected_by_time.values())
        for a in range(len(resampled)):
            for b in range(a + 1, len(resampled)):
                self.assertLess(
                    _normalized_cross_correlation(resampled[a], resampled[b]),
                    0.99,
                    "resampled kernels are too similar for this test to discriminate",
                )

        for parallelize in [False, True]:
            with self.subTest(parallelize=parallelize):
                reprojected = reproject_work_unit(
                    self.test_wunit,
                    self.common_wcs,
                    parallelize=parallelize,
                    show_progress=False,
                )

                self.assertEqual(len(reprojected.im_stack), len(unique_times))
                for i, time in enumerate(reprojected.im_stack.times):
                    expected = expected_by_time[float(time)]
                    actual_kernel = reprojected.im_stack.psfs[i]
                    self.assertEqual(
                        actual_kernel.shape,
                        expected.shape,
                        f"image {i} at obstime {time} has the wrong kernel shape "
                        f"(parallelize={parallelize})",
                    )
                    np.testing.assert_allclose(
                        actual_kernel,
                        expected,
                        rtol=1e-5,
                        atol=1e-7,
                        err_msg=(
                            f"image {i} at obstime {time} carries the wrong PSF "
                            f"(parallelize={parallelize})"
                        ),
                    )

    def test_reprojection_provenance_is_recorded_and_round_trips(self):
        """A stored effective kernel must be traceable to the operator that made it.

        Provenance is what keeps a legacy native kernel from later being
        mistaken for an effective common-frame PSF. Its absence on an old file
        is meaningful, so that case is asserted too.
        """
        # A legacy file is explicitly labeled, not left silently empty.
        self.assertEqual(self.test_wunit.reprojection_provenance, {"psf_source": LEGACY_PSF_SOURCE})

        reprojected = reproject_work_unit(
            self.test_wunit,
            self.common_wcs,
            parallelize=False,
            show_progress=False,
            reprojection_config=LEGACY_CONFIG,
        )
        provenance = reprojected.reprojection_provenance
        self.assertEqual(provenance["psf_source"], "effective")
        self.assertEqual(provenance["preset"], "legacy")
        self.assertEqual(provenance["config_hash"], LEGACY_CONFIG.hexdigest)
        self.assertEqual(provenance["reproject_version"], LEGACY_CONFIG.provenance["reproject_version"])

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "provenance.fits")
            reprojected.to_fits(path)
            reloaded = WorkUnit.from_fits(path, show_progress=False)
        self.assertEqual(reloaded.reprojection_provenance, provenance)

    def test_configuration_reaches_science_and_psf_alike(self):
        """The supplied operator must be applied to the pixels, not just the PSF.

        Negative control for the project's central invariant: science and
        effective-PSF reprojection use identical options. A version that plumbed
        the config into the PSF path only still produced a WorkUnit whose header
        claimed a custom preset, while the science planes were resampled with
        the legacy operator -- provenance that was confidently false.

        The test fails if *either* side silently falls back, so it cannot be
        satisfied by fixing one of them.
        """
        custom = LEGACY_CONFIG.evolve(kernel_width=2.8, sample_region_width=7.0)

        def run(config):
            work_unit = WorkUnit.from_fits(self.data_path, show_progress=False)
            work_unit.org_img_meta["data_loc"] = self.data_locs
            return reproject_work_unit(
                work_unit,
                self.common_wcs,
                parallelize=False,
                show_progress=False,
                reprojection_config=config,
            )

        legacy_result = run(LEGACY_CONFIG)
        custom_result = run(custom)

        # The science planes must actually differ under a different operator.
        science_differences = [
            float(
                np.nanmax(
                    np.abs(
                        np.nan_to_num(legacy_result.im_stack.sci[i])
                        - np.nan_to_num(custom_result.im_stack.sci[i])
                    )
                )
            )
            for i in range(len(legacy_result.im_stack))
        ]
        self.assertGreater(
            max(science_differences),
            0.0,
            "science planes are identical under different operators: the configuration "
            "never reached reproject_image, so recorded provenance would be false",
        )

        # ... and so must the PSFs.
        psf_differs = any(
            legacy_result.im_stack.psfs[i].shape != custom_result.im_stack.psfs[i].shape
            or not np.allclose(legacy_result.im_stack.psfs[i], custom_result.im_stack.psfs[i])
            for i in range(len(legacy_result.im_stack))
        )
        self.assertTrue(psf_differs, "the configuration never reached the PSF path")

        # The recorded provenance must describe the operator that was used.
        self.assertEqual(custom_result.reprojection_provenance["config_hash"], custom.hexdigest)
        self.assertEqual(legacy_result.reprojection_provenance["config_hash"], LEGACY_CONFIG.hexdigest)

    def test_all_execution_paths_agree_under_a_non_default_config(self):
        """Serial, parallel, write-read, and lazy must produce the same result.

        Every WorkUnit test previously used the default configuration, so a
        defect that dropped the config on the science path changed nothing and
        stayed invisible. This test drives all four paths with a non-default
        operator and compares science, PSF, times, and provenance.

        Note what this does *not* prove: if the configuration were dropped on
        every path alike, all four would still agree here. Establishing that the
        operator is actually applied is the job of
        `test_configuration_reaches_science_and_psf_alike`, which compares two
        different configurations against each other. The two tests are
        complementary and both are needed.
        """
        custom = LEGACY_CONFIG.evolve(kernel_width=2.4, sample_region_width=6.0)

        def build():
            work_unit = WorkUnit.from_fits(self.data_path, show_progress=False)
            work_unit.org_img_meta["data_loc"] = self.data_locs
            return work_unit

        results = {}
        results["serial"] = reproject_work_unit(
            build(),
            self.common_wcs,
            parallelize=False,
            show_progress=False,
            reprojection_config=custom,
        )
        results["parallel"] = reproject_work_unit(
            build(),
            self.common_wcs,
            parallelize=True,
            show_progress=False,
            reprojection_config=custom,
        )

        with tempfile.TemporaryDirectory() as directory:
            reproject_work_unit(
                build(),
                self.common_wcs,
                parallelize=False,
                show_progress=False,
                write_output=True,
                directory=directory,
                filename="written.fits",
                reprojection_config=custom,
            )
            results["write_read"] = WorkUnit.from_sharded_fits("written.fits", directory)

        with tempfile.TemporaryDirectory() as directory:
            build().to_sharded_fits("lazy_in.fits", directory)
            lazy = WorkUnit.from_sharded_fits("lazy_in.fits", directory, lazy=True)
            reproject_lazy_work_unit(
                lazy,
                self.common_wcs,
                directory,
                "lazy_out.fits",
                show_progress=False,
                reprojection_config=custom,
            )
            results["lazy"] = WorkUnit.from_sharded_fits("lazy_out.fits", directory)

        reference = results["serial"]
        for name, result in results.items():
            with self.subTest(path=name):
                self.assertEqual(len(result.im_stack), len(reference.im_stack))
                npt.assert_allclose(result.im_stack.times, reference.im_stack.times)

                for i in range(len(reference.im_stack)):
                    npt.assert_allclose(
                        np.nan_to_num(result.im_stack.sci[i]),
                        np.nan_to_num(reference.im_stack.sci[i]),
                        rtol=1e-5,
                        atol=1e-6,
                        err_msg=f"science differs on the {name} path",
                    )
                    self.assertEqual(
                        result.im_stack.psfs[i].shape,
                        reference.im_stack.psfs[i].shape,
                        f"PSF shape differs on the {name} path",
                    )
                    npt.assert_allclose(
                        result.im_stack.psfs[i],
                        reference.im_stack.psfs[i],
                        rtol=1e-5,
                        atol=1e-7,
                        err_msg=f"PSF differs on the {name} path",
                    )

                # Provenance must describe the operator actually used, on every path.
                self.assertEqual(
                    result.reprojection_provenance.get("config_hash"),
                    custom.hexdigest,
                    f"provenance missing or wrong on the {name} path",
                )
                self.assertEqual(result.reprojection_provenance.get("psf_source"), "effective")
                self.assertEqual(result.reprojection_frame, "original")

    def test_validate_original_wcs(self):
        """Make sure that the original WCS is validated correctly."""
        wcs = self.test_wunit.get_wcs(0)
        _wcs = _validate_original_wcs(self.test_wunit, [0])
        assert np.all(wcs.pixel_scale_matrix == _wcs[0].pixel_scale_matrix)


if __name__ == "__main__":
    unittest.main()
