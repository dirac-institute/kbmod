import copy
import uuid
import unittest
from unittest import mock

from astropy.time import Time
import numpy as np

from utils import (
    DECamImdiffFactory,
    MockButler,
    MockFailedButler,
    DatasetRef,
    DatasetId,
    dafButler,
    lsstGeom,
    MockRubinPsf,
    BrokenRubinPsf,
    Point2D,
)
from kbmod import Standardizer, StandardizerConfig
from kbmod.core.psf import PSF
from kbmod.standardizers import ButlerStandardizer, ButlerStandardizerConfig, KBMODV1Config
from kbmod.standardizers.rubin_psf import RubinPsfError
from kbmod.psf_reprojection import measure_moments, summarize_shape

# Use a shared factory so that we can reference the same fits files in mocks
# and tests without having to untar the archive multiple times.
FitsFactory = DECamImdiffFactory()


@mock.patch.dict(
    "sys.modules",
    {
        "lsst.daf.butler": dafButler,
        "lsst.daf.butler.core.DatasetRef": DatasetRef,
        "lsst.daf.butler.core.DatasetId": DatasetId,
        "lsst.geom": lsstGeom,
    },
)
class TestButlerStandardizer(unittest.TestCase):
    """Test ButlerStandardizer."""

    def setUp(self):
        self.butler = MockButler("/far/far/away")
        self.failed_butler = MockFailedButler("futher/still")

    def test_init(self):
        """Test ButlerStandardizer can be built from DatasetRef, DatasetId and
        the dataset id."""
        # Just makes sure no errors are raised, whether it actually does what
        # we want is tested later.
        _ = ButlerStandardizer(uuid.uuid1(), butler=self.butler)
        _ = ButlerStandardizer(uuid.uuid1().hex, butler=self.butler)
        _ = ButlerStandardizer(DatasetRef(DatasetId(2)), butler=self.butler)
        _ = ButlerStandardizer(DatasetId(3), butler=self.butler)

        _ = Standardizer.get(DatasetRef(DatasetId(5)), butler=self.butler)
        _ = Standardizer.get(DatasetId(6), butler=self.butler)

        _ = Standardizer.get(DatasetId(6), butler=self.butler, force=ButlerStandardizer)

    def compare_to_expected(self, expected_idx, standardized):
        fits = FitsFactory.get_fits(expected_idx, spoof_data=True)
        hdr = fits["PRIMARY"].header
        expected = {
            "dataId": f"{expected_idx}",
            "datasetType": "test_datasettype_name",
            "visit": int(f"{hdr['EXPNUM']}{hdr['CCDNUM']}"),
            "detector": hdr["CCDNUM"],
            "exposureTime": hdr["EXPREQ"],
            "OBSID": hdr["OBSID"],
            "GAINA": hdr["GAINA"],
            "GAINB": hdr["GAINB"],
            "DTNSANAM": hdr["DTNSANAM"],
            "mjd_mid": Time(hdr["DATE-AVG"], format="isot", scale="tai").utc.mjd
            + (hdr["EXPREQ"] + 0.5) / 2.0 / 60.0 / 60.0 / 24.0,
            "filter": hdr["FILTER"],
        }
        expected["obs_day"] = ButlerStandardizer._mjd_to_obs_day(expected["mjd_mid"])

        for k, v in expected.items():
            with self.subTest("Value not standardized as expected.", key=k):
                if k == "mjd_mid":
                    self.assertAlmostEqual(v, standardized["meta"][k], 4)
                else:
                    self.assertEqual(v, standardized["meta"][k])

        # The CRVAL1/2 are with respect to the origin (CRPIX), Our center_ra
        # definition uses the pixel in the center of the CCD. The permissible
        # deviation should be on the scale of half a CCD's footprint, unless
        # it's DECam then it could be as big as half an FOV of the focal plane
        self.assertAlmostEqual(standardized["meta"]["ra"], fits[1].header["CRVAL1"], 0)
        self.assertAlmostEqual(standardized["meta"]["dec"], fits[1].header["CRVAL2"], 0)

        # compare standardized images
        # fmt: off
        np.testing.assert_equal([fits["IMAGE"].data,], standardized["science"])
        np.testing.assert_equal([fits["VARIANCE"].data,], standardized["variance"])
        np.testing.assert_equal([fits["MASK"].data,], standardized["mask"])
        # fmt: on

    def test_standardize(self):
        """Test ButlerStandardizer instantiates and standardizes as expected."""
        std = Standardizer.get(DatasetId(7, fill_metadata=True), butler=self.butler)
        standardized = std.standardize()
        self.compare_to_expected(7, standardized)

        # Test chained resolution works.
        std = Standardizer.get(DatasetId(7, fill_metadata=True), butler=[self.failed_butler, self.butler])
        standardized = std.standardize()
        self.compare_to_expected(7, standardized)

        # Test chained resolution fails expectedly
        with self.assertRaises(ValueError):
            std = Standardizer.get(
                DatasetId(7, fill_metadata=True), butler=[self.failed_butler, self.failed_butler]
            )

    def test_standardize_missing_wcs(self):
        """Test ButlerStandardizer instantiates and standardizes as expected een when fits appoximation of the WCS failed."""
        missing_wcs_butler = MockButler("/far/far/away", failed_fits_appoximation=True)
        std_config = StandardizerConfig()
        std_config["wcs_fallback_sips_degree"] = 4
        # Test that we raise an error when requesting negative points.
        with self.assertRaises(ValueError):
            std_config["wcs_fallback_points"] = -5
            std = Standardizer.get(
                DatasetId(7, fill_metadata=True),
                butler=missing_wcs_butler,
                config=std_config,
            )
            std.standardize()
        std_config["wcs_fallback_points"] = 10
        std = Standardizer.get(DatasetId(7, fill_metadata=True), config=std_config, butler=missing_wcs_butler)
        standardized = std.standardize()

        # Validate that getFitsMetadata raises an error forcing us to use a fallback WCS
        std._wcs is not None
        wcs_ref = std.ref.makeComponentRef("wcs")
        wcs = missing_wcs_butler.get(wcs_ref)
        with self.assertRaises(Exception):
            wcs.getFitsMetadata()

        fits = FitsFactory.get_fits(7, spoof_data=True)

        # The CRVAL1/2 are with respect to the origin (CRPIX), Our center_ra
        # definition uses the pixel in the center of the CCD. The permissible
        # deviation should be on the scale of half a CCD's footprint, unless
        # it's DECam then it could be as big as half an FOV of the focal plane
        self.assertAlmostEqual(standardized["meta"]["ra"], fits[1].header["CRVAL1"], 0)
        self.assertAlmostEqual(standardized["meta"]["dec"], fits[1].header["CRVAL2"], 0)

    def test_standardize_missing_headers(self):
        """Test ButlerStandardizer works even with certain missing headers."""
        # A list of optional headers that were present in the DEEP butler but
        # are not present on the current USDF embargo butler.
        missing_headers = [
            "GAINA",
            "GAINB",
            "DTNSANAM",
            "AIRMASS",
        ]
        missing_butler = MockButler("/far/far/away", missing_headers=missing_headers)
        std = Standardizer.get(DatasetId(7, fill_metadata=True), butler=missing_butler)
        standardized = std.standardize()

        fits = FitsFactory.get_fits(7, spoof_data=True)
        hdr = fits["PRIMARY"].header
        expected = {
            "dataId": "7",
            "datasetType": "test_datasettype_name",
            "visit": int(f"{hdr['EXPNUM']}{hdr['CCDNUM']}"),
            "detector": hdr["CCDNUM"],
            "exposureTime": hdr["EXPREQ"],
            "OBSID": hdr["OBSID"],
            "mjd_mid": Time(hdr["DATE-AVG"], format="isot", scale="tai").utc.mjd
            + (hdr["EXPREQ"] + 0.5) / 2.0 / 60.0 / 60.0 / 24.0,
            "filter": hdr["FILTER"],
        }

        # Assert that the "missing" headers were not standardized
        for header in missing_headers:
            self.assertNotIn(header, standardized["meta"])

        for k, v in expected.items():
            with self.subTest("Value not standardized as expected.", key=k):
                if k == "mjd_mid":
                    self.assertAlmostEqual(v, standardized["meta"][k], 4)
                else:
                    self.assertEqual(v, standardized["meta"][k])

    def test_roundtrip(self):
        """Test ButlerStandardizer can instantiate itself from standardized
        data and a Data Butler."""
        std = Standardizer.get(DatasetId(8), butler=self.butler)
        standardized = std.standardize()

        std2 = ButlerStandardizer(standardized["meta"]["dataId"], butler=self.butler)
        self.assertIsInstance(std2, ButlerStandardizer)

        standardized2 = std2.standardize()
        # TODO: I got to come up with some reasonable way of comparing this
        for k in [
            "mjd_mid",
            "filter",
            "dataId",
            "OBSID",
            "ra",
            "dec",
            "visit",
            "filter",
            "detector",
            "obs_lon",
            "obs_lat",
            "obs_elev",
        ]:
            with self.subTest("Failed to rounndtrip", key=k):
                self.assertEqual(standardized["meta"][k], standardized2["meta"][k])

    def test_psf_provenance_reaches_imagecollection_and_workunit(self):
        """PSF kernel and provenance survive the path into a WorkUnit.

        The kernel is only useful if it arrives intact at the thing that
        searches it, and the provenance is only useful if it arrives alongside.
        Both are checked end to end rather than at the standardizer alone.
        """
        from kbmod import ImageCollection

        std = Standardizer.get(DatasetId(7, fill_metadata=True), butler=self.butler)
        expected_kernel = std.standardizePSF()[0]

        collection = ImageCollection.fromStandardizers([std])

        for column in ("psf_source", "psf_eval_x", "psf_eval_y", "psf_native_sum", "psf_width"):
            self.assertIn(column, collection.data.colnames, f"{column} missing from ImageCollection")
        self.assertEqual(collection.data["psf_source"][0], "rubin:computeKernelImage")

        work_unit = collection.toWorkUnit(butler=self.butler)
        stored = work_unit.im_stack.psfs[0]

        # The Rubin kernel, not a Gaussian, and unaltered in transit.
        np.testing.assert_allclose(stored, expected_kernel, rtol=1e-6)
        self.assertEqual(stored.shape, expected_kernel.shape)
        self.assertAlmostEqual(float(np.sum(stored)), 1.0, places=5)

        # Provenance travels with it.
        self.assertIn("psf_source", work_unit.org_img_meta.colnames)
        self.assertEqual(work_unit.org_img_meta["psf_source"][0], "rubin:computeKernelImage")

    def test_imagecollection_roundtrip(self):
        """Test ButlerStandardizer can be reconstructed via ImageCollection's
        load_std path, which unpacks table columns as keyword arguments.

        The load_std() function in ImageCollection.get_standardizer() reconstructs
        standardizers via:
            std_cls(**kwargs, **row[no_conf_cols], config=config)
        so the __init__ parameter names must match the column names or else
        be interpreted as unknown kwargs and raise a TypeError.
        """
        from kbmod import ImageCollection

        # Create a ButlerStandardizer and build an ImageCollection from it
        std = Standardizer.get(DatasetId(7, fill_metadata=True), butler=self.butler)
        ic = ImageCollection.fromStandardizers([std])

        # Clear the cached standardizers to force reconstruction from the
        # serialized table row data via load_std(). Without this, the cached
        # standardizer would be returned directly, bypassing the kwargs path.
        n_stds = ic.meta["n_stds"]
        ic._standardizers = np.full((n_stds,), None)

        # get_standardizer will call load_std(), which unpacks the row columns as
        # **kwargs. If the __init__ param name (e.g. 'dataId') doesn't match the
        # metadata column name, this will raise a TypeError.
        recovered = ic.get_standardizer(0, butler=self.butler)
        self.assertIsInstance(recovered["std"], ButlerStandardizer)

        # Now rename the 'dataId' column so it no longer matches
        # the __init__ parameter, verifying we get a TypeError.
        ic.data.rename_column("dataId", "tgt")
        ic._standardizers = np.full((n_stds,), None)
        with self.assertRaises(TypeError):
            ic.get_standardizer(0, butler=self.butler)

    def mock_kbmodv1like_bitmasking(self, mockedexp):
        """Assign each flag that exists to a pixel, standardize, then expect
        the mask to only contain those pixels that are also in mask_flags.
        The grow_kernel is so large by default it would mask the nearly the
        whole image, so we turn it off.

        Because Rubin keeps flag map in the FITS file headers the
        ButlerStdConfig does not contain them. We mock these to match the
        DECam KBMODV1-like flags in MockButler, so we can set pixels to those
        flag values here.
        """
        mask_arr = mockedexp.mask.array
        for i, flag in enumerate(KBMODV1Config.bit_flag_map):
            mask_arr.ravel()[i] = KBMODV1Config.bit_flag_map[flag]

    # These tests are the same as KBMODV1 because the two hadn't diverged yet
    def test_bitmasking(self):
        """Test masking with direct config works as expected."""
        butler = MockButler("/far/far/away", mock_images_f=self.mock_kbmodv1like_bitmasking)

        conf = StandardizerConfig(grow_mask=False)
        std = Standardizer.get(DatasetId(9), butler=butler, config=conf)
        standardizedMask = std.standardizeMaskImage()

        for mask in standardizedMask:
            for i, flag in enumerate(KBMODV1Config.bit_flag_map):
                with self.subTest("Failed to mask expected", flag=flag):
                    if flag in ButlerStandardizerConfig.mask_flags:
                        self.assertEqual(mask.ravel()[i], True)
                    else:
                        self.assertEqual(mask.ravel()[i], False)

    def test_bitmasking_missing_flags(self):
        """Test masking succeeds when mask_flags config contains flags
        not present in the exposure's mask plane (e.g. 'SPIKE')."""
        butler = MockButler("/far/far/away", mock_images_f=self.mock_kbmodv1like_bitmasking)

        # Add flags that don't exist in the mock exposure's mask plane
        extra_flags = ButlerStandardizerConfig.mask_flags + ["SPIKE", "GHOST", "NONEXISTENT"]
        conf = StandardizerConfig(grow_mask=False, mask_flags=extra_flags)
        std = Standardizer.get(DatasetId(9), butler=butler, config=conf)

        # Should not raise KeyError
        standardizedMask = std.standardizeMaskImage()

        # Masking behavior should be identical to the default config
        # since the extra flags don't exist in the data
        for mask in standardizedMask:
            for i, flag in enumerate(KBMODV1Config.bit_flag_map):
                with self.subTest("Failed to mask expected", flag=flag):
                    if flag in ButlerStandardizerConfig.mask_flags:
                        self.assertEqual(mask.ravel()[i], True)
                    else:
                        self.assertEqual(mask.ravel()[i], False)

    def mock_kbmodv1like_thresholding(self, mockedexp):
        """Set image pixel [1, 1] to 1 and [2, 2] to 3."""
        mockedexp.image.array[1, 1] = 1
        mockedexp.image.array[2, 2] = 3

    def test_threshold_masking(self):
        """Test brightness threshold masking. Test config overrides."""
        butler = MockButler("/far/far/away", mock_images_f=self.mock_kbmodv1like_thresholding)

        conf = StandardizerConfig(
            {
                "grow_mask": False,
                "do_threshold": True,
                "brightness_threshold": 2,
            }
        )
        std = Standardizer.get(DatasetId(10), butler=butler, config=conf)
        mask = std.standardizeMaskImage()[0]

        self.assertFalse(mask[1, 1])
        self.assertTrue(mask[2, 2])

    def mock_kbmodv1like_growmask(self, mockedexp):
        """Flag image pixel [2, 2] as BAD, and expect grow_mask to grow that
        mask to all neighboring pixels. Again, because flags are not available
        through the butler, but exposures only, we mocked them to be the same
        like DECam KBMODV1 flags.
        """
        mockedexp.mask.array[2, 2] = KBMODV1Config.bit_flag_map["BAD"]

    def test_grow_mask(self):
        """Test mask grows as expected."""
        butler = MockButler("/far/far/away", mock_images_f=self.mock_kbmodv1like_growmask)

        conf = StandardizerConfig({"grow_mask": True, "grow_kernel_shape": (3, 3)})
        std = Standardizer.get(DatasetId(11), butler=butler, config=conf)
        mask = std.standardizeMaskImage()[0]

        self.assertTrue(mask[1:3, 1:3].all())
        self.assertFalse(mask[:, 0].all())
        self.assertFalse(mask[0, :].all())
        self.assertFalse(mask[-1, :].all())
        self.assertFalse(mask[:, -1].all())

    def test_psf_uses_rubin_model(self):
        """The Rubin Exposure.psf is used, not a Gaussian stand-in.

        Checks the properties a fixed Gaussian could not reproduce: the model's
        asymmetry, its measured width, and its orientation, all against the
        mock's analytic truth.
        """
        std = Standardizer.get(DatasetId(11), butler=self.butler)
        kernel = std.standardizePSF()[0]

        # Not the legacy Gaussian.
        legacy = PSF.make_gaussian_kernel(std.config["psf_std"])
        self.assertNotEqual(kernel.shape, legacy.shape)

        # Normalized, finite, non-negative, odd and square -- the invariants
        # KBMOD's PSF class enforces downstream.
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=6)
        self.assertTrue(np.all(np.isfinite(kernel)))
        self.assertTrue(np.all(kernel >= 0.0))
        self.assertEqual(kernel.shape[0], kernel.shape[1])
        self.assertEqual(kernel.shape[0] % 2, 1)
        PSF(kernel)  # must be acceptable as a search kernel

        # Shape agrees with the analytic truth of the very model that produced
        # the kernel. Reconstructing a second model from guessed constructor
        # arguments would test the guess, not the extraction.
        eval_x = std._psf_metadata["psf_eval_x"]
        eval_y = std._psf_metadata["psf_eval_y"]
        truth = std.exp.psf
        self.assertIsInstance(truth, MockRubinPsf)
        moments = measure_moments(kernel)
        shape = summarize_shape(moments.covariance)

        expected_major = 2.3548200450309493 * truth.sigma_major_at(eval_x, eval_y)
        expected_minor = 2.3548200450309493 * truth.sigma_minor_at(eval_x, eval_y)
        self.assertAlmostEqual(shape.fwhm_major / expected_major, 1.0, places=2)
        self.assertAlmostEqual(shape.fwhm_minor / expected_minor, 1.0, places=2)

        # Genuinely asymmetric: a Gaussian stand-in would be circular.
        self.assertGreater(shape.fwhm_major / shape.fwhm_minor, 1.2)

        # Kernel mode is centered, so the centroid sits on the middle pixel.
        center = (kernel.shape[0] - 1) / 2.0
        self.assertAlmostEqual(moments.centroid_x, center, places=2)
        self.assertAlmostEqual(moments.centroid_y, center, places=2)

    def test_psf_evaluation_position_is_honored(self):
        """The configured evaluation position actually changes the kernel.

        The mock PSF varies with position, so a standardizer that evaluates at
        the wrong place produces a measurably different kernel. With a constant
        PSF this class of bug would be invisible.
        """
        detector = Standardizer.get(DatasetId(11), butler=self.butler)
        detector.standardizeMetadata()  # populates the detector dimensions
        detector_width = detector._naxis1
        self.assertIsNotNone(detector_width)

        kernels = {}
        for name, position in (
            ("origin", (0.0, 0.0)),
            ("far", (float(detector_width * 8), 0.0)),
        ):
            config = ButlerStandardizerConfig(psf_eval_position=position)
            std = Standardizer.get(DatasetId(11), butler=self.butler, config=config)
            kernels[name] = std.standardizePSF()[0]
            self.assertEqual(std._psf_metadata["psf_eval_x"], position[0])
            self.assertEqual(std._psf_metadata["psf_eval_y"], position[1])

        far_shape = summarize_shape(measure_moments(kernels["far"]).covariance)
        origin_shape = summarize_shape(measure_moments(kernels["origin"]).covariance)
        # The mock widens with x, so evaluating further out must be wider.
        self.assertGreater(far_shape.fwhm_major, origin_shape.fwhm_major)

    def test_psf_eval_position_modes(self):
        """'average', 'center', and an explicit coordinate are distinguishable."""
        std = Standardizer.get(DatasetId(11), butler=self.butler)
        std.standardizePSF()
        self.assertEqual(std._psf_metadata["psf_source"], "rubin:computeKernelImage")

        config = ButlerStandardizerConfig(psf_eval_position="center")
        centered = Standardizer.get(DatasetId(11), butler=self.butler, config=config)
        centered.standardizePSF()
        self.assertAlmostEqual(centered._psf_metadata["psf_eval_x"], centered._naxis1 / 2.0)
        self.assertAlmostEqual(centered._psf_metadata["psf_eval_y"], centered._naxis2 / 2.0)

        config = ButlerStandardizerConfig(psf_eval_position="nonsense")
        bad = Standardizer.get(DatasetId(11), butler=self.butler, config=config)
        with self.assertRaises(ValueError):
            bad.standardizePSF()

    def test_psf_lazily_loads_exposure(self):
        """standardizePSF works when called first, without the exposure loaded.

        The sibling standardize* methods all lazily load the exposure; this one
        historically did not, because it never touched it. Calling it in
        isolation is the case that would have broken.
        """
        std = Standardizer.get(DatasetId(11), butler=self.butler)
        self.assertIsNone(std.exp)
        kernel = std.standardizePSF()[0]
        self.assertIsNotNone(std.exp)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=6)

    def test_psf_missing_model_raises(self):
        """A missing Rubin PSF raises rather than silently becoming a Gaussian."""
        butler = MockButler("/far/far/away", mock_psf=False)
        std = Standardizer.get(DatasetId(11), butler=butler)
        with self.assertRaises(RubinPsfError) as context:
            std.standardizePSF()
        # The message must point at the opt-in rather than being a bare failure.
        self.assertIn("psf_fallback_std", str(context.exception))

    def test_psf_broken_model_raises(self):
        """A model that raises is reported, not swallowed."""
        butler = MockButler("/far/far/away", mock_psf=BrokenRubinPsf("detector on fire"))
        std = Standardizer.get(DatasetId(11), butler=butler)
        with self.assertRaises(RubinPsfError) as context:
            std.standardizePSF()
        self.assertIn("detector on fire", str(context.exception))

    def test_psf_fallback_is_opt_in(self):
        """The Gaussian fallback is reachable only when explicitly configured."""
        butler = MockButler("/far/far/away", mock_psf=False)
        config = ButlerStandardizerConfig(psf_fallback_std=2)
        std = Standardizer.get(DatasetId(11), butler=butler, config=config)

        kernel = std.standardizePSF()[0]
        expected = PSF.make_gaussian_kernel(2)
        np.testing.assert_allclose(kernel, expected)
        # The fallback must be labeled, so a stored kernel is never mistaken
        # for a real Rubin model.
        self.assertEqual(std._psf_metadata["psf_source"], "gaussian_fallback")

    def test_psf_metadata_reaches_standardized_metadata(self):
        """PSF provenance columns are recorded during metadata standardization.

        These are what carry placement into the ImageCollection and then into
        WorkUnit.org_img_meta.
        """
        std = Standardizer.get(DatasetId(11), butler=self.butler)
        meta = std.standardizeMetadata()

        for key in (
            "psf_source",
            "psf_eval_x",
            "psf_eval_y",
            "psf_center_x",
            "psf_center_y",
            "psf_native_sum",
            "psf_width",
        ):
            self.assertIn(key, meta)

        self.assertEqual(meta["psf_source"], "rubin:computeKernelImage")
        self.assertGreater(meta["psf_width"], 1)
        self.assertGreater(meta["psf_native_sum"], 0.0)

    def test_psf_metadata_records_unavailable_without_raising(self):
        """Metadata degrades gracefully where the science kernel would raise.

        Refusing to describe an entire collection because one exposure has a
        bad PSF would be unhelpful; refusing to *search* it silently would not.
        """
        butler = MockButler("/far/far/away", mock_psf=False)
        std = Standardizer.get(DatasetId(11), butler=butler)
        meta = std.standardizeMetadata()
        self.assertEqual(meta["psf_source"], "unavailable")
        # ... but producing the science kernel still refuses.
        with self.assertRaises(RubinPsfError):
            std.standardizePSF()

    def test_psf_metadata_can_be_disabled(self):
        """The extra component fetch can be turned off for large collections."""
        config = ButlerStandardizerConfig(standardize_psf_metadata=False)
        std = Standardizer.get(DatasetId(11), butler=self.butler, config=config)
        meta = std.standardizeMetadata()
        self.assertNotIn("psf_source", meta)

    def test_to_layered_image(self):
        """Test ButlerStandardizer can create a LayeredImagePy."""
        std = Standardizer.get(DatasetId(8), butler=self.butler)
        self.assertIsInstance(std, ButlerStandardizer)

        # Get the expected FITS files and extract the MJD from the header
        fits = FitsFactory.get_fits(8, spoof_data=True)
        hdr = fits["PRIMARY"].header
        expected_mjd = Time(hdr["DATE-AVG"]).mjd + 120 / 24.0 / 60.0 / 60.0

        # Get list of layered images froom the standardizer
        butler_imgs = std.toLayeredImage()
        self.assertEqual(1, len(butler_imgs))
        img = butler_imgs[0]

        # Compare standardized images
        np.testing.assert_equal(fits["IMAGE"].data, img.sci)
        np.testing.assert_equal(fits["VARIANCE"].data, img.var)
        np.testing.assert_equal(fits["MASK"].data, img.mask)

        # Test that we correctly set metadata
        # times can only be compred approximately, because sometimes we
        # calculate the time in the middle of the exposure
        self.assertAlmostEqual(expected_mjd, img.time, 2)

    def test_mjd_to_obs_day(self):
        """Test that _mjd_to_obs_day works as expected."""
        # Check that all entries for the night of June 1-2, 2025 map to 20250601 as obs_day
        # We test from June 1, 2025 22:00 to June 2, 2025 08:00
        for hour in range(0, 10):
            mjd = 60827.91666667 + hour / 24.0
            obs_day = ButlerStandardizer._mjd_to_obs_day(mjd)
            self.assertEqual(obs_day, 20250601)

        # Check that all entries for the night of June 2-3, 2025 map to 20250602 as obs_day
        # We test from June 2, 2025 22:00 to June 3, 2025 08:00
        for hour in range(0, 10):
            mjd = 60828.91666667 + hour / 24.0
            obs_day = ButlerStandardizer._mjd_to_obs_day(mjd)
            self.assertEqual(obs_day, 20250602)

    def test_deepcopy(self):
        """Deep-copying a ButlerStandardizer yields an independent object
        whose butler attribute is the same instance as the original's."""
        std = ButlerStandardizer(DatasetId(7, fill_metadata=True), butler=self.butler)
        std._metadata = {"k": "v"}

        new_std = copy.deepcopy(std)

        self.assertIsNot(new_std, std)
        self.assertIs(new_std.butler, std.butler)
        # Other state is independent: mutating the copy's dict does not affect the original.
        self.assertIsNot(new_std._metadata, std._metadata)
        new_std._metadata["k"] = "mutated"
        self.assertEqual(std._metadata["k"], "v")


if __name__ == "__main__":
    unittest.main()
