"""Numerical and disk-backed regressions for the prototype WCS codec."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import DistortionLookupTable, Sip, WCS

from kbmod import ImageCollection, Standardizer
from kbmod.configuration import SearchConfiguration
from kbmod.core.image_stack_py import ImageStackPy
from kbmod.reprojection import reproject_work_unit
from kbmod.region_search import Patch, RegionSearch
from kbmod.results import Results
from kbmod.wcs_utils import (
    append_wcs_to_hdu_header,
    deserialize_wcs,
    extract_wcs_from_hdu_header,
    serialize_wcs,
)
from kbmod.work_unit import WorkUnit, hdu_to_image_metadata_table, image_metadata_table_to_hdu
from utils import DECamImdiffFactory


def precision_wcs(linear="cd", inverse_sip=True):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    wcs.wcs.crval = [187.03584703124386, 8.961019057678422]
    wcs.wcs.crpix = [17.123456789012345, 11.234567890123456]
    if linear == "cd":
        wcs.wcs.cd = [
            [-5.555555555555556e-5, 1.234567890123456e-7],
            [1.1123456789012345e-7, 5.555555555555556e-5],
        ]
    else:
        wcs.wcs.pc = [[0.9999238455631987, -0.012340987654321098], [0.012340987654321098, 0.9999238455631987]]
        wcs.wcs.cdelt = [-5.555555555555556e-5, 5.555555555555556e-5]
    wcs.wcs.radesys = "FK5"
    wcs.wcs.equinox = 2000.1234567890123
    wcs.wcs.name = "Precision test's TAN/SIP"
    a, b = np.zeros((4, 4)), np.zeros((4, 4))
    a[2, 0], a[2, 1] = 1.1234567890123456e-6, -7.345070484924044e-14
    b[0, 2], b[1, 2] = -2.1234567890123456e-6, 6.345070484924044e-14
    wcs.sip = Sip(a, b, -a if inverse_sip else None, -b if inverse_sip else None, wcs.wcs.crpix.copy())
    wcs.pixel_shape = (37, 23)
    wcs.pixel_bounds = [(-0.5, 36.5), (-0.5, 22.5)]
    wcs.wcs.set()
    return wcs


class TestWcsPersistence(unittest.TestCase):
    def assert_exact(self, left, right):
        # Compare actual parameters rather than rounded headers or angular tolerance.
        self.assertEqual(left.pixel_shape, right.pixel_shape)
        np.testing.assert_array_equal(left.pixel_bounds, right.pixel_bounds)
        self.assertEqual(left.wcs.has_cd(), right.wcs.has_cd())
        for key in ("crpix", "crval"):
            self.assertEqual(getattr(left.wcs, key).tobytes(), getattr(right.wcs, key).tobytes())
        for key in ("lonpole", "latpole", "equinox"):
            self.assertEqual(float(getattr(left.wcs, key)).hex(), float(getattr(right.wcs, key)).hex())
        for key in ("ctype", "cunit", "cname"):
            self.assertEqual(list(getattr(left.wcs, key)), list(getattr(right.wcs, key)))
        for key in ("radesys", "name", "alt"):
            self.assertEqual(getattr(left.wcs, key), getattr(right.wcs, key))
        if left.wcs.has_cd():
            self.assertEqual(left.wcs.cd.tobytes(), right.wcs.cd.tobytes())
        else:
            self.assertEqual(left.wcs.pc.tobytes(), right.wcs.pc.tobytes())
            self.assertEqual(left.wcs.cdelt.tobytes(), right.wcs.cdelt.tobytes())
        if left.sip is None:
            self.assertIsNone(right.sip)
        else:
            for key in ("a", "b", "ap", "bp", "crpix"):
                x, y = getattr(left.sip, key), getattr(right.sip, key)
                if x is None:
                    self.assertIsNone(y)
                else:
                    self.assertEqual(x.tobytes(), y.tobytes())
        points = np.array([[0.0, 0.0], [0.125, 22.0], [36.0, 0.25], [36.0, 22.0], [17.4, 11.2]])
        sky = left.all_pix2world(points, 0)
        np.testing.assert_array_equal(sky, right.all_pix2world(points, 0))
        np.testing.assert_array_equal(left.all_world2pix(sky, 0), right.all_world2pix(sky, 0))

    def test_numeric_first_and_second_roundtrip(self):
        for linear in ("cd", "pc"):
            for inverse in (False, True):
                with self.subTest(linear=linear, inverse=inverse):
                    original = precision_wcs(linear, inverse)
                    current = original
                    for _ in range(2):
                        encoded = serialize_wcs(current, require_exact=True)
                        current = deserialize_wcs(encoded)
                        self.assert_exact(original, current)
                        self.assertEqual(encoded, serialize_wcs(current))
        plain = precision_wcs()
        plain.sip = None
        plain.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        plain.pixel_shape = None
        self.assert_exact(plain, deserialize_wcs(serialize_wcs(plain)))

    def test_legacy_precision_failure_and_compatibility(self):
        original = precision_wcs()
        header = dict(original.to_header(relax=True))
        header.update(NAXIS1=37, NAXIS2=23)
        old = deserialize_wcs(json.dumps(header))
        self.assertEqual(old.pixel_shape, (37, 23))
        self.assertNotEqual(original.sip.a[2, 1], old.sip.a[2, 1])
        self.assertNotEqual(original.wcs.cd[0, 0], old.wcs.get_pc()[0, 0])
        # Legacy information already rounded away cannot be recovered by rewriting.
        self.assert_exact(old, deserialize_wcs(serialize_wcs(old)))
        for text in ("", "none", "None"):
            self.assertIsNone(deserialize_wcs(text))

    def test_unsupported_and_invalid_records(self):
        unsupported = precision_wcs()
        unsupported.sip = None
        unsupported.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        record = json.loads(serialize_wcs(unsupported))
        self.assertEqual(record["fidelity"], "fits-header")
        self.assertIn("reason", record)
        self.assertEqual(deserialize_wcs(json.dumps(record)).pixel_shape, (37, 23))
        with self.assertRaises(ValueError):
            serialize_wcs(unsupported, require_exact=True)
        lookup = precision_wcs()
        lookup.cpdis1 = DistortionLookupTable(np.zeros((2, 2), dtype=np.float32), (1, 1), (1, 1), (1, 1))
        self.assertEqual(json.loads(serialize_wcs(lookup))["fidelity"], "fits-header")
        with self.assertRaisesRegex(ValueError, "lookup"):
            serialize_wcs(lookup, require_exact=True)
        bounds = precision_wcs()
        bounds.wcs.bounds_check(False, False)
        with self.assertRaisesRegex(ValueError, "projection bounds"):
            serialize_wcs(bounds, require_exact=True)
        self.assertEqual(json.loads(serialize_wcs(bounds))["fidelity"], "fits-header")
        for key, value in (("__kbmod_wcs__", 99), ("__kbmod_wcs__", True), ("fidelity", "unknown")):
            bad = json.loads(serialize_wcs(precision_wcs()))
            bad[key] = value
            with self.assertRaises(ValueError):
                deserialize_wcs(json.dumps(bad))
        for mutate in (
            lambda x: x["state"]["linear"].update(shape=[4]),
            lambda x: x["state"].update(pixel_shape=[23]),
            lambda x: x["state"].update(linear_type="crota"),
        ):
            bad = json.loads(serialize_wcs(precision_wcs()))
            mutate(bad)
            with self.assertRaises(ValueError):
                deserialize_wcs(json.dumps(bad))
        bad = json.loads(serialize_wcs(precision_wcs()))
        del bad["state"]["linear"]
        with self.assertRaises(ValueError):
            deserialize_wcs(json.dumps(bad))
        for key in ("crpix", "crval", "cd"):
            invalid = precision_wcs()
            getattr(invalid.wcs, key).flat[0] = np.nan
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                serialize_wcs(invalid)
        invalid = precision_wcs()
        invalid.sip.a[2, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            serialize_wcs(invalid)

    def test_serialization_does_not_initialize_caller(self):
        original = WCS(naxis=2)
        original.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        self.assertTrue(np.isnan(original.wcs.lonpole))
        restored = deserialize_wcs(serialize_wcs(original, require_exact=True))
        self.assertTrue(np.isnan(original.wcs.lonpole))
        self.assertEqual(restored.wcs.lonpole, 180.0)
        header = fits.Header()
        append_wcs_to_hdu_header(original, header, include_exact=True)
        self.assertTrue(np.isnan(original.wcs.lonpole))
        self.assertEqual(extract_wcs_from_hdu_header(header).wcs.lonpole, 180.0)

    def test_header_authority_refresh_and_actual_continue_cards(self):
        original = precision_wcs()
        header = fits.Header()
        append_wcs_to_hdu_header(original, header, include_exact=True)
        self.assertGreater(len(header["KBWCS"]), 1000)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "header.fits"
            fits.PrimaryHDU(header=header).writeto(path)
            with fits.open(path) as hdul:
                self.assert_exact(original, extract_wcs_from_hdu_header(hdul[0].header))
        changed = original.deepcopy()
        changed.wcs.crval += 0.125
        changed.wcs.set()
        append_wcs_to_hdu_header(changed, header)
        self.assert_exact(changed, extract_wcs_from_hdu_header(header))
        append_wcs_to_hdu_header(dict(original.to_header(relax=True)), header)
        self.assertNotIn("KBWCS", header)

    def test_image_collection_disk_and_packed_roundtrip(self):
        original = precision_wcs()
        std = Standardizer.get(DECamImdiffFactory().get_n(1)[0])
        std._wcs = [original]
        ic = ImageCollection.fromStandardizers([std])
        self.assert_exact(original, ic.get_wcs(0))
        with tempfile.TemporaryDirectory() as tmp:
            for cycle in range(2):
                filename = Path(tmp) / f"images{cycle}.ecsv"
                ic.write(filename)
                ic = ImageCollection.read(filename)
                self.assert_exact(original, list(ic.wcs)[0])
                self.assert_exact(original, ic.get_wcs([0])[0])
                ic.pack()
                ic = ImageCollection.fromBinTableHDU(ic.toBinTableHDU())
                self.assert_exact(original, ic.get_wcs(0))

    def test_workunit_storage_and_reprojection(self):
        original = precision_wcs()
        target = original.deepcopy()
        target.wcs.crpix += [0.2, -0.3]
        stack = ImageStackPy(
            times=[60000.0],
            sci=[np.arange(23 * 37, dtype=np.float32).reshape(23, 37)],
            var=[np.full((23, 37), 2.0, dtype=np.float32)],
        )
        metadata = Table({"per_image_wcs": [original], "ebd_wcs": [original.deepcopy()]})
        wu = WorkUnit(stack, SearchConfiguration(), wcs=target, org_image_meta=metadata)
        reference = reproject_work_unit(wu, target, frame="ebd", parallelize=False, show_progress=False)
        with tempfile.TemporaryDirectory() as tmp:
            for sharded in (False, True):
                current = wu
                for cycle in range(2):
                    name = f"unit_{sharded}_{cycle}.fits"
                    if sharded:
                        current.to_sharded_fits(name, tmp, compression_type="NOCOMPRESS", quantize_level=0)
                        current = WorkUnit.from_sharded_fits(name, tmp, lazy=False)
                    else:
                        current.to_fits(Path(tmp) / name, compression_type="NOCOMPRESS", quantize_level=0)
                        current = WorkUnit.from_fits(Path(tmp) / name, show_progress=False)
                    self.assert_exact(target, current.wcs)
                    for key in ("per_image_wcs", "ebd_wcs"):
                        self.assert_exact(original, current.org_img_meta[key][0])
                    actual = reproject_work_unit(
                        current, current.wcs, frame="ebd", parallelize=False, show_progress=False
                    )
                    np.testing.assert_array_equal(reference.im_stack.sci[0], actual.im_stack.sci[0])
                    np.testing.assert_array_equal(reference.im_stack.var[0], actual.im_stack.var[0])
                    np.testing.assert_array_equal(reference.im_stack.get_mask(0), actual.im_stack.get_mask(0))
            # Old WorkUnits have only interoperable primary-header WCS cards.
            legacy = Path(tmp) / "unit_False_0.fits"
            with fits.open(legacy, mode="update") as hdul:
                del hdul[0].header["KBWCS"]
            self.assertIsInstance(WorkUnit.from_fits(legacy, show_progress=False).wcs, WCS)

    def test_metadata_column_starting_with_none(self):
        original = precision_wcs()
        hdu = image_metadata_table_to_hdu(Table({"ebd_wcs": [None, original]}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.fits"
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
            with fits.open(path) as hdul:
                result = hdu_to_image_metadata_table(hdul[1])
                self.assertIsNone(result["ebd_wcs"][0])
                self.assert_exact(original, result["ebd_wcs"][1])

    def test_region_export_global_wcs_and_legacy_header(self):
        ic = ImageCollection.fromTargets(DECamImdiffFactory().get_n(1))
        ic.data["detector"] = [1]
        search = RegionSearch(ic)
        patch = Patch(187.03584703124386, 8.961019057678422, 0.002, 0.001, 37, 23, 0.20000000000000004, 0)
        original = patch.to_wcs()
        original.wcs.set()  # Compare normalized WCSLIB defaults, not an unset lazy pole.
        exported = search.export_image_collection(patch=patch, in_place=False)
        self.assert_exact(original, exported.get_global_wcs())
        stale = exported.copy()
        stale.data["global_wcs_pixel_shape_0"] = 38
        with self.assertRaisesRegex(ValueError, "disagree"):
            stale.get_global_wcs()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "region.ecsv"
            exported.write(path)
            restored = ImageCollection.read(path)
            self.assert_exact(original, restored.get_global_wcs())
        # Both historical global-header forms continue to load.
        for legacy in (original.to_header_string(), json.dumps(dict(original.to_header()))):
            exported.data["global_wcs"] = [legacy]
            self.assertIsInstance(exported.get_global_wcs(), WCS)
            self.assertEqual(exported.get_global_wcs().pixel_shape, (37, 23))

    def test_results_ecsv_roundtrip(self):
        original = precision_wcs()
        result = Results(
            {name: [0] for name in ("x", "y", "vx", "vy", "likelihood", "flux", "obs_count")}, wcs=original
        )
        with tempfile.TemporaryDirectory() as tmp:
            for cycle in range(2):
                path = Path(tmp) / f"result{cycle}.ecsv"
                result.write_table(path)
                result = Results.read_table(path)
                self.assert_exact(original, result.wcs)


if __name__ == "__main__":
    unittest.main()
