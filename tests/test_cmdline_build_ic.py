"""Band selection at the Butler boundary, without an LSST installation."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import kbmod  # Import dependencies before the temporary LSST module patch.


class TestBuildIcBands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lsst = ModuleType("lsst")
        lsst.daf = ModuleType("lsst.daf")
        lsst.daf.butler = ModuleType("lsst.daf.butler")
        modules = {
            "lsst": lsst,
            "lsst.daf": lsst.daf,
            "lsst.daf.butler": lsst.daf.butler,
        }
        path = Path(__file__).parents[1] / "src/kbmod_cmdline/kbmod_build_ic.py"
        spec = importlib.util.spec_from_file_location("build_ic_under_test", path)
        cls.builder = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(cls.builder)

    def make_butler(self, bands=("y", "r", "g")):
        butler = Mock()

        def query(dataset_type, **kwargs):
            excluded = kwargs.get("bind", {}).get("excluded_bands", ())
            return [SimpleNamespace(band=band) for band in bands if band not in excluded]

        butler.registry.queryDatasets.side_effect = query
        butler.query_datasets.side_effect = query
        return butler

    def test_cli_default_and_explicit_overrides(self):
        base = ["repo", "difference_image", "--collections", "test"]
        self.assertEqual(self.builder.parse_args(base).bands_to_drop, ["y"])
        self.assertEqual(self.builder.parse_args(base + ["--bands-to-drop"]).bands_to_drop, [])
        self.assertEqual(
            self.builder.parse_args(base + ["--bands-to-drop", "u", "y"]).bands_to_drop,
            ["u", "y"],
        )

    def test_query_default_and_target_bindings(self):
        for target in (None, "target with ' quote"):
            with self.subTest(target=target):
                butler = self.make_butler()
                refs = self.builder.query_collection_refs(butler, "test", "difference_image", target)
                self.assertEqual([ref.band for ref in refs], ["r", "g"])
                call = butler.registry.queryDatasets if target is None else butler.query_datasets
                kwargs = call.call_args.kwargs
                self.assertIn("band NOT IN (:excluded_bands)", kwargs["where"])
                self.assertEqual(kwargs["bind"]["excluded_bands"], ("y",))
                if target:
                    self.assertEqual(kwargs["bind"]["target_name"], target)
                    self.assertNotIn(target, kwargs["where"])

    def test_explicit_empty_keeps_y(self):
        butler = self.make_butler()
        refs = self.builder.query_collection_refs(butler, "test", "difference_image", bands_to_drop=[])
        self.assertEqual([ref.band for ref in refs], ["y", "r", "g"])
        self.assertNotIn("where", butler.registry.queryDatasets.call_args.kwargs)

    def test_filter_before_limit_and_standardization(self):
        ic = Mock()
        ic.__len__ = Mock(return_value=1)
        ic.__setitem__ = Mock()
        with patch.object(self.builder.kbmod.ImageCollection, "fromTargets", return_value=ic) as build:
            self.builder.ingest_collection(
                self.make_butler(), "test", "difference_image", {}, max_exposures=1
            )
        self.assertEqual([ref.band for ref in build.call_args.args[0]], ["r"])

    def test_y_only_does_not_standardize_or_write(self):
        with patch.object(self.builder.kbmod.ImageCollection, "fromTargets") as build:
            self.builder.ingest_collection(self.make_butler(["y"]), "test", "difference_image", {})
        build.assert_not_called()

    def test_dry_run_uses_same_selection_and_per_collection_limit(self):
        args = self.builder.parse_args(
            [
                "repo",
                "difference_image",
                "--collections",
                "a",
                "b",
                "--dry",
                "--max_exposures",
                "1",
            ]
        )
        with patch.object(
            self.builder.dafButler,
            "Butler",
            return_value=self.make_butler(),
            create=True,
        ):
            with patch("builtins.print") as output:
                self.builder.execute(args)
        output.assert_any_call("Total exposures across all collections: 2")
