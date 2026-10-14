import math
import unittest
from types import SimpleNamespace

from kbmod.standardizers import VisitSummarySource

try:
    from lsst.afw.image import ExposureSummaryStats  # noqa: F401

    HAS_AFW = True
except ImportError:
    HAS_AFW = False


class FakeRecord:
    """visit_summary row with the accessors VisitSummarySource uses."""

    def __init__(self, detector, values):
        self.detector = detector
        self.values = values

    def getId(self):
        return self.detector

    def getVisitInfo(self):
        return f"visitInfo-{self.detector}"

    def getBBox(self):
        return f"bbox-{self.detector}"

    def getWcs(self):
        return None if self.detector == 99 else f"wcs-{self.detector}"

    def __getitem__(self, name):
        return self.values[name]


class FakeCatalog(list):
    def __init__(self, records, names):
        super().__init__(records)
        self.schema = SimpleNamespace(getNames=lambda: set(names))


class FakeButler:
    """Serves the catalogs listed per visit; visits not in `catalogs` have none.

    Each catalog is listed twice by the query, as a chained collection does
    when a run is reachable through several of its members. Image WCS reads
    return the catalog's value except for visits in `image_wcs`.
    """

    def __init__(self, catalogs, broken=(), image_wcs=None):
        self.catalogs = catalogs
        self.broken = broken
        self.image_wcs = image_wcs or {}
        self.reads = []
        self.image_reads = []

    def query_datasets(self, dataset_type, collections, find_first, where, bind, explain):
        visit = bind["visit_id"]
        if visit in self.broken:
            raise OSError("registry unavailable")
        refs = [SimpleNamespace(id=(visit, i)) for i in range(len(self.catalogs.get(visit, [])))]
        return refs + refs

    def get(self, ref):
        if hasattr(ref, "component"):
            visit, detector = ref.parent.dataId["visit"], ref.parent.dataId["detector"]
            self.image_reads.append((visit, detector))
            value = self.image_wcs.get(visit, f"wcs-{detector}")
            if isinstance(value, Exception):
                raise value
            return value
        visit, i = ref.id
        self.reads.append(visit)
        return self.catalogs[visit][i]


def ref(visit, detector):
    image = SimpleNamespace(dataId={"instrument": "LSSTCam", "visit": visit, "detector": detector})
    image.makeComponentRef = lambda component: SimpleNamespace(parent=image, component=component)
    return image


@unittest.skipUnless(HAS_AFW, "requires lsst.afw")
class TestVisitSummarySource(unittest.TestCase):
    def setUp(self):
        names = ["id", "psfSigma", "zeroPoint"]
        catalog = lambda dets: FakeCatalog(  # noqa: E731
            [FakeRecord(d, {"psfSigma": 2.0 + d, "zeroPoint": 31.5}) for d in dets], names
        )
        self.butler = FakeButler(
            {
                1: [catalog([10, 11, 99])],
                2: [catalog([10, 12, 12])],
                5: [catalog([10]), catalog([10])],
                6: [catalog([10, 11])],
                7: [catalog([10, 11])],
            },
            broken=(4,),
            image_wcs={6: "wcs-from-another-run", 7: OSError("unreadable")},
        )
        self.source = VisitSummarySource(self.butler, ["coll"])

    def test_supplies_components_per_detector(self):
        self.assertEqual(self.source.get(ref(1, 10), "visitInfo"), "visitInfo-10")
        self.assertEqual(self.source.get(ref(1, 11), "bbox"), "bbox-11")
        self.assertEqual(self.source.get(ref(1, 11), "wcs"), "wcs-11")
        stats = self.source.get(ref(1, 11), "summaryStats")
        self.assertEqual((stats.psfSigma, stats.zeroPoint), (13.0, 31.5))
        # Fields missing from the catalog schema keep their defaults.
        self.assertTrue(math.isnan(stats.skyNoise))
        # One catalog read serves every detector of the visit, and one image
        # WCS read checks it.
        self.assertEqual(self.butler.reads, [1])
        self.assertEqual(self.butler.image_reads, [(1, 10)])

    def test_reads_each_visit_when_it_changes(self):
        self.source.get(ref(1, 10), "wcs")
        self.source.get(ref(2, 10), "wcs")
        self.source.get(ref(1, 10), "wcs")
        self.assertEqual(self.butler.reads, [1, 2, 1])
        self.assertEqual(self.source.counts["catalog_reads"], 3)

    def test_unsupplied_components_return_none(self):
        # Unsupported component, missing detector, and an empty WCS.
        self.assertIsNone(self.source.get(ref(1, 10), "metadata"))
        self.assertIsNone(self.source.get(ref(1, 50), "wcs"))
        self.assertIsNone(self.source.get(ref(1, 99), "wcs"))
        # A duplicated detector is ambiguous; the others in the visit are still used.
        with self.assertLogs("kbmod.standardizers.visit_summary_source", level="WARNING"):
            self.assertIsNone(self.source.get(ref(2, 12), "wcs"))
        self.assertEqual(self.source.get(ref(2, 10), "wcs"), "wcs-10")

    def test_missing_or_unreadable_catalog(self):
        with self.assertLogs("kbmod.standardizers.visit_summary_source", level="WARNING"):
            self.assertIsNone(self.source.get(ref(3, 10), "wcs"))
        # The failure is remembered for the visit rather than retried per detector.
        self.assertIsNone(self.source.get(ref(3, 11), "wcs"))
        with self.assertLogs("kbmod.standardizers.visit_summary_source", level="WARNING"):
            self.assertIsNone(self.source.get(ref(4, 10), "wcs"))
        self.assertEqual(self.source.counts["catalog_failures"], 2)
        self.assertEqual(self.source.counts["not_supplied"], 3)

    def test_several_catalogs_for_a_visit(self):
        """A visit with catalogs from several runs falls back rather than guessing."""
        with self.assertLogs("kbmod.standardizers.visit_summary_source", level="WARNING"):
            self.assertIsNone(self.source.get(ref(5, 10), "wcs"))
        self.assertEqual(self.source.counts["ambiguous"], 1)
        self.assertEqual(self.butler.reads, [])

    def test_catalog_must_match_the_images(self):
        """A catalog that disagrees with its images is not used for the visit."""
        with self.assertLogs("kbmod.standardizers.visit_summary_source", level="WARNING") as logs:
            self.assertIsNone(self.source.get(ref(6, 10), "visitInfo"))
        self.assertIn("another processing run", logs.output[0])
        self.assertIsNone(self.source.get(ref(6, 11), "wcs"))
        with self.assertLogs("kbmod.standardizers.visit_summary_source", level="WARNING") as logs:
            self.assertIsNone(self.source.get(ref(7, 10), "wcs"))
        self.assertIn("could not be read", logs.output[0])
        self.assertEqual(self.source.counts["mismatched"], 2)
        self.assertEqual(self.butler.image_reads, [(6, 10), (7, 10)])

    def test_check_uses_an_image_with_a_catalog_wcs(self):
        # Detector 99 has no catalog WCS, so the next image of the visit is checked.
        self.assertIsNone(self.source.get(ref(1, 99), "wcs"))
        self.assertEqual(self.source.get(ref(1, 11), "wcs"), "wcs-11")
        self.assertEqual(self.butler.image_reads, [(1, 11)])

    def test_verify_can_be_disabled(self):
        source = VisitSummarySource(self.butler, ["coll"], verify=False)
        self.assertEqual(source.get(ref(6, 10), "wcs"), "wcs-10")
        self.assertEqual(self.butler.image_reads, [])


if __name__ == "__main__":
    unittest.main()
