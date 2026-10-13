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
    when a run is reachable through several of its members.
    """

    def __init__(self, catalogs, broken=()):
        self.catalogs = catalogs
        self.broken = broken
        self.reads = []

    def query_datasets(self, dataset_type, collections, find_first, where, bind, explain):
        visit = bind["visit_id"]
        if visit in self.broken:
            raise OSError("registry unavailable")
        refs = [SimpleNamespace(id=(visit, i)) for i in range(len(self.catalogs.get(visit, [])))]
        return refs + refs

    def get(self, ref):
        visit, i = ref.id
        self.reads.append(visit)
        return self.catalogs[visit][i]


def ref(visit, detector):
    return SimpleNamespace(dataId={"instrument": "LSSTCam", "visit": visit, "detector": detector})


@unittest.skipUnless(HAS_AFW, "requires lsst.afw")
class TestVisitSummarySource(unittest.TestCase):
    def setUp(self):
        names = ["id", "psfSigma", "zeroPoint"]
        catalog = lambda dets: FakeCatalog(  # noqa: E731
            [FakeRecord(d, {"psfSigma": 2.0 + d, "zeroPoint": 31.5}) for d in dets], names
        )
        self.butler = FakeButler(
            {1: [catalog([10, 11, 99])], 2: [catalog([10, 12, 12])], 5: [catalog([10]), catalog([10])]},
            broken=(4,),
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
        # One catalog read serves every detector of the visit.
        self.assertEqual(self.butler.reads, [1])

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


if __name__ == "__main__":
    unittest.main()
