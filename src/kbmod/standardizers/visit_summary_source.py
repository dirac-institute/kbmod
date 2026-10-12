"""Serve ButlerStandardizer component reads from a visit's ``visit_summary``.

Building an ImageCollection reads five components of every image. On large
Butler repositories the cost is dominated by opening each image file, while the
same per-detector VisitInfo, bounding box, WCS and summary statistics are
available from the visit's ``visit_summary`` catalog, read once per visit.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)

__all__ = ["VisitSummarySource"]


class VisitSummarySource:
    """Supplies VisitInfo, bounding box, WCS and summary statistics for each
    image from its visit's ``visit_summary`` catalog.

    Pass an instance to `ButlerStandardizer` as ``component_source``. Any
    component this source cannot supply (no catalog for the visit, a missing or
    duplicated detector row, an empty value, or any other component) is read
    from the image as usual.

    Only one visit is held in memory. Targets sorted by visit, as
    ``kbmod_build_ic`` sorts them, read each catalog once; unsorted targets are
    still correct but may re-read a catalog. The catalog itself is released
    after the per-detector objects are extracted.

    Notes
    -----
    The values are not bit-identical to the image's own components:
    ``visit_summary`` stores summary statistics as 32-bit floats and fills some
    fields (e.g. ``astromOffsetMean``) that difference images leave NaN. On
    DP2 difference images the WCS, bounding box and VisitInfo were identical.

    The first catalog read in a process imports the PSF model packages, which
    draws from NumPy's global random state.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler used to find and read the catalogs.
    collections : `str` or `list[str]`
        Collections searched for the catalog. Use the collections the targets
        were found in.
    dataset_type : `str`, optional
        Name of the per-visit summary dataset type.
    """

    components = ("visitInfo", "bbox", "wcs", "summaryStats")

    def __init__(self, butler, collections, dataset_type="visit_summary"):
        self.butler = butler
        self.collections = collections
        self.dataset_type = dataset_type
        self._visit = None
        self._rows = None
        self.counts = {"catalog_reads": 0, "catalog_failures": 0, "substituted": 0, "not_supplied": 0}

    def get(self, ref, component):
        """Return a component of the given image, or `None` if this source
        cannot supply it.

        Parameters
        ----------
        ref : `lsst.daf.butler.DatasetRef`
            Reference to the image.
        component : `str`
            Component name, e.g. ``"wcs"``.

        Returns
        -------
        value : `object` or `None`
            The component, or `None` when it should be read from the image.
        """
        if component not in self.components:
            return None
        visit = (ref.dataId["instrument"], ref.dataId["visit"])
        if visit != self._visit:
            self._visit, self._rows = visit, self._load(*visit)
        row = self._rows.get(ref.dataId["detector"]) if self._rows is not None else None
        value = row.get(component) if row is not None else None
        self.counts["not_supplied" if value is None else "substituted"] += 1
        return value

    def _load(self, instrument, visit):
        """Read a visit's catalog and extract its per-detector components.

        Returns `None`, which is remembered for the visit, when the catalog
        cannot be found or read; those images then use their own components.
        """
        try:
            ref = self.butler.find_dataset(
                self.dataset_type, instrument=instrument, visit=visit, collections=self.collections
            )
            catalog = None if ref is None else self.butler.get(ref)
        except Exception as e:
            # Reading the catalog is only an optimization: any failure falls
            # back to reading each image's components.
            self.counts["catalog_failures"] += 1
            logger.warning(f"Could not read {self.dataset_type} for visit {visit}: {e!r}")
            return None
        if catalog is None:
            self.counts["catalog_failures"] += 1
            logger.warning(f"No {self.dataset_type} found for visit {visit} in {self.collections}.")
            return None
        self.counts["catalog_reads"] += 1
        return self._extract(catalog)

    @staticmethod
    def _extract(catalog):
        """Map detector ID to its components, omitting duplicated detectors."""
        from lsst.afw.image import ExposureSummaryStats

        # The catalog schema can lag the stack's ExposureSummaryStats (so
        # ExposureSummaryStats.from_record fails); fields it lacks keep their defaults.
        names = set(catalog.schema.getNames())
        fields = [f.name for f in dataclasses.fields(ExposureSummaryStats) if f.name in names]

        rows, duplicates = {}, set()
        for record in catalog:
            detector = record.getId()
            if detector in rows:
                duplicates.add(detector)
                continue
            rows[detector] = {
                "visitInfo": record.getVisitInfo(),
                "bbox": record.getBBox(),
                "wcs": record.getWcs(),
                "summaryStats": ExposureSummaryStats(**{f: record[f] for f in fields}),
            }
        for detector in duplicates:
            logger.warning(f"Detector {detector} appears more than once in the catalog; not substituting it.")
            del rows[detector]
        return rows
