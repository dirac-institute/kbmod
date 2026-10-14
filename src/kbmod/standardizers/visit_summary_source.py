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
    component this source cannot supply (no catalog for the visit, several
    catalogs for the visit in the collections, a catalog that cannot be read
    or fails the check below, a missing or duplicated detector row, summary
    statistics missing a field the standardizer reads, an empty value, or any
    other component) is read from the image as usual.

    The catalog must come from the processing that made the images; a catalog
    from another processing run has a different calibration. When ``verify``
    is set, the catalog is checked separately for each image run and dataset
    type in a visit: the WCS of the first such image whose catalog row has a
    WCS is read from the image and compared with that row. Images of that run
    use the catalog only if they match, and nothing is supplied from the
    catalog for an image of that run before the check.

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

    The catalog's PhotoCalib is deliberately not supplied: it can describe
    instrumental counts while the delivered images are already calibrated.

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
    verify : `bool`, optional
        Check each visit's catalog against its images, at the cost of one
        extra WCS read per visit and image run.
    """

    components = ("visitInfo", "bbox", "wcs", "summaryStats")

    # Summary statistics ButlerStandardizer always reads. If the catalog lacks
    # any of them, summary statistics are read from the images instead.
    summary_fields = (
        "psfSigma",
        "psfArea",
        "nPsfStar",
        "zeroPoint",
        "skyBg",
        "skyNoise",
        "meanVar",
        "astromOffsetMean",
        "astromOffsetStd",
    )

    def __init__(self, butler, collections, dataset_type="visit_summary", verify=True):
        self.butler = butler
        self.collections = collections
        self.dataset_type = dataset_type
        self.verify = verify
        self._visit = None
        self._rows = None
        self._checks = {}
        self.counts = {
            "catalog_reads": 0,
            "catalog_failures": 0,
            "ambiguous": 0,
            "mismatched": 0,
            "substituted": 0,
            "not_supplied": 0,
        }

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
            self._visit, self._rows, self._checks = visit, self._load(*visit), {}
        row = self._rows.get(ref.dataId["detector"]) if self._rows is not None else None
        if row is not None and not self._is_verified(ref, row):
            row = None
        value = row.get(component) if row is not None else None
        self.counts["not_supplied" if value is None else "substituted"] += 1
        return value

    def _load(self, instrument, visit):
        """Read a visit's catalog and extract its per-detector components.

        Returns `None`, which is remembered for the visit, when the catalog
        cannot be found or read, or when the collections hold more than one
        catalog for the visit; those images then use their own components.
        """
        try:
            # Search every collection, not only the first match: a visit can
            # have catalogs from several processing runs, and the run listed
            # first is not necessarily the one the images were made from.
            candidates = {
                ref.id: ref
                for ref in self.butler.query_datasets(
                    self.dataset_type,
                    collections=self.collections,
                    find_first=False,
                    where="instrument = inst AND visit = visit_id",
                    bind={"inst": instrument, "visit_id": visit},
                    explain=False,
                )
            }
            if len(candidates) != 1:
                self.counts["ambiguous" if candidates else "catalog_failures"] += 1
                logger.warning(
                    f"Found {len(candidates)} {self.dataset_type} datasets for visit {visit} in "
                    f"{self.collections}; reading components from the images instead."
                )
                return None
            rows = self._extract(self.butler.get(next(iter(candidates.values()))), visit)
        except Exception as e:
            # Reading the catalog is only an optimization: any failure falls
            # back to reading each image's components.
            self.counts["catalog_failures"] += 1
            logger.warning(f"Could not read {self.dataset_type} for visit {visit}: {e!r}")
            return None
        self.counts["catalog_reads"] += 1
        return rows

    def _is_verified(self, ref, row):
        """Return whether the visit's catalog may be used for this image.

        The result is remembered per image run and dataset type within the
        visit. An image whose row has no WCS cannot be checked; it uses its
        own components until another image of its run has passed the check.
        """
        if not self.verify:
            return True
        key = (ref.run, ref.datasetType.name)
        if key not in self._checks:
            if row["wcs"] is None:
                return False
            self._checks[key] = self._matches_image(ref, row["wcs"])
        return self._checks[key]

    def _matches_image(self, ref, catalog_wcs):
        """Return whether an image's own WCS equals its catalog row's WCS."""
        try:
            image_wcs = self.butler.get(ref.makeComponentRef("wcs"))
        except Exception as e:
            image_wcs = None
            reason = f"its WCS could not be read ({e!r})"
        else:
            reason = "its WCS differs from the catalog's"
        if image_wcs is not None and image_wcs == catalog_wcs:
            return True
        self.counts["mismatched"] += 1
        logger.warning(
            f"{self.dataset_type} for visit {self._visit[1]} in {self.collections} does not match image "
            f"{ref.dataId} of run {ref.run}: {reason}. The catalog may be from another processing run; "
            "reading components from that run's images instead."
        )
        return False

    def _extract(self, catalog, visit):
        """Map detector ID to its components, omitting duplicated detectors."""
        from lsst.afw.image import ExposureSummaryStats

        # The catalog schema can lag the stack's ExposureSummaryStats (so
        # ExposureSummaryStats.from_record fails); fields it lacks keep their
        # defaults unless the standardizer reads them.
        names = set(catalog.schema.getNames())
        fields = [f.name for f in dataclasses.fields(ExposureSummaryStats) if f.name in names]
        missing = [f for f in self.summary_fields if f not in names]
        if missing:
            logger.warning(
                f"{self.dataset_type} for visit {visit} lacks {missing}; reading summary statistics "
                "from the images instead."
            )

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
                "summaryStats": None if missing else ExposureSummaryStats(**{f: record[f] for f in fields}),
            }
        for detector in duplicates:
            logger.warning(f"Detector {detector} appears more than once in the catalog; not substituting it.")
            del rows[detector]
        return rows
