"""Configuration for Sorcha-backed injection."""

import dataclasses
import logging
from typing import Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Default on-disk locations at UW / DiRAC. Everything here is world-readable.
DEFAULT_SORCHA_ROOT = "/astro/store/shire/murtagh/sorcha/dp2__outputs"
DEFAULT_POINTING_DB = "/astro/store/shire/murtagh/sorcha/surveysetup/exposures_from_rubin_fixed.db"

# Populations present under DEFAULT_SORCHA_ROOT. All ten Sorcha runs are complete
# (each population's log ends "Sorcha process is completed"; last writes 2026-08-21).
ALL_POPULATIONS = (
    "cc", "cen", "de", "hc", "sc",
    "re_21", "re_32", "re_52", "re_73", "re_74",
)

# Trans-Neptunian classes only -- centaurs (``cen``) are interlopers between Jupiter
# and Neptune, not TNOs, so they are excluded from TNO recoverability estimates.
# ``sc`` (scattered) and the ``re_*`` mean-motion resonances (2:1, 3:2 plutinos, 5:2,
# 7:3, 7:4) are all trans-Neptunian.
TNO_POPULATIONS = (
    "cc", "hc", "de", "sc",
    "re_21", "re_32", "re_52", "re_73", "re_74",
)

# Historical default (the four populations the first ``index_dp2_mag27`` was built
# from). Retained for backward compatibility; new indexes use ``ALL_POPULATIONS``.
DEFAULT_POPULATIONS = ("cc", "hc", "cen", "de")


@dataclasses.dataclass
class SorchaInjectionConfig:
    """Configuration controlling which Sorcha objects get injected, and how.

    Attributes
    ----------
    index_path : `str`
        Path to the partitioned parquet dataset written by
        :func:`kbmod.sorcha_injection.build_sorcha_index`.
    populations : `tuple` of `str`
        Sorcha populations to draw from. Must be a subset of what the index
        was built with. `None` means "every population present in the index".
    mag_range : `tuple` of (`float` or `None`, `float` or `None`)
        Inclusive/exclusive ``(min, max)`` bounds on ``trailedSourceMag``.
        `None` on either end disables that bound. Note the index itself was
        built with a magnitude ceiling; this can only narrow it further.
    bands : `tuple` of `str` or `None`
        Restrict to these Sorcha ``optFilter`` values (e.g. ``("r", "i", "z")``).
        `None` keeps every band. Injection is done per-exposure, so the band of
        the row always matches the band of the exposure it lands in.
    max_objs_per_patch : `int` or `None`
        Cap on the number of *distinct* objects injected into one collection.
        Sampling is always done on whole tracks (by ``ObjID``), never per-row,
        so a selected object keeps every one of its in-footprint detections.
    min_obs_per_object : `int`
        Drop objects with fewer than this many in-footprint detections in the
        collection. Objects seen once or twice cannot be recovered by KBMOD and
        only dilute the truth table. Set to 1 to keep everything.
    source_type : `str`
        Value written into the catalog's ``source_type`` column, consumed by
        ``lsst.source.injection``. ``"Star"`` renders a PSF-convolved point source.
    seed : `int`
        Seed for the per-collection object subsampling, so a given
        (collection, config) pair always selects the same objects.
    require_on_detector : `bool`
        Require that a row lands inside the pixel bounds of the specific
        (visit, detector) exposure it is being injected into. This is what makes
        the injection actually render, and it correctly reproduces chip gaps.
    require_in_patch : `bool`
        Additionally require that the row lands inside the patch's ``global_wcs``
        footprint (in the reflex-corrected frame, when the patch is reflex
        corrected). Objects failing this are on a detector but outside the
        searched region, so they would be un-recoverable by construction.
    correct_epoch_to_mid_exposure : `bool`
        Sorcha evaluated positions at the *start* of each exposure rather than at
        mid-exposure (see :mod:`kbmod.sorcha_injection.visits`). When True, each
        row's position is linearly propagated forward by ``visitTime / 2`` using
        the object's own on-sky rate, so it corresponds to ``ic["mjd_mid"]``.
        The correction is small -- 0.011" median, 0.023" at the 99th percentile
        for cold classicals, i.e. about a tenth of an LSSTCam pixel -- but it is
        free and removes a known systematic.
    healpix_nside : `int`
        NSIDE of the ``healpix`` column stored in the index, used for the cheap
        spatial pre-filter. Must match the value the index was built with.
    """

    index_path: str
    populations: Optional[Sequence[str]] = None
    mag_range: Tuple[Optional[float], Optional[float]] = (None, 27.0)
    bands: Optional[Sequence[str]] = None
    max_objs_per_patch: Optional[int] = None
    min_obs_per_object: int = 3
    source_type: str = "Star"
    seed: int = 0
    require_on_detector: bool = True
    require_in_patch: bool = True
    correct_epoch_to_mid_exposure: bool = True
    healpix_nside: int = 64

    @classmethod
    def from_dict(cls, params):
        """Build a config from a plain dict, e.g. an ``injection`` block in a YAML config.

        Recognises an ``injection.source`` key so a single block can express both
        paths; ``source: "random"`` returns `None`, meaning "use the stock generator".

        Parameters
        ----------
        params : `dict`
            Keys matching this dataclass's fields, plus an optional ``source``.

        Returns
        -------
        config : `SorchaInjectionConfig` or `None`
        """
        params = dict(params or {})
        source = params.pop("source", "sorcha")
        if source == "random":
            return None
        if source != "sorcha":
            raise ValueError(f"injection.source must be 'random' or 'sorcha', got {source!r}")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(params) - known
        if unknown:
            raise ValueError(f"Unknown injection config key(s): {sorted(unknown)}. Valid: {sorted(known)}")
        if "mag_range" in params and params["mag_range"] is not None:
            params["mag_range"] = tuple(params["mag_range"])
        return cls(**params)

    @classmethod
    def from_search_config(cls, search_config):
        """Pull an ``injection`` block out of a `~kbmod.configuration.SearchConfiguration`.

        Returns `None` when no block is present or when it selects the random path, so
        callers can write ``cfg = SorchaInjectionConfig.from_search_config(sc)`` and
        pass the result straight to
        :func:`~kbmod.sorcha_injection.generate_injection_catalog_for_ic`.
        """
        try:
            block = search_config["injection"]
        except (KeyError, TypeError):
            return None
        if not block:
            return None
        return cls.from_dict(block)

    def __post_init__(self):
        if self.populations is not None:
            bad = set(self.populations) - set(ALL_POPULATIONS)
            if bad:
                raise ValueError(
                    f"Unknown Sorcha population(s): {sorted(bad)}. Known: {list(ALL_POPULATIONS)}"
                )
            self.populations = tuple(self.populations)
        if self.bands is not None:
            self.bands = tuple(self.bands)
        if self.min_obs_per_object < 1:
            raise ValueError("min_obs_per_object must be >= 1")
        lo, hi = self.mag_range
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(f"mag_range must be increasing, got {self.mag_range}")
