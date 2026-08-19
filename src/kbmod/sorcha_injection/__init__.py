"""Inject a globally-consistent synthetic population from Sorcha into KBMOD ImageCollections.

The stock :func:`kbmod.injection.generate_injection_catalog` fabricates a *random*
ecliptic trajectory set per ImageCollection. Those objects are IC-local: they carry
no identity that is shared between collections, so they cannot be used to test whether
KBMOD's linking reconnects an object that crosses from one patch into another.

This subpackage replaces the random draw with a *pre-built ephemeris* matched to each
patch spatially and temporally. Objects are selected from Sorcha DP2 synthetic
populations and injected at their real observed positions and magnitudes, preserving
the Sorcha ``ObjID`` as the injection ``obj_ids``. Because a synthetic object's track
naturally crosses several patches over time, the same ``ObjID`` is injected into every
overlapping ImageCollection -- which is what makes cross-collection linking
recall/precision measurable against a ground truth.

The output of :func:`generate_injection_catalog_from_sorcha` has the identical schema
to the stock generator, so :func:`kbmod.injection.inject_sources_into_ic` consumes it
unchanged.

Typical use::

    from kbmod.sorcha_injection import (
        SorchaInjectionConfig,
        SorchaIndex,
        generate_injection_catalog_from_sorcha,
    )

    cfg = SorchaInjectionConfig(index_path="/path/to/sorcha_index")
    index = SorchaIndex(cfg.index_path)
    catalog = generate_injection_catalog_from_sorcha(patch_ic, index, cfg)

    from kbmod.injection import inject_sources_into_ic
    injected_ic, injected_cats = inject_sources_into_ic(patch_ic, catalog, butler)
"""

from .config import SorchaInjectionConfig
from .visits import (
    PointingDB,
    ic_visit_table,
    sorcha_epoch_from_ic,
)
from .index import SorchaIndex, build_sorcha_index
from .tracks import build_object_tracks
from .selector import select_injections_for_ic
from .catalog import (
    generate_injection_catalog_from_sorcha,
    generate_injection_catalog_for_ic,
    write_injection_truth,
)

__all__ = [
    "SorchaInjectionConfig",
    "PointingDB",
    "SorchaIndex",
    "build_sorcha_index",
    "build_object_tracks",
    "ic_visit_table",
    "sorcha_epoch_from_ic",
    "select_injections_for_ic",
    "generate_injection_catalog_from_sorcha",
    "generate_injection_catalog_for_ic",
    "write_injection_truth",
]
