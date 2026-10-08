"""Build injection catalogs from Sorcha, and the truth table that scores them.

:func:`generate_injection_catalog_from_sorcha` returns the *identical* astropy Table
schema as :func:`kbmod.injection.generate_injection_catalog`, so
:func:`kbmod.injection.inject_sources_into_ic` and
:func:`kbmod.injection.match_injection_results` consume it unchanged.

Coordinate frames follow the same convention as the stock generator, which is worth
spelling out because it is easy to get backwards:

* ``ra``/``dec`` are always **observed-frame** coordinates -- what ``VisitInjectTask``
  needs, because it renders into the original (un-reprojected) exposures. Sorcha's
  ``RA_deg``/``Dec_deg`` are already observed topocentric positions, so unlike the
  stock generator there is nothing to invert; they go straight in.
* When the patch is reflex-corrected at a guess distance ``D``, ``ra_{D}``/``dec_{D}``
  hold the **patch-frame** (reflex-corrected) coordinates. Those are what KBMOD
  results live in, so ``match_injection_results`` reads them, and they are what
  ``plot_x``/``plot_y`` are computed from.

A useful consequence of joining on visit id: the Sorcha row's ``optFilter`` is by
construction the band of the visit it came from, which is the band of the exposure it
is injected into. So ``mag`` is always already in the correct band -- no colour
transformation is needed or applied.
"""

import logging
import os

import numpy as np
from astropy.table import Table

logger = logging.getLogger(__name__)


def generate_injection_catalog_from_sorcha(
    ic,
    index,
    config,
    global_wcs=None,
    guess_distance=None,
    earth_loc=None,
    return_selection=False,
):
    """Generate an injection catalog for an ImageCollection from the Sorcha index.

    Parameters
    ----------
    ic : `kbmod.ImageCollection`
        Collection to inject into. Normally a patch collection carrying ``global_wcs``.
    index : `kbmod.sorcha_injection.SorchaIndex`
        Prebuilt ephemeris index.
    config : `kbmod.sorcha_injection.SorchaInjectionConfig`
        Selection configuration.
    global_wcs : `astropy.wcs.WCS`, optional
        Patch WCS. Defaults to ``ic.get_global_wcs()``.
    guess_distance : `float`, optional
        Guess distance (AU) the patch is reflex-corrected at. Defaults to the
        collection's ``helio_guess_dist`` column. ``None``/``0.0`` means the patch is
        in the observed frame.
    earth_loc : `astropy.coordinates.EarthLocation`, optional
        Observatory location; defaults to ``ic.get_observatory()``.
    return_selection : `bool`, optional
        Also return the raw selection dict from
        :func:`~kbmod.sorcha_injection.select_injections_for_ic`, which carries
        provenance (population, visit, detector, per-detector pixel position) that the
        catalog schema has no room for.

    Returns
    -------
    catalog : `astropy.table.Table`
        Columns ``injection_id, ra, dec, mag, guess_distance, source_type, obj_ids,
        obstime, plot_x, plot_y`` plus ``ra_{d}, dec_{d}`` when reflex-corrected --
        exactly the stock generator's schema. ``obj_ids`` holds the Sorcha ``ObjID``,
        stable across every collection the object appears in.
    selection : `dict`, optional
        Returned only when ``return_selection`` is True.
    """
    from .selector import select_injections_for_ic

    data = getattr(ic, "data", ic)
    if global_wcs is None and hasattr(ic, "get_global_wcs"):
        global_wcs = ic.get_global_wcs()
    if guess_distance is None and "helio_guess_dist" in data.colnames:
        guess_distance = float(data["helio_guess_dist"][0])
    if guess_distance is not None and guess_distance == 0.0:
        guess_distance = None

    sel = select_injections_for_ic(
        ic, index, config, global_wcs=global_wcs, guess_distance=guess_distance, earth_loc=earth_loc
    )

    n = len(sel["ObjID"])
    plot_x = sel.get("patch_x", np.full(n, np.nan))
    plot_y = sel.get("patch_y", np.full(n, np.nan))

    catalog_dict = {
        "injection_id": np.arange(n, dtype=np.int64),
        "ra": np.asarray(sel["ra"], dtype=float),
        "dec": np.asarray(sel["dec"], dtype=float),
        "mag": np.asarray(sel["mag"], dtype=float),
        # Matches the stock generator, which writes the scalar guess_distance on every
        # row (None when the patch is in the observed frame).
        "guess_distance": [guess_distance] * n,
        "source_type": [config.source_type] * n,
        "obj_ids": np.asarray(sel["ObjID"], dtype=str),
        "obstime": np.asarray(sel["obstime"], dtype=float),
        "plot_x": np.asarray(plot_x, dtype=float),
        "plot_y": np.asarray(plot_y, dtype=float),
    }
    if guess_distance is not None and "ra_reflex" in sel:
        catalog_dict[f"ra_{float(guess_distance)}"] = np.asarray(sel["ra_reflex"], dtype=float)
        catalog_dict[f"dec_{float(guess_distance)}"] = np.asarray(sel["dec_reflex"], dtype=float)

    catalog = Table(catalog_dict)
    catalog.meta["injection_source"] = "sorcha"
    catalog.meta["sorcha_index"] = getattr(index, "path", "")
    catalog.meta["sorcha_populations"] = list(index.populations)
    catalog.meta["guess_distance"] = guess_distance
    catalog.meta["n_objects"] = int(len(np.unique(sel["ObjID"]))) if n else 0

    logger.info(
        "Sorcha injection catalog: %d rows, %d distinct ObjIDs, mag %.2f-%.2f",
        n,
        catalog.meta["n_objects"],
        float(np.min(catalog["mag"])) if n else np.nan,
        float(np.max(catalog["mag"])) if n else np.nan,
    )

    if return_selection:
        return catalog, sel
    return catalog


def generate_injection_catalog_for_ic(
    ic,
    search_config,
    global_wcs,
    injection_config=None,
    index=None,
    **random_kwargs,
):
    """Dispatch to either the random or the Sorcha injection generator.

    This is the ``injection.source`` switch. The default is ``"random"``, so existing
    callers are unaffected.

    Parameters
    ----------
    ic : `kbmod.ImageCollection`
        Collection to inject into.
    search_config : `kbmod.configuration.SearchConfiguration`
        Search config, used by the random generator for its trajectory generator.
    global_wcs : `astropy.wcs.WCS`
        Shared WCS for the collection.
    injection_config : `kbmod.sorcha_injection.SorchaInjectionConfig` or `None`, optional
        When supplied, the Sorcha path is taken. When `None`, the stock random
        generator runs.
    index : `kbmod.sorcha_injection.SorchaIndex`, optional
        Prebuilt index; constructed from ``injection_config.index_path`` if omitted.
    **random_kwargs
        Forwarded to :func:`kbmod.injection.generate_injection_catalog` on the random
        path (``n_objs_per_ic``, ``guess_distance``, ``mag_range``, ``source_type``).

    Returns
    -------
    catalog : `astropy.table.Table`
    """
    if injection_config is None:
        from kbmod.injection import generate_injection_catalog

        return generate_injection_catalog(ic, search_config, global_wcs, **random_kwargs)

    from .index import SorchaIndex

    if index is None:
        index = SorchaIndex(injection_config.index_path)
    return generate_injection_catalog_from_sorcha(
        ic,
        index,
        injection_config,
        global_wcs=global_wcs,
        guess_distance=random_kwargs.get("guess_distance"),
    )


def write_injection_truth(path, catalog, selection, ic_name=None, patch_id=None, append=True):
    """Append this collection's injections to the cross-collection truth table.

    The truth table is the ground truth for both single-collection recovery and
    cross-collection linking: because ``obj_ids`` is the global Sorcha ``ObjID``, an
    object appearing under two different ``ic_name`` values is one that a correct
    linker must reconnect into a single track.

    Parameters
    ----------
    path : `str`
        Destination ``.parquet`` file.
    catalog : `astropy.table.Table`
        Catalog from :func:`generate_injection_catalog_from_sorcha`.
    selection : `dict`
        The matching selection dict (pass ``return_selection=True`` to obtain it).
    ic_name : `str`, optional
        Identifier for the collection/patch these injections went into.
    patch_id : `int` or `str`, optional
        Patch identifier, when known.
    append : `bool`, optional
        Append to an existing file rather than replacing it. Rows for the same
        ``ic_name`` are replaced, so re-running one patch does not duplicate it.

    Returns
    -------
    n_rows : `int`
        Total number of rows in the truth table after the write.
    """
    import pandas as pd

    n = len(catalog)
    df = pd.DataFrame(
        {
            "ic_name": np.full(n, ic_name if ic_name is not None else "", dtype=object),
            "patch_id": np.full(n, patch_id if patch_id is not None else -1),
            "injection_id": np.asarray(catalog["injection_id"]),
            "obj_ids": np.asarray(catalog["obj_ids"], dtype=str),
            "population": np.asarray(selection["population"], dtype=str),
            "visit": np.asarray(selection["visit"], dtype=np.int64),
            "detector": np.asarray(selection["detector"], dtype=np.int64),
            "visit_mjd": np.asarray(catalog["obstime"], dtype=float),
            "ra": np.asarray(catalog["ra"], dtype=float),
            "dec": np.asarray(catalog["dec"], dtype=float),
            "mag": np.asarray(catalog["mag"], dtype=float),
            "band": np.asarray(selection["band"], dtype=str),
            "det_x": np.asarray(selection["det_x"], dtype=float),
            "det_y": np.asarray(selection["det_y"], dtype=float),
            "plot_x": np.asarray(catalog["plot_x"], dtype=float),
            "plot_y": np.asarray(catalog["plot_y"], dtype=float),
        }
    )
    gd = catalog.meta.get("guess_distance")
    df["guess_distance"] = np.nan if gd is None else float(gd)
    if gd is not None and f"ra_{float(gd)}" in catalog.colnames:
        df["ra_reflex"] = np.asarray(catalog[f"ra_{float(gd)}"], dtype=float)
        df["dec_reflex"] = np.asarray(catalog[f"dec_{float(gd)}"], dtype=float)

    if append and os.path.exists(path):
        prior = pd.read_parquet(path)
        if ic_name is not None and "ic_name" in prior.columns:
            prior = prior[prior.ic_name != ic_name]
        df = pd.concat([prior, df], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Truth table %s now holds %d rows (%d objects)", path, len(df), df.obj_ids.nunique())
    return len(df)
