Sorcha Synthetic Source Injection
=================================

KBMOD can inject synthetic moving objects into real exposures so that recovery
efficiency can be measured against a known truth. There are two ways to produce the
injection catalog.

The stock generator, :py:func:`~kbmod.injection.generate_injection_catalog`, fabricates
a set of *random* ecliptic trajectories for each ImageCollection. That is fine for
single-collection recovery efficiency, but the objects it creates are collection-local:
they carry no identity shared between collections, so nothing about cross-collection
linking can be tested with them.

The :py:mod:`kbmod.sorcha_injection` subpackage instead draws from a *pre-built
ephemeris* -- the Sorcha DP2 synthetic populations -- and matches objects to each patch
spatially and temporally. Objects are injected at their real observed positions and
magnitudes, and the Sorcha ``ObjID`` is preserved as the injection ``obj_ids``. Because
a synthetic object's track naturally crosses several patches over time, the *same*
``ObjID`` is injected into every overlapping collection, which is what makes
cross-collection linking recall and precision measurable.

Both paths emit the identical astropy Table schema, so
:py:func:`~kbmod.injection.inject_sources_into_ic` consumes either unchanged.


Building the index
------------------

The raw Sorcha outputs are roughly 3 TB across 10\ :sup:`9` rows, one row per
(object, visit) detection. Scanning that per patch is hopeless, so the scan is done once
into a compact parquet dataset partitioned by population and night:

.. code-block:: bash

    python -m kbmod.sorcha_injection build-index \
        --out /path/to/sorcha_index \
        --populations cc hc cen de \
        --mag-max 27 \
        --collections /path/to/combined_all.collection \
        --workers 32

Passing ``--collections`` is strongly recommended. It restricts the index to visits that
appear in the collections you actually intend to inject into; for the DP2 collections
that is about 6% of the simulated visits and is the single largest size reduction
available. Combined with the magnitude ceiling, a four-population index over the DP2
visits reduces 1.29 billion scanned rows to about 10 million (roughly 350 MB) in well
under a minute on a many-core machine.

Inspect an existing index with::

    python -m kbmod.sorcha_injection inspect /path/to/sorcha_index


Generating a catalog
--------------------

.. code-block:: python

    from kbmod import ImageCollection
    from kbmod.injection import inject_sources_into_ic
    from kbmod.sorcha_injection import (
        SorchaIndex,
        SorchaInjectionConfig,
        generate_injection_catalog_from_sorcha,
        write_injection_truth,
    )

    ic = ImageCollection.read("patch_167.collection")
    index = SorchaIndex("/path/to/sorcha_index")
    cfg = SorchaInjectionConfig(index_path=index.path, mag_range=(None, 27.0))

    catalog, selection = generate_injection_catalog_from_sorcha(
        ic, index, cfg, return_selection=True
    )
    injected_ic, injected_cats = inject_sources_into_ic(ic, catalog, butler)

    # Ground truth for recovery and cross-collection linking.
    write_injection_truth("injection_truth.parquet", catalog, selection, ic_name="patch_167")

The patch WCS and guess distance are read from the collection's ``global_wcs`` and
``helio_guess_dist`` columns when present, so a patch ImageCollection produced by
:py:class:`~kbmod.region_search.RegionSearch` needs no extra arguments.

To switch between the two generators from configuration, use
:py:func:`~kbmod.sorcha_injection.generate_injection_catalog_for_ic`. It takes the random
path when ``injection_config`` is ``None``, which keeps existing callers unaffected.
:py:meth:`~kbmod.sorcha_injection.SorchaInjectionConfig.from_dict` understands a
``source: "random" | "sorcha"`` key so a single config block can express either.


How the matching works
----------------------

**Temporal.** Sorcha was driven by a pointing database whose ``observationId`` values are
real Rubin exposure ids, and the ImageCollections built from the DP2 butler carry the
same identifiers in their ``visit`` column. The join is therefore an exact integer match
on visit id, with no time tolerance involved. Verify this for a new collection with
:py:func:`~kbmod.sorcha_injection.validate.check_ic_pointing_alignment`; if the visit-id
overlap is not essentially complete, the Sorcha run and the collections describe
different survey realizations and nothing downstream can be trusted.

**Epoch.** Sorcha's ``fieldMJD_TAI`` equals ``observationStartMJD + visitTime / 2`` in the
pointing database, which turns out to be the same physical instant as the
ImageCollection's exposure **start** (differing only by the 37 s TAI-UTC offset). KBMOD's
``obstime`` is mid-exposure, about 15.5 s later. The selector propagates each position
forward across that gap using the object's own on-sky rate. The correction is small --
sub-milliarcsecond for cold classicals, up to roughly 0.8 pixels for the fastest
centaurs -- and can be disabled with ``correct_epoch_to_mid_exposure=False``.

**Spatial.** Each row is tested against its *own* per-visit position, not a mean position,
so fast movers land correctly. A row must fall inside the pixel bounds of a specific
(visit, detector) exposure in the collection, which is also what reproduces chip gaps.
When ``require_in_patch`` is set (the default) it must additionally fall inside the
patch's ``global_wcs`` footprint, so injected objects are actually searchable.

**Density.** ``max_objs_per_patch`` samples whole tracks by ``ObjID``. Thinning row by row
would punch holes in tracks and destroy the property that makes this useful for linking.


Coordinate frames
-----------------

The frame convention matches the stock generator exactly:

* ``ra`` / ``dec`` are always **observed-frame** coordinates, which is what
  ``VisitInjectTask`` needs because it renders into the original, un-reprojected
  exposures. Sorcha's ``RA_deg`` / ``Dec_deg`` are already observed topocentric
  positions, so unlike the stock generator nothing has to be inverted.
* When the patch is reflex-corrected at a guess distance ``D``, ``ra_{D}`` / ``dec_{D}``
  hold the patch-frame coordinates. KBMOD results live in that frame, so
  :py:func:`~kbmod.injection.match_injection_results` reads those columns, and
  ``plot_x`` / ``plot_y`` are computed from them.

Because the join is on visit id, a row's Sorcha ``optFilter`` is by construction the band
of the exposure it is injected into, so ``mag`` is always already in the right band and no
colour transformation is applied.


A note on track completeness
----------------------------

Sorcha output is often a *detection* table rather than a full ephemeris, which would
leave gaps in faint objects' tracks and invalidate whole-track injection. That is not the
case for the DP2 run used here: it was configured with ``default_snr_cut = False``, with
both the fading function and the linking filter disabled, and with the pointing database's
``fiveSigmaDepth`` hardcoded to 34.5 for every visit, so every source sits far "above
depth" and nothing is filtered photometrically. The only losses are geometric --
``camera_model = footprint``, which correctly removes objects landing in chip gaps -- and
a saturation cut at magnitude 16 that is irrelevant at these magnitudes.

If you point this machinery at a different Sorcha run, re-check that configuration before
assuming tracks are complete.


Validation
----------

:py:mod:`kbmod.sorcha_injection.validate` provides the checks in the order you should
trust them:

.. code-block:: python

    from kbmod.sorcha_injection.validate import validate_end_to_end

    passed, checks = validate_end_to_end(
        ic, index, catalog, pointing_db=None, truth_path="injection_truth.parquet"
    )

Individual checks cover pointing-database alignment, index integrity, catalog schema and
astrometric self-consistency, per-object track lengths, and the cross-collection shared
``ObjID`` property.
