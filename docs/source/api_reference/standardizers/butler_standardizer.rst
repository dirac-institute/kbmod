Module: butler_standardizer
===========================

.. automodule:: kbmod.standardizers.butler_standardizer
   :members:

Timestamp convention and migration
----------------------------------

``mjd_mid`` is the native ``VisitInfo.date`` converted to UTC MJD. That date
already represents the exposure midpoint; no additional half-exposure or
half-second offset is applied. ``mjd_start`` is a nominal start derived by
subtracting half the exposure duration. It is not an independent shutter-open
measurement. The observing-day helper accepts UTC MJD and retains the existing
noon-TAI boundary convention.

Previously generated ImageCollections may contain a midpoint advanced by
``exposureTime / 2 + 0.5`` seconds and a ``mjd_start`` that actually represents
the native midpoint. Loading such a collection does not automatically migrate
its stored metadata. Rebuild affected ImageCollections and dependent WorkUnits,
reprojection products, and time-dependent truth associations into new cache
locations. Preserve the original artifacts for reproducibility.

When regenerating synthetic truth, use each ephemeris row's actual epoch and
record its time scale. Do not infer that epoch from an old collection's
``mjd_start``, or change timestamps without checking the associated positions.
Preserve the saved standardizer configuration, including mask policy, when
rebuilding a matched comparison.
