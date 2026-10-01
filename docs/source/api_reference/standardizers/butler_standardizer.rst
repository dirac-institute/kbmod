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

Why the native date is the midpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rubin's `VisitInfo API contract
<https://github.com/lsst/afw/blob/41b6eb56439ac6a47191309589c181fbfa4659e0/include/lsst/afw/image/VisitInfo.h>`_
defines its date as the exposure midpoint in TAI. The `DateTime.toAstropy()
implementation
<https://github.com/lsst/daf_base/blob/8ea07a8fe4ad399a23e32cd9cfd14d3c70f5cef5/python/lsst/daf/base/dateTime/dateTimeContinued.py>`_
returns an Astropy ``Time`` constructed from TAI MJD with ``scale="tai"``.
Consequently, ``butler.get(visit_ref).date.toAstropy()`` already carries both
the midpoint and its time scale. Calling ``.utc.mjd`` converts that same instant
to UTC and represents it as a floating-point MJD.

This is the Butler ``VisitInfo`` contract for both Rubin and DECam exposures,
including DEEP data ingested into Butler. It does not establish the meaning of
arbitrary standalone FITS date keywords. FITS standardizers must interpret each
supported product's timestamp keyword and time-scale convention separately.

Astropy distinguishes `input time scales
<https://docs.astropy.org/en/stable/time/index.html#creating-a-time-object>`_
from `conversion between scales
<https://docs.astropy.org/en/stable/time/index.html#convert-time-scale>`_. For
example, for a 30.5-second exposure:

.. code-block:: python

    from astropy.time import Time
    import astropy.units as u

    midpoint = Time("2025-05-02T01:02:21.750", scale="tai")
    midpoint.utc.isot  # '2025-05-02T01:01:44.750'
    (midpoint - 15.25 * u.s).utc.isot  # '2025-05-02T01:01:29.500'
    mjd_mid = midpoint.utc.mjd

The 37-second TAI/UTC difference in this example is date-dependent; let Astropy
perform the conversion rather than hard-coding it. Neither converting the scale
nor obtaining ``.mjd`` requires adding half the exposure duration.

The observing-day fix is separate from the removal of the extra midpoint offset.
Its input ``mjd_mid`` is a UTC number, so
``Time(mjd_mid, format="mjd", scale="utc").tai`` preserves the intended instant
before applying the noon-TAI boundary. Constructing
``Time(mjd_mid, format="mjd", scale="tai")`` instead interprets the same numeric
value as TAI and therefore refers to a different instant. The stored KBMOD
midpoints remain UTC MJD.

Existing products and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously generated ImageCollections may contain a midpoint advanced by
``exposureTime / 2 + 0.5`` seconds and a ``mjd_start`` that actually represents
the native midpoint. Loading such a collection does not automatically migrate
its stored metadata. Rebuild affected ImageCollections and dependent WorkUnits,
reprojection products, and time-dependent truth associations into new cache
locations. Preserve the original artifacts for reproducibility.

``ImageCollection.toWorkUnit()`` checks that each reconstructed image timestamp
and its collection row's ``mjd_mid`` are finite and agree within 1 millisecond,
using an absolute tolerance with no relative tolerance. This allows numerical
roundoff while rejecting mixed timing conventions. A mismatch raises an error
identifying the row, available visit/detector identifiers, both epochs, and the
difference in seconds. Historical collections remain readable; the check neither
migrates nor relabels them. Agreement is an internal consistency check, not proof
that two matching legacy timestamps are scientifically correct.

Butler injection performs a separate preflight against fresh ``VisitInfo`` dates
before loading pixels, and checks that rebuilding the injected collection does
not change its epochs. Each nonempty catalog epoch must match exactly one
distinct collection epoch within 1 ms; multiple detectors at the same epoch are
supported. Subset precomputed catalogs to the intended collection before use.
Unmatched or ambiguous catalog epochs raise an error. Empty catalogs and
individual exposures without catalog sources remain valid. This tolerance
accommodates rounding only: catalog coordinates and timestamps are preserved,
not propagated or relabeled.

Retain original products and their software revisions for historical
reproduction. Use rebuilt products for corrected analyses; there is no option
to restore the erroneous Butler midpoint arithmetic. Audit any manual timestamp
offsets in downstream notebooks before using rebuilt inputs to avoid applying
a correction twice. Do not apply a universal offset to unverified products:
exposure duration, ingestion path, and source time scale determine the correction.

When regenerating synthetic truth, use each ephemeris row's actual epoch and
record its time scale. Do not infer that epoch from an old collection's
``mjd_start``, or change timestamps without checking the associated positions.
Preserve the saved standardizer configuration, including mask policy, when
rebuilding a matched comparison.
