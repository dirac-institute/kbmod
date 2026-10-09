Module: butler_standardizer
===========================

.. automodule:: kbmod.standardizers.butler_standardizer
   :members:

Photometric calibration and multi-night tests
--------------------------------------------

Science pixels are multiplied by
``10**((config["zero_point"] - pixel_zero_point) / 2.5)``; variance pixels are
multiplied by the square of that factor. ``pixel_zero_point`` is obtained from
the delivered Exposure's ``photoCalib.instFluxToMagnitude(1.0)``. Missing or
non-finite calibration is rejected rather than replaced by a summary value.
The source Exposure is not modified. The configured target is honored, including
``zero_point=31.4`` to retain nJy units for identity-calibrated inputs.

Rubin can compute ``summaryStats.zeroPoint`` before calibrating image pixels.
For native visit and difference images already calibrated to nJy, the attached
PhotoCalib is the identity and the pixel zero point is 31.4 AB mag. Conversion
to the default KBMOD zero point 31 therefore multiplies science by approximately
0.691831 and variance by 0.478630. Reusing the instrumental summary zero point
would apply an exposure-dependent calibration again. The metadata field
``zeroPoint`` remains the original summary statistic for compatibility and
provenance; it is not the zero point of the standardized output pixels.

For legacy images still in instrumental counts, the attached PhotoCalib supplies
the corresponding input calibration. The current standardizer retains an
exposure-wide scalar approximation: a PhotoCalib call without a position uses
its mean calibration. This change does not apply a spatial calibration map,
propagate calibration uncertainty, homogenize seeing, or impose source colors.
Separate per-exposure PSFs and variances continue to determine search weights.

Testing this change
~~~~~~~~~~~~~~~~~~~

Run the focused regressions from a built checkout::

    PYTHONPATH=src:tests python -m unittest test_butlerstd test_imagecollection test_standardizer

For a native Rubin exposure, independently compare ``summaryStats.zeroPoint``
with ``photoCalib.instFluxToMagnitude(1.0)``. For an identity PhotoCalib and
``BUNIT=nJy``, standardized science and variance should equal the delivered
arrays times the two factors above, irrespective of the summary zero point.
Use finite nonzero pixels when calculating ratios; retain masks and verify
that the source arrays are unchanged. Use real PhotoCalib values, not BUNIT
alone, to decide the conversion.

For a multi-night A/B search, build fresh WorkUnits from identical native
dataset UUIDs under the base revision and this change. Keep the input selection,
target zero point, timestamps, PSFs, masks, reprojection, trajectory grid,
filtering, and color settings fixed. Use separate output/cache directories and
record each software revision. Compare per-exposure psi/phi and fluxes before
comparing combined likelihoods and candidate retention. A non-null
``color_scale`` currently has a separate runner issue; leave it unset for this
isolated calibration comparison.

Previously saved WorkUnits, reprojected images, and psi/phi arrays retain their
old pixel scaling. Loading them with new code does not recalibrate them. Rebuild
dependent products and rerun the search; changing final catalog fluxes alone
cannot restore candidates lost at earlier thresholds. This test branch does
not automatically migrate stored products or establish a recovery improvement.

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

Retain original products and their software revisions for historical
reproduction. Use rebuilt products for corrected analyses; there is no option
to restore the erroneous Butler midpoint arithmetic.
