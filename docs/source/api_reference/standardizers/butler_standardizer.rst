Module: butler_standardizer
===========================

.. automodule:: kbmod.standardizers.butler_standardizer
   :members:

Photometric calibration
-----------------------

Science pixels are multiplied by
``10**((config["zero_point"] - pixel_zero_point) / 2.5)``; variance pixels are
multiplied by the square of that factor. ``pixel_zero_point`` comes from the
delivered Exposure's ``photoCalib.instFluxToMagnitude(1.0)`` (the mean calibration,
without a position). Missing PhotoCalib or invalid calibration raises
``ValueError``; the source Exposure is not modified. A target ``zero_point=31.4``
retains nJy units for identity-calibrated inputs. Spatial calibration maps,
calibration uncertainty, PSF homogenization, and source colors are separate.

Rubin can compute ``summaryStats.zeroPoint`` before calibrating pixels to nJy.
For identity-calibrated nJy images, the pixel zero point is 31.4 AB mag; the
default target 31 multiplies science by approximately 0.691831 and variance by
0.478630. ``zeroPoint`` remains the original summary statistic for provenance,
not the zero point of standardized pixels. ``unravel_results`` uses the stored
standardizer ``config["zero_point"]`` for catalog magnitudes and rejects missing
or mixed output calibrations unless a known flux zero point is supplied explicitly.

Previously saved WorkUnits, reprojected images, and psi/phi arrays retain their
old scaling. Loading them with new code does not recalibrate them. Rebuild
these products and rerun the search; changing catalog magnitudes cannot restore
candidates discarded by earlier thresholds or repair incorrectly scaled pixels.

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
