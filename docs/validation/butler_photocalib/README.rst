Butler photometry validation
===========================

This evidence accompanies the calibration change; it is not a recovery or
completeness measurement.

Native inputs
-------------

``native_calibration.json`` records read-only checks from slacd against the
DP2 preparation repository. Each exact dataset UUID and processing run is
included. All ten sampled inputs (five visit images and five difference
images, across grizy) have identity PhotoCalib and nJy pixel units. For example,
the g-band difference image ``019d847e-a6f2-78a5-a7b3-147b70153596`` gives::

    photoCalib.getCalibrationMean()           = 1.0
    photoCalib.instFluxToMagnitude(1.0)        = 31.4
    info.getSummaryStats().zeroPoint          = 32.224116464399806

The other bands also have pixel zero point 31.4, while their summary values
vary. These are exposure-specific examples, not per-band correction constants.
This validates the sampled DP2 products; it does not assert that every DP1,
DP2, legacy calexp, or future pipeline product has identity PhotoCalib. The
implementation reads the attached calibration rather than assuming identity.

To independently inspect a native input, activate a compatible Rubin stack and
run from the repository root::

    python docs/validation/butler_photocalib/inspect_native_calibration.py \
        --repo /sdf/group/rubin/repo/dp2_prep \
        --dataset-id 019d847e-a6f2-78a5-a7b3-147b70153596

The script retrieves the persisted Exposure components, without downloading
pixels or importing KBMOD. It prints the tuple above along with the UUID/run.
Repository access is required; this is not part of offline CI.

Pixel validation already performed
----------------------------------

At commit ``1a64c66ead7a17798e3bd533ffc66e80a95a0186``, the complete branch
ButlerStandardizer module was loaded in memory using existing slacd KBMOD/LSST
dependencies. The native constructor and cold science/variance methods passed
on ten 32x32 cutouts (10,240 checked pixels). Both target zero points 31 and
31.4 were checked against Rubin PhotoCalib. Science and variance ratios to the
independent calibration were 1.0. Two full 4004x4096 g-band images, one visit
and one difference image, also passed ``toLayeredImage`` export with masking
disabled to isolate calibration. Input arrays and summary zero points were
preserved, and the standardizer released its exposure after export.

The original source and evidence hashes are recorded in
``pixel_validation.json``. This was not a fresh full-package build on slacd,
a reprojection test, or a multi-night GPU search. Subsequent catalog-export
and metadata-option fixes are covered by offline regression tests; this dated
pixel evidence should not be described as a new full-workflow validation.

Offline regressions
-------------------

The review follow-up passed 93 tests in the ``kbmod`` conda environment with
a CPU build of the native extension::

    PYTHONPATH=src:tests python -m unittest test_util_functions test_butlerstd test_imagecollection test_standardizer test_injection

These include standardized-flux-to-catalog magnitudes at targets 31 and 31.4,
packed collection serialization, inconsistent calibration rejection, and all
three optional metadata configuration overrides. A finite positive flux scale
can still overflow or underflow when squared, so the variance-scale guard is
retained and tested with scales of ``1e200`` and ``1e-200``.

Compatibility and comparison
----------------------------

Missing PhotoCalib now raises ValueError instead of relying on the summary
zero point. Missing or non-finite calibration must be repaired upstream.
The exposure-wide mean calibration approximation remains; this change does
not apply a spatial calibration map or a source-color model.

``unravel_results`` uses the persisted standardizer target zero point, including
packed collections. It rejects missing, invalid, or inconsistent output units.
Callers without this provenance can supply ``zero_point=...`` only when the
result flux units are independently known. The exported magnitude repeats the
fitted common-flux magnitude, not separate measurements in each exposure/band.
Neither a stored target nor an explicit argument repairs historical pixel
scaling or reverses color-template scaling.

For an A/B search, use identical native UUIDs and fresh, separate WorkUnit and
psi/phi caches for the base and changed revisions. Keep timing, PSFs, masks,
reprojection, trajectory grid, filtering, and color assumptions fixed. First
compare per-exposure flux/psi/phi, then joint likelihoods and candidate retention.
Do not infer improved recovery solely from the calibration tests.
