PSF and Reprojection Contracts
==============================

This note defines the units, normalization, centering, and coordinate
conventions that the PSF and reprojection code must obey. It exists because
these choices are individually small and collectively decisive: a half-pixel
convention error or a squared-versus-unsquared scale factor changes recovered
fluxes without producing an obvious failure anywhere.

.. note::
   Status: Phase 0. The configuration and measurement contracts below are
   implemented. Variance and covariance handling is **not** yet correct; see
   `Variance semantics`_ for what is deferred and what must not be claimed in
   the meantime.


Coordinate and array conventions
--------------------------------

These four rules apply throughout ``kbmod.psf_reprojection`` and
``kbmod.reprojection``:

* Images are NumPy arrays indexed ``[y, x]``.
* Pixel coordinates are ``(x, y)`` floats in the Astropy convention, where
  ``(0, 0)`` is the **center** of the first pixel.
* Covariance and second-moment matrices are ordered ``[[xx, xy], [xy, yy]]``.
* Angular quantities are arcseconds, with the right-ascension axis multiplied
  by ``cos(dec)`` so that the units are true angle on the sky.

Rubin and Astropy do not share a pixel origin convention, and Rubin stamps carry
their own ``xy0`` bounding-box origin. Converting a Rubin image to an array with
``.array`` and discarding that origin loses the stamp's placement, which is the
single most likely source of a one-pixel error in this code. Retain the origin
alongside the array until the stamp is deliberately placed.

Tests that exercise these conventions must use **asymmetric** sources. A
circular Gaussian centered in a square frame is invariant under transpose and
under x/y swaps, so it cannot detect the errors this section is meant to
prevent.


Science units and the scaling contract
--------------------------------------

Two independent scalings apply to a KBMOD image, and they compose.

**Zeropoint harmonization.** ``ButlerStandardizer`` already rescales each
exposure to a common photometric zeropoint, and it correctly squares the factor
for variance:

.. code-block:: python

    zp_correct = 10 ** ((metadata["zeroPoint"] - config.zero_point) / 2.5)
    science = exposure.image.array / zp_correct
    variance = exposure.variance.array / zp_correct**2

This is the existing, correct precedent for the contract: **a linear scale on
science is a squared scale on variance.** Any new scaling introduced anywhere in
the pipeline must follow the same rule.

**Resampling flux mode.** ``reproject_adaptive`` can either preserve surface
brightness (``conserve_flux=False``) or total flux (``conserve_flux=True``).
The difference is exactly the pixel-area Jacobian, and it is not subtle when
the frames differ in scale. Measured on generated isolated sources:

.. list-table::
   :header-rows: 1

   * - Transform
     - Pixel-area ratio
     - Recovered flux, ``conserve_flux=False``
     - Recovered flux, ``conserve_flux=True``
   * - identity
     - 1.000
     - 1.0006
     - 1.0006
   * - rotation, 30 deg
     - 1.000
     - 1.0001
     - 1.0001
   * - scale 0.77
     - 0.593
     - 1.6854
     - 0.9993
   * - scale 1.30
     - 1.690
     - 0.5925
     - 1.0014
   * - anisotropic 1.25 in y
     - 1.250
     - 0.7996
     - 0.9995

The flux-conserving operator recovers the injected flux under every transform.
The surface-brightness operator changes it by the pixel-area ratio: reprojecting
to pixels 1.3x coarser loses 41% of the summed counts.

**Therefore, for point-source photometry the production flux mode is**
``conserve_flux=True``. Note that a rotation alone does not distinguish the two
modes, so a test suite that only rotates will not detect a wrong choice.

.. important::
   The recommendation above is recorded, but the shipped default remains
   ``conserve_flux=False`` so that Phase 0 changes no results. Flipping it is a
   deliberate step, taken together with the end-to-end photometric validation
   that confirms it, not as a side effect of making the options explicit.

A normalized PSF kernel is invariant under both scalings. That is a useful
consistency check, not a coincidence to depend on silently.


The reprojection configuration
------------------------------

Every ``reproject_adaptive`` option that changes the numerical result is modeled
by :class:`~kbmod.reprojection_config.AdaptiveReprojectionConfig` and passed
explicitly. KBMOD previously passed two options and inherited twelve, which made
results depend on the installed ``reproject`` release rather than on KBMOD
source.

Two fields deliberately differ from the library defaults because KBMOD has
always overridden them. **Do not "correct" either one:**

* ``bad_value_mode="ignore"`` (library default ``"strict"``)
* ``roundtrip_coords=False`` (library default ``True``)

The science image and its effective PSF must be produced by the same operator
with the same options, so a single configuration object is shared by both paths
rather than each specifying its own arguments. The configuration carries a
provenance record including the ``reproject`` version and a stable hash suitable
as a cache key.

Canary tests fail loudly if a library default drifts, if a new numerically
relevant option appears, or if the resampler's numerics change on a pinned
input. Library defaults have changed historically; the canaries exist so that
such a change is a test failure rather than a silent shift in the science.


PSF normalization and centering
-------------------------------

For kernels this project generates:

* finite and non-negative in every pixel;
* odd-sized and square;
* normalized to sum to 1 only **after** any clipped or lost flux has been
  measured and reported. Normalizing a truncated kernel hides the truncation;
* large enough that further support changes nothing measurable. A fixed
  "3 sigma" support is **not** valid for realistic PSFs: Moffat-like wings
  require materially more support than a Gaussian at the same FWHM to reach the
  same enclosed energy.

For kernels read from existing files:

.. warning::
   Legacy kernels are returned **byte-for-byte** and are never renormalized on
   read. Stored kernels are not necessarily normalized — those in the shipped
   reprojection fixture sum to 0.994608 — and silently renormalizing them would
   change the results of reprocessing old data. Legacy kernels carry unknown
   provenance and must never be labeled as effective common-frame PSFs.


Separating pixel scale from interpolation
-----------------------------------------

Reprojection changes the point-source response in two ways that must not be
conflated:

**Geometry** changes the PSF's pixel width without losing information. A source
in a frame resampled to finer pixels covers more pixels; nothing has been
blurred.

**Interpolation** adds width. This is real information loss and is what the
effective PSF must capture.

A single pixel FWHM cannot tell these apart. Compare in angular units, using the
local WCS Jacobian ``J`` that maps pixel offsets to angular offsets:

.. math::

   C_{\rm sky} = J \, C_{\rm pix} \, J^{T}

and compare the measured output moments against the purely geometric prediction
``A C_native A^T``, where ``A`` is the locally affine native-to-output pixel
map. Attribute the residual to interpolation only when the **matrix** difference
is meaningful. Do not reduce an anisotropic or non-Gaussian PSF to a single
quadrature FWHM.

Worked example. A source of 5.28 px FWHM reprojected to pixels 0.8x the native
size measures 6.87 px in the output, an apparent 30% broadening:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Value
     - Interpretation
   * - native FWHM
     - 5.282 px
     - as injected
   * - geometric prediction
     - 6.603 px
     - pixel-scale change alone, ``5.282 / 0.8``
   * - measured output
     - 6.868 px
     - what the science path produces
   * - interpolation residual
     - 1.040x
     - the only part that is blur

So roughly 25 of the 30 percentage points are pixel-scale geometry and about 4
are interpolation. Reporting the 30% as "blur" would be wrong, and correcting
for it as though it were would bias the PSF.

The interpolation residual is close to an additive width in **output** pixels,
so its fractional effect shrinks as the PSF widens: under an identity transform
it is about 6.7% for a 4 px FWHM source and about 4.0% for a 5.3 px source. It
is not a fixed multiplicative factor and must not be modeled as one.


Variance semantics
------------------

.. warning::
   KBMOD does not currently propagate variance correctly through reprojection,
   and this project does not fix it.

Science, variance, and mask are presently passed through the same interpolator.
For a linear resampler with weights :math:`w_{ij}`, science transforms as

.. math::

   y_i = \sum_j w_{ij} x_j

but independent input variances transform as

.. math::

   \mathrm{Var}(y_i) = \sum_j w_{ij}^2 \, \mathrm{Var}(x_j)

Interpolating variance values with :math:`w_{ij}` is a different operation.
Resampling also induces off-diagonal covariance between output pixels, which
remains even once the diagonal is correct, and which a matched filter is
directly sensitive to.

Consequences that hold until the variance and covariance work lands:

* **Do not claim calibrated signal-to-noise.** S/N comparisons between kernel
  choices are meaningful as *relative* statements under uncorrected covariance,
  and must be labeled that way.
* Do not treat a visually plausible variance plane as evidence of correctness.
* ``bad_value_mode="ignore"`` renormalizes interpolation weights near masked
  pixels, perturbing both flux and noise there. Regions within a PSF radius of a
  mask deserve separate treatment.
* ``boundary_mode="strict"`` can clip a PSF stamp or produce NaN. Report the
  loss rather than normalizing it away.
