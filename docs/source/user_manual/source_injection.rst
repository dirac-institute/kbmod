Injecting point sources and trails
=================================

``kbmod.injection.inject_sources_into_ic`` accepts a user-supplied Astropy
table and calls Rubin's ``VisitInjectTask`` on native Butler exposures.
It requires an LSST Science Pipelines environment with ``source_injection``.
The catalog can mix ``Star`` and ``Trail`` rows; no KBMOD search or
reprojection is required to inject them.

Catalog conventions
-------------------

* ``ra``, ``dec``: source center in degrees. A trail is centered here.
* ``obstime``: exposure midpoint MJD, copied exactly from ``ic["mjd_mid"]``.
* ``injection_id``: unique injection-row ID; ``obj_ids``: persistent object ID
  if the same object appears in multiple exposures.
* ``source_type``: ``Star`` or ``Trail`` (other Rubin-supported types pass through).
* ``trail_length``: positive length in pixels of the native exposure.
* ``beta``: angle in degrees counterclockwise from image +x, not a sky position
  angle. Omit the column for horizontal trails (zero degrees). If present,
  trail values must be finite and unmasked. Mask unused geometry on stars.

Positions and geometry must describe each target exposure's pixel frame.
Rows are selected by ``obstime``; the wrapper does not derive trail geometry
from velocities, durations, or ephemerides, or select rows by detector ID.
For simultaneous detectors with different pixel geometry, inject each detector's
collection with its corresponding catalog.

Total magnitudes versus Rubin magnitudes
---------------------------------------

The existing injection API passes ``mag`` through unchanged. In Rubin's
``Trail`` renderer, total flux is the flux implied by ``mag`` multiplied by
``trail_length`` in pixels. Thus the same native ``mag`` produces more total
flux for a longer trail.

Use ``prepare_injection_catalog`` when starting from total object magnitudes:
it copies the input, retains ``mag_total`` and truth columns, and sets
``mag = mag_total + 2.5 * log10(trail_length)`` for trails. For other sources,
``mag = mag_total``. Repeating preparation is safe because it always uses
``mag_total``. Supply zero-length sources as ``Star`` rows instead.

.. code-block:: python

   import numpy as np
   from astropy.table import Table, MaskedColumn
   from kbmod.injection import prepare_injection_catalog, inject_sources_into_ic

   # ic is a Butler-backed ImageCollection for one detector exposure;
   # choose ra_deg, dec_deg inside that exposure. butler accesses its repository.
   truth = Table({
       "injection_id": [0, 1],
       "obj_ids": [0, 1],
       "obstime": [ic["mjd_mid"][0]] * 2,
       "ra": [ra_deg, ra_deg + 0.01],
       "dec": [dec_deg, dec_deg],
       "source_type": ["Star", "Trail"],
       "mag_total": [22.0, 22.0],
   })
   truth["trail_length"] = MaskedColumn([0.0, 10.0], mask=[True, False])
   truth["beta"] = MaskedColumn([0.0, 45.0], mask=[True, False])
   catalog = prepare_injection_catalog(truth)
   np.testing.assert_allclose(catalog["mag"], [22.0, 24.5])
   injected_ic, rendered_catalog = inject_sources_into_ic(ic, catalog, butler)

The helper uses standard column names. It accepts units convertible to
magnitudes, pixels, and degrees; angular trail lengths must first be converted
to pixels for the target exposure. Existing Rubin-format catalogs can instead
be passed directly to ``inject_sources_into_ic``, including configured column
aliases. The helper leaves the injector's noise and variance configuration alone.

Keep both the requested and returned catalogs: rendering can reject or clip
sources. Serializing an ImageCollection alone does not persist injected pixels;
materialize its WorkUnit or explicitly save the modified exposures.

Validation
----------

``tests/test_injection.py`` covers catalog conversion and wrapper behavior
without LSST. ``tests/test_trail_injection_lsst.py`` additionally checks actual
Rubin rendering for total flux, centroid, length, orientation, and the
point-source limit. It skips when LSST is unavailable. Run both from ``tests/``
with ``python -m unittest test_injection test_trail_injection_lsst``.

.. autofunction:: kbmod.injection.prepare_injection_catalog
