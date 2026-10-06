# Prototype: numeric WCS persistence

Status: sample implementation for engineering and scientific review, not a cache migration or a general WCS interchange standard. Base: `origin/main` at `42356120f3e68bff1f62bddda6d2bf20366d7516`. Branch: `prototype/lossless-wcs-roundtrip`.

## Problem and contract

The previous `serialize_wcs` first called `WCS.to_header(relax=True)`, losing low bits in core parameters through WCSLIB formatting. `deserialize_wcs` then constructed `WCS(dict)`, passing SIP coefficients through FITS card float formatting and losing additional bits. ImageCollection duplicated that path. A header comparison can therefore pass while the original in-memory transformation differs from the reloaded transformation. Reprojection can amplify such differences; an angular tolerance is not evidence of identical numerical inputs.

Version 1 preserves the **encoded, normalized binary64 TAN/SIP transformation parameters**. It does not preserve every WCS Python attribute, FITS card/comment ordering, NaN payload bits, arbitrary distortion models, or all ancillary time/observer metadata. It does not promise identical WCSLIB/reprojection output across library versions, operating systems, or hardware.

The implementation normalizes lazy WCSLIB defaults on a deep copy before capture. An unset pole may become its calculated value in the saved state. Finite encoded parameters are then bit-preserved; caller data are not refitted, rounded, or migrated. Existing NaN equinox denotes an unset value. Nonfinite active CRPIX/CRVAL/linear/SIP parameters are rejected.

## Record

`serialize_wcs` emits a compact JSON object with:

- `__kbmod_wcs__: 1`: format version; unknown versions fail closed.
- `fidelity: "tan-sip-binary64"`: supported numeric state is authoritative.
- `header`: ordinary FITS-compatible WCS values for interchange, explicitly approximate.
- `state`: binary64 values stored as hexadecimal strings with explicit array shapes, plus string metadata and dimensions.

State includes CRPIX/CRVAL, CD or PC and CDELT, longitude/latitude pole, equinox, CTYPE/CUNIT/CNAME, RADESYS/name/alternate key, pixel shape/bounds, and SIP A/B/AP/BP arrays and SIP CRPIX. CD versus PC representation is retained. Readback constructs the ancillary header metadata, restores numeric state through Astropy setters/Sip, and initializes WCSLIB. No alternate implementation of the transformation is introduced.

Supported models are two-dimensional celestial TAN or TAN-SIP in degrees with default WCSLIB projection bounds, without PV/PS/CROTA, simultaneous CD+PC, or lookup-table distortions. Nondefault `bounds_check` settings can alter finite/invalid coordinate classification and are explicitly outside the exact subset. Optional inverse SIP arrays and unknown image dimensions are supported. Out-of-scope models default to `fidelity: "fits-header"` with a reason, retaining the historical approximate behavior. **Header-only records can lose lookup-table distortions and nondefault projection-bounds settings because those are not serialized.** Callers requiring exact support use `serialize_wcs(wcs, require_exact=True)` to reject them. There is no per-row warning flood or claim of general losslessness.

Legacy flat JSON headers and empty/None strings remain readable. New JSON envelopes require updated KBMOD readers: old code that passes the JSON dictionary straight into `WCS` is not compatible. This reader compatibility change needs a release/version decision before production adoption.

## Persistence paths

One shared codec now handles:

1. ImageCollection creation from Standardizers, iteration/get_wcs, ECSV and packed FITS tables.
2. RegionSearch patch/global WCS export and ImageCollection global WCS loading; the result-matcher CLI uses that getter. Legacy FITS header strings and flat JSON global records remain readable. Redundant legacy shape columns must agree with a versioned payload, rather than silently overriding its dimensions.
3. WorkUnit `org_img_meta` WCS columns, including `per_image_wcs` and `ebd_wcs`. Mixed columns beginning with None use the WCS codec instead of falling through to object stringification.
4. WorkUnit global WCS in both monolithic and sharded primary metadata: an authoritative `KBWCS` string card alongside ordinary WCS cards. FITS CONTINUE cards carry long JSON; no HDU positions change. Legacy files without `KBWCS` retain the old header path.
5. Results WCS metadata through its existing shared-codec calls.

`append_wcs_to_hdu_header(..., include_exact=True)` creates the authoritative header payload. Reusing this helper on a header that already has `KBWCS` refreshes the payload for WCS objects, or removes it for ordinary header dictionaries. **External tools that edit WCS cards directly must also update or remove `KBWCS`; otherwise the numeric payload remains authoritative.** Ordinary FITS readers see approximate interchange cards. Per-image SCI/VAR/MSK headers still serve interchange; WorkUnit's authoritative constituent geometry resides in its metadata table.

This proposal does not repair already rounded artifacts, automatically invalidate old caches, select a new astrometric fit, or change image quantization, noise, photometry, masks, or timing policy. Reproducibility records should include the codec version and runtime versions; a rollout should preserve old artifacts rather than silently replacing them.

## Validation

`tests/test_wcs_persistence.py` uses a non-square 37×23 TAN/SIP fixture with long-mantissa coefficients. It first reproduces the legacy core/SIP precision loss. It checks exact parameter bytes and exact forward/inverse coordinates across first and second new-codec cycles, CD and PC/CDELT, with/without inverse SIP, and no SIP. It also checks legacy records, unsupported models/strict behavior, version/shape/missing-field validation and invalid transform values.

Actual ECSV and FITS disk tests cover Standardizer→ImageCollection, packed collections, RegionSearch global metadata, Results, WorkUnit monolithic and sharded metadata, None-leading WCS columns, long FITS CONTINUE strings containing apostrophes, and refreshing an existing authoritative payload. A small serial CPU `reproject_work_unit` regression compares SCI/VAR/mask exactly before and after two lossless storage cycles. Image writes explicitly use `quantize_level=0` so image quantization cannot obscure the geometry check.

Tests use the existing `kbmod` conda Python3.12/Astropy8.0.1 environment. Its installed extension predates current main, so a CPU-only extension was compiled inside this worktree from current source using local copies of the exact pinned Eigen/pybind11 references and existing CMake; no package installation or environment change was needed. These are genuine local persistence and reprojection tests, not Rubin/USDF/GPU or cross-version qualification.

## Review decisions before production

- Agree on schema/version naming and the deliberate old-reader incompatibility.
- Decide whether production callers should demand exact support rather than permit labeled header-only fallback.
- Consider metadata-size/large-catalog benchmarks; hex encodings and the interchange header increase storage, and global FITS CONTINUE cards trade convenience for header length.
- Require any future model extensions to add explicit state coverage and numerical/disk regressions. Preserve the bounded exactness claim rather than treating header equality as a universal guarantee.
