"""Version 1 numeric TAN/SIP persistence; FITS headers are interchange only.

Finite binary64 parameters are encoded with float.hex, avoiding both WCSLIB's
header formatting and FITS card float formatting. This is not a general WCS
object serializer or a cross-version WCSLIB numerical reproducibility promise.
"""

import warnings

import numpy as np
from astropy.wcs import Sip, WCS

_MARKER = "__kbmod_wcs__"
_FIDELITY = "tan-sip-binary64"
_STRINGS = ("ctype", "cunit", "cname")
_SCALARS = ("lonpole", "latpole", "equinox")


def _array(value):
    value = np.asarray(value, dtype=np.float64)
    return {"shape": list(value.shape), "hex": [float(x).hex() for x in value.flat]}


def _read_array(value, shape=None):
    if not isinstance(value, dict) or set(value) != {"shape", "hex"}:
        raise ValueError("Invalid numeric WCS array")
    dims = value["shape"]
    if (
        not isinstance(dims, list)
        or any(type(n) is not int or n < 0 for n in dims)
        or (shape is not None and dims != list(shape))
    ):
        raise ValueError("Invalid numeric WCS array shape")
    if len(value["hex"]) != int(np.prod(dims)):
        raise ValueError("Invalid numeric WCS array length")
    return np.array([float.fromhex(x) for x in value["hex"]], dtype=np.float64).reshape(dims)


def _unsupported(wcs):
    core = wcs.wcs
    if wcs.pixel_n_dim != 2 or wcs.world_n_dim != 2 or not wcs.is_celestial:
        return "requires two celestial axes"
    if any(str(c).removesuffix("-SIP")[-4:] != "-TAN" for c in core.ctype):
        return "only TAN and TAN-SIP projections are supported"
    if any(getattr(wcs, key) is not None for key in ("cpdis1", "cpdis2", "det2im1", "det2im2")):
        return "lookup-table distortion is not supported"
    if core.get_pv() or core.get_ps() or core.has_crota() or (core.has_cd() and core.has_pc()):
        return "PV/PS/CROTA or simultaneous CD and PC is not supported"
    if any(str(unit) not in ("", "deg") for unit in core.cunit):
        return "only degree-valued celestial axes are supported"
    if core.cel.prj.bounds != 7:
        return "nondefault WCSLIB projection bounds are not supported"
    active = [core.crpix, core.crval, core.cd if core.has_cd() else core.get_pc()]
    if not core.has_cd():
        active.append(core.cdelt)
    if wcs.sip is not None:
        active.extend(
            getattr(wcs.sip, key)
            for key in ("a", "b", "ap", "bp", "crpix")
            if getattr(wcs.sip, key) is not None
        )
    if any(not np.all(np.isfinite(value)) for value in active):
        return "nonfinite active transform parameters"
    return None


def encode_wcs(wcs, *, require_exact=False):
    # Even model introspection can initialize WCSLIB's lazy state.
    # Astropy deepcopy also resets custom bounds, so inspect this flag first.
    projection_bounds = wcs.wcs.cel.prj.bounds
    normalized = wcs.deepcopy()
    reason = (
        "nondefault WCSLIB projection bounds are not supported"
        if projection_bounds != 7
        else _unsupported(normalized)
    )
    if reason and (require_exact or reason == "nonfinite active transform parameters"):
        raise ValueError(f"WCS cannot be serialized exactly: {reason}")
    # Normalize lazy WCSLIB defaults on a copy, never mutate the caller.
    normalized.wcs.set()
    header = dict(normalized.to_header(relax=True))
    if normalized.pixel_shape is not None:
        for axis, size in enumerate(normalized.pixel_shape, 1):
            header[f"NAXIS{axis}"] = size
    record = {_MARKER: 1, "fidelity": "fits-header" if reason else _FIDELITY, "header": header}
    if reason:
        record["reason"] = reason
        return record
    core = normalized.wcs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # CDELT is inactive when CD is present.
        state = {key: _array(getattr(core, key)) for key in ("crpix", "crval", "cdelt")}
    state.update({key: [str(x) for x in getattr(core, key)] for key in _STRINGS})
    state.update({key: float(getattr(core, key)).hex() for key in _SCALARS})
    state.update({key: getattr(core, key) for key in ("radesys", "name", "alt")})
    state["linear_type"] = "cd" if core.has_cd() else "pc"
    state["linear"] = _array(core.cd if core.has_cd() else core.get_pc())
    state["pixel_shape"] = list(normalized.pixel_shape) if normalized.pixel_shape is not None else None
    state["pixel_bounds"] = _array(normalized.pixel_bounds) if normalized.pixel_bounds is not None else None
    state["sip"] = None
    if normalized.sip is not None:
        sip = normalized.sip
        state["sip"] = {
            key: None if getattr(sip, key) is None else _array(getattr(sip, key))
            for key in ("a", "b", "ap", "bp", "crpix")
        }
    record["state"] = state
    return record


def decode_wcs(record):
    """Read legacy headers or validate and reconstruct a versioned payload."""
    try:
        return _decode_wcs(record)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed KBMOD WCS record") from error


def _decode_wcs(record):
    if not isinstance(record, dict):
        raise ValueError("Expected a WCS JSON object")
    if _MARKER not in record:
        # Historical flat-header JSON, including ImageCollection records.
        return WCS(record, relax=True)
    if type(record[_MARKER]) is not int or record[_MARKER] != 1:
        raise ValueError("Unsupported KBMOD WCS serialization version")
    if record.get("fidelity") not in (_FIDELITY, "fits-header"):
        raise ValueError("Unknown KBMOD WCS fidelity")
    result = WCS(record["header"], relax=True)
    if record["fidelity"] == "fits-header":
        return result
    state = record["state"]
    core = result.wcs
    for key in _STRINGS:
        if not isinstance(state[key], list) or len(state[key]) != 2:
            raise ValueError(f"Invalid numeric WCS {key}")
        setattr(core, key, state[key])
    for key in ("radesys", "name", "alt"):
        setattr(core, key, state[key])
    for key in _SCALARS:
        setattr(core, key, float.fromhex(state[key]))
    for key in ("crpix", "crval", "cdelt"):
        setattr(core, key, _read_array(state[key], (2,)))
    if core.has_cd():
        del core.cd
    if core.has_pc():
        del core.pc
    if state["linear_type"] not in ("pc", "cd"):
        raise ValueError("Invalid numeric WCS linear representation")
    setattr(core, state["linear_type"], _read_array(state["linear"], (2, 2)))
    sip = state["sip"]
    result.sip = None
    if sip is not None:
        arrays = []
        for key in ("a", "b", "ap", "bp"):
            value = None if sip[key] is None else _read_array(sip[key])
            if value is not None and (value.ndim != 2 or value.shape[0] != value.shape[1]):
                raise ValueError("Invalid numeric WCS SIP shape")
            arrays.append(value)
        result.sip = Sip(*arrays, _read_array(sip["crpix"], (2,)))
    shape = state["pixel_shape"]
    if shape is not None and (
        not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n < 0 for n in shape)
    ):
        raise ValueError("Invalid numeric WCS pixel shape")
    result.pixel_shape = shape
    bounds = state["pixel_bounds"]
    result.pixel_bounds = None if bounds is None else _read_array(bounds, (2, 2)).tolist()
    core.set()
    if _unsupported(result):
        raise ValueError("Numeric WCS payload is outside the supported TAN/SIP model")
    return result
