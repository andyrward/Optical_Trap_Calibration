"""DNA force-extension models and looped-construct distance helpers.

The models in this module are deliberately lightweight: they are appropriate for
rapidly fitting and comparing delta-extension curves, rather than as a full
biophysical simulation engine.  All lengths are interpreted in nanometers and
all forces in piconewtons unless otherwise noted.

The parallel target/loop solve is a first-pass approximation: it treats the
looped target as a single effective parallel block and uses a numerical force
partition solve to match the same extension on both arms.  This is useful for
fitting a delta-extension curve, but it is not a replacement for a full
statistical-mechanical model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq

_DEFAULT_DSDNA = {
    "rise_per_base_nm": 0.34,
    "persistence_length_nm": 50.0,
    "stretch_modulus_pn": 1200.0,
    "model": "wlc",
}

_DEFAULT_SSDNA = {
    "rise_per_base_nm": 0.63,
    "persistence_length_nm": 0.75,
    "stretch_modulus_pn": 800.0,
    "model": "fjc",
}

_DEFAULT_TEMPERATURE_K = 298.15
_DEFAULT_KBT_PN_NM = 4.114  # pN·nm at ~298 K


def _resolve_kbt(temperature_k: float | None = None, kbt_pn_nm: float | None = None) -> float:
    if kbt_pn_nm is not None:
        return float(kbt_pn_nm)
    if temperature_k is not None:
        return 1.380649e-23 * temperature_k / 1e-21  # J to pN·nm
    return _DEFAULT_KBT_PN_NM


def _coerce_contour_length(length_nm: float | int | None, rise_per_unit_nm: float | int = 1.0) -> float:
    if length_nm is None:
        raise ValueError("Contour length is required.")
    value = float(length_nm) * float(rise_per_unit_nm)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Contour length must be positive and finite.")
    return value


def _normalise_model_name(model: str) -> str:
    key = str(model).strip().lower().replace("-", "_")
    aliases = {
        "markosiggia": "wlc",
        "marko_siggia": "wlc",
        "wlc_ms": "wlc",
        "wlc_marko_siggia": "wlc",
        "ms_wlc": "wlc",
        "odijk": "extensible_wlc",
        "extensible": "extensible_wlc",
        "extensible_wlc_odijk": "extensible_wlc",
        "extensible_wlc_ms": "extensible_wlc",
        "extensible_wlc_marko_siggia": "extensible_wlc",
        "fjc": "fjc",
        "extensible_fjc": "extensible_fjc",
        "wormlikechain": "wlc",
    }
    return aliases.get(key, key)


def _validate_parameters(contour_length_nm: float, persistence_length_nm: float, stretch_modulus_pn: float) -> tuple[float, float, float]:
    if contour_length_nm <= 0.0:
        raise ValueError("Contour length must be > 0")
    if persistence_length_nm <= 0.0:
        raise ValueError("Persistence length must be > 0")
    if stretch_modulus_pn <= 0.0:
        raise ValueError("Stretch modulus must be > 0")
    return float(contour_length_nm), float(persistence_length_nm), float(stretch_modulus_pn)


def _force_from_position_wlc(
    extension_nm: float,
    contour_length_nm: float,
    persistence_length_nm: float,
    kbt_pn_nm: float,
    *,
    stretch_modulus_pn: float | None = None,
) -> float:
    """Return the WLC force for a given extension, in pN.

    Uses the standard Marko-Siggia form with the small extension approximation,
    and optionally adds a linear elastic correction via the stretch modulus.
    """
    x = float(extension_nm)
    L = float(contour_length_nm)
    lp = float(persistence_length_nm)
    if x <= 0.0:
        return 0.0
    if x >= L:
        return np.inf
    x_over_L = x / L
    wlc_force = (kbt_pn_nm / lp) * (x_over_L + 1.0 / (4.0 * (1.0 - x_over_L) ** 2) - 0.25)
    if stretch_modulus_pn is not None:
        elastic_term = (stretch_modulus_pn / L) * (x_over_L - 0.0)
        return wlc_force + elastic_term
    return wlc_force


def _wlc_extension_from_force(force_pn: np.ndarray, contour_length_nm: float, persistence_length_nm: float, kbt_pn_nm: float) -> np.ndarray:
    """Return x(F) for an inextensible WLC by numerically inverting the force law."""
    force_arr = np.asarray(np.atleast_1d(force_pn), dtype=float)
    result = np.empty_like(force_arr, dtype=float)
    for idx, force in enumerate(force_arr):
        if not np.isfinite(force) or force <= 0.0:
            result[idx] = 0.0
            continue
        if force == np.inf:
            result[idx] = contour_length_nm
            continue
        lower = 1e-12
        upper = contour_length_nm * (1.0 - 1e-9)

        def objective(extension: float) -> float:
            return _force_from_position_wlc(extension, contour_length_nm, persistence_length_nm, kbt_pn_nm) - force

        result[idx] = brentq(objective, lower, upper, maxiter=200)
    return result[0] if np.isscalar(force_pn) else result


def _extensible_wlc_extension_from_force(
    force_pn: np.ndarray,
    contour_length_nm: float,
    persistence_length_nm: float,
    kbt_pn_nm: float,
    stretch_modulus_pn: float,
) -> np.ndarray:
    """Return the extensible WLC extension using the additive stretch term.

    The Odijk high-force limit is x = L * (1 - 0.5 * sqrt(kBT / (F * lp)) + F/S),
    so the linear strain correction is added to the inextensible WLC extension
    rather than multiplying by a factor of (1 + F/S).
    """
    force_arr = np.asarray(np.atleast_1d(force_pn), dtype=float)
    L = float(contour_length_nm)
    lp = float(persistence_length_nm)
    S = float(stretch_modulus_pn)
    wlc_term = np.asarray(_wlc_extension_from_force(force_arr, L, lp, kbt_pn_nm), dtype=float)
    result = np.maximum(wlc_term + L * np.clip(force_arr / S, 0.0, None), 0.0)
    return result[0] if np.isscalar(force_pn) else result


def _langevin(u: np.ndarray | float) -> np.ndarray | float:
    """Return Langevin(u) = coth(u) - 1/u with a stable low-u expansion."""
    u_arr = np.asarray(u, dtype=float)
    result = np.empty_like(u_arr, dtype=float)
    small = np.abs(u_arr) < 1e-6
    result[small] = u_arr[small] / 3.0
    valid = ~small
    if np.any(valid):
        u_valid = np.asarray(u_arr[valid], dtype=float)
        e2 = np.exp(-2.0 * u_valid)
        coth = (1.0 + e2) / (1.0 - e2)
        result[valid] = coth - 1.0 / u_valid
    return result[()] if np.ndim(u_arr) == 0 else result


def _fjc_extension_from_force(force_pn: np.ndarray, contour_length_nm: float, persistence_length_nm: float, kbt_pn_nm: float) -> np.ndarray:
    """Return x(F) for a 3D freely-jointed chain.

    The standard FJC relation is x = L * (coth(u) - 1/u) with u = F * b / kBT,
    where the Kuhn length is b = 2 * lp for a persistence-length model.
    """
    force_arr = np.asarray(np.atleast_1d(force_pn), dtype=float)
    L = float(contour_length_nm)
    lp = float(persistence_length_nm)
    kuhn_length_nm = max(2.0 * lp, 1e-12)
    u = np.clip(force_arr * kuhn_length_nm / max(kbt_pn_nm, 1e-12), 0.0, None)
    result = L * _langevin(u)
    return result[0] if np.isscalar(force_pn) else result


def _extensible_fjc_extension_from_force(
    force_pn: np.ndarray,
    contour_length_nm: float,
    persistence_length_nm: float,
    kbt_pn_nm: float,
    stretch_modulus_pn: float,
) -> np.ndarray:
    """Return the extensible FJC from Smith, Cui & Bustamante (Science 271, 795, 1996).

    The multiplicative form is
        x = L_c * Langevin(F * b / kBT) * (1 + F / S),
    with b = 2 * lp for a persistence-length-based model.  This preserves the
    standard 3D FJC response while applying the finite stretch modulus as a
    multiplicative elastic correction.
    """
    force_arr = np.asarray(np.atleast_1d(force_pn), dtype=float)
    L = float(contour_length_nm)
    lp = float(persistence_length_nm)
    kuhn_length_nm = max(2.0 * lp, 1e-12)
    u = np.clip(force_arr * kuhn_length_nm / max(kbt_pn_nm, 1e-12), 0.0, None)
    result = L * np.asarray(_langevin(u), dtype=float) * (1.0 + np.clip(force_arr / max(stretch_modulus_pn, 1e-12), 0.0, None))
    return result[0] if np.isscalar(force_pn) else result


def _extension_from_force_core(
    force_pn: float | np.ndarray,
    contour_length_nm: float,
    persistence_length_nm: float,
    stretch_modulus_pn: float,
    *,
    model: str,
    kbt_pn_nm: float | None = None,
    temperature_k: float | None = None,
) -> float | np.ndarray:
    model_name = _normalise_model_name(model)
    kbt = _resolve_kbt(temperature_k, kbt_pn_nm)
    contour_length_nm, persistence_length_nm, stretch_modulus_pn = _validate_parameters(
        contour_length_nm,
        persistence_length_nm,
        stretch_modulus_pn,
    )
    force_arr = np.asarray(force_pn, dtype=float)
    if model_name == "wlc":
        return _wlc_extension_from_force(force_arr, contour_length_nm, persistence_length_nm, kbt)
    if model_name == "extensible_wlc":
        return _extensible_wlc_extension_from_force(
            force_arr,
            contour_length_nm,
            persistence_length_nm,
            kbt,
            stretch_modulus_pn,
        )
    if model_name == "fjc":
        return _fjc_extension_from_force(force_arr, contour_length_nm, persistence_length_nm, kbt)
    if model_name == "extensible_fjc":
        return _extensible_fjc_extension_from_force(
            force_arr,
            contour_length_nm,
            persistence_length_nm,
            kbt,
            stretch_modulus_pn,
        )
    raise ValueError(f"Unsupported model: {model!r}")


def _force_from_extension_core(
    extension_nm: float | np.ndarray,
    contour_length_nm: float,
    persistence_length_nm: float,
    stretch_modulus_pn: float,
    *,
    model: str,
    kbt_pn_nm: float | None = None,
    temperature_k: float | None = None,
) -> float | np.ndarray:
    model_name = _normalise_model_name(model)
    kbt = _resolve_kbt(temperature_k, kbt_pn_nm)
    contour_length_nm, persistence_length_nm, stretch_modulus_pn = _validate_parameters(
        contour_length_nm,
        persistence_length_nm,
        stretch_modulus_pn,
    )
    extension_arr = np.asarray(extension_nm, dtype=float)
    result = np.empty_like(extension_arr, dtype=float)
    for idx, extension in enumerate(extension_arr):
        if not np.isfinite(extension) or extension <= 0.0:
            result[idx] = 0.0
            continue
        if extension >= contour_length_nm and model_name not in {"extensible_wlc", "extensible_fjc"}:
            result[idx] = np.inf
            continue

        def objective(force: float) -> float:
            return _extension_from_force_core(force, contour_length_nm, persistence_length_nm, stretch_modulus_pn, model=model_name, kbt_pn_nm=kbt) - extension

        lower = 1e-12
        upper = max(1.0, 10.0 * stretch_modulus_pn)
        if model_name in {"wlc", "extensible_wlc"}:
            upper = max(upper, 1e4)
        found_root = False
        for _ in range(50):
            try:
                root = brentq(objective, lower, upper, maxiter=200)
                result[idx] = root
                found_root = True
                break
            except ValueError:
                upper *= 10.0
                if upper > 1e12:
                    result[idx] = np.nan
                    break
        if not found_root and not np.isfinite(result[idx]):
            result[idx] = np.nan
    return result[0] if np.asarray(extension_nm).ndim == 0 else result


def single_polymer_extension(
    force_pn: float | np.ndarray,
    contour_length_nm: float | int | None,
    persistence_length_nm: float | int | None,
    stretch_modulus_pn: float | int | None,
    *,
    model: str = "wlc",
    temperature_k: float | None = None,
    kbt_pn_nm: float | None = None,
    rise_per_unit_nm: float | int = 1.0,
    dna_type: str | None = None,
) -> float | np.ndarray:
    """Return extension as a function of force for a single dsDNA or ssDNA segment.

    Parameters
    ----------
    force_pn:
        Force in pN.  Scalar or NumPy array supported.
    contour_length_nm:
        Physical contour length in nm, or the total number of bases/nt when
        *rise_per_unit_nm* is supplied.
    persistence_length_nm:
        Persistence length in nm.
    stretch_modulus_pn:
        Stretch modulus in pN.
    model:
        One of ``"wlc"``, ``"extensible_wlc"``, ``"fjc"``,
        ``"extensible_fjc"``.
    temperature_k:
        Temperature in Kelvin.  If not given, a default of 298.15 K is used.
    kbt_pn_nm:
        Override the thermal energy in pN·nm.
    rise_per_unit_nm:
        Extension per base / nt.  Defaults to 1.0, meaning the contour-length
        input is already in nm.
    dna_type:
        Optional string of ``"dsdna"`` or ``"ssdna"``; if supplied, it will
        provide sensible defaults when parameters are left as ``None``.

    Returns
    -------
    float or np.ndarray
        Extension in nm for each input force.
    """
    contour_length_nm = _coerce_contour_length(contour_length_nm, rise_per_unit_nm)
    if dna_type is not None:
        dna_type = str(dna_type).lower()
        if dna_type == "dsdna":
            default = _DEFAULT_DSDNA.copy()
            if persistence_length_nm is None:
                persistence_length_nm = float(default["persistence_length_nm"])
            if stretch_modulus_pn is None:
                stretch_modulus_pn = float(default["stretch_modulus_pn"])
        elif dna_type == "ssdna":
            default = _DEFAULT_SSDNA.copy()
            if persistence_length_nm is None:
                persistence_length_nm = float(default["persistence_length_nm"])
            if stretch_modulus_pn is None:
                stretch_modulus_pn = float(default["stretch_modulus_pn"])
    if persistence_length_nm is None:
        raise ValueError("persistence_length_nm is required unless supplied by dna_type.")
    if stretch_modulus_pn is None:
        raise ValueError("stretch_modulus_pn is required unless supplied by dna_type.")
    return _extension_from_force_core(
        force_pn,
        contour_length_nm,
        float(persistence_length_nm),
        float(stretch_modulus_pn),
        model=model,
        kbt_pn_nm=kbt_pn_nm,
        temperature_k=temperature_k,
    )


def single_polymer_force(
    extension_nm: float | np.ndarray,
    contour_length_nm: float | int | None,
    persistence_length_nm: float | int | None,
    stretch_modulus_pn: float | int | None,
    *,
    model: str = "wlc",
    temperature_k: float | None = None,
    kbt_pn_nm: float | None = None,
    rise_per_unit_nm: float | int = 1.0,
    dna_type: str | None = None,
) -> float | np.ndarray:
    """Return force as a function of extension for a single DNA segment."""
    contour_length_nm = _coerce_contour_length(contour_length_nm, rise_per_unit_nm)
    if dna_type is not None:
        dna_type = str(dna_type).lower()
        if dna_type == "dsdna":
            default = _DEFAULT_DSDNA.copy()
            if persistence_length_nm is None:
                persistence_length_nm = float(default["persistence_length_nm"])
            if stretch_modulus_pn is None:
                stretch_modulus_pn = float(default["stretch_modulus_pn"])
        elif dna_type == "ssdna":
            default = _DEFAULT_SSDNA.copy()
            if persistence_length_nm is None:
                persistence_length_nm = float(default["persistence_length_nm"])
            if stretch_modulus_pn is None:
                stretch_modulus_pn = float(default["stretch_modulus_pn"])
    if persistence_length_nm is None:
        raise ValueError("persistence_length_nm is required unless supplied by dna_type.")
    if stretch_modulus_pn is None:
        raise ValueError("stretch_modulus_pn is required unless supplied by dna_type.")
    result = _force_from_extension_core(
        extension_nm,
        contour_length_nm,
        float(persistence_length_nm),
        float(stretch_modulus_pn),
        model=model,
        kbt_pn_nm=kbt_pn_nm,
        temperature_k=temperature_k,
    )
    return result




def full_length_curve(
    force_pn: float | np.ndarray,
    ds_contour_length_nm: float | int,
    loop_contour_length_nm: float | int,
    ds_params: dict[str, Any],
    loop_params: dict[str, Any],
    *,
    offset_nm: float = 0.0,
) -> float | np.ndarray:
    """Extension of the full-length construct with no loop closure.

    This is the dsDNA region in series with the loop region; the two segments see
    the same force and their extensions add. If the dsDNA and loop region have
    the same mechanical parameters, the construct collapses to a single polymer
    of summed contour length exactly as expected in the dsDNA-only limit.
    """
    ds_length = _coerce_contour_length(ds_contour_length_nm, ds_params.get("rise_per_unit_nm", 1.0))
    loop_length = _coerce_contour_length(loop_contour_length_nm, loop_params.get("rise_per_unit_nm", 1.0))
    ds_model = ds_params.get("model", "wlc")
    loop_model = loop_params.get("model", "wlc")
    same_material = (
        _normalise_model_name(ds_model) == _normalise_model_name(loop_model)
        and ds_params.get("persistence_length_nm", 50.0) == loop_params.get("persistence_length_nm", 50.0)
        and ds_params.get("stretch_modulus_pn", 1200.0) == loop_params.get("stretch_modulus_pn", 1200.0)
        and ds_params.get("rise_per_unit_nm", 1.0) == loop_params.get("rise_per_unit_nm", 1.0)
        and ds_params.get("temperature_k", _DEFAULT_TEMPERATURE_K) == loop_params.get("temperature_k", _DEFAULT_TEMPERATURE_K)
        and ds_params.get("kbt_pn_nm", _DEFAULT_KBT_PN_NM) == loop_params.get("kbt_pn_nm", _DEFAULT_KBT_PN_NM)
    )
    if same_material:
        total_length = ds_length + loop_length
        return np.asarray(
            single_polymer_extension(
                force_pn,
                total_length,
                ds_params.get("persistence_length_nm", 50.0),
                ds_params.get("stretch_modulus_pn", 1200.0),
                model=ds_model,
                temperature_k=ds_params.get("temperature_k"),
                kbt_pn_nm=ds_params.get("kbt_pn_nm"),
            ),
            dtype=float,
        ) + float(offset_nm)

    ds_extension = single_polymer_extension(
        force_pn,
        ds_length,
        ds_params.get("persistence_length_nm", 50.0),
        ds_params.get("stretch_modulus_pn", 1200.0),
        model=ds_model,
        temperature_k=ds_params.get("temperature_k"),
        kbt_pn_nm=ds_params.get("kbt_pn_nm"),
    )
    loop_extension = single_polymer_extension(
        force_pn,
        loop_length,
        loop_params.get("persistence_length_nm", 50.0),
        loop_params.get("stretch_modulus_pn", 1200.0),
        model=loop_model,
        temperature_k=loop_params.get("temperature_k"),
        kbt_pn_nm=loop_params.get("kbt_pn_nm"),
    )
    return np.asarray(ds_extension, dtype=float) + np.asarray(loop_extension, dtype=float) + float(offset_nm)


def zero_target_curve(
    force_pn: float | np.ndarray,
    ds_contour_length_nm: float | int,
    ds_params: dict[str, Any],
    *,
    offset_nm: float = 0.0,
) -> float | np.ndarray:
    """Calibration reference: extension of the dsDNA-only construct."""
    ds_length = _coerce_contour_length(ds_contour_length_nm, ds_params.get("rise_per_unit_nm", 1.0))
    ds_model = ds_params.get("model", "wlc")
    extension = single_polymer_extension(
        force_pn,
        ds_length,
        ds_params.get("persistence_length_nm", 50.0),
        ds_params.get("stretch_modulus_pn", 1200.0),
        model=ds_model,
        temperature_k=ds_params.get("temperature_k"),
        kbt_pn_nm=ds_params.get("kbt_pn_nm"),
    )
    return np.asarray(extension, dtype=float) + float(offset_nm)


def _solve_parallel_force_split(
    total_force_pn: float,
    target_params: dict[str, Any],
    loop_params: dict[str, Any],
    *,
    target_contour_length_nm: float,
    loop_contour_length_nm: float,
) -> tuple[float, float, float]:
    target_length = _coerce_contour_length(target_contour_length_nm, target_params.get("rise_per_unit_nm", 1.0))
    loop_length = _coerce_contour_length(loop_contour_length_nm, loop_params.get("rise_per_unit_nm", 1.0))

    def objective(force_target: float) -> float:
        force_loop = max(total_force_pn - force_target, 0.0)
        if force_target <= 0.0:
            return np.inf
        if force_loop <= 0.0:
            return np.inf
        xt = np.asarray(
            single_polymer_extension(
                force_target,
                target_length,
                target_params.get("persistence_length_nm", 50.0),
                target_params.get("stretch_modulus_pn", 1200.0),
                model=target_params.get("model", "wlc"),
                temperature_k=target_params.get("temperature_k"),
                kbt_pn_nm=target_params.get("kbt_pn_nm"),
            ),
            dtype=float,
        ).reshape(-1)[0]
        xl = np.asarray(
            single_polymer_extension(
                force_loop,
                loop_length,
                loop_params.get("persistence_length_nm", 50.0),
                loop_params.get("stretch_modulus_pn", 1200.0),
                model=loop_params.get("model", "wlc"),
                temperature_k=loop_params.get("temperature_k"),
                kbt_pn_nm=loop_params.get("kbt_pn_nm"),
            ),
            dtype=float,
        ).reshape(-1)[0]
        return float(xt) - float(xl)

    left = 1e-12
    right = max(total_force_pn - 1e-12, 1e-12)
    # Search for the sign change, which must exist for a monotone extension.
    span_left = left
    span_right = right
    found = False
    for _ in range(60):
        try:
            root = brentq(objective, span_left, span_right, maxiter=200)
            found = True
            break
        except ValueError:
            span_left *= 10.0
            span_right = max(total_force_pn - 1e-12, span_left)
            if span_left > max(total_force_pn, 1e9):
                break
    if not found:
        # Fall back to the equal-force limit: a physically sensible zero-force
        # or half-force split is still useful for fitting workflows.
        force_target = total_force_pn / 2.0
        force_loop = total_force_pn - force_target
    else:
        force_target = float(root)
        force_loop = float(total_force_pn - root)

    xt = np.asarray(
        single_polymer_extension(
            force_target,
            target_length,
            target_params.get("persistence_length_nm", 50.0),
            target_params.get("stretch_modulus_pn", 1200.0),
            model=target_params.get("model", "wlc"),
            temperature_k=target_params.get("temperature_k"),
            kbt_pn_nm=target_params.get("kbt_pn_nm"),
        ),
        dtype=float,
    ).reshape(-1)[0]
    xl = np.asarray(
        single_polymer_extension(
            force_loop,
            loop_length,
            loop_params.get("persistence_length_nm", 50.0),
            loop_params.get("stretch_modulus_pn", 1200.0),
            model=loop_params.get("model", "wlc"),
            temperature_k=loop_params.get("temperature_k"),
            kbt_pn_nm=loop_params.get("kbt_pn_nm"),
        ),
        dtype=float,
    ).reshape(-1)[0]
    return float(force_target), float(force_loop), float(xt)


def target_curve(
    force_pn: float | np.ndarray,
    ds_contour_length_nm: float | int,
    target_contour_length_nm: float | int,
    loop_contour_length_nm: float | int,
    ds_params: dict[str, Any],
    target_params: dict[str, Any],
    loop_params: dict[str, Any],
    *,
    offset_nm: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Construct the target-state curve with a dsDNA series arm and parallel target/loop block.

    The output includes the force split between the target and loop arms as a
    diagnostic, together with the total extension of the resulting construct.
    """
    force_arr = np.asarray(force_pn, dtype=float)
    ds_series = zero_target_curve(force_arr, ds_contour_length_nm, ds_params)
    target_extension = np.empty_like(force_arr, dtype=float)
    target_force = np.empty_like(force_arr, dtype=float)
    loop_force = np.empty_like(force_arr, dtype=float)
    parallel_extension = np.empty_like(force_arr, dtype=float)
    for idx, total_force in enumerate(force_arr):
        if not np.isfinite(total_force) or total_force <= 0.0:
            target_force[idx] = 0.0
            loop_force[idx] = 0.0
            parallel_extension[idx] = 0.0
            target_extension[idx] = ds_series[idx] + 0.0 + offset_nm
            continue
        ft, fl, x_parallel = _solve_parallel_force_split(
            float(total_force),
            target_params,
            loop_params,
            target_contour_length_nm=target_contour_length_nm,
            loop_contour_length_nm=loop_contour_length_nm,
        )
        target_force[idx] = ft
        loop_force[idx] = fl
        parallel_extension[idx] = x_parallel
        target_extension[idx] = float(ds_series[idx]) + x_parallel + float(offset_nm)

    return {
        "force_pn": force_arr,
        "force_target_pn": target_force,
        "force_loop_pn": loop_force,
        "parallel_extension_nm": parallel_extension,
        "extension_nm": target_extension,
        "series_extension_nm": ds_series,
    }


def delta_extension_zero_target(
    force_pn: float | np.ndarray,
    ds_contour_length_nm: float | int,
    loop_contour_length_nm: float | int,
    ds_params: dict[str, Any],
    loop_params: dict[str, Any],
    *,
    offset_nm: float = 0.0,
) -> float | np.ndarray:
    """Return Δ_0T(F) = x_full(F) - x_0T(F)."""
    return full_length_curve(force_pn, ds_contour_length_nm, loop_contour_length_nm, ds_params, loop_params, offset_nm=offset_nm) - zero_target_curve(force_pn, ds_contour_length_nm, ds_params, offset_nm=offset_nm)


def delta_extension_target(
    force_pn: float | np.ndarray,
    ds_contour_length_nm: float | int,
    target_contour_length_nm: float | int,
    loop_contour_length_nm: float | int,
    ds_params: dict[str, Any],
    target_params: dict[str, Any],
    loop_params: dict[str, Any],
    *,
    offset_nm: float = 0.0,
) -> float | np.ndarray:
    """Return Δ_target(F) = x_full(F) - x_target(F)."""
    target = target_curve(
        force_pn,
        ds_contour_length_nm,
        target_contour_length_nm,
        loop_contour_length_nm,
        ds_params,
        target_params,
        loop_params,
        offset_nm=offset_nm,
    )
    full = full_length_curve(force_pn, ds_contour_length_nm, loop_contour_length_nm, ds_params, loop_params, offset_nm=offset_nm)
    return np.asarray(full, dtype=float) - np.asarray(target["extension_nm"], dtype=float)


def distance_estimate(
    force_pn: float | np.ndarray,
    ds_contour_length_nm: float | int,
    target_contour_length_nm: float | int,
    loop_contour_length_nm: float | int,
    ds_params: dict[str, Any],
    target_params: dict[str, Any],
    loop_params: dict[str, Any],
    *,
    offset_nm: float = 0.0,
) -> float | np.ndarray:
    """Return the target/loop distance estimate: Δ_0T(F) - Δ_target(F)."""
    delta_0T = delta_extension_zero_target(
        force_pn,
        ds_contour_length_nm,
        loop_contour_length_nm,
        ds_params,
        loop_params,
        offset_nm=offset_nm,
    )
    delta_target = delta_extension_target(
        force_pn,
        ds_contour_length_nm,
        target_contour_length_nm,
        loop_contour_length_nm,
        ds_params,
        target_params,
        loop_params,
        offset_nm=offset_nm,
    )
    return np.asarray(delta_0T, dtype=float) - np.asarray(delta_target, dtype=float)


def compare_delta_extension(
    measured_delta_nm: float | np.ndarray,
    force_pn: float | np.ndarray,
    model_delta_nm: float | np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compare a measured delta-extension curve against a model curve."""
    measured = np.asarray(measured_delta_nm, dtype=float)
    model = np.asarray(model_delta_nm, dtype=float)
    residual = measured - model
    return {
        "force_pn": np.asarray(force_pn, dtype=float),
        "measured_delta_nm": measured,
        "model_delta_nm": model,
        "residual_nm": residual,
        "rmse_nm": float(np.sqrt(np.mean(residual ** 2))),
        "mad_nm": float(np.median(np.abs(residual))),
    }


polymer_extension = single_polymer_extension
polymer_force = single_polymer_force
x_of_f = single_polymer_extension
f_of_x = single_polymer_force
force_extension_curve = single_polymer_extension
force_from_extension = single_polymer_force
looped_construct_full = full_length_curve
looped_construct_zero_target = zero_target_curve
parallel_target_curve = target_curve

__all__ = [
    "single_polymer_extension",
    "single_polymer_force",
    "polymer_extension",
    "polymer_force",
    "x_of_f",
    "f_of_x",
    "force_extension_curve",
    "force_from_extension",
    "full_length_curve",
    "zero_target_curve",
    "target_curve",
    "looped_construct_full",
    "looped_construct_zero_target",
    "parallel_target_curve",
    "delta_extension_zero_target",
    "delta_extension_target",
    "distance_estimate",
    "compare_delta_extension",
    "_DEFAULT_DSDNA",
    "_DEFAULT_SSDNA",
]
