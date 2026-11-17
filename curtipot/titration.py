"""Potentiometric titration simulation engine.

Reimplements the titration simulation algorithm from CurTiPot (Simulador.bas and Titulador.bas),
supporting volume-step and pH-step simulations, continuous sample and titrant dilution,
counterion tracking for exact ionic strength, and numerical derivative calculations.
"""

from __future__ import annotations

import math
from typing import Sequence, Any
import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks

from curtipot.models import (
    ChemicalSystem,
    TitrationPoint,
    TitrationConfig,
    TitrationResult,
)
from curtipot.equilibrium import (
    solve_equilibrium,
    _determine_initial_protons,
)


def _infer_counterions(systems: Sequence[ChemicalSystem]) -> list[tuple[float, int]]:
    """Infer common spectator counterions (e.g. Na+ for NaOH, K+ for KOH) for proper ionic strength.

    When strong bases or salts are added without explicitly modeling the inert counterion,
    this function adds the spectator ion (charge and concentration).
    """
    ions: list[tuple[float, int]] = []
    for sys in systems:
        if sys.concentration <= 0:
            continue
        name_lower = sys.name.lower()
        formula_lower = sys.formula.lower() if sys.formula else ""
        if "sódio" in name_lower or "sodium" in name_lower or "naoh" in formula_lower:
            ions.append((sys.concentration, 1))
        elif "potássio" in name_lower or "potassium" in name_lower or "koh" in formula_lower:
            ions.append((sys.concentration, 1))
        elif "lítio" in name_lower or "lithium" in name_lower or "lioh" in formula_lower:
            ions.append((sys.concentration, 1))
        elif "cálcio" in name_lower or "calcium" in name_lower or "ca(oh)2" in formula_lower:
            ions.append((sys.concentration, 2))
        elif "bário" in name_lower or "barium" in name_lower or "ba(oh)2" in formula_lower:
            ions.append((sys.concentration, 2))
    return ions


def _solve_titration_point(
    v: float,
    config: TitrationConfig,
    ch_sample_0: float,
    ch_titrant_0: float,
    sample_inert: list[tuple[float, int]],
    titrant_inert: list[tuple[float, int]],
) -> TitrationPoint:
    """Solve equilibrium at a single added titrant volume V (mL)."""
    v_total = config.sample_volume + config.water_volume + v
    d_sample = config.sample_volume / v_total
    d_titrant = v / v_total

    ch_total = ch_sample_0 * d_sample + ch_titrant_0 * d_titrant

    diluted_systems = [
        s.with_concentration(s.concentration * d_sample) for s in config.sample_systems
    ] + [
        s.with_concentration(s.concentration * d_titrant) for s in config.titrant_systems
    ]

    diluted_inert = [
        (c * d_sample, z) for c, z in sample_inert
    ] + [
        (c * d_titrant, z) for c, z in titrant_inert
    ]

    eq_res = solve_equilibrium(
        systems=diluted_systems,
        initial_protons=ch_total,
        inert_salts=diluted_inert,
        calc_activity=config.calc_ionic_strength,
        temp_c=config.temp_c,
    )

    return TitrationPoint(
        volume_added=float(v),
        ph=float(eq_res.ph),
        ionic_strength=float(eq_res.ionic_strength),
        buffer_capacity=float(eq_res.buffer_capacity),
        species_concentrations=eq_res.species_concentrations,
    )


def _compute_derivatives_and_equivalence(
    points: list[TitrationPoint],
) -> list[dict[str, Any]]:
    """Compute numerical 1st (dpH/dV) and 2nd (d2pH/dV2) derivatives and locate equivalence points."""
    if len(points) < 3:
        return []

    vols = np.array([p.volume_added for p in points], dtype=float)
    phs = np.array([p.ph for p in points], dtype=float)

    # Compute numerical derivatives
    dpH = np.gradient(phs, vols)
    d2pH = np.gradient(dpH, vols)

    for i, p in enumerate(points):
        p.dpH_dV = float(dpH[i])
        p.d2pH_dV2 = float(d2pH[i])

    abs_dpH = np.abs(dpH)
    max_abs = float(np.max(abs_dpH))
    prominence_threshold = max(0.2, 0.08 * max_abs)
    peaks, _ = find_peaks(abs_dpH, prominence=prominence_threshold)

    eq_points: list[dict[str, Any]] = []
    for idx in peaks:
        eq_points.append(
            {
                "index": int(idx),
                "volume": float(vols[idx]),
                "ph": float(phs[idx]),
                "dpH_dV": float(dpH[idx]),
                "d2pH_dV2": float(d2pH[idx]),
            }
        )

    # Fallback if find_peaks didn't identify a sharp interior peak
    if not eq_points and len(points) >= 3:
        interior_abs = abs_dpH[1:-1]
        if len(interior_abs) > 0:
            max_inner_idx = int(np.argmax(interior_abs)) + 1
            if abs_dpH[max_inner_idx] > 2.0 * float(np.median(abs_dpH)):
                eq_points.append(
                    {
                        "index": int(max_inner_idx),
                        "volume": float(vols[max_inner_idx]),
                        "ph": float(phs[max_inner_idx]),
                        "dpH_dV": float(dpH[max_inner_idx]),
                        "d2pH_dV2": float(d2pH[max_inner_idx]),
                    }
                )

    return eq_points


def simulate_volume_steps(config: TitrationConfig) -> TitrationResult:
    """Simulate a titration curve with regular volume increments (dV).

    Parameters
    ----------
    config : TitrationConfig
        Titration parameters including sample, titrant, volumes, and step size.

    Returns
    -------
    TitrationResult
        Complete titration simulation result with points and equivalence points.
    """
    ch_sample_0 = _determine_initial_protons(config.sample_systems, None)
    ch_titrant_0 = _determine_initial_protons(config.titrant_systems, None)

    sample_inert = (
        list(config.sample_inert_salts)
        if config.sample_inert_salts
        else _infer_counterions(config.sample_systems)
    )
    titrant_inert = (
        list(config.titrant_inert_salts)
        if config.titrant_inert_salts
        else _infer_counterions(config.titrant_systems)
    )

    num_steps = int(math.ceil(config.max_volume / config.step_size))
    vols = [i * config.step_size for i in range(num_steps + 1)]
    if vols[-1] < config.max_volume - 1e-9:
        vols.append(config.max_volume)
    elif vols[-1] > config.max_volume + 1e-9:
        vols[-1] = config.max_volume

    points: list[TitrationPoint] = []
    for v in vols:
        pt = _solve_titration_point(
            v=v,
            config=config,
            ch_sample_0=ch_sample_0,
            ch_titrant_0=ch_titrant_0,
            sample_inert=sample_inert,
            titrant_inert=titrant_inert,
        )
        points.append(pt)

    eq_points = _compute_derivatives_and_equivalence(points)
    return TitrationResult(config=config, points=points, equivalence_points=eq_points)


def simulate_ph_steps(config: TitrationConfig) -> TitrationResult:
    """Simulate a titration curve with regular pH increments (dpH).

    Finds the titrant volume corresponding to each target pH using root finding (Brent's method).

    Parameters
    ----------
    config : TitrationConfig
        Titration parameters. step_size represents the target delta pH.

    Returns
    -------
    TitrationResult
        Complete titration simulation result.
    """
    ch_sample_0 = _determine_initial_protons(config.sample_systems, None)
    ch_titrant_0 = _determine_initial_protons(config.titrant_systems, None)

    sample_inert = (
        list(config.sample_inert_salts)
        if config.sample_inert_salts
        else _infer_counterions(config.sample_systems)
    )
    titrant_inert = (
        list(config.titrant_inert_salts)
        if config.titrant_inert_salts
        else _infer_counterions(config.titrant_systems)
    )

    def ph_at_vol(v_val: float) -> float:
        pt = _solve_titration_point(
            v=v_val,
            config=config,
            ch_sample_0=ch_sample_0,
            ch_titrant_0=ch_titrant_0,
            sample_inert=sample_inert,
            titrant_inert=titrant_inert,
        )
        return pt.ph

    ph_start = ph_at_vol(0.0)
    ph_end = ph_at_vol(config.max_volume)

    delta = ph_end - ph_start
    if abs(delta) < 1e-4:
        # No pH change across titration; fallback to volume steps
        return simulate_volume_steps(config)

    step = config.step_size if delta > 0 else -config.step_size
    num_steps = max(2, int(math.ceil(abs(delta) / config.step_size)))

    target_phs = [ph_start + i * step for i in range(num_steps + 1)]
    if (delta > 0 and target_phs[-1] < ph_end - 1e-5) or (delta < 0 and target_phs[-1] > ph_end + 1e-5):
        target_phs.append(ph_end)
    elif (delta > 0 and target_phs[-1] > ph_end + 1e-5) or (delta < 0 and target_phs[-1] < ph_end - 1e-5):
        target_phs[-1] = ph_end

    vols: list[float] = [0.0]
    for target in target_phs[1:-1]:
        try:
            # Brent's method to find root V in [0, max_volume]
            v_sol = brentq(lambda x: ph_at_vol(x) - target, 0.0, config.max_volume, xtol=1e-4)
            vols.append(float(v_sol))
        except ValueError:
            continue
    vols.append(config.max_volume)

    # Sort and remove duplicates
    vols = sorted(list(set(vols)))

    points: list[TitrationPoint] = []
    for v in vols:
        pt = _solve_titration_point(
            v=v,
            config=config,
            ch_sample_0=ch_sample_0,
            ch_titrant_0=ch_titrant_0,
            sample_inert=sample_inert,
            titrant_inert=titrant_inert,
        )
        points.append(pt)

    eq_points = _compute_derivatives_and_equivalence(points)
    return TitrationResult(config=config, points=points, equivalence_points=eq_points)


def simulate_titration(config: TitrationConfig) -> TitrationResult:
    """Simulate a potentiometric titration based on the given configuration.

    Parameters
    ----------
    config : TitrationConfig
        Titration parameters. mode can be 'volume' or 'ph'.

    Returns
    -------
    TitrationResult
        Complete titration simulation result.
    """
    mode = config.mode.lower().strip()
    if mode in ("volume", "vol", "v", "dvol"):
        return simulate_volume_steps(config)
    elif mode in ("ph", "dp_h", "ph_steps", "dph"):
        return simulate_ph_steps(config)
    else:
        raise ValueError(
            f"Unsupported titration mode: '{config.mode}'. Must be 'volume' or 'ph'."
        )
