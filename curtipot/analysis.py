"""Analysis of experimental titration curves.

Reimplements:
1. Análise I (Análise_I_Interpola.bas):
   - Natural cubic spline interpolation of experimental (V, pH) data.
   - Continuous 1st (dpH/dV) and 2nd (d2pH/dV2) derivatives.
   - High-precision equivalence point detection via zero-crossing refinement of d2pH/dV2.
2. Análise II (Análise_II_pH_ajustado.bas):
   - Non-linear least-squares curve fitting (RNLMP) using `scipy.optimize.least_squares`.
   - Simultaneous determination of unknown analyte concentrations and dissociation constants (pKas).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.signal import find_peaks

from curtipot.models import (
    ChemicalSystem,
    TitrationConfig,
    TitrationResult,
)
from curtipot.equilibrium import _determine_initial_protons
from curtipot.titration import _infer_counterions, _solve_titration_point


@dataclass
class EquivalencePoint:
    """Detected equivalence point with volume, pH, and derivatives.

    Attributes
    ----------
    volume : float
        Titrant volume at the equivalence point in mL.
    ph : float
        pH at the equivalence point.
    dpH_dV : float
        First derivative (slope) dpH/dV at the equivalence point.
    d2pH_dV2 : float
        Second derivative d2pH/dV2 at the equivalence point (approximately 0.0).
    """

    volume: float
    ph: float
    dpH_dV: float
    d2pH_dV2: float

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like attribute access."""
        return getattr(self, key)

    def to_dict(self) -> dict[str, float]:
        """Convert equivalence point to a dictionary."""
        return {
            "volume": self.volume,
            "ph": self.ph,
            "dpH_dV": self.dpH_dV,
            "d2pH_dV2": self.d2pH_dV2,
        }


@dataclass
class SplineAnalysisResult:
    """Result of cubic spline interpolation and derivative analysis.

    Attributes
    ----------
    interpolated_volumes : np.ndarray
        Evenly spaced volume grid in mL.
    interpolated_phs : np.ndarray
        Interpolated pH values on the grid.
    dpH_dV : np.ndarray
        First derivative values dpH/dV on the grid.
    d2pH_dV2 : np.ndarray
        Second derivative values d2pH/dV2 on the grid.
    equivalence_points : list[EquivalencePoint]
        List of identified equivalence points.
    spline : CubicSpline
        Underlying SciPy CubicSpline object.
    rmse : float
        Root-mean-square error of spline fit against experimental points.
    r_squared : float
        Coefficient of determination (R^2) of spline fit against experimental points.
    """

    interpolated_volumes: np.ndarray
    interpolated_phs: np.ndarray
    dpH_dV: np.ndarray
    d2pH_dV2: np.ndarray
    equivalence_points: list[EquivalencePoint]
    spline: CubicSpline
    rmse: float
    r_squared: float

    @property
    def volumes(self) -> np.ndarray:
        """Alias for interpolated_volumes."""
        return self.interpolated_volumes

    @property
    def phs(self) -> np.ndarray:
        """Alias for interpolated_phs."""
        return self.interpolated_phs


@dataclass
class FitResult:
    """Result of non-linear least-squares titration curve fitting.

    Attributes
    ----------
    fitted_parameters : dict[str, float]
        Estimated parameter values (e.g. {'concentration': 0.0851, 'pKa_1': 4.76}).
    residuals : np.ndarray
        Difference (fitted_pH - measured_pH) at each experimental volume.
    rmse : float
        Root-mean-square error of the fit.
    r_squared : float
        Coefficient of determination R^2.
    fitted_ph : np.ndarray
        Calculated pH values at the experimental volumes using fitted parameters.
    success : bool
        Whether the optimization algorithm converged successfully.
    message : str
        Termination message from optimizer.
    """

    fitted_parameters: dict[str, float]
    residuals: np.ndarray
    rmse: float
    r_squared: float
    fitted_ph: np.ndarray
    success: bool = True
    message: str = ""


def analyze_spline(
    volumes: np.ndarray | Sequence[float] | TitrationResult,
    ph_values: np.ndarray | Sequence[float] | None = None,
    num_points: int = 500,
) -> SplineAnalysisResult:
    """Fit a natural cubic spline to titration data, evaluate derivatives, and locate equivalence points.

    Reimplements Análise I from CurTiPot (Análise_I_Interpola.bas) using SciPy's
    CubicSpline with natural boundary conditions and analytical derivatives.

    Parameters
    ----------
    volumes : np.ndarray, Sequence[float], or TitrationResult
        Experimental titrant volumes in mL, or a TitrationResult object.
    ph_values : np.ndarray or Sequence[float], optional
        Measured pH values corresponding to volumes. Required if volumes is not TitrationResult.
    num_points : int
        Number of interpolated points to generate along the volume range (default: 500).

    Returns
    -------
    SplineAnalysisResult
        Analysis result containing continuous grid, derivatives, and equivalence points.

    Raises
    ------
    ValueError
        If fewer than 4 points are supplied or volume and pH arrays have mismatched lengths.
    """
    if isinstance(volumes, TitrationResult):
        v_data = volumes.volumes
        ph_data = volumes.phs
    else:
        if ph_values is None:
            raise ValueError("ph_values must be provided when volumes is an array or list")
        v_data = np.asarray(volumes, dtype=float)
        ph_data = np.asarray(ph_values, dtype=float)

    if len(v_data) != len(ph_data):
        raise ValueError(
            f"Mismatched array lengths: {len(v_data)} volumes vs {len(ph_data)} pH values"
        )
    if len(v_data) < 4:
        raise ValueError(
            f"At least 4 points are required for cubic spline analysis, got {len(v_data)}"
        )

    # Sort data by volume and average any duplicate volume entries
    sort_order = np.argsort(v_data)
    v_sorted = v_data[sort_order]
    ph_sorted = ph_data[sort_order]

    unique_vols, inv_indices = np.unique(v_sorted, return_inverse=True)
    if len(unique_vols) < len(v_sorted):
        # Average duplicates
        unique_phs = np.zeros_like(unique_vols)
        counts = np.bincount(inv_indices)
        for i, ph_val in enumerate(ph_sorted):
            unique_phs[inv_indices[i]] += ph_val
        unique_phs /= counts
        v_sorted = unique_vols
        ph_sorted = unique_phs

    if len(v_sorted) < 4:
        raise ValueError("At least 4 unique volume points are required for spline analysis")

    # Fit natural cubic spline (bc_type='natural' enforces zero 2nd derivative at endpoints)
    cs = CubicSpline(v_sorted, ph_sorted, bc_type="natural")
    d1_cs = cs.derivative(nu=1)
    d2_cs = cs.derivative(nu=2)

    # Evaluate on fine grid
    v_min, v_max = float(v_sorted[0]), float(v_sorted[-1])
    grid_volumes = np.linspace(v_min, v_max, max(num_points, 100))
    grid_phs = cs(grid_volumes)
    grid_d1 = d1_cs(grid_volumes)
    grid_d2 = d2_cs(grid_volumes)

    # Locate candidate equivalence points via peaks in |dpH/dV|
    abs_d1 = np.abs(grid_d1)
    max_d1 = float(np.max(abs_d1))
    median_d1 = float(np.median(abs_d1))

    prominence = max(0.08 * max_d1, 2.0 * median_d1, 0.2)
    min_distance = max(3, len(grid_volumes) // 50)
    peaks, _ = find_peaks(abs_d1, prominence=prominence, distance=min_distance)

    # Refine equivalence points around peaks of |dpH/dV|
    equivalence_points: list[EquivalencePoint] = []

    for p_idx in peaks:
        v_center = float(grid_volumes[p_idx])
        v_min = float(grid_volumes[max(0, p_idx - 3)])
        v_max = float(grid_volumes[min(len(grid_volumes) - 1, p_idx + 3)])

        # Search immediately adjacent intervals around peak for zero-crossing of d2pH/dV2
        start_i = max(0, p_idx - 4)
        end_i = min(len(grid_volumes) - 1, p_idx + 4)
        candidates: list[tuple[float, float]] = []

        for i in range(start_i, end_i):
            vi = float(grid_volumes[i])
            vnext = float(grid_volumes[i + 1])
            d2_i = float(d2_cs(vi))
            d2_next = float(d2_cs(vnext))
            if d2_i * d2_next <= 0:
                try:
                    r = float(brentq(d2_cs, vi, vnext))
                    candidates.append((r, abs(float(d1_cs(r)))))
                except Exception:
                    pass

        if candidates:
            # Select the zero-crossing with the highest first derivative
            candidates.sort(key=lambda item: item[1], reverse=True)
            v_eq = candidates[0][0]
        else:
            # Fallback: bounded maximization of |dpH/dV|
            try:
                opt = minimize_scalar(
                    lambda v: -abs(float(d1_cs(v))),
                    bounds=(v_min, v_max),
                    method="bounded",
                )
                v_eq = float(opt.x)
            except Exception:
                v_eq = v_center

        ph_eq = float(cs(v_eq))
        d1_eq = float(d1_cs(v_eq))
        d2_eq = float(d2_cs(v_eq))

        # Avoid duplicate equivalence points within 0.15 mL
        is_duplicate = any(abs(v_eq - ep.volume) < 0.15 for ep in equivalence_points)
        if not is_duplicate:
            equivalence_points.append(
                EquivalencePoint(
                    volume=v_eq,
                    ph=ph_eq,
                    dpH_dV=d1_eq,
                    d2pH_dV2=d2_eq,
                )
            )

    # Sort equivalence points by volume
    equivalence_points.sort(key=lambda ep: ep.volume)

    # Calculate statistics against experimental data
    ph_pred = cs(v_sorted)
    residuals = ph_sorted - ph_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((ph_sorted - np.mean(ph_sorted)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    return SplineAnalysisResult(
        interpolated_volumes=grid_volumes,
        interpolated_phs=grid_phs,
        dpH_dV=grid_d1,
        d2pH_dV2=grid_d2,
        equivalence_points=equivalence_points,
        spline=cs,
        rmse=rmse,
        r_squared=r_squared,
    )


def analyze_spline_derivatives(
    volumes: np.ndarray | Sequence[float] | TitrationResult,
    ph_values: np.ndarray | Sequence[float] | None = None,
    num_points: int = 500,
) -> SplineAnalysisResult:
    """Alias for `analyze_spline` for compatibility with plan specification."""
    return analyze_spline(volumes=volumes, ph_values=ph_values, num_points=num_points)


def fit_titration_curve(
    volumes: np.ndarray | Sequence[float],
    ph_measured: np.ndarray | Sequence[float],
    config: TitrationConfig,
    fit_concentration: bool = True,
    fit_pka_indices: list[int] | None = None,
    max_nfev: int = 200,
) -> FitResult:
    """Non-linear least squares regression of titration parameters (CurTiPot Análise II).

    Minimizes the sum of squared pH residuals sum((pH_calc(V_i) - pH_exp,i)^2) by adjusting
    analyte analytical concentration and/or stepwise acid dissociation constants (pKas).

    Parameters
    ----------
    volumes : np.ndarray or Sequence[float]
        Experimental added titrant volumes in mL.
    ph_measured : np.ndarray or Sequence[float]
        Experimentally measured pH values.
    config : TitrationConfig
        Titration system configuration. `config.sample_systems[0]` is treated as the analyte to fit.
    fit_concentration : bool
        Whether to optimize the analyte analytical concentration (default: True).
    fit_pka_indices : list[int], optional
        List of 0-based indices of `sample_systems[0].pKas` to optimize (e.g. [0] or [0, 1]).
    max_nfev : int
        Maximum number of function evaluations (default: 200).

    Returns
    -------
    FitResult
        Result containing fitted parameters, residuals, RMSE, R^2, and calculated pH curve.
    """
    v_exp = np.asarray(volumes, dtype=float)
    ph_exp = np.asarray(ph_measured, dtype=float)

    if len(v_exp) != len(ph_exp):
        raise ValueError(
            f"Mismatched array lengths: {len(v_exp)} volumes vs {len(ph_exp)} pH values"
        )
    if len(v_exp) < 3:
        raise ValueError(f"At least 3 data points required for curve fitting, got {len(v_exp)}")

    base_sample = config.sample_systems[0]
    pka_indices = list(fit_pka_indices) if fit_pka_indices is not None else []

    # Prepare parameter vector and bounds
    x0: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    param_names: list[str] = []

    if fit_concentration:
        c_init = max(1e-5, float(base_sample.concentration))
        x0.append(c_init)
        lb.append(1e-7)
        ub.append(10.0)
        param_names.append("concentration")

    for idx in pka_indices:
        if idx < 0 or idx >= len(base_sample.pKas):
            raise IndexError(
                f"pKa index {idx} out of range for system with {len(base_sample.pKas)} pKas"
            )
        pka_val = float(base_sample.pKas[idx])
        x0.append(pka_val)
        lb.append(-2.0)
        ub.append(16.0)
        param_names.append(f"pKa_{idx + 1}")

    if not x0:
        raise ValueError("No parameters selected for fitting. Enable fit_concentration or fit_pka_indices.")

    # Pre-calculate titrant properties since titrant does not change
    ch_titrant_0 = _determine_initial_protons(config.titrant_systems, None)
    titrant_inert = (
        list(config.titrant_inert_salts)
        if config.titrant_inert_salts
        else _infer_counterions(config.titrant_systems)
    )

    def _build_candidate_sample(params: np.ndarray) -> ChemicalSystem:
        curr_c = float(base_sample.concentration)
        curr_pkas = list(base_sample.pKas)
        param_idx = 0

        if fit_concentration:
            curr_c = float(params[param_idx])
            param_idx += 1

        for idx in pka_indices:
            curr_pkas[idx] = float(params[param_idx])
            param_idx += 1

        # Keep pKas sorted for validity
        curr_pkas.sort()

        return ChemicalSystem(
            name=base_sample.name,
            charge_base=base_sample.charge_base,
            pKas=curr_pkas,
            concentration=curr_c,
            formula=base_sample.formula,
            molar_mass=base_sample.molar_mass,
            temp_c=base_sample.temp_c,
            ionic_strength=base_sample.ionic_strength,
        )

    def _calc_ph_profile(sample_sys: ChemicalSystem) -> np.ndarray:
        cand_sample_systems = [sample_sys] + list(config.sample_systems[1:])
        ch_sample_0 = _determine_initial_protons(cand_sample_systems, None)
        sample_inert = (
            list(config.sample_inert_salts)
            if config.sample_inert_salts
            else _infer_counterions(cand_sample_systems)
        )

        cand_config = TitrationConfig(
            sample_systems=cand_sample_systems,
            sample_volume=config.sample_volume,
            titrant_systems=config.titrant_systems,
            water_volume=config.water_volume,
            max_volume=config.max_volume,
            step_size=config.step_size,
            calc_ionic_strength=config.calc_ionic_strength,
            temp_c=config.temp_c,
            mode=config.mode,
            sample_inert_salts=config.sample_inert_salts,
            titrant_inert_salts=config.titrant_inert_salts,
        )

        calc_phs = np.zeros(len(v_exp), dtype=float)
        last_valid_ph = float(ph_exp[0])
        for i, v in enumerate(v_exp):
            try:
                pt = _solve_titration_point(
                    v=v,
                    config=cand_config,
                    ch_sample_0=ch_sample_0,
                    ch_titrant_0=ch_titrant_0,
                    sample_inert=sample_inert,
                    titrant_inert=titrant_inert,
                )
                calc_phs[i] = pt.ph
                last_valid_ph = pt.ph
            except Exception:
                calc_phs[i] = last_valid_ph  # Smooth penalty preserving continuity

        return calc_phs

    def _residuals(params: np.ndarray) -> np.ndarray:
        sample_cand = _build_candidate_sample(params)
        ph_calc = _calc_ph_profile(sample_cand)
        return ph_calc - ph_exp

    opt_res = least_squares(
        _residuals,
        x0=np.array(x0, dtype=float),
        bounds=(np.array(lb, dtype=float), np.array(ub, dtype=float)),
        max_nfev=max_nfev,
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )

    opt_sample = _build_candidate_sample(opt_res.x)
    final_ph = _calc_ph_profile(opt_sample)
    residuals = final_ph - ph_exp

    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((ph_exp - np.mean(ph_exp)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    fitted_params: dict[str, float] = {}
    for name, val in zip(param_names, opt_res.x):
        fitted_params[name] = float(val)

    return FitResult(
        fitted_parameters=fitted_params,
        residuals=residuals,
        rmse=rmse,
        r_squared=r_squared,
        fitted_ph=final_ph,
        success=bool(opt_res.success),
        message=str(opt_res.message),
    )
