"""Species distribution diagrams and buffer capacity calculations.

Reimplements the distribution calculations from CurTiPot (Distribuição.bas)
in vectorized NumPy form.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from curtipot.models import ChemicalSystem, DistributionResult

LN10 = 2.302585092994046
DEFAULT_KW = 1.0069e-14  # Equivalent to CurTiPot 10^-13.997 at 25 °C


def calculate_distribution(
    system: ChemicalSystem,
    ph_values: np.ndarray | Sequence[float] | None = None,
    gamma_h: float = 1.0,
    kw: float = DEFAULT_KW,
) -> DistributionResult:
    """Calculate species distribution (alpha fractions) and buffer capacity across a pH range.

    Parameters
    ----------
    system : ChemicalSystem
        Chemical system with protonation constants and base charge.
    ph_values : np.ndarray | Sequence[float] | None
        Array of pH values. If None, defaults to 141 points from 0.0 to 14.0 (step 0.1).
    gamma_h : float
        Activity coefficient for H+ (default: 1.0).
    kw : float
        Water ionization constant Kw (default: 1.0069e-14).

    Returns
    -------
    DistributionResult
        Container holding pH array, alphas, log_alphas, n_bar, average_charge,
        buffer_capacity, and species labels.
    """
    if ph_values is None:
        ph_arr = np.linspace(0.0, 14.0, 141, dtype=float)
    else:
        ph_arr = np.atleast_1d(np.asarray(ph_values, dtype=float))

    if len(ph_arr) == 0:
        return DistributionResult(
            ph_array=ph_arr,
            alphas=np.empty((system.num_protons + 1, 0), dtype=float),
            log_alphas=np.empty((system.num_protons + 1, 0), dtype=float),
            n_bar=np.empty(0, dtype=float),
            average_charge=np.empty(0, dtype=float),
            buffer_capacity=np.empty(0, dtype=float),
            species_labels=list(system.species_names),
        )

    # Effective H+ concentration: H = (10^-pH) / gamma_H
    h_conc = (10.0 ** (-ph_arr)) / max(gamma_h, 1e-12)
    safe_h = np.maximum(h_conc, 1e-30)

    n_protons = system.num_protons
    m_points = len(ph_arr)

    if n_protons == 0:
        alphas = np.ones((1, m_points), dtype=float)
        log_alphas = np.zeros((1, m_points), dtype=float)
        n_bar = np.zeros(m_points, dtype=float)
        average_charge = np.full(m_points, float(system.charge_base), dtype=float)
        buff = LN10 * (safe_h + kw / safe_h)
    else:
        betas = system.betas
        terms = np.zeros((n_protons + 1, m_points), dtype=float)
        terms[0, :] = 1.0
        for i in range(1, n_protons + 1):
            terms[i, :] = betas[i] * (safe_h**i)

        f0 = np.sum(terms, axis=0)
        # Avoid division by zero
        f0 = np.maximum(f0, 1e-300)
        alphas = terms / f0

        # Bound log_alphas using 1e-20 threshold like CurTiPot
        log_alphas = np.log10(np.maximum(alphas, 1e-20))

        # Average protonation number n_bar
        i_vec = np.arange(n_protons + 1, dtype=float).reshape(-1, 1)
        n_bar = np.sum(i_vec * alphas, axis=0)

        # Average charge of species = z_B + n_bar
        average_charge = system.charge_base + n_bar

        # Buffer capacity beta = ln(10) * ( [H+] + Kw/[H+] + Ct * Var(i) )
        # Var(i) = sum( i^2 * alpha_i ) - (n_bar)^2
        var_n = np.sum((i_vec**2) * alphas, axis=0) - (n_bar**2)
        var_n = np.maximum(var_n, 0.0)

        buff = LN10 * (safe_h + (kw / safe_h) + system.concentration * var_n)

    return DistributionResult(
        ph_array=ph_arr,
        alphas=alphas,
        log_alphas=log_alphas,
        n_bar=n_bar,
        average_charge=average_charge,
        buffer_capacity=buff,
        species_labels=list(system.species_names),
    )


def calculate_alpha_species(
    system: ChemicalSystem,
    h_conc: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Calculate alpha fractions matrix (shape: [num_species, len(h_conc)]) from [H+]."""
    h_arr = np.atleast_1d(np.asarray(h_conc, dtype=float))
    safe_h = np.maximum(h_arr, 1e-30)
    n_protons = system.num_protons
    m_points = len(h_arr)

    if n_protons == 0:
        return np.ones((1, m_points), dtype=float)

    betas = system.betas
    terms = np.zeros((n_protons + 1, m_points), dtype=float)
    terms[0, :] = 1.0
    for i in range(1, n_protons + 1):
        terms[i, :] = betas[i] * (safe_h**i)

    f0 = np.sum(terms, axis=0)
    f0 = np.maximum(f0, 1e-300)
    return terms / f0


def calculate_buffer_capacity(
    system: ChemicalSystem,
    ph_values: np.ndarray | Sequence[float],
    gamma_h: float = 1.0,
    kw: float = DEFAULT_KW,
) -> np.ndarray:
    """Calculate buffer capacity beta across pH values."""
    res = calculate_distribution(system, ph_values=ph_values, gamma_h=gamma_h, kw=kw)
    return res.buffer_capacity


def find_isoelectric_point(result: DistributionResult) -> float | None:
    """Find the approximate isoelectric point (pI) where average charge equals zero.

    Interpolates linearly between the two closest points where the sign of
    average charge changes, matching CurTiPot's Piso algorithm.

    Parameters
    ----------
    result : DistributionResult
        Distribution result containing average_charge and ph_array.

    Returns
    -------
    float | None
        Approximate pH of isoelectric point, or None if charge does not cross 0.
    """
    charges = result.average_charge
    phs = result.ph_array

    if len(charges) < 2:
        return None

    for k in range(len(charges) - 1):
        c1 = float(charges[k])
        c2 = float(charges[k + 1])
        if (c1 <= 0.0 and c2 >= 0.0) or (c1 >= 0.0 and c2 <= 0.0):
            if math.isclose(c1, c2, abs_tol=1e-15):
                return float(phs[k])
            p1 = float(phs[k])
            p2 = float(phs[k + 1])
            # Linear interpolation: pI = p1 + (p2 - p1) * |c1| / |c2 - c1|
            return float(p1 + (p2 - p1) * abs(c1) / abs(c2 - c1))

    return None
