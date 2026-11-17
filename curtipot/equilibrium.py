"""Chemical equilibrium calculation engine with Davies activity model.

Reimplements the equilibrium algorithm from CurTiPot (pH.bas), ensuring
exact compatibility with CurTiPot Excel calculations.
"""

from __future__ import annotations

import math
from typing import Sequence, Any
import numpy as np
from scipy.optimize import brentq

from curtipot.models import ChemicalSystem, EquilibriumResult

# Reference physical constants at 25 °C
DEFAULT_PKW = 13.997  # Exact CurTiPot value at 25 °C
DEFAULT_DAVIES_A = 0.509  # CurTiPot K16 value at 25 °C (0.509)
DEFAULT_DAVIES_B = 0.30  # CurTiPot K17 value


def pkw_at_temperature(temp_c: float) -> float:
    """Calculate pKw at a given temperature in °C.

    At 25.0 °C, returns exactly 13.997 (matching CurTiPot).
    For other temperatures, uses the Harned-Owen equation.
    """
    if abs(temp_c - 25.0) < 1e-6:
        return DEFAULT_PKW
    t_k = temp_c + 273.15
    # Empirical formulation for pKw(T)
    pkw = 4470.99 / t_k - 6.0875 + 0.01706 * t_k
    return float(pkw)


def davies_a_at_temperature(temp_c: float) -> float:
    """Calculate Debye-Hückel / Davies A parameter at a given temperature in °C.

    At 25.0 °C, returns 0.509 (matching CurTiPot K16).
    """
    if abs(temp_c - 25.0) < 1e-6:
        return DEFAULT_DAVIES_A
    # Standard dielectric constant formula for water
    t = temp_c
    eps = 87.740 - 0.40008 * t + 9.398e-4 * (t**2) - 1.410e-6 * (t**3)
    t_k = temp_c + 273.15
    a = 1.8248e6 * ((eps * t_k) ** -1.5)
    return float(a)


def davies_activity_coefficient(
    ionic_strength: float,
    charge: int,
    a_param: float = DEFAULT_DAVIES_A,
    b_param: float = DEFAULT_DAVIES_B,
) -> float:
    """Calculate single ion activity coefficient using the Davies equation.

    Parameters
    ----------
    ionic_strength : float
        Ionic strength in mol/L.
    charge : int
        Electrical charge of the ion (z).
    a_param : float
        Davies A parameter (default: 0.509 at 25 °C).
    b_param : float
        Davies b parameter (default: 0.30).

    Returns
    -------
    float
        Activity coefficient gamma.
    """
    if ionic_strength <= 0.0 or charge == 0:
        return 1.0
    sqrt_i = math.sqrt(ionic_strength)
    termo = -a_param * (sqrt_i / (1.0 + sqrt_i) - b_param * ionic_strength)
    return float(10.0 ** (termo * (charge**2)))


def calculate_ionic_strength(
    h_conc: float,
    oh_conc: float,
    systems: Sequence[ChemicalSystem] | None = None,
    apparent_betas: list[list[float]] | None = None,
    inert_salts: Sequence[tuple[float, int]] | None = None,
) -> float:
    """Calculate total ionic strength of the solution.

    I = 0.5 * ( [H+] + [OH-] + sum_ions( C_ion * z_ion^2 ) + sum_systems( C_j * sum_i( alpha_i * (z_B + i)^2 ) ) )
    """
    # Inert ions contribution
    sum_z2_inert = 0.0
    if inert_salts:
        for conc, charge in inert_salts:
            if conc > 0 and charge != 0:
                sum_z2_inert += conc * (charge**2)

    # Chemical systems contribution
    sum_z2_systems = 0.0
    if systems:
        for j, sys in enumerate(systems):
            if sys.concentration <= 0:
                continue
            betas = apparent_betas[j] if apparent_betas is not None else sys.betas
            np_prot = sys.num_protons
            sden = 1.0
            for i in range(1, np_prot + 1):
                sden += betas[i] * (h_conc**i)

            sys_term = 0.0
            for i in range(0, np_prot + 1):
                term = (1.0 if i == 0 else betas[i]) * (h_conc**i)
                alpha_i = term / sden
                sys_term += alpha_i * ((sys.charge_base + i) ** 2)
            sum_z2_systems += sys.concentration * sys_term

    fi = 0.5 * (h_conc + oh_conc + sum_z2_inert + sum_z2_systems)
    return float(fi)


def proton_balance(
    h_conc: float,
    systems: Sequence[ChemicalSystem],
    kw: float = 10.0 ** (-DEFAULT_PKW),
    apparent_betas: list[list[float]] | None = None,
) -> float:
    """Calculate total proton concentration CH_calc at a given [H+].

    CH_calc = [H+] + sum_j( C_j * n_bar_j ) - Kw / [H+]
    """
    if h_conc <= 0:
        raise ValueError("h_conc must be positive")

    h_prot = 0.0
    for j, sys in enumerate(systems):
        if sys.concentration <= 0 or sys.num_protons == 0:
            continue
        betas = apparent_betas[j] if apparent_betas is not None else sys.betas
        np_prot = sys.num_protons
        num = 0.0
        sden = 1.0
        for i in range(1, np_prot + 1):
            term = betas[i] * (h_conc**i)
            num += i * term
            sden += term
        h_prot += sys.concentration * (num / sden)

    return float(h_conc + h_prot - kw / h_conc)


def _determine_initial_protons(
    systems: Sequence[ChemicalSystem],
    initial_protons: float | Sequence[float] | None,
) -> float:
    """Determine CH_total from user specification or default state."""
    if initial_protons is not None:
        if isinstance(initial_protons, (int, float)):
            return float(initial_protons)
        if isinstance(initial_protons, (list, tuple)):
            if len(initial_protons) == len(systems):
                total = 0.0
                for p, sys in zip(initial_protons, systems):
                    total += float(p) * sys.concentration
                return total
            raise ValueError(
                f"Length of initial_protons ({len(initial_protons)}) must match systems ({len(systems)})"
            )

    # Inferred CH_total from species nature
    ch_total = 0.0
    for sys in systems:
        if sys.concentration <= 0:
            continue
        name_lower = sys.name.lower()
        # Hydroxides / strong bases are added as OH- (0 protons bound)
        if "hidróxido" in name_lower or "hydroxide" in name_lower or "base" in name_lower:
            n_init = 0
        else:
            # If the neutral form exists (zB + i = 0), neutral molecule was added
            neutral_i = -sys.charge_base
            if 0 <= neutral_i <= sys.num_protons:
                n_init = neutral_i
            else:
                # Fully protonated acid form
                n_init = sys.num_protons
        ch_total += n_init * sys.concentration

    return float(ch_total)


def _calculate_buffer_capacity(
    h_conc: float,
    oh_conc: float,
    systems: Sequence[ChemicalSystem],
    apparent_betas: list[list[float]],
) -> float:
    """Calculate buffer capacity beta = dCb/dpH in mol/(L*pH).

    Uses CurTiPot formula:
    beta = ln(10) * ( [H+] + [OH-] + sum_j( C_j * sum_{k<l} (l-k)^2 alpha_k alpha_l ) )
    """
    buff = 0.0
    for j, sys in enumerate(systems):
        if sys.concentration <= 0 or sys.num_protons == 0:
            continue
        betas = apparent_betas[j]
        np_prot = sys.num_protons
        sden = 1.0
        for i in range(1, np_prot + 1):
            sden += betas[i] * (h_conc**i)

        alphas = [1.0 / sden]
        for i in range(1, np_prot + 1):
            alphas.append((betas[i] * (h_conc**i)) / sden)

        sys_buff = 0.0
        for k in range(np_prot):
            if k + 1 <= np_prot:
                sys_buff += alphas[k] * alphas[k + 1]
            if k + 2 <= np_prot:
                sys_buff += 4.0 * alphas[k] * alphas[k + 2]
            if k + 3 <= np_prot:
                sys_buff += 9.0 * alphas[k] * alphas[k + 3]
        buff += sys.concentration * sys_buff

    buff += h_conc + oh_conc
    return float(math.log(10.0) * buff)


def solve_equilibrium(
    systems: Sequence[ChemicalSystem],
    initial_protons: float | Sequence[float] | None = None,
    inert_salts: Sequence[tuple[float, int]] | None = None,
    calc_activity: bool = True,
    temp_c: float = 25.0,
    a_davies: float | None = None,
    b_davies: float = DEFAULT_DAVIES_B,
    max_iter: int = 60,
    tolerance: float = 1e-7,
) -> EquilibriumResult:
    """Solve the acid-base equilibrium and calculate pH, ionic strength, and speciation.

    Parameters
    ----------
    systems : Sequence[ChemicalSystem]
        List of chemical systems participating in acid-base equilibria.
    initial_protons : float | Sequence[float] | None
        Total proton concentration CH_total (mol/L) or protons per system.
        If None, automatically inferred from the neutral or added forms.
    inert_salts : Sequence[tuple[float, int]] | None
        List of (concentration, charge) for inert electrolytes (e.g. [(0.1, 1)] for Na+).
    calc_activity : bool
        Whether to perform iterative Davies activity corrections. Default: True.
    temp_c : float
        Temperature in °C (default: 25.0).
    a_davies : float | None
        Davies A parameter. If None, calculated from temperature (0.509 at 25 °C).
    b_davies : float
        Davies b parameter (default: 0.30).
    max_iter : int
        Maximum iterations for activity correction loop (default: 60).
    tolerance : float
        Convergence threshold in |dp[H]| (default: 1e-7).

    Returns
    -------
    EquilibriumResult
        Complete calculation results (pH, ionic strength, species concentrations, etc.)
    """
    ch_total = _determine_initial_protons(systems, initial_protons)
    pkw = pkw_at_temperature(temp_c)
    kw = 10.0 ** (-pkw)
    a_dav = a_davies if a_davies is not None else davies_a_at_temperature(temp_c)

    # Collect any inert ions present in systems as electrolytes with 0 protons
    combined_inert: list[tuple[float, int]] = []
    if inert_salts:
        combined_inert.extend(inert_salts)
    for sys in systems:
        if sys.num_protons == 0 and sys.concentration > 0 and sys.charge_base != 0:
            combined_inert.append((sys.concentration, sys.charge_base))

    # Base protonation constants
    base_betas = [list(sys.betas) for sys in systems]
    app_betas = [list(b) for b in base_betas]
    kw_app = kw

    # Define root-finding function for a fixed set of apparent betas and kw_app
    def root_fn(p_h_val: float, curr_betas: list[list[float]], curr_kw: float) -> float:
        h = 10.0 ** (-p_h_val)
        return proton_balance(h, systems, curr_kw, curr_betas) - ch_total

    # Bracket search helper
    def find_bracket(curr_betas: list[list[float]], curr_kw: float) -> tuple[float, float]:
        low, high = -2.0, 16.0
        f_low = root_fn(low, curr_betas, curr_kw)
        f_high = root_fn(high, curr_betas, curr_kw)
        while f_low * f_high > 0 and (low > -6.0 or high < 20.0):
            low -= 1.0
            high += 1.0
            f_low = root_fn(low, curr_betas, curr_kw)
            f_high = root_fn(high, curr_betas, curr_kw)
        return low, high

    # Step 1: Solve for concentration p[H] with zero ionic strength
    b_low, b_high = find_bracket(base_betas, kw)
    p_h_conc = brentq(root_fn, b_low, b_high, args=(base_betas, kw), xtol=1e-12)
    p_h = p_h_conc

    gamma_h = 1.0
    fi = calculate_ionic_strength(
        10.0 ** (-p_h),
        kw / (10.0 ** (-p_h)),
        systems,
        base_betas,
        combined_inert,
    )

    iterations = 0
    converged = True

    # Step 2: Iterative Davies activity refinement if requested
    if calc_activity:
        for it in range(1, max_iter + 1):
            iterations = it
            h_curr = 10.0 ** (-p_h)
            oh_curr = kw_app / h_curr

            fi = calculate_ionic_strength(
                h_curr,
                oh_curr,
                systems,
                app_betas,
                combined_inert,
            )

            # Davies activity coefficient
            sqrt_fi = math.sqrt(fi) if fi > 0 else 0.0
            termo = -a_dav * (sqrt_fi / (1.0 + sqrt_fi) - b_davies * fi) if fi > 0 else 0.0
            gamma_h = float(10.0**termo)
            kw_app = kw / (gamma_h**2)

            # Update apparent betas
            for j, sys in enumerate(systems):
                if sys.num_protons == 0:
                    continue
                zb = sys.charge_base
                gama_0 = float(10.0 ** (termo * (zb**2)))
                for i in range(1, sys.num_protons + 1):
                    gama_i = float(10.0 ** (termo * ((zb + i) ** 2)))
                    app_betas[j][i] = base_betas[j][i] * gama_0 * (gamma_h**i) / gama_i

            # Solve for new p[H]
            b_low, b_high = find_bracket(app_betas, kw_app)
            new_p_h = brentq(root_fn, b_low, b_high, args=(app_betas, kw_app), xtol=1e-12)

            dp_h = abs(new_p_h - p_h)
            p_h = new_p_h

            if dp_h < tolerance:
                converged = True
                break
        else:
            converged = False

    # Final quantities
    # pH = p[H] - log10(gamma_H)
    ph = float(p_h - math.log10(gamma_h))
    poh = float(pkw - ph)
    h_conc = float(10.0 ** (-p_h))
    oh_conc = float(kw_app / h_conc)

    # Species concentrations
    species_concs: dict[str, list[float]] = {}
    for j, sys in enumerate(systems):
        if sys.num_protons == 0:
            species_concs[sys.name] = [sys.concentration]
            continue
        sden = 1.0
        for i in range(1, sys.num_protons + 1):
            sden += app_betas[j][i] * (h_conc**i)
        sys_species: list[float] = []
        for i in range(sys.num_protons + 1):
            term = (1.0 if i == 0 else app_betas[j][i]) * (h_conc**i)
            alpha_i = term / sden
            sys_species.append(float(sys.concentration * alpha_i))
        species_concs[sys.name] = sys_species

    buffer_cap = _calculate_buffer_capacity(h_conc, oh_conc, systems, app_betas)

    return EquilibriumResult(
        ph=ph,
        poh=poh,
        h_conc=h_conc,
        oh_conc=oh_conc,
        ionic_strength=fi,
        activity_coeff_h=gamma_h,
        buffer_capacity=buffer_cap,
        species_concentrations=species_concs,
        converged=converged,
        iterations=iterations,
    )


def solve_ph(
    systems: Sequence[ChemicalSystem],
    inert_ions: Sequence[tuple[float, int]] | None = None,
    initial_protons: float | Sequence[float] | None = None,
    calc_activity: bool = True,
    temp_c: float = 25.0,
    **kwargs: Any,
) -> EquilibriumResult:
    """Convenience alias for solve_equilibrium supporting inert_ions parameter name."""
    return solve_equilibrium(
        systems=systems,
        initial_protons=initial_protons,
        inert_salts=inert_ions,
        calc_activity=calc_activity,
        temp_c=temp_c,
        **kwargs,
    )
