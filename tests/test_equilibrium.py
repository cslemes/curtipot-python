"""Tests for equilibrium pH calculation and Davies activity model."""

import pytest
import math
from curtipot.models import ChemicalSystem, EquilibriumResult
from curtipot.equilibrium import (
    solve_equilibrium,
    solve_ph,
    calculate_ionic_strength,
    davies_activity_coefficient,
    proton_balance,
)


def test_davies_activity_coefficient():
    """Test Davies activity coefficient calculation."""
    # At I = 0, gamma = 1.0
    gamma = davies_activity_coefficient(ionic_strength=0.0, charge=1)
    assert gamma == pytest.approx(1.0)

    # At I = 0.1, z = 1, A = 0.509, b = 0.3
    # log10(gamma) = -0.509 * (sqrt(0.1)/(1 + sqrt(0.1)) - 0.3*0.1) * 1^2
    # sqrt(0.1) = 0.3162277
    # 0.3162277 / 1.3162277 = 0.24025
    # 0.24025 - 0.03 = 0.21025
    # log10(gamma) = -0.509 * 0.21025 = -0.1070
    # gamma = 10^(-0.1070) = 0.7816
    gamma_1 = davies_activity_coefficient(ionic_strength=0.1, charge=1, a_param=0.509, b_param=0.30)
    assert gamma_1 == pytest.approx(0.7816, abs=1e-3)

    # For z = 2, log10(gamma_2) = 4 * log10(gamma_1)
    gamma_2 = davies_activity_coefficient(ionic_strength=0.1, charge=2, a_param=0.509, b_param=0.30)
    assert gamma_2 == pytest.approx(gamma_1**4, rel=1e-4)

    # For neutral species z = 0, gamma = 1.0
    gamma_0 = davies_activity_coefficient(ionic_strength=0.1, charge=0)
    assert gamma_0 == pytest.approx(1.0)


def test_strong_acid_hcl():
    """Test strong acid HCl 0.01 M without and with Davies activity."""
    # HCl as strong acid: charge_base = -1, pKa = -7.0, conc = 0.01 M
    hcl = ChemicalSystem(name="Ácido clorídrico", charge_base=-1, pkas=[-7.0], concentration=0.01)

    # Without activity: pHconc = -log10(0.01) = 2.0
    res_no_act = solve_equilibrium([hcl], calc_activity=False)
    assert isinstance(res_no_act, EquilibriumResult)
    assert res_no_act.ph == pytest.approx(2.0, abs=1e-3)
    assert res_no_act.h_conc == pytest.approx(0.01, rel=1e-3)

    # With activity: I ~ 0.01 M, pH ~ 2.045
    res_act = solve_equilibrium([hcl], calc_activity=True)
    assert res_act.ph == pytest.approx(2.045, abs=0.05)
    assert res_act.ionic_strength == pytest.approx(0.01, abs=1e-3)
    assert res_act.activity_coeff_h < 1.0


def test_strong_base_naoh():
    """Test strong base NaOH 0.01 M -> pH ~ 12.0."""
    naoh = ChemicalSystem(name="Hidróxido de sódio", charge_base=-1, pkas=[13.997], concentration=0.01)
    # NaOH added as base: initial_protons = 0.0, inert counterion Na+ = 0.01 M (z=+1)
    res = solve_equilibrium([naoh], initial_protons=0.0, inert_salts=[(0.01, 1)], calc_activity=False)
    # Kw = 10^-13.997, [OH-] = 0.01 M -> [H+] = 10^-11.997 -> pH = 11.997 ~ 12.0
    assert res.ph == pytest.approx(11.997, abs=1e-2)

    # Also test solve_ph alias
    res_alias = solve_ph([naoh], initial_protons=0.0, inert_ions=[(0.01, 1)], calc_activity=False)
    assert res_alias.ph == pytest.approx(res.ph, abs=1e-6)


def test_weak_acid_acetic():
    """Test weak acid acetic acid 0.1 M (pKa 4.757) -> pHconc ~ 2.878 - 2.881."""
    acetic = ChemicalSystem(name="Ácido acético", charge_base=-1, pkas=[4.757], concentration=0.1)

    # Without activity
    res_conc = solve_equilibrium([acetic], calc_activity=False)
    assert res_conc.ph == pytest.approx(2.881, abs=0.005)

    # With activity
    res_act = solve_equilibrium([acetic], calc_activity=True)
    assert res_act.ph == pytest.approx(2.881, abs=0.005)
    assert res_act.ionic_strength > 0.0
    assert res_act.buffer_capacity > 0.0


def test_phosphate_buffer_benchmark():
    """CurTiPot exact validation benchmark: Phosphate buffer.

    Conditions from CurTiPot pH.bas:
    - Phosphate: zB = -3, pKas = [2.148, 7.199, 12.35]
    - [HPO4^2-] = 0.0265 M (1 proton), [H2PO4^-] = 0.0200 M (2 protons)
    - Total C = 0.0465 M, CH_total = 0.0665 M
    - Counterion: Na+ = 0.073 M, z = +1
    - Temperature = 25 °C

    Expected Excel results:
    - pHconc = 7.3212 +- 0.001
    - pH (with activity) = 7.0006 +- 0.001
    - gamma_H = 0.7819 +- 0.001
    - Ionic Strength = 0.0995 +- 0.001
    """
    phosphate = ChemicalSystem(
        name="Ácido fosfórico",
        charge_base=-3,
        pkas=[2.148, 7.199, 12.35],
        concentration=0.0465,
    )
    ch_total = 0.0265 * 1 + 0.0200 * 2  # 0.0665 M
    inert_salts = [(0.073, 1)]  # 0.073 M Na+

    res = solve_equilibrium(
        systems=[phosphate],
        initial_protons=ch_total,
        inert_salts=inert_salts,
        calc_activity=True,
        temp_c=25.0,
    )

    # Check convergence
    assert res.converged is True

    # 1. pHconc (without activity)
    res_conc = solve_equilibrium(
        systems=[phosphate],
        initial_protons=ch_total,
        inert_salts=inert_salts,
        calc_activity=False,
        temp_c=25.0,
    )
    assert res_conc.ph == pytest.approx(7.3212, abs=1e-3)

    # 2. pH with activity
    assert res.ph == pytest.approx(7.0006, abs=1e-3)

    # 3. gamma_H
    assert res.activity_coeff_h == pytest.approx(0.7819, abs=1e-3)

    # 4. Ionic strength
    assert res.ionic_strength == pytest.approx(0.0995, abs=1e-3)

    # 5. Species concentrations
    assert "Ácido fosfórico" in res.species_concentrations
    species = res.species_concentrations["Ácido fosfórico"]
    assert len(species) == 4  # PO4^3-, HPO4^2-, H2PO4^-, H3PO4
    assert species[1] == pytest.approx(0.0265, abs=1e-3)  # HPO4^2-
    assert species[2] == pytest.approx(0.0200, abs=1e-3)  # H2PO4^-


def test_proton_balance_helper():
    """Test proton_balance helper function directly."""
    phosphate = ChemicalSystem(
        name="Ácido fosfórico",
        charge_base=-3,
        pkas=[2.148, 7.199, 12.35],
        concentration=0.0465,
    )
    # At pH 7.321204, with beta without activity, proton_balance should equal CH_total = 0.0665
    h_conc = 10.0**(-7.321204)
    ch_calc = proton_balance(
        h_conc=h_conc,
        systems=[phosphate],
        kw=10.0**(-13.997),
    )
    assert ch_calc == pytest.approx(0.0665, abs=1e-4)


def test_calculate_ionic_strength_helper():
    """Test calculate_ionic_strength helper function directly."""
    # 0.1 M NaCl: Na+ 0.1 M (+1), Cl- 0.1 M (-1)
    fi = calculate_ionic_strength(
        h_conc=1e-7,
        oh_conc=1e-7,
        systems=[],
        inert_salts=[(0.1, 1), (0.1, -1)],
    )
    # I = 0.5 * (1e-7 + 1e-7 + 0.1*1 + 0.1*1) = 0.1000
    assert fi == pytest.approx(0.1, abs=1e-4)


def test_pure_water():
    """Test pure water neutrality at 25 °C."""
    res = solve_equilibrium([], calc_activity=True)
    assert res.ph == pytest.approx(6.9985, abs=1e-3)
    assert res.poh == pytest.approx(6.9985, abs=1e-3)
    assert res.ionic_strength == pytest.approx(1e-7, abs=1e-6)


def test_acetate_buffer():
    """Test acetate buffer (0.1 M acetic acid + 0.1 M sodium acetate)."""
    # Acetic acid pKa = 4.757. Equal conc of HAc and Ac- gives pH ~ 4.757 (without activity)
    acetic = ChemicalSystem(name="Ácido acético", charge_base=-1, pkas=[4.757], concentration=0.2)
    # Total acetate = 0.2 M. Half is protonated (0.1 M HAc), so CH_total = 0.1 M.
    # Na+ = 0.1 M counterion.
    res_conc = solve_equilibrium(
        [acetic],
        initial_protons=0.1,
        inert_salts=[(0.1, 1)],
        calc_activity=False,
    )
    assert res_conc.ph == pytest.approx(4.757, abs=1e-3)

    res_act = solve_equilibrium(
        [acetic],
        initial_protons=0.1,
        inert_salts=[(0.1, 1)],
        calc_activity=True,
    )
    # With ionic strength ~ 0.1 M, gamma_H ~ 0.78, pH is slightly shifted
    assert res_act.ionic_strength == pytest.approx(0.1, abs=0.01)
    assert res_act.ph < 4.757  # pH drops slightly due to activity


def test_phosphoric_acid_polyprotic():
    """Test phosphoric acid 0.1 M solution."""
    phosphoric = ChemicalSystem(
        name="Ácido fosfórico",
        charge_base=-3,
        pkas=[2.148, 7.199, 12.35],
        concentration=0.1,
    )
    # Pure H3PO4 (fully protonated: 3 protons)
    res = solve_equilibrium([phosphoric], calc_activity=True)
    # First dissociation dominates: pKa1 = 2.148, C = 0.1 M -> pH ~ 1.6
    assert 1.5 < res.ph < 1.7
    assert res.converged is True

