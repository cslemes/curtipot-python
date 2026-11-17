"""Tests for potentiometric titration simulation (curtipot.titration)."""

import pytest
import numpy as np

from curtipot.models import ChemicalSystem, TitrationConfig, TitrationResult
from curtipot.titration import (
    simulate_titration,
    simulate_volume_steps,
    simulate_ph_steps,
)


@pytest.fixture
def strong_acid_hcl():
    return ChemicalSystem(
        name="Ácido clorídrico",
        charge_base=-1,
        pkas=[-7.0],
        concentration=0.1,
    )


@pytest.fixture
def strong_base_naoh():
    return ChemicalSystem(
        name="Hidróxido de sódio",
        charge_base=-1,
        pkas=[13.997],
        concentration=0.1,
    )


@pytest.fixture
def weak_acid_acetic():
    return ChemicalSystem(
        name="Ácido acético",
        charge_base=-1,
        pkas=[4.757],
        concentration=0.1,
    )


@pytest.fixture
def weak_base_nh3():
    return ChemicalSystem(
        name="Amônia",
        charge_base=0,
        pkas=[9.244],
        concentration=0.1,
    )


def test_strong_acid_strong_base_titration(strong_acid_hcl, strong_base_naoh):
    """Test titration of 25.0 mL HCl 0.1 M with NaOH 0.1 M (0 to 50 mL)."""
    cfg = TitrationConfig(
        sample_systems=[strong_acid_hcl],
        sample_volume=25.0,
        titrant_systems=[strong_base_naoh],
        water_volume=0.0,
        max_volume=50.0,
        step_size=0.5,
        calc_ionic_strength=False,
        mode="volume",
    )

    result = simulate_titration(cfg)
    assert isinstance(result, TitrationResult)
    assert len(result.points) > 0

    # Initial point V = 0: pH ~ 1.0
    pt_0 = result.points[0]
    assert pt_0.volume_added == pytest.approx(0.0)
    assert pt_0.ph == pytest.approx(1.0, abs=0.05)

    # Equivalence point at V = 25.0 mL: pH ~ 7.0
    pt_25 = [p for p in result.points if abs(p.volume_added - 25.0) < 1e-5][0]
    assert pt_25.ph == pytest.approx(7.0, abs=0.1)

    # Final point V = 50.0 mL: pH ~ 12.5 (excess NaOH diluted 25/75 = 0.0333 M)
    pt_50 = result.points[-1]
    assert pt_50.volume_added == pytest.approx(50.0)
    assert pt_50.ph == pytest.approx(12.5, abs=0.1)

    # Numerical derivative dpH/dV: peak at V = 25.0 mL
    dpH = result.dpH_dV
    max_idx = int(np.argmax(dpH))
    assert result.volumes[max_idx] == pytest.approx(25.0, abs=0.1)

    # Equivalence point detection in result
    assert len(result.equivalence_points) >= 1
    eq_vols = [eq["volume"] for eq in result.equivalence_points]
    assert any(abs(v - 25.0) <= 0.5 for v in eq_vols)


def test_weak_acid_strong_base_titration(weak_acid_acetic, strong_base_naoh):
    """Test titration of 25.0 mL Acetic acid 0.1 M with NaOH 0.1 M."""
    cfg = TitrationConfig(
        sample_systems=[weak_acid_acetic],
        sample_volume=25.0,
        titrant_systems=[strong_base_naoh],
        water_volume=0.0,
        max_volume=50.0,
        step_size=0.5,
        calc_ionic_strength=False,
        mode="volume",
    )

    result = simulate_volume_steps(cfg)
    assert isinstance(result, TitrationResult)

    # Initial point V = 0: pH ~ 2.88
    pt_0 = result.points[0]
    assert pt_0.ph == pytest.approx(2.88, abs=0.03)

    # Half-titration at V = 12.5 mL: pH ~ pKa (4.76)
    pt_half = [p for p in result.points if abs(p.volume_added - 12.5) < 1e-5][0]
    assert pt_half.ph == pytest.approx(4.757, abs=0.03)

    # Equivalence point at V = 25.0 mL: pH basic ~ 8.72
    pt_25 = [p for p in result.points if abs(p.volume_added - 25.0) < 1e-5][0]
    assert pt_25.ph == pytest.approx(8.72, abs=0.10)

    # Equivalence points list
    assert len(result.equivalence_points) == 1
    assert result.equivalence_points[0]["volume"] == pytest.approx(25.0, abs=0.5)


def test_water_dilution_effect(strong_acid_hcl, strong_base_naoh):
    """Test dilution effect with addition of water (V_water > 0)."""
    # Without water: 25 mL sample
    cfg_no_water = TitrationConfig(
        sample_systems=[strong_acid_hcl],
        sample_volume=25.0,
        titrant_systems=[strong_base_naoh],
        water_volume=0.0,
        max_volume=50.0,
        step_size=1.0,
        calc_ionic_strength=False,
    )
    res_no_water = simulate_titration(cfg_no_water)

    # With 25 mL water: total initial volume = 50 mL -> [HCl] is halved to 0.05 M
    cfg_with_water = TitrationConfig(
        sample_systems=[strong_acid_hcl],
        sample_volume=25.0,
        titrant_systems=[strong_base_naoh],
        water_volume=25.0,
        max_volume=50.0,
        step_size=1.0,
        calc_ionic_strength=False,
    )
    res_with_water = simulate_titration(cfg_with_water)

    # Initial pH with water: -log10(0.05) ~ 1.30 vs 1.00 without water
    assert res_with_water.points[0].ph == pytest.approx(1.30, abs=0.03)
    assert res_no_water.points[0].ph == pytest.approx(1.00, abs=0.03)

    # Equivalence point is still at V = 25.0 mL (same number of moles)
    pt_eq = [p for p in res_with_water.points if abs(p.volume_added - 25.0) < 1e-5][0]
    assert pt_eq.ph == pytest.approx(7.0, abs=0.1)


def test_weak_base_strong_acid_titration(weak_base_nh3, strong_acid_hcl):
    """Test titration of 25.0 mL NH3 0.1 M with HCl 0.1 M."""
    cfg = TitrationConfig(
        sample_systems=[weak_base_nh3],
        sample_volume=25.0,
        titrant_systems=[strong_acid_hcl],
        water_volume=0.0,
        max_volume=50.0,
        step_size=0.5,
        calc_ionic_strength=False,
    )
    result = simulate_titration(cfg)

    # Initial point: basic pH ~ 11.12
    assert result.points[0].ph == pytest.approx(11.12, abs=0.05)

    # Half-titration V = 12.5 mL: pH ~ pKa (9.24)
    pt_half = [p for p in result.points if abs(p.volume_added - 12.5) < 1e-5][0]
    assert pt_half.ph == pytest.approx(9.24, abs=0.05)

    # Equivalence point V = 25.0 mL: acidic pH ~ 5.27
    pt_eq = [p for p in result.points if abs(p.volume_added - 25.0) < 1e-5][0]
    assert pt_eq.ph == pytest.approx(5.27, abs=0.10)

    # Derivative |dpH/dV| has peak at 25.0 mL
    abs_dpH = np.abs(result.dpH_dV)
    max_idx = int(np.argmax(abs_dpH))
    assert result.volumes[max_idx] == pytest.approx(25.0, abs=0.1)


def test_polyprotic_phosphoric_acid(strong_base_naoh):
    """Test diprotic/triprotic titration showing two clear equivalence points."""
    h3po4 = ChemicalSystem(
        name="Ácido fosfórico",
        charge_base=-3,
        pkas=[2.148, 7.199, 12.35],
        concentration=0.1,
    )
    cfg = TitrationConfig(
        sample_systems=[h3po4],
        sample_volume=25.0,
        titrant_systems=[strong_base_naoh],
        water_volume=0.0,
        max_volume=60.0,
        step_size=0.5,
        calc_ionic_strength=False,
    )
    result = simulate_titration(cfg)

    # Should detect 2 equivalence points: ~25.0 mL and ~50.0 mL
    eq_vols = [eq["volume"] for eq in result.equivalence_points]
    assert len(eq_vols) >= 2
    assert any(abs(v - 25.0) <= 0.5 for v in eq_vols)
    assert any(abs(v - 50.0) <= 0.5 for v in eq_vols)


def test_ph_steps_simulation(weak_acid_acetic, strong_base_naoh):
    """Test simulate_ph_steps with regular pH intervals."""
    cfg = TitrationConfig(
        sample_systems=[weak_acid_acetic],
        sample_volume=25.0,
        titrant_systems=[strong_base_naoh],
        water_volume=0.0,
        max_volume=50.0,
        step_size=0.5,  # 0.5 pH unit steps
        calc_ionic_strength=False,
        mode="ph",
    )

    result = simulate_ph_steps(cfg)
    assert isinstance(result, TitrationResult)
    assert len(result.points) > 5

    # Volumes should be non-decreasing
    vols = result.volumes
    assert np.all(np.diff(vols) >= -1e-6)

    # Equivalence point detected
    assert len(result.equivalence_points) >= 1
    eq_vol = result.equivalence_points[0]["volume"]
    assert eq_vol == pytest.approx(25.0, abs=1.0)


def test_invalid_mode_raises(strong_acid_hcl, strong_base_naoh):
    """Test that invalid titration mode raises ValueError."""
    cfg = TitrationConfig(
        sample_systems=[strong_acid_hcl],
        sample_volume=25.0,
        titrant_systems=[strong_base_naoh],
        mode="invalid_mode",
    )
    with pytest.raises(ValueError, match="mode"):
        simulate_titration(cfg)


def test_titration_with_activity_and_counterions(strong_acid_hcl, strong_base_naoh):
    """Test titration with Davies activity correction enabled and exact ionic strength."""
    cfg = TitrationConfig(
        sample_systems=[strong_acid_hcl],
        sample_volume=25.0,
        titrant_systems=[strong_base_naoh],
        water_volume=0.0,
        max_volume=50.0,
        step_size=0.5,
        calc_ionic_strength=True,
    )
    res = simulate_titration(cfg)

    # Point 0: 0.1 M HCl -> I ~ 0.10 M
    assert res.points[0].ionic_strength == pytest.approx(0.10, abs=1e-3)
    assert res.points[0].ph > 1.0  # Davies gamma_H < 1 raises pH slightly from 1.00 to ~ 1.09

    # Equivalence point at 25 mL: NaCl diluted 25/50 = 0.05 M -> I ~ 0.05 M
    pt_eq = [p for p in res.points if abs(p.volume_added - 25.0) < 1e-5][0]
    assert pt_eq.ionic_strength == pytest.approx(0.05, abs=1e-3)
    assert pt_eq.ph == pytest.approx(7.0, abs=0.1)

    # Equivalence point detected
    assert len(res.equivalence_points) == 1
    assert res.equivalence_points[0]["volume"] == pytest.approx(25.0, abs=0.1)

