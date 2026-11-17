"""Unit and integration tests for titration curve analysis and non-linear regression.

Covers:
- Cubic spline interpolation and continuous derivatives (dpH/dV, d2pH/dV2).
- Automatic equivalence point detection and precision refinement (VBA Análise I).
- Non-linear least-squares fitting of unknown parameters (concentration, pKas) (VBA Análise II).
"""

from __future__ import annotations

import numpy as np
import pytest

from curtipot.models import ChemicalSystem, TitrationConfig
from curtipot.titration import simulate_titration
from curtipot.analysis import (
    EquivalencePoint,
    SplineAnalysisResult,
    FitResult,
    analyze_spline,
    analyze_spline_derivatives,
    fit_titration_curve,
)


class TestSplineAnalysis:
    """Tests for cubic spline interpolation, numerical derivatives, and equivalence points."""

    def test_strong_acid_titration_equivalence_point(self):
        """Test equivalence point detection for HCl 0.1 M titrated with NaOH 0.1 M.

        Theoretical equivalence point for 25.0 mL of HCl 0.1 M titrated with NaOH 0.1 M:
        V_eq = 25.0 mL, pH_eq = 7.00.
        """
        # 1. Generate clean synthetic curve
        hcl = ChemicalSystem(name="Ácido clorídrico", charge_base=-1, pKas=[-2.0], concentration=0.10)
        naoh = ChemicalSystem(name="Hidróxido de sódio", charge_base=0, pKas=[14.0], concentration=0.10)
        config = TitrationConfig(
            sample_systems=[hcl],
            sample_volume=25.0,
            titrant_systems=[naoh],
            max_volume=50.0,
            step_size=0.5,
        )
        sim_res = simulate_titration(config)

        vols = sim_res.volumes
        phs = sim_res.phs

        # Add light numerical noise (std = 0.005 pH units)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.005, size=len(phs))
        phs_noisy = phs + noise

        # 2. Run analyze_spline
        result = analyze_spline(vols, phs_noisy, num_points=600)

        assert isinstance(result, SplineAnalysisResult)
        assert len(result.interpolated_volumes) == 600
        assert len(result.interpolated_phs) == 600
        assert len(result.dpH_dV) == 600
        assert len(result.d2pH_dV2) == 600
        assert result.r_squared > 0.999
        assert result.rmse < 0.05

        # Verify equivalence point
        assert len(result.equivalence_points) >= 1
        eq = result.equivalence_points[0]
        assert isinstance(eq, EquivalencePoint)
        assert eq.volume == pytest.approx(25.0, abs=0.05)
        assert eq.ph == pytest.approx(7.0, abs=0.2)
        assert eq.dpH_dV > 8.0  # Sharp slope at equivalence
        assert abs(eq.d2pH_dV2) < 1.0  # Second derivative passes through zero

    def test_consume_titration_result_directly(self):
        """Test analyze_spline accepting a TitrationResult object directly."""
        hcl = ChemicalSystem(name="HCl", charge_base=-1, pKas=[-2.0], concentration=0.05)
        naoh = ChemicalSystem(name="NaOH", charge_base=0, pKas=[14.0], concentration=0.10)
        config = TitrationConfig(
            sample_systems=[hcl],
            sample_volume=20.0,
            titrant_systems=[naoh],
            max_volume=20.0,
            step_size=0.2,
        )
        sim_res = simulate_titration(config)

        # Expected V_eq = 20.0 * 0.05 / 0.10 = 10.0 mL
        result = analyze_spline(sim_res)
        assert isinstance(result, SplineAnalysisResult)
        assert len(result.equivalence_points) >= 1
        eq = result.equivalence_points[0]
        assert eq.volume == pytest.approx(10.0, abs=0.05)

    def test_alias_analyze_spline_derivatives(self):
        """Test analyze_spline_derivatives alias function."""
        v = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        ph = np.array([2.0, 2.5, 3.2, 7.0, 10.8, 11.5, 12.0])
        res = analyze_spline_derivatives(v, ph)
        assert isinstance(res, SplineAnalysisResult)
        assert len(res.equivalence_points) >= 1

    def test_diprotic_acid_multiple_equivalence_points(self):
        """Test detection of two equivalence points in maleic acid titration."""
        # Maleic acid: pKa1 = 1.92, pKa2 = 6.27
        maleic = ChemicalSystem(
            name="Ácido maleico",
            charge_base=-2,
            pKas=[1.92, 6.27],
            concentration=0.05,
        )
        naoh = ChemicalSystem(name="NaOH", charge_base=0, pKas=[14.0], concentration=0.10)
        config = TitrationConfig(
            sample_systems=[maleic],
            sample_volume=20.0,
            titrant_systems=[naoh],
            max_volume=30.0,
            step_size=0.2,
        )
        sim_res = simulate_titration(config)
        # Expected V_eq1 = 10.0 mL, V_eq2 = 20.0 mL
        result = analyze_spline(sim_res)
        assert len(result.equivalence_points) == 2
        eq1, eq2 = result.equivalence_points
        assert eq1.volume == pytest.approx(10.0, abs=0.1)
        assert eq2.volume == pytest.approx(20.0, abs=0.1)

    def test_base_titrated_with_acid(self):
        """Test equivalence point detection when base is titrated with acid (negative dpH/dV)."""
        naoh = ChemicalSystem(name="NaOH", charge_base=0, pKas=[14.0], concentration=0.10)
        hcl = ChemicalSystem(name="HCl", charge_base=-1, pKas=[-2.0], concentration=0.10)
        config = TitrationConfig(
            sample_systems=[naoh],
            sample_volume=25.0,
            titrant_systems=[hcl],
            max_volume=50.0,
            step_size=0.5,
        )
        sim = simulate_titration(config)
        res = analyze_spline(sim)
        assert len(res.equivalence_points) == 1
        eq = res.equivalence_points[0]
        assert eq.volume == pytest.approx(25.0, abs=0.05)
        assert eq.ph == pytest.approx(7.0, abs=0.2)
        assert eq.dpH_dV < -8.0
        assert abs(eq.d2pH_dV2) < 1.0

    def test_insufficient_points_raises_error(self):
        """Test that fewer than 4 points raises a ValueError."""
        with pytest.raises(ValueError, match=r"(?i)at least 4 points"):
            analyze_spline([1.0, 2.0, 3.0], [3.0, 4.0, 5.0])


class TestNonLinearRegression:
    """Tests for non-linear curve fitting (VBA Análise II)."""

    def test_fit_acetic_acid_concentration(self):
        """Test fitting unknown concentration of acetic acid from titration data.

        Synthetic true concentration: C = 0.085 M, pKa = 4.757.
        Initial guess: C_0 = 0.10 M.
        Verify convergence to C ~ 0.085 M (+- 0.001) and R^2 > 0.999.
        """
        # True sample
        true_hac = ChemicalSystem(
            name="Ácido acético",
            charge_base=-1,
            pKas=[4.757],
            concentration=0.085,
        )
        naoh = ChemicalSystem(name="NaOH", charge_base=0, pKas=[14.0], concentration=0.10)
        true_config = TitrationConfig(
            sample_systems=[true_hac],
            sample_volume=25.0,
            titrant_systems=[naoh],
            max_volume=35.0,
            step_size=1.0,
        )
        true_res = simulate_titration(true_config)
        v_exp = true_res.volumes
        ph_exp = true_res.phs

        # Initial guess configuration with C = 0.10 M
        guess_hac = ChemicalSystem(
            name="Ácido acético",
            charge_base=-1,
            pKas=[4.757],
            concentration=0.10,
        )
        guess_config = TitrationConfig(
            sample_systems=[guess_hac],
            sample_volume=25.0,
            titrant_systems=[naoh],
            max_volume=35.0,
            step_size=1.0,
        )

        fit_res = fit_titration_curve(
            volumes=v_exp,
            ph_measured=ph_exp,
            config=guess_config,
            fit_concentration=True,
        )

        assert isinstance(fit_res, FitResult)
        assert fit_res.success is True
        assert fit_res.r_squared > 0.999
        assert fit_res.rmse < 0.05
        fitted_c = fit_res.fitted_parameters["concentration"]
        assert fitted_c == pytest.approx(0.085, abs=0.001)

    def test_fit_concentration_and_pka(self):
        """Test simultaneous fitting of analyte concentration and pKa."""
        true_acid = ChemicalSystem(
            name="Ácido monoprótico fraco",
            charge_base=-1,
            pKas=[3.80],
            concentration=0.050,
        )
        naoh = ChemicalSystem(name="NaOH", charge_base=0, pKas=[14.0], concentration=0.10)
        true_config = TitrationConfig(
            sample_systems=[true_acid],
            sample_volume=20.0,
            titrant_systems=[naoh],
            max_volume=20.0,
            step_size=1.0,
        )
        true_res = simulate_titration(true_config)
        v_exp = true_res.volumes
        ph_exp = true_res.phs

        # Guesses: C = 0.065 M, pKa = 4.20
        guess_acid = ChemicalSystem(
            name="Ácido monoprótico fraco",
            charge_base=-1,
            pKas=[4.20],
            concentration=0.065,
        )
        guess_config = TitrationConfig(
            sample_systems=[guess_acid],
            sample_volume=20.0,
            titrant_systems=[naoh],
            max_volume=20.0,
            step_size=1.0,
        )

        fit_res = fit_titration_curve(
            volumes=v_exp,
            ph_measured=ph_exp,
            config=guess_config,
            fit_concentration=True,
            fit_pka_indices=[0],
        )

        assert fit_res.success is True
        assert fit_res.fitted_parameters["concentration"] == pytest.approx(0.050, abs=0.002)
        assert fit_res.fitted_parameters["pKa_1"] == pytest.approx(3.80, abs=0.05)
        assert fit_res.r_squared > 0.998
