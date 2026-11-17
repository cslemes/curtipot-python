"""Tests for species distribution and buffer capacity calculations."""

import math
import numpy as np
import pytest

from curtipot.models import ChemicalSystem, DistributionResult
from curtipot.database import get_system_by_name
from curtipot.distribution import calculate_distribution, find_isoelectric_point


class TestDistribution:
    def test_alpha_sum_equals_one(self):
        """For any chemical system and pH, sum of alpha fractions must equal 1.0 within 1e-12."""
        ph_range = np.linspace(0.0, 14.0, 141)

        # Monoprotic: Acetic acid
        acetic = ChemicalSystem(name="Ácido acético", charge_base=-1, pKas=[4.757], concentration=0.1)
        res_ac = calculate_distribution(acetic, ph_values=ph_range)
        sum_ac = np.sum(res_ac.alphas, axis=0)
        np.testing.assert_allclose(sum_ac, 1.0, atol=1e-12)

        # Polyprotic: Phosphoric acid (triprotic)
        phosphoric = ChemicalSystem(
            name="Ácido fosfórico",
            charge_base=-3,
            pKas=[2.148, 7.199, 12.35],
            concentration=0.1,
        )
        res_phos = calculate_distribution(phosphoric, ph_values=ph_range)
        sum_phos = np.sum(res_phos.alphas, axis=0)
        np.testing.assert_allclose(sum_phos, 1.0, atol=1e-12)

        # Tetraprotic: EDTA
        edta = ChemicalSystem(
            name="EDTA",
            charge_base=-4,
            pKas=[1.99, 2.67, 6.16, 10.26],
            concentration=0.05,
        )
        res_edta = calculate_distribution(edta, ph_values=ph_range)
        sum_edta = np.sum(res_edta.alphas, axis=0)
        np.testing.assert_allclose(sum_edta, 1.0, atol=1e-12)

    def test_acetic_acid_half_equivalence_and_buffer_peak(self):
        """At pH = pKa (4.757), alpha_0 == alpha_1 == 0.5 and buffer capacity has a local peak."""
        pka = 4.757
        c_t = 0.1
        acetic = ChemicalSystem(name="Ácido acético", charge_base=-1, pKas=[pka], concentration=c_t)

        # Single exact point at pH = pKa
        res_exact = calculate_distribution(acetic, ph_values=np.array([pka]))
        assert res_exact.alphas.shape == (2, 1)
        # alpha_0 (Ac-) and alpha_1 (HAc)
        assert math.isclose(res_exact.alphas[0, 0], 0.5, abs_tol=1e-10)
        assert math.isclose(res_exact.alphas[1, 0], 0.5, abs_tol=1e-10)
        assert math.isclose(res_exact.n_bar[0], 0.5, abs_tol=1e-10)
        assert math.isclose(res_exact.average_charge[0], -0.5, abs_tol=1e-10)

        # Buffer capacity profile around pKa
        phs = np.linspace(3.0, 6.5, 36)
        res = calculate_distribution(acetic, ph_values=phs)

        # Peak of buffer capacity should be near pKa = 4.757
        idx_pka = np.argmin(np.abs(phs - pka))
        beta_at_pka = res.buffer_capacity[idx_pka]

        # Points 0.7 pH units away must have lower buffer capacity
        idx_low = np.argmin(np.abs(phs - (pka - 0.7)))
        idx_high = np.argmin(np.abs(phs - (pka + 0.7)))
        assert beta_at_pka > res.buffer_capacity[idx_low]
        assert beta_at_pka > res.buffer_capacity[idx_high]

        # Theoretical buffer capacity contribution of weak acid at pH = pKa:
        # beta_acid = ln(10) * C_t * 0.25 = 2.302585 * 0.1 * 0.25 ≈ 0.05756
        expected_beta = math.log(10.0) * (
            10.0 ** (-pka) + 1.0069e-14 / (10.0 ** (-pka)) + c_t * 0.25
        )
        assert math.isclose(beta_at_pka, expected_beta, rel_tol=1e-2)

    def test_phosphoric_acid_species_dominance(self):
        """Phosphoric acid: check dominance of H3PO4, H2PO4-, HPO4^2-, PO4^3- in respective pH ranges."""
        # pKa1 = 2.148, pKa2 = 7.199, pKa3 = 12.35
        # Species:
        # index 0: PO4^3-  (charge -3)
        # index 1: HPO4^2- (charge -2)
        # index 2: H2PO4^- (charge -1)
        # index 3: H3PO4   (charge 0)
        phosphoric = ChemicalSystem(
            name="Ácido fosfórico",
            charge_base=-3,
            pKas=[2.148, 7.199, 12.35],
            concentration=0.1,
        )

        test_phs = np.array([1.0, 4.5, 9.5, 13.5])
        res = calculate_distribution(phosphoric, ph_values=test_phs)

        # pH 1.0: H3PO4 dominates (index 3)
        assert res.alphas[3, 0] > 0.90
        assert res.alphas[2, 0] < 0.10

        # pH 4.5: H2PO4^- dominates (index 2)
        assert res.alphas[2, 1] > 0.99
        assert res.alphas[3, 1] < 0.01
        assert res.alphas[1, 1] < 0.01

        # pH 9.5: HPO4^2- dominates (index 1)
        assert res.alphas[1, 2] > 0.99
        assert res.alphas[2, 2] < 0.01
        assert res.alphas[0, 2] < 0.01

        # pH 13.5: PO4^3- dominates (index 0)
        assert res.alphas[0, 3] > 0.90
        assert res.alphas[1, 3] < 0.10

    def test_n_bar_and_average_charge_monotonicity(self):
        """Average protonation n_bar and average charge z(pH) must be monotonically decreasing with pH."""
        ph_range = np.linspace(0.0, 14.0, 281)

        # Test on phosphoric acid
        phosphoric = ChemicalSystem(
            name="Ácido fosfórico",
            charge_base=-3,
            pKas=[2.148, 7.199, 12.35],
            concentration=0.1,
        )
        res = calculate_distribution(phosphoric, ph_values=ph_range)

        # Monotonically non-increasing (decreasing with pH)
        diff_n_bar = np.diff(res.n_bar)
        diff_charge = np.diff(res.average_charge)

        assert np.all(diff_n_bar <= 1e-14), "n_bar must decrease as pH increases"
        assert np.all(diff_charge <= 1e-14), "average_charge must decrease as pH increases"

        # Check bounds: at pH 0, n_bar ~ 3, charge ~ 0; at pH 14, n_bar ~ 0, charge ~ -3
        assert math.isclose(res.n_bar[0], 3.0, abs_tol=0.05)
        assert math.isclose(res.average_charge[0], 0.0, abs_tol=0.05)
        assert math.isclose(res.n_bar[-1], 0.0, abs_tol=0.05)
        assert math.isclose(res.average_charge[-1], -3.0, abs_tol=0.05)

    def test_default_ph_values(self):
        """When ph_values is None, calculate_distribution should provide a standard default array."""
        acetic = ChemicalSystem(name="Ácido acético", charge_base=-1, pKas=[4.757], concentration=0.1)
        res = calculate_distribution(acetic)
        assert res.ph_array is not None
        assert len(res.ph_array) > 50
        assert res.ph_array[0] == 0.0
        assert res.ph_array[-1] == 14.0
        assert res.alphas.shape[1] == len(res.ph_array)

    def test_gamma_h_correction(self):
        """Gamma_H shifts effective [H+] = (10^-pH)/gamma_h."""
        acetic = ChemicalSystem(name="Ácido acético", charge_base=-1, pKas=[4.757], concentration=0.1)
        gamma = 0.8
        # H = (10^-pH) / gamma_h. For H = 10^-pKa, 10^-pH = gamma_h * 10^-pKa => pH = pKa - log10(gamma_h)
        ph_shift = 4.757 - math.log10(gamma)
        res = calculate_distribution(acetic, ph_values=np.array([ph_shift]), gamma_h=gamma)
        assert math.isclose(res.alphas[0, 0], 0.5, abs_tol=1e-10)
        assert math.isclose(res.alphas[1, 0], 0.5, abs_tol=1e-10)

    def test_inert_ion_zero_protons(self):
        """System with zero protons (e.g. Na+) should have alpha_0 = 1.0, n_bar = 0."""
        sodium = ChemicalSystem(name="Sódio", charge_base=1, pKas=[], concentration=0.1)
        res = calculate_distribution(sodium, ph_values=np.array([7.0]))
        assert res.alphas.shape == (1, 1)
        assert res.alphas[0, 0] == 1.0
        assert res.n_bar[0] == 0.0
        assert res.average_charge[0] == 1.0

    def test_log_alphas_values(self):
        """Check log_alphas correctly computes log10(max(alpha, 1e-20))."""
        acetic = ChemicalSystem(name="Ácido acético", charge_base=-1, pKas=[4.757], concentration=0.1)
        res = calculate_distribution(acetic, ph_values=np.array([4.757]))
        assert math.isclose(res.log_alphas[0, 0], math.log10(0.5), abs_tol=1e-10)
        assert math.isclose(res.log_alphas[1, 0], math.log10(0.5), abs_tol=1e-10)

    def test_isoelectric_point_glycine(self):
        """Glycine has z_B = -1, pKa1 = 2.34, pKa2 = 9.60. Isoelectric point pI = (2.34 + 9.60) / 2 = 5.97."""
        glycine = ChemicalSystem(
            name="Glicina",
            charge_base=-1,
            pKas=[2.34, 9.60],
            concentration=0.01,
        )
        res = calculate_distribution(glycine, ph_values=np.linspace(0.0, 14.0, 1401))
        pi = find_isoelectric_point(res)
        assert pi is not None
        assert math.isclose(pi, 5.97, abs_tol=0.02)
