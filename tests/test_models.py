import pytest
import numpy as np
import math
from curtipot.models import (
    ChemicalSystem,
    EquilibriumResult,
    DistributionResult,
    TitrationPoint,
    TitrationConfig,
    TitrationResult,
)


class TestChemicalSystem:
    def test_monoprotic_acid(self):
        """Test acetic acid (monoprotic weak acid)."""
        system = ChemicalSystem(
            name="Ácido acético",
            charge_base=-1,
            pKas=[4.756],
            concentration=0.1,
        )
        assert system.name == "Ácido acético"
        assert system.charge_base == -1
        assert system.pKas == [4.756]
        assert system.pkas == [4.756]
        assert system.concentration == 0.1
        assert system.num_protons == 1

        # Betas: beta_0 = 1.0, beta_1 = 10^4.756
        assert len(system.betas) == 2
        assert system.betas[0] == 1.0
        assert math.isclose(system.betas[1], 10**4.756, rel_tol=1e-9)

        # Log betas
        assert len(system.log_betas) == 2
        assert system.log_betas[0] == 0.0
        assert math.isclose(system.log_betas[1], 4.756, rel_tol=1e-9)

        # Species charges: [charge_base, charge_base + 1] -> [-1, 0]
        assert system.species_charges == [-1, 0]

    def test_triprotic_acid_phosphoric(self):
        """Test phosphoric acid (triprotic, pKas=[2.148, 7.199, 12.35], charge_base=-3)."""
        system = ChemicalSystem(
            name="Ácido fosfórico",
            charge_base=-3,
            pKas=[2.148, 7.199, 12.35],
            concentration=0.05,
            formula="H3PO4",
            molar_mass=97.994,
        )
        assert system.num_protons == 3
        assert system.formula == "H3PO4"
        assert system.molar_mass == 97.994

        # Betas verification:
        # beta_0 = 1.0 (PO4^3-)
        # beta_1 = 10^(12.35) (HPO4^2-)
        # beta_2 = 10^(12.35 + 7.199) = 10^(19.549) (H2PO4^-)
        # beta_3 = 10^(12.35 + 7.199 + 2.148) = 10^(21.697) (H3PO4)
        betas = system.betas
        assert len(betas) == 4
        assert betas[0] == 1.0
        assert math.isclose(betas[1], 10**12.35, rel_tol=1e-9)
        assert math.isclose(betas[2], 10**(12.35 + 7.199), rel_tol=1e-9)
        assert math.isclose(betas[3], 10**(12.35 + 7.199 + 2.148), rel_tol=1e-9)

        # Log betas
        log_betas = system.log_betas
        assert len(log_betas) == 4
        assert log_betas[0] == 0.0
        assert math.isclose(log_betas[1], 12.35, rel_tol=1e-9)
        assert math.isclose(log_betas[2], 19.549, rel_tol=1e-9)
        assert math.isclose(log_betas[3], 21.697, rel_tol=1e-9)

        # Species charges
        assert system.species_charges == [-3, -2, -1, 0]

    def test_cationic_system_ammonia(self):
        """Test ammonia (base is neutral NH3, protonated form is NH4+)."""
        system = ChemicalSystem(
            name="Amônia",
            charge_base=0,
            pKas=[9.25],
            concentration=0.02,
        )
        assert system.num_protons == 1
        assert system.species_charges == [0, 1]
        assert math.isclose(system.betas[1], 10**9.25, rel_tol=1e-9)

    def test_strong_electrolyte_no_pka(self):
        """Test strong acid or salt counterion with no pKa."""
        system = ChemicalSystem(
            name="Sódio",
            charge_base=1,
            pKas=[],
            concentration=0.1,
        )
        assert system.num_protons == 0
        assert system.betas == [1.0]
        assert system.log_betas == [0.0]
        assert system.species_charges == [1]

    def test_case_insensitive_pka_alias(self):
        """Test that ChemicalSystem accepts either pKas or pkas keyword."""
        s1 = ChemicalSystem(name="Teste", charge_base=-1, pkas=[4.0])
        assert s1.pKas == [4.0]
        assert s1.pkas == [4.0]

    def test_with_concentration_and_species_names(self):
        """Test with_concentration cloning and species_names labels."""
        system = ChemicalSystem(
            name="Ácido carbônico",
            charge_base=-2,
            pKas=[6.352, 10.329],
            concentration=0.1,
        )
        assert system.species_names == ["[B](2-)", "[HB](-)", "[H2B](0)"]

        clone = system.with_concentration(0.02)
        assert clone.concentration == 0.02
        assert clone.name == system.name
        assert clone.charge_base == system.charge_base
        assert clone.pKas == system.pKas

    def test_validations(self):
        """Test validation rules for ChemicalSystem."""
        # Empty name
        with pytest.raises(ValueError, match="name"):
            ChemicalSystem(name="", charge_base=-1, pKas=[4.0])

        # Negative concentration
        with pytest.raises(ValueError, match="concentration"):
            ChemicalSystem(name="A", charge_base=-1, pKas=[4.0], concentration=-0.1)

        # Non-integer charge_base
        with pytest.raises(TypeError, match="charge_base"):
            ChemicalSystem(name="A", charge_base=-1.5, pKas=[4.0])  # type: ignore

        # Inverted pKas (must be in ascending order: pKa1 <= pKa2 <= ...)
        with pytest.raises(ValueError, match="ascending"):
            ChemicalSystem(name="A", charge_base=-2, pKas=[7.0, 3.0])


class TestEquilibriumResult:
    def test_equilibrium_result_creation(self):
        res = EquilibriumResult(
            ph=4.75,
            poh=9.25,
            h_conc=1.778e-5,
            oh_conc=5.623e-10,
            ionic_strength=0.01,
            activity_coeff_h=0.90,
            buffer_capacity=0.025,
            species_concentrations={"Ácido acético": [0.05, 0.05]},
            converged=True,
            iterations=4,
        )
        assert math.isclose(res.ph, 4.75)
        assert math.isclose(res.poh, 9.25)
        assert res.converged is True
        assert res.iterations == 4


class TestDistributionResult:
    def test_distribution_result_creation(self):
        ph_arr = np.linspace(0, 14, 15)
        alphas = np.zeros((2, 15))
        alphas[0, :] = 0.5
        alphas[1, :] = 0.5
        log_alphas = np.log10(alphas)
        n_bar = np.ones(15) * 0.5
        avg_charge = np.ones(15) * (-0.5)
        buff = np.ones(15) * 0.01

        res = DistributionResult(
            ph_array=ph_arr,
            alphas=alphas,
            log_alphas=log_alphas,
            n_bar=n_bar,
            average_charge=avg_charge,
            buffer_capacity=buff,
            species_labels=["A-", "HA"],
        )
        assert len(res.ph_array) == 15
        assert res.alphas.shape == (2, 15)
        assert res.species_labels == ["A-", "HA"]


class TestTitrationModels:
    def test_titration_point(self):
        pt = TitrationPoint(
            volume_added=10.0,
            ph=4.5,
            ionic_strength=0.05,
            buffer_capacity=0.03,
            dpH_dV=0.15,
            d2pH_dV2=0.02,
        )
        assert pt.volume_added == 10.0
        assert pt.ph == 4.5
        assert pt.dpH_dV == 0.15

    def test_titration_config_and_result(self):
        sample = ChemicalSystem(name="HCl", charge_base=-1, pKas=[], concentration=0.1)
        titrant = ChemicalSystem(name="NaOH", charge_base=1, pKas=[], concentration=0.1)

        cfg = TitrationConfig(
            sample_systems=[sample],
            sample_volume=25.0,
            titrant_systems=[titrant],
            water_volume=25.0,
            max_volume=50.0,
            step_size=0.1,
            calc_ionic_strength=True,
        )
        assert cfg.sample_volume == 25.0
        assert cfg.water_volume == 25.0
        assert cfg.max_volume == 50.0
        assert cfg.step_size == 0.1
        assert cfg.calc_ionic_strength is True

        pt1 = TitrationPoint(volume_added=0.0, ph=1.3, ionic_strength=0.05)
        pt2 = TitrationPoint(volume_added=25.0, ph=7.0, ionic_strength=0.05)

        result = TitrationResult(config=cfg, points=[pt1, pt2])
        assert len(result.points) == 2
        assert np.array_equal(result.volumes, np.array([0.0, 25.0]))
        assert np.array_equal(result.phs, np.array([1.3, 7.0]))

    def test_titration_config_validation(self):
        sample = ChemicalSystem(name="HCl", charge_base=-1, pKas=[], concentration=0.1)
        titrant = ChemicalSystem(name="NaOH", charge_base=1, pKas=[], concentration=0.1)

        # sample_volume <= 0
        with pytest.raises(ValueError, match="sample_volume"):
            TitrationConfig(
                sample_systems=[sample],
                sample_volume=0.0,
                titrant_systems=[titrant],
            )

        # max_volume <= 0
        with pytest.raises(ValueError, match="max_volume"):
            TitrationConfig(
                sample_systems=[sample],
                sample_volume=25.0,
                titrant_systems=[titrant],
                max_volume=-10.0,
            )

        # step_size <= 0
        with pytest.raises(ValueError, match="step_size"):
            TitrationConfig(
                sample_systems=[sample],
                sample_volume=25.0,
                titrant_systems=[titrant],
                step_size=0.0,
            )

        # empty sample_systems
        with pytest.raises(ValueError, match="sample_systems"):
            TitrationConfig(
                sample_systems=[],
                sample_volume=25.0,
                titrant_systems=[titrant],
            )
