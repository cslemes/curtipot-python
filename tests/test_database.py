import math
import pytest
import pandas as pd
from curtipot.models import ChemicalSystem
from curtipot.database import (
    load_database,
    get_system_by_name,
    search_systems,
    get_database_dataframe,
    list_substance_names,
)


class TestDatabase:
    def test_load_database_count(self):
        """Database should contain all substances (>= 250 entries)."""
        systems = load_database()
        assert len(systems) >= 250
        assert all(isinstance(s, ChemicalSystem) for s in systems)

    def test_get_system_by_name_acetic_acid(self):
        """Test retrieving acetic acid by name with charge -1 and pKa 4.757."""
        system = get_system_by_name("Ácido acético")
        assert isinstance(system, ChemicalSystem)
        assert system.name == "Ácido acético"
        assert system.charge_base == -1
        assert len(system.pKas) == 1
        assert math.isclose(system.pKas[0], 4.757, abs_tol=1e-3)
        assert "CH3COOH" in system.formula

    def test_get_system_by_name_phosphoric_acid(self):
        """Test retrieving phosphoric acid with charge -3 and 3 pKas."""
        system = get_system_by_name("Ácido fosfórico")
        assert isinstance(system, ChemicalSystem)
        assert system.name == "Ácido fosfórico"
        assert system.charge_base == -3
        assert len(system.pKas) == 3
        assert math.isclose(system.pKas[0], 2.148, abs_tol=1e-3)
        assert math.isclose(system.pKas[1], 7.199, abs_tol=1e-3)
        assert math.isclose(system.pKas[2], 12.35, abs_tol=1e-2)
        assert system.formula == "H3PO4"

    def test_get_system_by_name_normalization(self):
        """Lookup should be case-insensitive, accent-insensitive, and strip whitespace."""
        sys1 = get_system_by_name("Ácido acético")
        sys2 = get_system_by_name("acido acetico")
        sys3 = get_system_by_name("  ÁCIDO ACÉTICO  ")
        assert sys1.name == sys2.name == sys3.name
        assert sys1.pKas == sys2.pKas == sys3.pKas

    def test_get_system_by_name_with_concentration(self):
        """Optional concentration parameter should set the system concentration."""
        sys = get_system_by_name("Ácido acético", concentration=0.05)
        assert sys.concentration == 0.05

    def test_get_system_not_found(self):
        """Non-existent substance should raise KeyError with informative message."""
        with pytest.raises(KeyError, match="Substância 'Inexistente' não encontrada"):
            get_system_by_name("Inexistente")

    def test_search_systems(self):
        """Search should filter by query, case and accent insensitive."""
        results = search_systems("ácido")
        assert len(results) > 20
        results_no_accent = search_systems("acido")
        assert len(results) == len(results_no_accent)

        # Search by formula
        results_formula = search_systems("H3PO4")
        assert any(r.name == "Ácido fosfórico" for r in results_formula)

        # Search with no match
        assert search_systems("xyz123nonexistent") == []

    def test_get_database_dataframe(self):
        """Should return a pandas DataFrame with core columns and rows >= 250."""
        df = get_database_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 250
        required_cols = ["name", "charge_base", "pKa1", "formula"]
        for col in required_cols:
            assert col in df.columns

        # Verify acetic acid row
        acetic = df[df["name"] == "Ácido acético"].iloc[0]
        assert acetic["charge_base"] == -1
        assert math.isclose(float(acetic["pKa1"]), 4.757, abs_tol=1e-3)

    def test_database_immutability(self):
        """Modifying a retrieved system should not contaminate future calls."""
        sys1 = get_system_by_name("Ácido acético", concentration=0.1)
        assert sys1.concentration == 0.1

        sys2 = get_system_by_name("Ácido acético")
        assert sys2.concentration == 0.0

    def test_list_substance_names(self):
        """list_substance_names returns all names as strings."""
        names = list_substance_names()
        assert len(names) >= 250
        assert "Ácido acético" in names
        assert "Ácido fosfórico" in names
