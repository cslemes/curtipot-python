"""CurTiPot - pH, Titration, Distribution and Analysis suite in Python."""

from curtipot.models import (
    ChemicalSystem,
    EquilibriumResult,
    DistributionResult,
    TitrationPoint,
    TitrationConfig,
    TitrationResult,
)
from curtipot.database import (
    load_database,
    get_system_by_name,
    search_systems,
    get_database_dataframe,
    list_substance_names,
)
from curtipot.equilibrium import (
    solve_equilibrium,
    solve_ph,
    calculate_ionic_strength,
    davies_activity_coefficient,
    proton_balance,
)
from curtipot.distribution import (
    calculate_distribution,
    calculate_alpha_species,
    calculate_buffer_capacity,
    find_isoelectric_point,
)
from curtipot.titration import (
    simulate_titration,
    simulate_volume_steps,
    simulate_ph_steps,
)
from curtipot.analysis import (
    EquivalencePoint,
    SplineAnalysisResult,
    FitResult,
    analyze_spline,
    analyze_spline_derivatives,
    fit_titration_curve,
)

__version__ = "0.1.0"

__all__ = [
    "ChemicalSystem",
    "EquilibriumResult",
    "DistributionResult",
    "TitrationPoint",
    "TitrationConfig",
    "TitrationResult",
    "EquivalencePoint",
    "SplineAnalysisResult",
    "FitResult",
    "load_database",
    "get_system_by_name",
    "search_systems",
    "get_database_dataframe",
    "list_substance_names",
    "solve_equilibrium",
    "solve_ph",
    "calculate_ionic_strength",
    "davies_activity_coefficient",
    "proton_balance",
    "calculate_distribution",
    "calculate_alpha_species",
    "calculate_buffer_capacity",
    "find_isoelectric_point",
    "simulate_titration",
    "simulate_volume_steps",
    "simulate_ph_steps",
    "analyze_spline",
    "analyze_spline_derivatives",
    "fit_titration_curve",
]
