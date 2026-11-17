"""Data models for CurTiPot physical chemistry calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ChemicalSystem:
    """Represents an acid-base or electrolyte chemical system.

    Parameters
    ----------
    name : str
        Substance name (e.g. 'Ácido acético', 'Ácido fosfórico').
    charge_base : int
        Electrical charge of the completely deprotonated base species (z_B).
        Example: -1 for acetate (CH3COO-), -3 for phosphate (PO4^3-), 0 for ammonia (NH3).
    pKas : list[float]
        Stepwise acid dissociation constants in ascending order [pKa1, pKa2, ...].
        Empty list for strong electrolytes/inert ions.
    concentration : float
        Total analytical concentration in mol/L (M).
    formula : str
        Chemical formula (e.g. 'H3PO4').
    molar_mass : float | None
        Molar mass in g/mol.
    temp_c : float
        Reference temperature in °C (default: 25.0).
    ionic_strength : float
        Reference ionic strength for constants (default: 0.0, thermodynamic constants).
    """

    name: str
    charge_base: int
    pKas: list[float] = field(default_factory=list)
    concentration: float = 0.0
    formula: str = ""
    molar_mass: float | None = None
    temp_c: float = 25.0
    ionic_strength: float = 0.0

    def __init__(
        self,
        name: str,
        charge_base: int,
        pKas: list[float] | None = None,
        concentration: float = 0.0,
        formula: str = "",
        molar_mass: float | None = None,
        temp_c: float = 25.0,
        ionic_strength: float = 0.0,
        pkas: list[float] | None = None,
    ):
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(charge_base, int) or isinstance(charge_base, bool):
            raise TypeError(f"charge_base must be an integer, got {type(charge_base).__name__}")
        if concentration < 0.0:
            raise ValueError(f"concentration must be non-negative, got {concentration}")

        self.name = name.strip()
        self.charge_base = int(charge_base)

        chosen_pkas = pkas if pkas is not None else (pKas if pKas is not None else [])
        self.pKas = [float(k) for k in chosen_pkas]

        # Validate ascending order
        for i in range(len(self.pKas) - 1):
            if self.pKas[i + 1] < self.pKas[i]:
                raise ValueError(
                    f"pKas must be in ascending order (pKa(i+1) >= pKa(i)), got {self.pKas}"
                )

        self.concentration = float(concentration)
        self.formula = formula
        self.molar_mass = float(molar_mass) if molar_mass is not None else None
        self.temp_c = float(temp_c)
        self.ionic_strength = float(ionic_strength)

    @property
    def pkas(self) -> list[float]:
        """Alias for pKas for case-insensitivity."""
        return self.pKas

    @property
    def num_protons(self) -> int:
        """Total number of protonation steps (Np = len(pKas))."""
        return len(self.pKas)

    @property
    def log_betas(self) -> list[float]:
        """Cumulative protonation constant logarithms [log(beta_0), log(beta_1), ...].

        beta_0 is 1.0 (log(beta_0) = 0.0) for the fully deprotonated base B.
        beta_i = 10^(sum_{j=1}^i pKa_{Np - j + 1}).
        """
        n = self.num_protons
        result = [0.0]
        accum = 0.0
        for j in range(1, n + 1):
            accum += self.pKas[n - j]
            result.append(accum)
        return result

    @property
    def betas(self) -> list[float]:
        """Cumulative protonation constants [beta_0, beta_1, ..., beta_Np].

        beta_0 = 1.0
        beta_1 = 10^(pKa_Np)
        beta_2 = 10^(pKa_Np + pKa_{Np-1})
        ...
        beta_Np = 10^(sum_{k=1}^{Np} pKa_k)
        """
        return [10.0**lb for lb in self.log_betas]

    @property
    def species_charges(self) -> list[int]:
        """List of integer charges for all species from base (0 protons) to acid (Np protons)."""
        return [self.charge_base + i for i in range(self.num_protons + 1)]

    @property
    def species_names(self) -> list[str]:
        """Default descriptive labels for each species."""
        n = self.num_protons
        charges = self.species_charges
        labels = []
        for i in range(n + 1):
            ch = charges[i]
            ch_str = f"{abs(ch)}-" if ch < -1 else ("-" if ch == -1 else (f"{ch}+" if ch > 1 else ("+" if ch == 1 else "0")))
            if i == 0:
                h_str = ""
            elif i == 1:
                h_str = "H"
            else:
                h_str = f"H{i}"
            labels.append(f"[{h_str}B]({ch_str})")
        return labels

    def with_concentration(self, new_concentration: float) -> ChemicalSystem:
        """Return a copy of this chemical system with a new analytical concentration."""
        return ChemicalSystem(
            name=self.name,
            charge_base=self.charge_base,
            pKas=list(self.pKas),
            concentration=new_concentration,
            formula=self.formula,
            molar_mass=self.molar_mass,
            temp_c=self.temp_c,
            ionic_strength=self.ionic_strength,
        )


@dataclass
class EquilibriumResult:
    """Result of an equilibrium pH calculation."""

    ph: float
    poh: float
    h_conc: float
    oh_conc: float
    ionic_strength: float = 0.0
    activity_coeff_h: float = 1.0
    buffer_capacity: float = 0.0
    species_concentrations: dict[str, list[float]] = field(default_factory=dict)
    converged: bool = True
    iterations: int = 0


@dataclass
class DistributionResult:
    """Result of species distribution calculation across a pH range."""

    ph_array: np.ndarray
    alphas: np.ndarray
    log_alphas: np.ndarray
    n_bar: np.ndarray
    average_charge: np.ndarray
    buffer_capacity: np.ndarray
    species_labels: list[str] = field(default_factory=list)


@dataclass
class TitrationPoint:
    """Single point along a simulated or experimental titration curve."""

    volume_added: float  # mL
    ph: float
    ionic_strength: float = 0.0
    buffer_capacity: float = 0.0
    species_concentrations: dict[str, list[float]] = field(default_factory=dict)
    dpH_dV: float | None = None
    d2pH_dV2: float | None = None


@dataclass
class TitrationConfig:
    """Configuration parameters for a titration simulation."""

    sample_systems: list[ChemicalSystem]
    sample_volume: float
    titrant_systems: list[ChemicalSystem]
    water_volume: float = 0.0
    max_volume: float = 50.0
    step_size: float = 0.1
    calc_ionic_strength: bool = True
    temp_c: float = 25.0
    mode: str = "volume"
    sample_inert_salts: list[tuple[float, int]] = field(default_factory=list)
    titrant_inert_salts: list[tuple[float, int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.sample_systems:
            raise ValueError("sample_systems cannot be empty")
        if self.sample_volume <= 0:
            raise ValueError(f"sample_volume must be positive, got {self.sample_volume}")
        if self.water_volume < 0:
            raise ValueError(f"water_volume cannot be negative, got {self.water_volume}")
        if self.max_volume <= 0:
            raise ValueError(f"max_volume must be positive, got {self.max_volume}")
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")


@dataclass
class TitrationResult:
    """Complete result of a titration simulation."""

    config: TitrationConfig
    points: list[TitrationPoint] = field(default_factory=list)
    equivalence_points: list[dict[str, Any]] = field(default_factory=list)

    @property
    def volumes(self) -> np.ndarray:
        return np.array([p.volume_added for p in self.points], dtype=float)

    @property
    def phs(self) -> np.ndarray:
        return np.array([p.ph for p in self.points], dtype=float)

    @property
    def ionic_strengths(self) -> np.ndarray:
        return np.array([p.ionic_strength for p in self.points], dtype=float)

    @property
    def buffer_capacities(self) -> np.ndarray:
        return np.array([p.buffer_capacity for p in self.points], dtype=float)

    @property
    def dpH_dV(self) -> np.ndarray:
        return np.array(
            [p.dpH_dV if p.dpH_dV is not None else np.nan for p in self.points], dtype=float
        )

    @property
    def d2pH_dV2(self) -> np.ndarray:
        return np.array(
            [p.d2pH_dV2 if p.d2pH_dV2 is not None else np.nan for p in self.points], dtype=float
        )
