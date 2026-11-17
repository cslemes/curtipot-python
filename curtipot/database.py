"""CurTiPot chemical constants database loader and lookup utilities."""

from __future__ import annotations

import json
from pathlib import Path
import unicodedata
from typing import Any
import pandas as pd

from curtipot.models import ChemicalSystem

_DATA_PATH = Path(__file__).parent / "data" / "constants.json"
_DATABASE_CACHE: list[dict[str, Any]] | None = None


def normalize_text(text: str) -> str:
    """Normalize text: NFKD decomposition, strip accents, lowercase, strip whitespace.

    Examples
    --------
    >>> normalize_text("Ácido acético")
    'acido acetico'
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_chars = [c for c in nfkd if not unicodedata.combining(c)]
    return "".join(ascii_chars).lower().strip()


def _load_raw_database(reload: bool = False) -> list[dict[str, Any]]:
    """Load raw records from constants.json with caching."""
    global _DATABASE_CACHE
    if _DATABASE_CACHE is None or reload:
        if not _DATA_PATH.exists():
            raise FileNotFoundError(
                f"Constants database file not found at {_DATA_PATH}. "
                "Run scripts/extract_constants.py to generate it."
            )
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _DATABASE_CACHE = json.load(f)
    return _DATABASE_CACHE


def _record_to_system(record: dict[str, Any], concentration: float = 0.0) -> ChemicalSystem:
    """Convert raw dictionary record to a new ChemicalSystem instance."""
    return ChemicalSystem(
        name=record["name"],
        charge_base=record["charge_base"],
        pKas=record.get("pkas", []),
        concentration=concentration,
        formula=record.get("formula", ""),
        molar_mass=record.get("molar_mass"),
        temp_c=record.get("temp_c", 25.0),
        ionic_strength=record.get("ionic_strength", 0.0),
    )


def load_database(reload: bool = False) -> list[ChemicalSystem]:
    """Load all chemical systems from the database.

    Parameters
    ----------
    reload : bool, default False
        Whether to force reloading the constants JSON from disk.

    Returns
    -------
    list[ChemicalSystem]
        List of chemical systems with default concentration 0.0.
    """
    records = _load_raw_database(reload=reload)
    return [_record_to_system(r) for r in records]


def get_system_by_name(name: str, concentration: float = 0.0) -> ChemicalSystem:
    """Look up a chemical system by its substance name.

    The search is case-insensitive, accent-insensitive, and ignores leading/trailing whitespace.

    Parameters
    ----------
    name : str
        Substance name (e.g. 'Ácido acético', 'acido fosforico', 'EDTA').
    concentration : float, default 0.0
        Initial concentration in mol/L.

    Returns
    -------
    ChemicalSystem
        New ChemicalSystem instance.

    Raises
    ------
    KeyError
        If no matching substance is found.
    """
    query_norm = normalize_text(name)
    if not query_norm:
        raise KeyError("Nome da substância não pode ser vazio.")

    records = _load_raw_database()

    # First pass: try exact normalized match
    for r in records:
        if normalize_text(r["name"]) == query_norm:
            return _record_to_system(r, concentration=concentration)

    # Second pass: check formula match
    for r in records:
        if r.get("formula") and normalize_text(r["formula"]) == query_norm:
            return _record_to_system(r, concentration=concentration)

    raise KeyError(f"Substância '{name}' não encontrada no banco de dados CurTiPot.")


def search_systems(query: str) -> list[ChemicalSystem]:
    """Search substances whose name or formula contains the query string.

    Search is case-insensitive and accent-insensitive. Results are deduplicated by name.

    Parameters
    ----------
    query : str
        Search term.

    Returns
    -------
    list[ChemicalSystem]
        List of matching chemical systems.
    """
    query_norm = normalize_text(query)
    if not query_norm:
        return []

    records = _load_raw_database()
    results: list[ChemicalSystem] = []
    seen_names: set[str] = set()

    for r in records:
        name_norm = normalize_text(r["name"])
        formula_norm = normalize_text(r.get("formula", ""))

        if query_norm in name_norm or (formula_norm and query_norm in formula_norm):
            if name_norm not in seen_names:
                seen_names.add(name_norm)
                results.append(_record_to_system(r))

    return results


def get_database_dataframe() -> pd.DataFrame:
    """Return the entire CurTiPot database as a pandas DataFrame.

    Columns include: index, name, category, charge_base, pKa1 to pKa6,
    temp_c, ionic_strength, formula, molar_mass.

    Returns
    -------
    pd.DataFrame
    """
    records = _load_raw_database()
    rows = []
    for r in records:
        pkas = r.get("pkas", [])
        row = {
            "index": r["index"],
            "name": r["name"],
            "category": r.get("category", ""),
            "charge_base": r["charge_base"],
            "pKa1": pkas[0] if len(pkas) > 0 else None,
            "pKa2": pkas[1] if len(pkas) > 1 else None,
            "pKa3": pkas[2] if len(pkas) > 2 else None,
            "pKa4": pkas[3] if len(pkas) > 3 else None,
            "pKa5": pkas[4] if len(pkas) > 4 else None,
            "pKa6": pkas[5] if len(pkas) > 5 else None,
            "temp_c": r.get("temp_c", 25.0),
            "ionic_strength": r.get("ionic_strength", 0.0),
            "formula": r.get("formula", ""),
            "molar_mass": r.get("molar_mass"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def list_substance_names() -> list[str]:
    """Return a sorted list of unique substance names in the database.

    Returns
    -------
    list[str]
    """
    records = _load_raw_database()
    seen = set()
    names = []
    for r in records:
        name = r["name"].strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names
