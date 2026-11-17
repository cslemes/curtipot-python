"""Smoke tests for CurTiPot Streamlit UI modules."""

from __future__ import annotations

import inspect
import py_compile
from pathlib import Path
import pytest


def test_ui_modules_import():
    """Verify all UI tab modules and main app import cleanly."""
    import curtipot.ui as ui
    import curtipot.ui.tab_ph as tab_ph
    import curtipot.ui.tab_distribution as tab_distribution
    import curtipot.ui.tab_simulator as tab_simulator
    import curtipot.ui.tab_analysis_spline as tab_analysis_spline
    import curtipot.ui.tab_analysis_fit as tab_analysis_fit
    import curtipot.ui.tab_database as tab_database
    import curtipot.app as app

    assert hasattr(tab_ph, "render")
    assert hasattr(tab_distribution, "render")
    assert hasattr(tab_simulator, "render")
    assert hasattr(tab_analysis_spline, "render")
    assert hasattr(tab_analysis_fit, "render")
    assert hasattr(tab_database, "render")
    assert hasattr(app, "main")


def test_ui_render_signatures():
    """Verify render functions are callable without mandatory positional arguments."""
    from curtipot.ui import (
        tab_ph,
        tab_distribution,
        tab_simulator,
        tab_analysis_spline,
        tab_analysis_fit,
        tab_database,
    )
    import curtipot.app as app

    modules = [
        tab_ph,
        tab_distribution,
        tab_simulator,
        tab_analysis_spline,
        tab_analysis_fit,
        tab_database,
    ]

    for mod in modules:
        assert callable(mod.render)
        sig = inspect.signature(mod.render)
        for param in sig.parameters.values():
            assert param.default is not inspect.Parameter.empty, (
                f"{mod.__name__}.render should not require mandatory positional arguments"
            )

    assert callable(app.main)


def test_ui_files_syntax_compilation():
    """Verify that all UI Python files compile without syntax or bytecode errors."""
    ui_dir = Path(__file__).parent.parent / "curtipot" / "ui"
    for py_file in ui_dir.glob("*.py"):
        py_compile.compile(str(py_file), doraise=True)

    app_file = Path(__file__).parent.parent / "curtipot" / "app.py"
    py_compile.compile(str(app_file), doraise=True)

    root_entry = Path(__file__).parent.parent / "streamlit_app.py"
    if root_entry.exists():
        py_compile.compile(str(root_entry), doraise=True)


def test_ui_example_generators():
    """Verify built-in dataset generators in analysis tabs return valid titration arrays."""
    from curtipot.ui.tab_analysis_spline import _generate_example_data
    from curtipot.ui.tab_analysis_fit import _generate_example_fit_data

    v_spline, ph_spline = _generate_example_data()
    assert len(v_spline) == len(ph_spline)
    assert len(v_spline) >= 4
    assert (v_spline[1:] >= v_spline[:-1]).all()

    v_fit, ph_fit = _generate_example_fit_data()
    assert len(v_fit) == len(ph_fit)
    assert len(v_fit) >= 3
    assert (v_fit[1:] >= v_fit[:-1]).all()


def test_brazilian_comma_decimal_csv_loading():
    """Verify CSV files with semicolon delimiters and Brazilian comma decimals parse accurately."""
    from curtipot.ui.data_loader import (
        extract_matched_volume_ph,
        load_experimental_dataframe,
    )
    import numpy as np

    raw_csv = b"Volume;pH\n0,0;2,88\n10,5;4,76\n25,0;8,72\n50,0;12,45\n"
    df = load_experimental_dataframe(raw_csv)
    assert len(df.columns) == 2
    assert "Volume" in df.columns
    assert "pH" in df.columns

    v_arr, ph_arr = extract_matched_volume_ph(df, "Volume", "pH")
    assert len(v_arr) == 4
    assert np.isclose(v_arr[1], 10.5)
    assert np.isclose(ph_arr[1], 4.76)
    assert np.isclose(v_arr[2], 25.0)
    assert np.isclose(ph_arr[2], 8.72)

