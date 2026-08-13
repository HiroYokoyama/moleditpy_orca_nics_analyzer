"""Shared fixtures. Qt runs offscreen; every heavy dependency is optional."""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "sample_outputs")
ROOT = os.path.dirname(HERE)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def sample(name):
    return os.path.join(SAMPLES, name)


@pytest.fixture
def single_out():
    return sample("benzene_nics_single.out")


@pytest.fixture
def plane_out():
    return sample("benzene_nics_plane.out")


@pytest.fixture
def volume_out():
    return sample("benzene_nics_volume.out")


@pytest.fixture
def no_ghosts_out():
    return sample("benzene_no_ghosts.out")


@pytest.fixture
def real_out():
    """A genuine ORCA 5.0.4 benzene NMR output (no ghosts)."""
    return sample("real_benzene_nmr_orca5.out")


@pytest.fixture
def real_grid_out():
    """A genuine ORCA 6.1.1 benzene NMR output with a 9x9x7 ghost probe grid."""
    return sample("benzene-opt-nmr.out")


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Keep the user's real preferences file out of the test run.

    A dialog built without an explicit ``settings_file`` falls back to the one
    inside the package, so closing it in a test would rewrite what the user
    last chose in the app — and leak that state into the next test.
    """
    try:
        from orca_nics_analyzer import gui, settings
    except ImportError:  # PyQt6 is optional in the bare-pytest CI tier
        return
    path = str(tmp_path / "test_settings.json")
    monkeypatch.setattr(settings, "SETTINGS_FILE", path)
    monkeypatch.setattr(gui, "SETTINGS_FILE", path)


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session; skips when PyQt6 is absent."""
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


@pytest.fixture
def no_modals(monkeypatch):
    """Neutralize every modal, so a warning path can never block the run."""
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    calls = []
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            name,
            lambda *a, _n=name, **k: calls.append((_n, a[2] if len(a) > 2 else "")),
        )
    return calls


@pytest.fixture
def fake_plotter():
    """Stand-in for the host's PyVista plotter."""
    from unittest.mock import MagicMock

    return MagicMock()


@pytest.fixture
def fake_context(qapp, fake_plotter):
    """Minimal PluginContext: a main window carrying the plotter.

    The window is a real QWidget, not a mock — the plugin passes it to
    QDialog as a parent, which rejects anything that is not a QWidget.

    ``show_xyz_data`` is a MagicMock so tests can assert on calls to it
    without needing the full host application.
    """
    from unittest.mock import MagicMock
    from PyQt6.QtWidgets import QWidget

    main_window = QWidget()
    main_window.plotter = fake_plotter

    context = MagicMock()
    context.get_main_window.return_value = main_window
    context.get_window.return_value = None
    context.show_xyz_data.return_value = None  # realistic: may return a mol or None
    context._main_window = main_window  # keep it alive for the test's duration
    return context
