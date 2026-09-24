"""Regression tests for bugs found in the 0.5.2 code review."""

import json
import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_fixtures as mf

from orca_nics_analyzer import cube_io
from orca_nics_analyzer.analysis import NicsField
from orca_nics_analyzer.parser import NicsParser
from orca_nics_analyzer.settings import DEFAULT_SETTINGS, load_settings


def _field(points, tmp_path, name="probe.out"):
    path = tmp_path / name
    path.write_text(mf.build_output(points, name), encoding="utf-8")
    parser = NicsParser()
    parser.load(str(path))
    return NicsField(parser)


class TestXYScan:
    def test_scan_parallel_to_ring_plane_keeps_its_abscissa(self, tmp_path):
        # A line at constant height: every probe is 1 A above the ring, so the
        # height cannot be the x axis.
        xs = np.linspace(-3.0, 3.0, 13)
        field = _field([(x, 0.0, 1.0) for x in xs], tmp_path)
        data = field.line_data()
        assert field.layout["kind"] == "line"
        assert data["distance"] == pytest.approx(xs - xs.min(), abs=1e-4)
        assert "along the scan" in data["label"]

    def test_scan_along_the_normal_still_uses_height(self, tmp_path):
        zs = np.linspace(0.0, 3.0, 7)
        field = _field([(0.0, 0.0, z) for z in zs], tmp_path)
        data = field.line_data()
        assert "height" in data["label"]
        assert np.abs(data["distance"]) == pytest.approx(zs, abs=1e-4)


class TestGridRegularity:
    XS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)

    def test_uneven_spacing_is_detected(self, tmp_path):
        field = _field([(x, y, 1.0) for x in self.XS for y in self.XS], tmp_path)
        assert field.is_gridded
        assert not field.is_uniform

    def test_uneven_grid_refuses_a_cube(self, tmp_path):
        field = _field([(x, y, 1.0) for x in self.XS for y in self.XS], tmp_path)
        with pytest.raises(ValueError, match="unevenly spaced"):
            field.write_cube("iso")
        assert field.cached_cube("iso") is None

    def test_grid_points_are_the_real_probe_positions(self, tmp_path):
        field = _field([(x, y, 1.0) for x in self.XS for y in self.XS], tmp_path)
        pts = field.grid_points().reshape(-1, 3)
        for probe in field.probe_coords:
            assert np.min(np.linalg.norm(pts - probe, axis=1)) < 1e-4

    def test_even_grid_is_uniform(self, tmp_path):
        xs = np.linspace(-2.0, 2.0, 5)
        field = _field([(x, y, 1.0) for x in xs for y in xs], tmp_path)
        assert field.is_uniform

    def test_two_probes_in_one_cell_are_not_regular(self, tmp_path):
        xs = [-1.0, 0.0, 1.0]
        pts = [(x, y, 1.0) for x in xs for y in xs]
        # Replace one probe with a duplicate: the count still multiplies out.
        pts[-1] = pts[0]
        field = _field(pts, tmp_path)
        assert not field.layout["regular"]
        assert field.layout["kind"] == "scattered"


class TestSettingsValidation:
    def test_wrongly_typed_values_fall_back_to_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "nics_analyzer_settings": {
                        "map_levels": "abc",
                        "map_range": float("nan"),
                        "show_probes": "yes",
                        "map_colormap": 3,
                        "icss_opacity": 0.3,
                    }
                }
            ),
            encoding="utf-8",
        )
        settings = load_settings(str(path))
        assert settings["map_levels"] == DEFAULT_SETTINGS["map_levels"]
        assert settings["map_range"] == DEFAULT_SETTINGS["map_range"]
        assert settings["show_probes"] == DEFAULT_SETTINGS["show_probes"]
        assert settings["map_colormap"] == DEFAULT_SETTINGS["map_colormap"]
        assert settings["icss_opacity"] == pytest.approx(0.3)


class TestCubeUnits:
    def test_negative_voxel_counts_mean_angstrom(self, tmp_path):
        path = tmp_path / "ang.cube"
        lines = [
            "comment",
            "stamp",
            "    0    1.000000    2.000000    3.000000",
            "   -2    0.500000    0.000000    0.000000",
            "   -1    0.000000    0.500000    0.000000",
            "   -1    0.000000    0.000000    0.500000",
            " 1.0 2.0",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cube = cube_io.read_cube(str(path))
        assert cube["origin"] == pytest.approx([1.0, 2.0, 3.0])
        assert cube["vectors"][0] == pytest.approx([0.5, 0.0, 0.0])


@pytest.fixture
def plane_dialog(qapp, fake_context, no_modals, plane_out):
    pytest.importorskip("matplotlib")
    from orca_nics_analyzer.gui import NicsAnalyzerDialog

    dlg = NicsAnalyzerDialog(None, fake_context)
    assert dlg.load_file(plane_out)
    yield dlg
    dlg.close()


class TestDialogReload:
    def test_tab_handler_is_connected_once_across_reloads(
        self, plane_dialog, plane_out
    ):
        signal = plane_dialog.tabs.currentChanged
        before = plane_dialog.tabs.receivers(signal)
        assert plane_dialog.load_file(plane_out)
        assert plane_dialog.load_file(plane_out)
        assert plane_dialog.tabs.receivers(signal) == before

    def test_previous_tabs_are_deleted_on_reload(self, plane_dialog, plane_out):
        from PyQt6 import sip
        from PyQt6.QtCore import QCoreApplication, QEvent

        old = plane_dialog.probe_tab
        assert plane_dialog.load_file(plane_out)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        assert sip.isdeleted(old)
        assert plane_dialog.tabs.count() == 5


class TestStaleSlice:
    def test_axis_change_recuts_the_1d_slice(self, plane_dialog):
        dlg = plane_dialog
        field = dlg.field
        dlg.scan_tab.show_slice(field.extract_line("zz", 0, 4))
        before = dlg.scan_tab._slice_data["values"].copy()

        dlg.axis_combo.setCurrentIndex(dlg.axis_combo.findData("x"))

        after = dlg.scan_tab._slice_data["values"]
        expected = field.extract_line("zz", 0, 4)["values"]
        assert after == pytest.approx(expected, nan_ok=True)
        assert not np.allclose(before, after, equal_nan=True)


class TestDialogLifetime:
    def test_closed_dialog_is_deleted(self, qapp, fake_context, no_modals, plane_out):
        pytest.importorskip("matplotlib")
        from PyQt6 import sip
        from PyQt6.QtCore import QCoreApplication, QEvent

        from orca_nics_analyzer.gui import NicsAnalyzerDialog

        dlg = NicsAnalyzerDialog(None, fake_context, parent=fake_context._main_window)
        assert dlg.load_file(plane_out)
        dlg.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        assert sip.isdeleted(dlg)


class TestProbeTableScale:
    def test_large_volume_builds_a_model_not_widgets(self, qapp, tmp_path):
        from orca_nics_analyzer.probe_tab import ProbeTab

        axis = np.linspace(-2.0, 2.0, 12)
        field = _field([(x, y, z) for x in axis for y in axis for z in axis], tmp_path)
        tab = ProbeTab(field)
        assert tab.model.rowCount() == 12**3
        column = tab.HEADERS.index("NICS(iso)/ppm")
        assert tab.cell_text(0, column) != ""
