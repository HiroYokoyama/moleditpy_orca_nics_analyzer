"""GUI tests with a real offscreen Qt, a mocked host plotter and no modals."""

import os
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PyQt6.QtWidgets")

from orca_nics_analyzer.parser import NicsParser  # noqa: E402
from orca_nics_analyzer.gui import NicsAnalyzerDialog  # noqa: E402

pytestmark = pytest.mark.usefixtures("qapp", "no_modals")


@pytest.fixture
def make_dialog(fake_context, request):
    """Build a dialog for a fixture output and close it afterwards."""
    created = []

    def _make(path):
        parser = NicsParser()
        parser.load(path)
        dialog = NicsAnalyzerDialog(parser, fake_context)
        created.append(dialog)
        return dialog

    yield _make
    for dialog in created:
        dialog.close()


class TestOpening:
    @pytest.mark.parametrize(
        "fixture,tab",
        [
            ("single_out", "1D Scan"),
            ("plane_out", "2D Map"),
            ("volume_out", "3D ICSS"),
            ("no_ghosts_out", "Probes"),
        ],
    )
    def test_opens_on_the_tab_matching_the_layout(
        self, request, make_dialog, fixture, tab
    ):
        dialog = make_dialog(request.getfixturevalue(fixture))
        assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == tab

    def test_all_tabs_are_present(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
        assert titles == ["Probes", "1D Scan", "2D Map", "3D ICSS", "Summary"]

    def test_title_names_the_file(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        assert os.path.basename(volume_out) in dialog.windowTitle()

    def test_summary_tab_is_populated(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        assert "volume" in dialog.summary.toPlainText()


class TestProbeTable:
    def test_row_per_probe(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        assert dialog.probe_tab.table.rowCount() == 125

    def test_columns_match_the_csv(self, make_dialog, single_out):
        dialog = make_dialog(single_out)
        table = dialog.probe_tab.table
        assert table.columnCount() == len(dialog.field.CSV_COLUMNS)

    def test_values_are_formatted(self, make_dialog, single_out):
        dialog = make_dialog(single_out)
        table = dialog.probe_tab.table
        headers = [
            table.horizontalHeaderItem(c).text() for c in range(table.columnCount())
        ]
        column = headers.index("NICS_zz/ppm")
        assert table.item(0, column).text().replace("-", "").replace(".", "").isdigit()

    def test_numeric_columns_sort_by_value(self, make_dialog, volume_out):
        """Text sorting would put -10 after -9."""
        dialog = make_dialog(volume_out)
        table = dialog.probe_tab.table
        headers = [
            table.horizontalHeaderItem(c).text() for c in range(table.columnCount())
        ]
        column = headers.index("NICS_zz/ppm")
        table.sortItems(column)
        values = [table.item(r, column).value for r in range(table.rowCount())]
        assert values == sorted(values)

    def test_colouring_can_be_switched_off(self, make_dialog, single_out):
        dialog = make_dialog(single_out)
        dialog.probe_tab.colour_chk.setChecked(False)
        assert dialog.probe_tab.table.rowCount() == 3

    def test_copy_csv_reaches_the_clipboard(self, make_dialog, single_out):
        from PyQt6.QtGui import QGuiApplication

        dialog = make_dialog(single_out)
        dialog.probe_tab.copy_csv()
        assert "NICS_zz/ppm" in QGuiApplication.clipboard().text()

    def test_export_csv_writes_the_file(self, make_dialog, single_out, tmp_path):
        dialog = make_dialog(single_out)
        target = str(tmp_path / "out.csv")
        with patch(
            "orca_nics_analyzer.probe_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.probe_tab.export_csv()
        assert len(open(target, encoding="utf-8").read().splitlines()) == 4

    def test_export_csv_cancelled_writes_nothing(
        self, make_dialog, single_out, tmp_path
    ):
        dialog = make_dialog(single_out)
        with patch(
            "orca_nics_analyzer.probe_tab.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ):
            dialog.probe_tab.export_csv()
        assert list(tmp_path.iterdir()) == []


class TestScanTab:
    def test_plots_the_profile(self, make_dialog, request):
        dialog = make_dialog(
            os.path.join(
                os.path.dirname(__file__), "sample_outputs", "benzene_nics_scan.out"
            )
        )
        assert "11 probes" in dialog.scan_tab.info.text()
        assert "largest" in dialog.scan_tab.info.text()

    def test_reports_a_non_line_layout(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        assert dialog.scan_tab.info.text() == ""

    def test_curve_toggles(self, make_dialog, single_out):
        dialog = make_dialog(single_out)
        dialog.scan_tab.show_zz.setChecked(False)
        assert "NICS(iso)" in dialog.scan_tab.info.text()

    def test_export_csv(self, make_dialog, single_out, tmp_path):
        dialog = make_dialog(single_out)
        target = str(tmp_path / "scan.csv")
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.scan_tab.export_csv()
        assert open(target, encoding="utf-8").read().startswith("Index,Distance")

    def test_export_png(self, make_dialog, single_out, tmp_path):
        dialog = make_dialog(single_out)
        target = str(tmp_path / "scan.png")
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.scan_tab.export_png()
        assert os.path.getsize(target) > 0


class TestMapTab:
    def test_draws_a_plane(self, make_dialog, plane_out):
        dialog = make_dialog(plane_out)
        assert dialog.map_tab.figure.axes

    def test_auto_range_fills_the_spin_box(self, make_dialog, plane_out):
        dialog = make_dialog(plane_out)
        assert dialog.map_tab.vmax.value() > 0

    def test_manual_range_is_kept(self, make_dialog, plane_out):
        dialog = make_dialog(plane_out)
        dialog.map_tab.auto_range.setChecked(False)
        dialog.map_tab.vmax.setValue(12.5)
        assert dialog.map_tab.vmax.value() == pytest.approx(12.5)

    def test_slice_slider_hidden_for_a_single_layer(self, make_dialog, plane_out):
        dialog = make_dialog(plane_out)
        assert dialog.map_tab.slice_slider.maximum() == 0

    def test_slice_slider_spans_a_volume(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        assert dialog.map_tab.slice_slider.maximum() == 4
        dialog.map_tab.slice_slider.setValue(4)
        assert dialog.map_tab.figure.axes

    def test_display_toggles_redraw(self, make_dialog, plane_out):
        dialog = make_dialog(plane_out)
        for widget in (
            dialog.map_tab.show_molecule,
            dialog.map_tab.show_contours,
            dialog.map_tab.show_probes,
        ):
            widget.setChecked(not widget.isChecked())
        assert dialog.map_tab.figure.axes

    def test_message_when_there_is_no_grid(self, make_dialog, single_out):
        dialog = make_dialog(single_out)
        texts = [t.get_text() for ax in dialog.map_tab.figure.axes for t in ax.texts]
        assert any("do not form a regular" in t for t in texts)

    def test_export_grid_csv(self, make_dialog, plane_out, tmp_path):
        dialog = make_dialog(plane_out)
        target = str(tmp_path / "map.csv")
        with patch(
            "orca_nics_analyzer.map2d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.map_tab.export_csv()
        lines = open(target, encoding="utf-8").read().splitlines()
        assert len(lines) == 10  # header + 9 rows
        assert lines[0].startswith("axis2\\axis1")

    def test_export_png(self, make_dialog, plane_out, tmp_path):
        dialog = make_dialog(plane_out)
        target = str(tmp_path / "map.png")
        with patch(
            "orca_nics_analyzer.map2d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.map_tab.export_png()
        assert os.path.getsize(target) > 0


class TestIcssTab:
    def test_draws_isosurfaces(self, make_dialog, volume_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.draw()
        assert fake_plotter.add_mesh.call_count == 2
        assert "isosurface" in dialog.icss_tab.status.text()

    def test_one_sign_only(self, make_dialog, volume_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.show_positive.setChecked(False)
        dialog.icss_tab.draw()
        assert fake_plotter.add_mesh.call_count == 1

    def test_auto_isovalue_is_positive(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        assert dialog.icss_tab.isovalue.value() > 0

    def test_isovalue_slider_and_spin_stay_in_sync(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        dialog.icss_tab.isovalue.setValue(4.0)
        assert dialog.icss_tab.iso_slider.value() == 40
        dialog.icss_tab.iso_slider.setValue(80)
        assert dialog.icss_tab.isovalue.value() == pytest.approx(8.0)

    def test_refuses_without_a_grid(self, make_dialog, single_out, fake_plotter):
        dialog = make_dialog(single_out)
        dialog.icss_tab.draw()
        assert fake_plotter.add_mesh.call_count == 0

    def test_generates_and_reuses_the_cube(self, make_dialog, volume_out, tmp_path):
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        dialog = make_dialog(str(source))
        path = dialog.icss_tab.generate_cube()
        assert os.path.exists(path)
        assert "Wrote cube" in dialog.icss_tab.status.text()
        dialog.icss_tab.generate_cube()
        assert "Reused cached" in dialog.icss_tab.status.text()

    def test_cache_label_tracks_the_file(self, make_dialog, volume_out, tmp_path):
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        dialog = make_dialog(str(source))
        assert "Not generated yet" in dialog.icss_tab.cache_label.text()
        dialog.icss_tab.generate_cube()
        assert "Cached:" in dialog.icss_tab.cache_label.text()

    def test_save_cube_as(self, make_dialog, volume_out, tmp_path):
        dialog = make_dialog(volume_out)
        target = str(tmp_path / "manual.cube")
        with patch(
            "orca_nics_analyzer.icss3d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.icss_tab.save_cube_as()
        assert os.path.getsize(target) > 0

    def test_component_switch_updates_the_cache_label(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        dialog.icss_tab.component.setCurrentIndex(1)
        assert "NICS_iso" in dialog.icss_tab.cache_label.text()

    def test_show_plane_in_3d(self, make_dialog, plane_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(plane_out)
        dialog.map_tab._emit_show_in_3d()
        assert fake_plotter.add_mesh.call_count == 1
        assert "Map plane added" in dialog.icss_tab.status.text()

    def test_clear_removes_every_actor(self, make_dialog, volume_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.draw()
        fake_plotter.reset_mock()
        dialog.icss_tab.clear_actors()
        removed = {c.args[0] for c in fake_plotter.remove_actor.call_args_list}
        from orca_nics_analyzer.icss3d_tab import ALL_ACTORS

        assert removed == set(ALL_ACTORS)


class TestAxisSwitching:
    def test_changing_the_axis_updates_every_tab(self, make_dialog, single_out):
        dialog = make_dialog(single_out)
        before = [p["zz"] for p in dialog.field.probes]
        dialog.axis_combo.setCurrentIndex(2)  # Lab X
        assert [p["zz"] for p in dialog.field.probes] != before
        assert (
            "Lab X" in dialog.summary.toPlainText()
            or "x" in dialog.summary.toPlainText()
        )

    def test_axis_is_disabled_without_tensors(
        self, make_dialog, single_out, monkeypatch
    ):
        parser = NicsParser()
        parser.load(single_out)
        for entry in parser.data["shieldings"].values():
            entry["tensor"] = None
        parser.data["has_tensors"] = False
        dialog = NicsAnalyzerDialog(parser, MagicMock())
        try:
            assert not dialog.axis_combo.isEnabled()
            assert not dialog.icss_tab.component.isEnabled()
        finally:
            dialog.close()


class TestExportAll:
    def test_writes_everything(self, make_dialog, volume_out, tmp_path):
        dialog = make_dialog(volume_out)
        with patch(
            "orca_nics_analyzer.gui.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dialog.export_all()
        names = sorted(p.name for p in tmp_path.iterdir())
        assert len(names) == 4
        assert sum(1 for n in names if n.endswith(".cube")) == 2

    def test_cancelled_writes_nothing(self, make_dialog, volume_out, tmp_path):
        dialog = make_dialog(volume_out)
        with patch(
            "orca_nics_analyzer.gui.QFileDialog.getExistingDirectory", return_value=""
        ):
            dialog.export_all()
        assert list(tmp_path.iterdir()) == []


class TestLifecycle:
    def test_close_releases_the_window_slot(self, volume_out, fake_context):
        parser = NicsParser()
        parser.load(volume_out)
        dialog = NicsAnalyzerDialog(parser, fake_context)
        dialog.close()
        fake_context.register_window.assert_called_with("nics_analyzer", None)

    def test_close_clears_the_actors(self, volume_out, fake_context, fake_plotter):
        pytest.importorskip("pyvista")
        parser = NicsParser()
        parser.load(volume_out)
        dialog = NicsAnalyzerDialog(parser, fake_context)
        dialog.icss_tab.draw()
        fake_plotter.reset_mock()
        dialog.close()
        assert fake_plotter.remove_actor.called

    def test_escape_also_cleans_up(self, volume_out, fake_context):
        """Rejecting a dialog never raises a close event."""
        parser = NicsParser()
        parser.load(volume_out)
        dialog = NicsAnalyzerDialog(parser, fake_context)
        dialog.reject()
        fake_context.register_window.assert_called_with("nics_analyzer", None)

    def test_cleanup_runs_once(self, volume_out, fake_context):
        parser = NicsParser()
        parser.load(volume_out)
        dialog = NicsAnalyzerDialog(parser, fake_context)
        dialog.close()
        fake_context.register_window.reset_mock()
        dialog.close()
        fake_context.register_window.assert_not_called()
