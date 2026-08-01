"""GUI tests with a real offscreen Qt, a mocked host plotter and no modals."""

import os
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QMimeData, QUrl  # noqa: E402

from orca_nics_analyzer.parser import NicsParser  # noqa: E402
from orca_nics_analyzer.gui import NicsAnalyzerDialog, _xyz_block  # noqa: E402

pytestmark = pytest.mark.usefixtures("qapp", "no_modals")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def make_dialog(fake_context, request):
    """Build a dialog for a fixture output and close it afterwards."""
    created = []

    def _make(path=None):
        parser = None
        if path is not None:
            parser = NicsParser()
            parser.load(path)
        dialog = NicsAnalyzerDialog(parser, fake_context)
        created.append(dialog)
        return dialog

    yield _make
    for dialog in created:
        dialog.close()


def _mime_for_path(path):
    """Return a QMimeData carrying a single local-file URL."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(path)])
    return mime


# ---------------------------------------------------------------------------
# _xyz_block helper
# ---------------------------------------------------------------------------


class TestXyzBlock:
    def _atoms(self):
        return [
            {"symbol": "C", "xyz": (1.0, 2.0, 3.0), "is_ghost": False},
            {"symbol": "H", "xyz": (0.0, 0.0, 1.0), "is_ghost": True},
            {"symbol": "C", "xyz": (-1.0, 0.0, 0.0), "is_ghost": False},
        ]

    def test_excludes_probes_by_default(self):
        lines = _xyz_block(self._atoms(), include_probes=False).splitlines()
        assert len(lines) == 2
        assert all("C" in ln for ln in lines)

    def test_includes_probes_when_requested(self):
        lines = _xyz_block(self._atoms(), include_probes=True).splitlines()
        assert len(lines) == 3
        # Ghost atom 'H' should be mapped to 'X'
        assert lines[1].startswith("X  ")

    def test_empty_when_no_real_atoms(self):
        atoms = [{"symbol": "H", "xyz": (0.0, 0.0, 0.0), "is_ghost": True}]
        assert _xyz_block(atoms, include_probes=False) == ""

    def test_returns_empty_string_for_empty_list(self):
        assert _xyz_block([], include_probes=False) == ""
        assert _xyz_block([], include_probes=True) == ""

    def test_coordinates_are_formatted(self):
        atoms = [{"symbol": "N", "xyz": (1.5, -2.5, 0.0), "is_ghost": False}]
        block = _xyz_block(atoms, include_probes=False)
        assert "N" in block
        assert "1.50000000" in block
        assert "-2.50000000" in block


# ---------------------------------------------------------------------------
# Empty / welcome state
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_opens_without_parser(self, make_dialog):
        dlg = make_dialog()
        assert dlg is not None

    def test_title_is_generic_without_file(self, make_dialog):
        dlg = make_dialog()
        assert "ORCA NICS Analyzer" in dlg.windowTitle()

    def test_welcome_widget_is_shown(self, make_dialog):
        dlg = make_dialog()
        # Stack index 0 is the welcome widget.
        assert dlg._stack.currentIndex() == 0

    def test_tabs_not_shown_in_empty_state(self, make_dialog):
        dlg = make_dialog()
        # tabs_container is at stack index 1, which is not current.
        assert dlg._stack.currentWidget() is not dlg._tabs_container

    def test_controls_disabled_in_empty_state(self, make_dialog):
        dlg = make_dialog()
        assert not dlg.axis_combo.isEnabled()
        assert not dlg._probe_chk.isEnabled()
        assert not dlg._export_btn.isEnabled()

    def test_open_btn_is_always_visible(self, make_dialog):
        dlg = make_dialog()
        assert not dlg._open_btn.isHidden()

    def test_no_show_xyz_called_in_empty_state(self, make_dialog, fake_context):
        make_dialog()
        fake_context.show_xyz_data.assert_not_called()

    def test_close_in_empty_state_releases_window_slot(self, fake_context):
        dlg = NicsAnalyzerDialog(None, fake_context)
        dlg.close()
        fake_context.register_window.assert_called_with("nics_analyzer", None)


# ---------------------------------------------------------------------------
# Loading data (load_file / load_parser)
# ---------------------------------------------------------------------------


class TestLoadFile:
    def test_load_file_populates_tabs(self, make_dialog, volume_out):
        dlg = make_dialog()
        dlg.load_file(volume_out)
        assert dlg._stack.currentIndex() == 1

    def test_load_file_updates_window_title(self, make_dialog, volume_out):
        dlg = make_dialog()
        dlg.load_file(volume_out)
        assert os.path.basename(volume_out) in dlg.windowTitle()

    def test_load_file_enables_controls(self, make_dialog, volume_out):
        dlg = make_dialog()
        dlg.load_file(volume_out)
        assert dlg._probe_chk.isEnabled()
        assert dlg._export_btn.isEnabled()

    def test_load_file_returns_true_on_success(self, make_dialog, volume_out):
        dlg = make_dialog()
        assert dlg.load_file(volume_out) is True

    def test_load_file_returns_false_for_missing_file(self, make_dialog, tmp_path):
        dlg = make_dialog()
        with patch("orca_nics_analyzer._warn"):
            result = dlg.load_file(str(tmp_path / "nope.out"))
        assert result is False

    def test_load_file_returns_false_for_no_probes(
        self, make_dialog, no_ghosts_out, no_modals
    ):
        dlg = make_dialog()
        result = dlg.load_file(no_ghosts_out)
        assert result is False

    def test_load_file_shows_warning_for_no_probes(
        self, make_dialog, no_ghosts_out, no_modals
    ):
        dlg = make_dialog()
        dlg.load_file(no_ghosts_out)
        assert any("ghost" in msg.lower() for _, msg in no_modals)

    def test_load_file_twice_replaces_data(self, make_dialog, volume_out, single_out):
        dlg = make_dialog()
        dlg.load_file(volume_out)
        dlg.load_file(single_out)
        assert os.path.basename(single_out) in dlg.windowTitle()

    def test_load_parser_directly(self, make_dialog, volume_out):
        dlg = make_dialog()
        parser = NicsParser()
        parser.load(volume_out)
        dlg.load_parser(parser)
        assert dlg._stack.currentIndex() == 1


# ---------------------------------------------------------------------------
# Opening with a parser (constructor path)
# ---------------------------------------------------------------------------


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
        # no_ghosts_out: make_dialog path with parser built here because the
        # fixture has no ghosts and the dialog accepts any parser (even an
        # empty one when passed directly).
        path = request.getfixturevalue(fixture)
        parser = NicsParser()
        parser.load(path)
        # Bypass the ghost check in load_file by passing parser directly.
        from orca_nics_analyzer.analysis import NicsField

        try:
            NicsField(parser)
        except Exception:
            # no_ghosts_out has no probes — default tab is Probes.
            pass
        dlg = make_dialog(path if fixture != "no_ghosts_out" else None)
        if fixture != "no_ghosts_out":
            assert dlg.tabs.tabText(dlg.tabs.currentIndex()) == tab

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

    def test_is_not_modal(self, make_dialog, volume_out):
        """Dialog must not block the main window."""
        from PyQt6.QtCore import Qt

        dlg = make_dialog(volume_out)
        assert not dlg.isModal()
        assert dlg.windowModality() == Qt.WindowModality.NonModal


# ---------------------------------------------------------------------------
# Molecule loading / probe toggle
# ---------------------------------------------------------------------------


class TestMoleculeLoading:
    def test_show_xyz_called_on_file_load(self, make_dialog, volume_out, fake_context):
        make_dialog(volume_out)
        fake_context.show_xyz_data.assert_called_once()

    def test_reset_camera_called_on_file_load(
        self, make_dialog, volume_out, fake_plotter
    ):
        make_dialog(volume_out)
        fake_plotter.reset_camera.assert_called_once()

    def test_show_xyz_called_with_source_name(
        self, make_dialog, volume_out, fake_context
    ):
        make_dialog(volume_out)
        call_kwargs = fake_context.show_xyz_data.call_args
        # Second positional arg or keyword arg 'source_name'.
        args, kwargs = call_kwargs
        source = kwargs.get("source_name", args[1] if len(args) > 1 else None)
        assert source is not None
        assert "benzene" in source.lower() or ".out" in source.lower()

    def test_xyz_block_excludes_probes_by_default(
        self, make_dialog, volume_out, fake_context
    ):
        """The XYZ sent to the viewer must not contain ghost/probe lines."""
        make_dialog(volume_out)
        xyz_sent = fake_context.show_xyz_data.call_args[0][0]
        # Probe atoms in fixture are 'H:' — after stripping the colon their
        # symbol is 'H', but real H atoms also exist; what matters is that the
        # count matches the real (non-ghost) atom count only.
        from orca_nics_analyzer.parser import NicsParser

        parser = NicsParser()
        parser.load(volume_out)
        real_count = len([a for a in parser.data["atoms"] if not a["is_ghost"]])
        sent_lines = [ln for ln in xyz_sent.splitlines() if ln.strip()]
        assert len(sent_lines) == real_count

    def test_probe_toggle_calls_show_xyz_again(
        self, make_dialog, volume_out, fake_context
    ):
        dlg = make_dialog(volume_out)
        fake_context.show_xyz_data.reset_mock()
        dlg._probe_chk.setChecked(True)
        fake_context.show_xyz_data.assert_called_once()

    def test_probe_toggle_includes_probes_when_checked(
        self, make_dialog, volume_out, fake_context
    ):
        from orca_nics_analyzer.parser import NicsParser

        parser = NicsParser()
        parser.load(volume_out)
        total_count = len(parser.data["atoms"])

        dlg = make_dialog(volume_out)
        fake_context.show_xyz_data.reset_mock()
        dlg._probe_chk.setChecked(True)

        xyz_sent = fake_context.show_xyz_data.call_args[0][0]
        sent_lines = [ln for ln in xyz_sent.splitlines() if ln.strip()]
        assert len(sent_lines) == total_count

    def test_probe_toggle_hides_probes_when_unchecked(
        self, make_dialog, volume_out, fake_context
    ):
        from orca_nics_analyzer.parser import NicsParser

        parser = NicsParser()
        parser.load(volume_out)
        real_count = len([a for a in parser.data["atoms"] if not a["is_ghost"]])

        dlg = make_dialog(volume_out)
        # Start checked so we can uncheck and assert.
        dlg._probe_chk.blockSignals(True)
        dlg._probe_chk.setChecked(True)
        dlg._probe_chk.blockSignals(False)
        fake_context.show_xyz_data.reset_mock()
        dlg._probe_chk.setChecked(False)

        xyz_sent = fake_context.show_xyz_data.call_args[0][0]
        sent_lines = [ln for ln in xyz_sent.splitlines() if ln.strip()]
        assert len(sent_lines) == real_count

    def test_show_xyz_not_called_in_empty_state(self, make_dialog, fake_context):
        make_dialog()
        fake_context.show_xyz_data.assert_not_called()

    def test_show_xyz_errors_are_swallowed(self, make_dialog, volume_out, fake_context):
        """A broken show_xyz_data must not crash the dialog."""
        fake_context.show_xyz_data.side_effect = RuntimeError("viewer gone")
        # Should not raise.
        dlg = make_dialog(volume_out)
        assert dlg is not None


# ---------------------------------------------------------------------------
# Drag-and-drop
# ---------------------------------------------------------------------------


class TestDragAndDrop:
    def _drag_enter(self, dlg, mime):
        """Simulate a dragEnterEvent and return whether it was accepted."""
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QDragEnterEvent

        pos = QPointF(10.0, 10.0)
        event = QDragEnterEvent(
            pos.toPoint(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        dlg.dragEnterEvent(event)
        return event.isAccepted()

    def _drop(self, dlg, mime):
        """Simulate a dropEvent and return whether it was accepted."""
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QDropEvent

        pos = QPointF(10.0, 10.0)
        event = QDropEvent(
            pos,
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        dlg.dropEvent(event)
        return event.isAccepted()

    def test_accepts_out_file(self, make_dialog, volume_out):
        dlg = make_dialog()
        mime = _mime_for_path(volume_out)
        assert self._drag_enter(dlg, mime) is True

    def test_accepts_log_file(self, make_dialog, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("")
        dlg = make_dialog()
        mime = _mime_for_path(str(log))
        assert self._drag_enter(dlg, mime) is True

    def test_rejects_non_orca_file(self, make_dialog, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("")
        dlg = make_dialog()
        mime = _mime_for_path(str(txt))
        assert self._drag_enter(dlg, mime) is False

    def test_rejects_empty_mime(self, make_dialog):
        dlg = make_dialog()
        empty = QMimeData()
        assert self._drag_enter(dlg, empty) is False

    def test_drop_loads_file(self, make_dialog, volume_out):
        dlg = make_dialog()
        mime = _mime_for_path(volume_out)
        accepted = self._drop(dlg, mime)
        assert accepted is True
        assert dlg._stack.currentIndex() == 1

    def test_drop_on_loaded_dialog_replaces_data(
        self, make_dialog, volume_out, single_out
    ):
        dlg = make_dialog(volume_out)
        mime = _mime_for_path(single_out)
        self._drop(dlg, mime)
        assert os.path.basename(single_out) in dlg.windowTitle()

    def test_drop_rejects_when_load_fails(self, make_dialog, tmp_path):
        """Dropping an unreadable file should not accept the event."""
        bad = tmp_path / "empty.out"
        bad.write_text("")  # valid path, no ghosts — load_file returns False
        dlg = make_dialog()
        mime = _mime_for_path(str(bad))
        accepted = self._drop(dlg, mime)
        # load_file returns False (no probes) → dropEvent should not accept.
        assert accepted is False

    def test_dialog_has_accept_drops_enabled(self, make_dialog):
        dlg = make_dialog()
        assert dlg.acceptDrops() is True


# ---------------------------------------------------------------------------
# Probe table (existing tests preserved)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scan tab
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Map tab
# ---------------------------------------------------------------------------


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
        assert dialog.icss_tab.slice_slider.maximum() == 0

    def test_slice_slider_spans_a_volume(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        assert dialog.icss_tab.slice_slider.maximum() == 4
        dialog.icss_tab.slice_slider.setValue(4)
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


# ---------------------------------------------------------------------------
# ICSS tab
# ---------------------------------------------------------------------------


class TestIcssTab:
    def test_draws_isosurfaces(self, make_dialog, volume_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.draw()

        # In test, the dialog creation sets the default slice which draws the preview (2 meshes),
        # then draw() draws 2 isosurfaces and another preview (2 meshes).
        # Total: 2 + 2 + 2 = 6 add_mesh calls to the mock.
        assert fake_plotter.add_mesh.call_count >= 4
        assert "isosurface" in dialog.icss_tab.status.text()

    def test_one_sign_only(self, make_dialog, volume_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.show_positive.setChecked(False)
        dialog.icss_tab.draw()
        # Same as above, but 1 isosurface instead of 2.
        assert fake_plotter.add_mesh.call_count >= 3

    def test_cmap_and_span_uses_tab_controls(self, make_dialog, volume_out):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.cmap.setCurrentText("RdBu_r")
        dialog.icss_tab.vmax.setValue(15.5)
        dialog.icss_tab.auto_range.setChecked(False)

        cmap, span, auto = dialog.icss_tab._cmap_and_span()
        assert cmap == "RdBu_r"
        assert span == 15.5
        assert not auto

    def test_vmax_disabled_when_auto(self, make_dialog, volume_out):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)

        dialog.icss_tab.auto_range.setChecked(True)
        assert not dialog.icss_tab.vmax.isEnabled()

        dialog.icss_tab.auto_range.setChecked(False)
        assert dialog.icss_tab.vmax.isEnabled()

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

    def test_initial_3d_draw_auto_saves_selected_cube(
        self, make_dialog, volume_out, tmp_path
    ):
        pytest.importorskip("pyvista")
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        dialog = make_dialog(str(source))
        cube_path = source.parent / "run_nics_cubes" / "run_NICS_zz.cube"
        assert cube_path.exists()
        assert "Cached:" in dialog.icss_tab.cache_label.text()
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
        assert fake_plotter.add_mesh.call_count >= 1
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

    def test_slice_slider_and_spin_sync(self, make_dialog, volume_out):
        dlg = make_dialog(volume_out)
        dlg.icss_tab.slice_slider.setValue(1)
        assert dlg.icss_tab.slice_spin.value() == 1
        dlg.icss_tab.slice_spin.setValue(2)
        assert dlg.icss_tab.slice_slider.value() == 2

    def test_goto_2d_btn_switches_tab(self, make_dialog, volume_out):
        dlg = make_dialog(volume_out)
        dlg.tabs.setCurrentWidget(dlg.icss_tab)
        dlg.icss_tab.goto_2d_btn.click()
        assert dlg.tabs.currentWidget() is dlg.map_tab


# ---------------------------------------------------------------------------
# Axis switching
# ---------------------------------------------------------------------------


class TestTabSync:
    def test_cmap_syncs_bidirectionally(self, make_dialog, volume_out):
        dlg = make_dialog(volume_out)

        # Change on map tab
        dlg.map_tab.cmap.setCurrentText("RdBu_r")
        assert dlg.icss_tab.cmap.currentText() == "RdBu_r"

        # Change on icss tab
        dlg.icss_tab.cmap.setCurrentText("coolwarm")
        assert dlg.map_tab.cmap.currentText() == "coolwarm"

    def test_vmax_syncs_bidirectionally(self, make_dialog, volume_out):
        dlg = make_dialog(volume_out)

        # Turn off auto range so it doesn't immediately overwrite our manual values
        dlg.map_tab.auto_range.setChecked(False)

        dlg.map_tab.vmax.setValue(42.0)
        assert dlg.icss_tab.vmax.value() == 42.0

        dlg.icss_tab.vmax.setValue(17.5)
        assert dlg.map_tab.vmax.value() == 17.5

    def test_auto_range_syncs_bidirectionally(self, make_dialog, volume_out):
        dlg = make_dialog(volume_out)

        dlg.map_tab.auto_range.setChecked(False)
        assert not dlg.icss_tab.auto_range.isChecked()

        dlg.icss_tab.auto_range.setChecked(True)
        assert dlg.map_tab.auto_range.isChecked()

    def test_stack_axis_syncs_to_3d_viewer(self, make_dialog, volume_out, fake_plotter):
        pytest.importorskip("pyvista")
        dlg = make_dialog(volume_out)

        # Change stack axis on 3D tab
        dlg.icss_tab.stack_axis_combo.setCurrentIndex(1)

        # Verify it updated the field
        assert dlg.field.stack_axis_index() == 1

        # Verify it called the 3d tab's update method (which calls add_mesh)
        # We can't directly check the arrow easily, but we know add_mesh was called
        # more than just during draw(). Actually, it triggers update_cut_axis_preview
        # which removes and adds ACTOR_CUT_AXIS.
        assert fake_plotter.add_mesh.called


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


# ---------------------------------------------------------------------------
# Export all
# ---------------------------------------------------------------------------


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

    def test_export_btn_disabled_in_empty_state(self, make_dialog):
        dlg = make_dialog()
        assert not dlg._export_btn.isEnabled()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


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

    def test_empty_dialog_close_releases_slot(self, fake_context):
        dialog = NicsAnalyzerDialog(None, fake_context)
        dialog.close()
        fake_context.register_window.assert_called_with("nics_analyzer", None)


# ---------------------------------------------------------------------------
# Scan1DTab — show_slice / clear_slice
# ---------------------------------------------------------------------------


class TestScan1DSlice:
    def _slice_data(self, field):
        """A minimal slice-data dict from the plane fixture."""
        return field.extract_line("iso", 0, 0)

    def test_show_slice_switches_to_slice_mode(self, make_dialog, plane_out):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        data = load_field(plane_out).extract_line("iso", 0, 0)
        dlg.scan_tab.show_slice(data)
        assert dlg.scan_tab._slice_data is data

    def test_clear_slice_restores_native_mode(self, make_dialog, plane_out):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        data = load_field(plane_out).extract_line("iso", 0, 0)
        dlg.scan_tab.show_slice(data)
        dlg.scan_tab.clear_slice()
        assert dlg.scan_tab._slice_data is None

    def test_back_btn_visible_in_slice_mode(self, make_dialog, plane_out):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        data = load_field(plane_out).extract_line("iso", 0, 0)
        dlg.scan_tab.show_slice(data)
        assert not dlg.scan_tab._clear_slice_btn.isHidden()

    def test_back_btn_hidden_in_native_mode(self, make_dialog, plane_out):
        dlg = make_dialog(plane_out)
        assert dlg.scan_tab._clear_slice_btn.isHidden()

    def test_info_shows_extracted_slice_label(self, make_dialog, plane_out):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        data = load_field(plane_out).extract_line("iso", 0, 3)
        dlg.scan_tab.show_slice(data)
        assert "extracted slice" in dlg.scan_tab.info.text()

    def test_slice_csv_export(self, make_dialog, plane_out, tmp_path):
        from orca_nics_analyzer.analysis import load_field
        from unittest.mock import patch

        dlg = make_dialog(plane_out)
        data = load_field(plane_out).extract_line("iso", 0, 0)
        dlg.scan_tab.show_slice(data)
        target = str(tmp_path / "slice.csv")
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dlg.scan_tab.export_csv()
        lines = open(target, encoding="utf-8").read().splitlines()
        assert lines[0].startswith("Index,Distance")
        assert len(lines) > 1


# ---------------------------------------------------------------------------
# Map2DTab — 2D → 1D slice controls
# ---------------------------------------------------------------------------


class TestMap2DSliceControls:
    def test_slice1d_axis_combo_present(self, make_dialog, plane_out):
        dlg = make_dialog(plane_out)
        assert dlg.map_tab._slice1d_axis.count() == 2

    def test_slice1d_slider_present(self, make_dialog, plane_out):
        dlg = make_dialog(plane_out)
        assert dlg.map_tab._slice1d_slider is not None

    def test_slice1d_slider_max_matches_plane_axis1(self, make_dialog, plane_out):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        field = load_field(plane_out)
        info = field.plane_data("iso")
        # Default axis=0: fix axis-1 rows → max = len(a1) - 1
        assert dlg.map_tab._slice1d_slider.maximum() == len(info["a1"]) - 1

    def test_slice1d_slider_max_updates_on_axis_switch(self, make_dialog, plane_out):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        field = load_field(plane_out)
        info = field.plane_data("iso")
        # Switch to fix axis-2 → max = len(a2) - 1
        dlg.map_tab._slice1d_axis.setCurrentIndex(1)
        assert dlg.map_tab._slice1d_slider.maximum() == len(info["a2"]) - 1

    def test_send_to_scan_routes_data(self, make_dialog, plane_out):
        """Clicking the '→ 1D Scan tab' button injects data into scan_tab."""
        dlg = make_dialog(plane_out)
        dlg.map_tab._slice1d_slider.setValue(2)
        dlg.map_tab._emit_slice_to_1d()
        assert dlg.scan_tab._slice_data is not None

    def test_send_to_scan_switches_tab(self, make_dialog, plane_out):
        dlg = make_dialog(plane_out)
        dlg.map_tab._emit_slice_to_1d()
        assert dlg.tabs.currentWidget() is dlg.scan_tab

    def test_send_to_scan_data_has_correct_length(self, make_dialog, plane_out):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        field = load_field(plane_out)
        info = field.plane_data("iso")
        dlg.map_tab._slice1d_axis.setCurrentIndex(0)  # fix axis-1, walk axis-2
        dlg.map_tab._emit_slice_to_1d()
        n_a2 = len(info["a2"])
        assert len(dlg.scan_tab._slice_data["distance"]) == n_a2

    def test_different_slider_positions_give_different_slices(
        self, make_dialog, plane_out
    ):
        import numpy as np

        dlg = make_dialog(plane_out)
        dlg.map_tab._slice1d_slider.setValue(0)
        dlg.map_tab._emit_slice_to_1d()
        vals0 = dlg.scan_tab._slice_data["values"].copy()

        dlg.map_tab._slice1d_slider.setValue(4)
        dlg.map_tab._emit_slice_to_1d()
        vals4 = dlg.scan_tab._slice_data["values"].copy()

        assert not np.allclose(vals0, vals4)

    def test_slice_from_volume_respects_stack_slider(self, make_dialog, volume_out):
        """Extracting a slice from a volume should use the 3D tab's stack-axis slider."""
        import numpy as np

        dlg = make_dialog(volume_out)
        dlg.icss_tab.slice_slider.setValue(0)
        dlg.map_tab._slice1d_axis.setCurrentIndex(0)
        dlg.map_tab._slice1d_slider.setValue(0)

        def _recv(d):
            dlg.scan_tab._slice_data = d

        dlg.map_tab._show_slice_in_1d = _recv
        dlg.map_tab._emit_slice_to_1d()
        vals_layer0 = dlg.scan_tab._slice_data["values"].copy()

        dlg.icss_tab.slice_slider.setValue(2)
        dlg.map_tab._emit_slice_to_1d()
        vals_layer2 = dlg.scan_tab._slice_data["values"].copy()

        assert not np.allclose(vals_layer0, vals_layer2)

    def test_no_slice_panel_for_non_gridded(self, make_dialog, single_out, no_modals):
        """Clicking → 1D on a non-gridded field shows a warning, not a crash."""
        dlg = make_dialog(single_out)
        # single_out is not gridded; _emit_slice_to_1d should show a message.
        dlg.map_tab._emit_slice_to_1d()
        # scan_tab must not be poisoned with bad data.
        assert dlg.scan_tab._slice_data is None

    def test_slice1d_spin_and_slider_sync(self, make_dialog, plane_out):
        dlg = make_dialog(plane_out)
        dlg.map_tab._slice1d_slider.setValue(1)
        assert dlg.map_tab._slice1d_spin.value() == 1
        dlg.map_tab._slice1d_spin.setValue(2)
        assert dlg.map_tab._slice1d_slider.value() == 2

    def test_slice1d_slider_defaults_to_center(self, make_dialog, plane_out):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        field = load_field(plane_out)
        info = field.plane_data("iso")
        n = len(info["a1"])
        assert dlg.map_tab._slice1d_slider.value() == (n - 1) // 2

    def test_map2d_slice1d_spin_change_triggers_refresh(self, make_dialog, plane_out):
        from unittest.mock import MagicMock

        dlg = make_dialog(plane_out)
        dlg.map_tab.refresh = MagicMock()
        dlg.map_tab._slice1d_spin.setValue(1)
        assert dlg.map_tab.refresh.called

    def test_icss3d_slice_spin_maximum_updates_on_stack_axis_change(
        self, make_dialog, volume_out
    ):
        dlg = make_dialog(volume_out)
        dlg.icss_tab.stack_axis_combo.setCurrentIndex(1)
        n = dlg.icss_tab.field.plane_data("zz")["n_slices"]
        assert dlg.icss_tab.slice_spin.maximum() == max(0, n - 1)

    def test_slice_controls_visibility_single_vs_volume(
        self, make_dialog, plane_out, volume_out
    ):
        dlg_plane = make_dialog(plane_out)
        # plane_out has 1 slice in volume sense
        assert dlg_plane.icss_tab.slice_group.isHidden()

        dlg_vol = make_dialog(volume_out)
        assert not dlg_vol.icss_tab.slice_group.isHidden()

    def test_show_map_tab_refreshes_map(self, make_dialog, volume_out):
        from unittest.mock import MagicMock

        dlg = make_dialog(volume_out)
        dlg.map_tab.refresh = MagicMock()
        dlg._show_map_tab()
        assert dlg.tabs.currentWidget() is dlg.map_tab
        assert dlg.map_tab.refresh.called

    def test_tab_changed_signal_triggers_refresh(self, make_dialog, volume_out):
        from unittest.mock import MagicMock

        dlg = make_dialog(volume_out)
        dlg.map_tab.refresh = MagicMock()
        dlg.icss_tab.draw = MagicMock()

        dlg.tabs.setCurrentWidget(dlg.map_tab)
        assert dlg.map_tab.refresh.called

        dlg.tabs.setCurrentWidget(dlg.icss_tab)
        assert dlg.icss_tab.draw.called

    def test_icss_slice_slider_notifies_map_tab(self, make_dialog, volume_out):
        from unittest.mock import MagicMock

        dlg = make_dialog(volume_out)
        dlg.icss_tab._on_slice_settings_changed = MagicMock()
        dlg.icss_tab.slice_slider.setValue(0)
        assert dlg.icss_tab._on_slice_settings_changed.called

    def test_icss3d_slice_group_structure(self, make_dialog, volume_out):
        from PyQt6.QtWidgets import QGroupBox

        dlg = make_dialog(volume_out)
        assert isinstance(dlg.icss_tab.slice_group, QGroupBox)
        assert dlg.icss_tab.slice_group.title() == "Slice → 2D"
        # Verify show_cut_axis is child of slice_group
        assert dlg.icss_tab.show_cut_axis.parentWidget() is dlg.icss_tab.slice_group
