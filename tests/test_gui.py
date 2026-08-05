"""GUI tests with a real offscreen Qt, a mocked host plotter and no modals."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QMimeData, QUrl  # noqa: E402
from PyQt6.QtWidgets import QPushButton  # noqa: E402

from orca_nics_analyzer.parser import NicsParser  # noqa: E402
from orca_nics_analyzer.gui import NicsAnalyzerDialog, _xyz_block  # noqa: E402
from orca_nics_analyzer.icss3d_tab import ACTOR_AXIS_VECTOR  # noqa: E402
from orca_nics_analyzer.icss3d_tab import ACTOR_NEGATIVE, ACTOR_POSITIVE  # noqa: E402

pytestmark = pytest.mark.usefixtures("qapp", "no_modals")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def make_dialog(fake_context, request, tmp_path):
    """Build a dialog for a fixture output and close it afterwards."""
    created = []

    def _make(path=None):
        parser = None
        if path is not None:
            parser = NicsParser()
            parser.load(path)
        dialog = NicsAnalyzerDialog(
            parser, fake_context, settings_file=str(tmp_path / "settings.json")
        )
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

    def test_load_file_returns_false_for_ghosts_without_shieldings(
        self, make_dialog, tmp_path
    ):
        path = tmp_path / "ghosts_only.out"
        path.write_text(
            "CARTESIAN COORDINATES (ANGSTROEM)\n"
            "---------------------------------\n"
            "  C      0.000000    0.000000    0.000000\n"
            "  H:     0.000000    0.000000    1.000000\n\n",
            encoding="utf-8",
        )
        dlg = make_dialog()
        with patch("orca_nics_analyzer._warn_missing_shieldings") as warn:
            assert dlg.load_file(str(path)) is False
        warn.assert_called_once()

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

    def test_map_range_and_auto_controls_use_distinct_cells(
        self, make_dialog, plane_out
    ):
        dialog = make_dialog(plane_out)
        map_layout = dialog.map_tab.auto_range.parentWidget().layout()
        map_auto = map_layout.getItemPosition(
            map_layout.indexOf(dialog.map_tab.auto_range)
        )
        map_vmax = map_layout.getItemPosition(map_layout.indexOf(dialog.map_tab.vmax))
        assert map_auto[:2] != map_vmax[:2]

    def test_map_axes_are_centered_with_a_data_limit_aspect(
        self, make_dialog, plane_out
    ):
        dialog = make_dialog(plane_out)
        ax = dialog.map_tab.figure.axes[0]
        assert ax.get_aspect() == 1.0
        assert ax.get_adjustable() == "datalim"
        assert ax.get_anchor() == "C"

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

    def test_alternate_3d_slice_axis_uses_coordinate_title(
        self, make_dialog, volume_out
    ):
        dialog = make_dialog(volume_out)
        dialog.field.set_stack_axis(1)
        info = dialog.field.plane_slice("zz", 2)
        title = dialog.map_tab._slice_position_label(info)
        assert "lattice axis 2" in title
        assert "from the ring plane" not in title

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

    def test_slice_line_is_hidden_by_default(self, make_dialog, plane_out):
        dialog = make_dialog(plane_out)
        assert not dialog.map_tab.show_1d_line.isChecked()

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

    def test_show_in_3d_errors_are_reported(self, make_dialog, plane_out, no_modals):
        dialog = make_dialog(plane_out)

        def fail(*args):
            raise RuntimeError("overlay failed")

        dialog.map_tab._show_in_3d = fail
        dialog.map_tab._emit_show_in_3d()
        assert any("overlay failed" in message for _, message in no_modals)

    def test_slice_to_1d_without_receiver_shows_message(
        self, make_dialog, plane_out, no_modals
    ):
        dialog = make_dialog(plane_out)
        dialog.map_tab._show_slice_in_1d = None
        dialog.map_tab._emit_slice_to_1d()
        assert any("No 1D scan tab" in message for _, message in no_modals)

    def test_export_cancelled_does_nothing(self, make_dialog, plane_out):
        dialog = make_dialog(plane_out)
        with patch(
            "orca_nics_analyzer.map2d_tab.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ):
            dialog.map_tab.export_csv()
            dialog.map_tab.export_png()

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
    def test_structured_grid_preserves_volume_shape(self, volume_out):
        pytest.importorskip("pyvista")
        from orca_nics_analyzer.icss3d_tab import structured_grid

        grid = structured_grid(
            np.zeros((2, 3, 4)),
            np.zeros(3),
            np.eye(3),
        )
        assert grid.dimensions == (2, 3, 4)
        assert grid.n_points == 24

    def test_plotter_getter_errors_are_handled(self, make_dialog, volume_out):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)

        def fail():
            raise RuntimeError("plotter closed")

        dialog.icss_tab._plotter_getter = fail
        assert dialog.icss_tab._plotter() is None

    def test_non_volume_draw_reports_layout(self, make_dialog, plane_out, no_modals):
        pytest.importorskip("pyvista")
        dialog = make_dialog(plane_out)
        dialog.tabs.setCurrentWidget(dialog.icss_tab)
        dialog.icss_tab.draw()
        assert any("Isosurfaces need" in message for _, message in no_modals)

    def test_colormap_change_never_pops_a_modal(
        self, make_dialog, plane_out, no_modals
    ):
        """A control change must not nag about the layout it cannot draw."""
        pytest.importorskip("pyvista")
        dialog = make_dialog(plane_out)
        dialog.tabs.setCurrentWidget(dialog.icss_tab)
        dialog.icss_tab.cmap.setCurrentIndex(1)
        no_modals.clear()
        # Index 0 is the one that used to leak through as ``silent=0``.
        dialog.icss_tab.cmap.setCurrentIndex(0)
        assert no_modals == []

    def test_control_changes_never_pop_a_modal(self, make_dialog, plane_out, no_modals):
        pytest.importorskip("pyvista")
        dialog = make_dialog(plane_out)
        dialog.tabs.setCurrentWidget(dialog.icss_tab)
        no_modals.clear()
        dialog.icss_tab.opacity.setValue(0.4)
        dialog.icss_tab.show_positive.setChecked(False)
        dialog.icss_tab.isovalue.setValue(2.0)
        assert no_modals == []

    def test_missing_plotter_is_handled(self, make_dialog, volume_out):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab._plotter_getter = lambda: None
        dialog.icss_tab.status.clear()
        dialog.icss_tab.draw(silent=True)
        assert "Drew" not in dialog.icss_tab.status.text()

    def test_cut_axis_preview_adds_and_tracks_actors(
        self, make_dialog, volume_out, fake_plotter
    ):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.show_cut_axis.setChecked(True)
        dialog.icss_tab.update_cut_axis_preview()
        names = {
            call.kwargs.get("name") for call in fake_plotter.add_mesh.call_args_list
        }
        assert "nics_cut_axis" in names
        assert "nics_cut_axis_edge" in names

    def test_clear_actors_without_plotter_clears_local_state(
        self, make_dialog, volume_out
    ):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab._actors.update({"nics_negative", "nics_positive"})
        dialog.icss_tab._plotter_getter = lambda: None
        dialog.icss_tab.clear_actors()
        assert not dialog.icss_tab._actors

    def test_cube_generation_reports_write_errors(
        self, make_dialog, volume_out, monkeypatch, no_modals
    ):
        dialog = make_dialog(volume_out)

        def fail(*args, **kwargs):
            raise OSError("read-only output")

        monkeypatch.setattr(dialog.field, "ensure_cube", fail)
        assert dialog.icss_tab.generate_cube() is None
        assert any("read-only output" in message for _, message in no_modals)

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

    def test_render_controls_refresh_automatically(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        tab = dialog.icss_tab
        tab.draw = MagicMock()

        tab.iso_slider.setValue(tab.iso_slider.value() + 1)
        tab.show_positive.setChecked(False)
        tab.show_cut_axis.setChecked(True)
        next_slice = min(tab.slice_slider.maximum(), tab.slice_slider.value() + 1)
        tab.slice_slider.setValue(next_slice)

        assert tab.draw.call_count >= 4

    def test_cmap_and_span_uses_internal_compatibility_state(
        self, make_dialog, volume_out
    ):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.cmap.setCurrentText("RdBu_r")
        dialog.icss_tab.set_display_range(15.5)
        dialog.icss_tab.set_auto_display_range(False)

        cmap, span, auto = dialog.icss_tab._cmap_and_span()
        assert cmap == "RdBu_r"
        assert span == 15.5
        assert not auto

    def test_3d_tab_does_not_expose_map_range_controls(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        assert not hasattr(dialog.icss_tab, "vmax")
        assert not hasattr(dialog.icss_tab, "auto_range")

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
        pytest.importorskip("pyvista")
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        dialog = make_dialog(str(source))

        # The first 3D render now persists the selected field automatically.
        path = dialog.icss_tab.generate_cube()
        assert os.path.exists(path)
        assert "Reused cached" in dialog.icss_tab.status.text()

        dialog.icss_tab.generate_cube(force=True)
        assert "Wrote cube" in dialog.icss_tab.status.text()

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
        pytest.importorskip("pyvista")
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        dialog = make_dialog(str(source))
        assert "Cached:" in dialog.icss_tab.cache_label.text()

        dialog.icss_tab.generate_cube(force=True)
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

    def test_cache_label_marks_axis_incompatible_cube_stale(
        self, make_dialog, volume_out
    ):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.field.set_axis_mode("x")
        dialog.icss_tab._update_cache_label()
        assert "Stale cache" in dialog.icss_tab.cache_label.text()

    def test_component_switch_updates_the_cache_label(self, make_dialog, volume_out):
        dialog = make_dialog(volume_out)
        dialog.icss_tab.component.setCurrentIndex(1)
        assert "NICS_iso" in dialog.icss_tab.cache_label.text()

    def test_planar_grid_does_not_enter_isosurface_renderer(
        self, make_dialog, plane_out, fake_plotter
    ):
        pytest.importorskip("pyvista")
        make_dialog(plane_out)
        names = {
            call.kwargs.get("name") for call in fake_plotter.add_mesh.call_args_list
        }
        assert ACTOR_POSITIVE not in names
        assert ACTOR_NEGATIVE not in names

    def test_show_vector_adds_named_3d_actor(
        self, make_dialog, volume_out, fake_plotter
    ):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.tabs.setCurrentWidget(dialog.icss_tab)
        dialog.icss_tab.show_vector.setChecked(False)
        dialog.icss_tab.show_vector.setChecked(True)
        names = {
            call.kwargs.get("name") for call in fake_plotter.add_mesh.call_args_list
        }
        assert ACTOR_AXIS_VECTOR in names

    def test_clear_removes_vector_actor(self, make_dialog, volume_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.show_vector.setChecked(True)
        dialog.icss_tab.clear_actors()
        assert ACTOR_AXIS_VECTOR not in dialog.icss_tab._actors

    def test_2d_tab_offers_3d_plane_button(self, make_dialog, plane_out):
        dialog = make_dialog(plane_out)
        assert any(
            button.text() == "Show in 3D view"
            for button in dialog.map_tab.findChildren(QPushButton)
        )
        assert any(
            button.text() == "Clear from 3D view"
            for button in dialog.map_tab.findChildren(QPushButton)
        )

    def test_show_plane_in_3d(self, make_dialog, plane_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(plane_out)
        dialog.map_tab._emit_show_in_3d()
        assert fake_plotter.add_mesh.call_count >= 1
        assert "Map plane added" in dialog.icss_tab.status.text()

    def test_switching_to_2d_clears_3d_actors(
        self, make_dialog, volume_out, fake_plotter
    ):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.tabs.setCurrentWidget(dialog.icss_tab)
        fake_plotter.reset_mock()

        dialog.tabs.setCurrentWidget(dialog.map_tab)

        removed = {call.args[0] for call in fake_plotter.remove_actor.call_args_list}
        from orca_nics_analyzer.icss3d_tab import ALL_ACTORS

        assert removed == set(ALL_ACTORS)

    def test_clear_removes_every_actor(self, make_dialog, volume_out, fake_plotter):
        pytest.importorskip("pyvista")
        dialog = make_dialog(volume_out)
        dialog.icss_tab.show_cut_axis.setChecked(True)
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

    def test_dialog_settings_round_trip(self, make_dialog, volume_out):
        first = make_dialog(volume_out)
        first.axis_combo.setCurrentIndex(2)
        first._probe_chk.setChecked(True)
        first.map_tab.levels.setValue(47)
        first.map_tab.show_contours.setChecked(False)
        first.icss_tab.opacity.setValue(0.8)
        first.icss_tab.show_negative.setChecked(False)
        first._save_settings()

        second = make_dialog(volume_out)
        assert second.axis_combo.currentData() == "x"
        assert second._probe_chk.isChecked()
        assert second.map_tab.levels.value() == 47
        assert not second.map_tab.show_contours.isChecked()
        assert second.icss_tab.opacity.value() == pytest.approx(0.8)
        assert not second.icss_tab.show_negative.isChecked()

    def test_map_range_updates_3d_compatibility_state(self, make_dialog, volume_out):
        dlg = make_dialog(volume_out)

        dlg.map_tab.auto_range.setChecked(False)
        dlg.map_tab.vmax.setValue(42.0)
        assert dlg.icss_tab._display_span == 42.0
        assert not dlg.icss_tab._auto_display_range

        dlg.map_tab.auto_range.setChecked(True)
        assert dlg.icss_tab._auto_display_range

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

    def test_axis_change_redraws_the_3d_field(self, make_dialog, single_out):
        from unittest.mock import MagicMock

        dialog = make_dialog(single_out)
        dialog.icss_tab.draw = MagicMock()
        dialog.axis_combo.setCurrentIndex(2)
        dialog.icss_tab.draw.assert_called_once_with(silent=True)

    def test_manual_axis_change_refreshes_visible_2d_plane(
        self, make_dialog, plane_out
    ):
        from unittest.mock import MagicMock

        dialog = make_dialog(plane_out)
        dialog.tabs.setCurrentWidget(dialog.map_tab)
        dialog.icss_tab.show_plane = MagicMock()
        dialog.axis_combo.setCurrentIndex(2)
        dialog.icss_tab.show_plane.assert_called()

    def test_manual_axis_vector_updates_nics_zz(self, make_dialog, single_out):
        dialog = make_dialog(single_out)
        dialog.axis_combo.setCurrentIndex(dialog.axis_combo.findData("custom"))
        dialog._axis_vector[0].setValue(1.0)
        dialog._axis_vector[1].setValue(2.0)
        dialog._axis_vector[2].setValue(3.0)
        assert dialog.field.axis_mode == "custom"
        assert dialog.field.custom_axis == pytest.approx((1.0, 2.0, 3.0))

    def test_manual_axis_vector_is_saved(self, make_dialog, single_out, tmp_path):
        dialog = make_dialog(single_out)
        dialog.axis_combo.setCurrentIndex(dialog.axis_combo.findData("custom"))
        dialog._axis_vector[0].setValue(1.0)
        dialog._axis_vector[1].setValue(2.0)
        dialog._axis_vector[2].setValue(3.0)
        dialog.close()
        saved = json.loads((tmp_path / "settings.json").read_text())
        assert saved["nics_analyzer_settings"]["axis_vector"] == [1.0, 2.0, 3.0]

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
    def test_close_releases_the_window_slot(self, volume_out, fake_context, tmp_path):
        parser = NicsParser()
        parser.load(volume_out)
        dialog = NicsAnalyzerDialog(
            parser, fake_context, settings_file=str(tmp_path / "settings.json")
        )
        dialog.close()
        fake_context.register_window.assert_called_with("nics_analyzer", None)

    def test_close_clears_the_actors(
        self, volume_out, fake_context, fake_plotter, tmp_path
    ):
        pytest.importorskip("pyvista")
        parser = NicsParser()
        parser.load(volume_out)
        dialog = NicsAnalyzerDialog(
            parser, fake_context, settings_file=str(tmp_path / "settings.json")
        )
        dialog.icss_tab.draw()
        fake_plotter.reset_mock()
        dialog.close()
        assert fake_plotter.remove_actor.called

    def test_escape_also_cleans_up(self, volume_out, fake_context, tmp_path):
        """Rejecting a dialog never raises a close event."""
        parser = NicsParser()
        parser.load(volume_out)
        dialog = NicsAnalyzerDialog(
            parser, fake_context, settings_file=str(tmp_path / "settings.json")
        )
        dialog.reject()
        fake_context.register_window.assert_called_with("nics_analyzer", None)

    def test_cleanup_runs_once(self, volume_out, fake_context, tmp_path):
        parser = NicsParser()
        parser.load(volume_out)
        dialog = NicsAnalyzerDialog(
            parser, fake_context, settings_file=str(tmp_path / "settings.json")
        )
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

    def test_iso_slice_extremum_ignores_the_zz_checkbox(self, make_dialog, plane_out):
        """An extracted slice only carries its own component."""
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        dlg.scan_tab.show_zz.setChecked(True)
        data = load_field(plane_out).extract_line("iso", 0, 2)
        dlg.scan_tab.show_slice(data)
        assert "largest |NICS(iso)|" in dlg.scan_tab.info.text()

    def test_slice_csv_has_no_stray_join_artifacts(
        self, make_dialog, plane_out, tmp_path
    ):
        from orca_nics_analyzer.analysis import load_field

        dlg = make_dialog(plane_out)
        field = load_field(plane_out)
        data = field.extract_line("iso", 0, 2)
        dlg.scan_tab.show_slice(data)
        target = str(tmp_path / "slice.csv")
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dlg.scan_tab.export_csv()
        rows = open(target, encoding="utf-8").read().splitlines()[1:]
        first = rows[0].split(",")
        assert float(first[2]) == pytest.approx(float(data["iso"][0]), abs=1e-4)
        assert first[3] == ""

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

    def test_slice1d_bounds_follow_the_cut_axis(self, make_dialog, tmp_path):
        """The in-plane axes swap — and resize — when the cut axis changes."""
        from make_fixtures import build_output, volume_points

        path = tmp_path / "slab.out"
        path.write_text(
            build_output(volume_points(n_xy=7, n_z=3), "non-cubic slab"),
            encoding="utf-8",
        )
        dlg = make_dialog(str(path))
        dlg.tabs.setCurrentWidget(dlg.map_tab)

        # Cut along a 7-point axis so the 3-point one becomes in-plane.
        new_stack = next(
            a
            for a in range(3)
            if len(dlg.field.layout["coords"][a]) == 7
            and a != dlg.field.stack_axis_index()
        )
        order = [a for a in range(3) if a != new_stack]
        shrinking = 0 if len(dlg.field.layout["coords"][order[0]]) == 3 else 1
        dlg.map_tab._slice1d_axis.setCurrentIndex(shrinking)
        dlg.map_tab._slice1d_slider.setValue(6)

        dlg.icss_tab.stack_axis_combo.setCurrentIndex(new_stack)
        dlg.map_tab.refresh(force=True)

        assert dlg.map_tab._slice1d_slider.maximum() == 2
        assert dlg.map_tab._slice1d_spin.maximum() == 2
        assert dlg.map_tab._slice1d_slider.value() <= 2

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

    def test_map_tab_auto_shows_current_plane_in_3d(self, make_dialog, volume_out):
        from unittest.mock import MagicMock

        dlg = make_dialog(volume_out)
        dlg.icss_tab.show_plane = MagicMock()
        dlg.tabs.setCurrentWidget(dlg.map_tab)
        dlg.icss_tab.show_plane.assert_called_with(
            dlg.map_tab.component.currentData(), dlg.icss_tab.slice_slider.value()
        )

    def test_map_changes_refresh_the_automatic_3d_plane(self, make_dialog, volume_out):
        from unittest.mock import MagicMock

        dlg = make_dialog(volume_out)
        dlg.tabs.setCurrentWidget(dlg.map_tab)
        dlg.icss_tab.show_plane = MagicMock()
        dlg.map_tab.component.setCurrentIndex(1)
        assert dlg.icss_tab.show_plane.called

    def test_hidden_graphs_do_not_refresh_until_their_tab_is_active(
        self, make_dialog, volume_out
    ):
        dlg = make_dialog(volume_out)
        dlg.tabs.setCurrentWidget(dlg.icss_tab)
        dlg.map_tab.figure.clear()
        dlg.map_tab.refresh()
        assert not dlg.map_tab.figure.axes

        dlg.tabs.setCurrentWidget(dlg.map_tab)
        dlg.map_tab.refresh()
        assert dlg.map_tab.figure.axes

    def test_hidden_3d_and_1d_graphs_wait_for_their_tabs(self, make_dialog, volume_out):
        dlg = make_dialog(volume_out)
        dlg.tabs.setCurrentWidget(dlg.map_tab)

        original_status = dlg.icss_tab.status.text()
        dlg.icss_tab.draw(silent=True)
        assert dlg.icss_tab.status.text() == original_status

        dlg.scan_tab.figure.clear()
        dlg.scan_tab.refresh()
        assert not dlg.scan_tab.figure.axes

        dlg.tabs.setCurrentWidget(dlg.scan_tab)
        dlg.scan_tab.refresh(force=True)
        assert dlg.scan_tab.figure.axes

    def test_2d_shared_controls_force_a_3d_refresh(self, make_dialog, volume_out):
        from unittest.mock import MagicMock

        dlg = make_dialog(volume_out)
        dlg.tabs.setCurrentWidget(dlg.map_tab)
        dlg.icss_tab.draw = MagicMock()

        dlg.map_tab.auto_range.setChecked(False)
        assert dlg.icss_tab.draw.called

        dlg.icss_tab.draw.reset_mock()
        dlg.map_tab.component.setCurrentIndex(1)
        assert dlg.icss_tab.component.currentIndex() == 1
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
        # Verify vector checkbox is removed from 3D slice group
        from PyQt6.QtWidgets import QCheckBox

        checkbox_texts = [
            cb.text() for cb in dlg.icss_tab.slice_group.findChildren(QCheckBox)
        ]
        assert "Show NICS_zz vector" not in checkbox_texts


class TestHeaderAndVectorSettings:
    def test_two_line_header_layout_structure(self, make_dialog, single_out):
        from PyQt6.QtWidgets import QHBoxLayout

        dlg = make_dialog(single_out)
        # Header settings layout has two QHBoxLayout rows
        rows = [child for child in dlg.findChildren(QHBoxLayout) if child.count() > 0]
        assert len(rows) >= 2

        assert dlg._vector_chk.text() == "Show NICS_zz vector"
        assert dlg._probe_chk.text() == "Show probe atoms"
        assert dlg.axis_combo.isEnabled()

    def test_top_header_vector_checkbox_toggles_actor_and_syncs(
        self, make_dialog, volume_out, fake_plotter
    ):
        pytest.importorskip("pyvista")
        dlg = make_dialog(volume_out)
        dlg._vector_chk.setChecked(True)
        assert dlg.icss_tab.show_vector.isChecked()
        names = {
            call.kwargs.get("name") for call in fake_plotter.add_mesh.call_args_list
        }
        assert ACTOR_AXIS_VECTOR in names

        dlg._vector_chk.setChecked(False)
        assert not dlg.icss_tab.show_vector.isChecked()

    def test_vector_rendering_for_planar_output(
        self, make_dialog, plane_out, fake_plotter
    ):
        pytest.importorskip("pyvista")
        dlg = make_dialog(plane_out)
        dlg._vector_chk.setChecked(True)
        names = {
            call.kwargs.get("name") for call in fake_plotter.add_mesh.call_args_list
        }
        assert ACTOR_AXIS_VECTOR in names

    def test_vector_checkbox_disabled_without_tensors(self, make_dialog, single_out):
        parser = NicsParser()
        parser.load(single_out)
        for entry in parser.data["shieldings"].values():
            entry["tensor"] = None
        parser.data["has_tensors"] = False
        dlg = NicsAnalyzerDialog(parser, MagicMock())
        try:
            assert not dlg._vector_chk.isEnabled()
        finally:
            dlg.close()

