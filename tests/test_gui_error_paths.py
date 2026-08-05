"""Export targets that cannot be written, cancelled dialogs, refused layouts.

Every save path here goes through a real Qt widget with the file chooser
patched, so the guard clauses and the except branches are exercised rather
than the happy path.
"""

import os
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PyQt6.QtWidgets")

from orca_nics_analyzer.analysis import load_field  # noqa: E402
from orca_nics_analyzer.gui import NicsAnalyzerDialog  # noqa: E402
from orca_nics_analyzer.parser import NicsParser  # noqa: E402

pytestmark = pytest.mark.usefixtures("qapp", "no_modals")


def _dialog(path, fake_context, tmp_path):
    parser = NicsParser()
    parser.load(path)
    return NicsAnalyzerDialog(
        parser, fake_context, settings_file=str(tmp_path / "settings.json")
    )


@pytest.fixture
def dialog(fake_context, tmp_path, plane_out):
    dlg = _dialog(plane_out, fake_context, tmp_path)
    yield dlg
    dlg.close()


@pytest.fixture
def line_dialog(fake_context, tmp_path, single_out):
    """The single/line fixture: no grid, so cube and map exports are refused."""
    dlg = _dialog(single_out, fake_context, tmp_path)
    yield dlg
    dlg.close()


def _kinds(no_modals):
    return [kind for kind, _ in no_modals]


def _texts(no_modals):
    return [text for _, text in no_modals]


def _unwritable(tmp_path, name):
    """A path whose parent directory does not exist, so opening it raises."""
    return str(tmp_path / "no_such_dir" / name)


class TestMapExportFailures:
    def test_csv_reports_an_unwritable_target(self, dialog, tmp_path, no_modals):
        target = _unwritable(tmp_path, "map.csv")
        with patch(
            "orca_nics_analyzer.map2d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.map_tab.export_csv()
        assert "critical" in _kinds(no_modals)
        assert any("Could not write the file" in t for t in _texts(no_modals))
        assert not os.path.exists(target)

    def test_image_reports_an_unwritable_target(self, dialog, tmp_path, no_modals):
        target = _unwritable(tmp_path, "map.png")
        with patch(
            "orca_nics_analyzer.map2d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.map_tab.export_png()
        assert "critical" in _kinds(no_modals)
        assert any("Could not write the image" in t for t in _texts(no_modals))

    def test_cancelling_writes_nothing(self, dialog, no_modals):
        with patch(
            "orca_nics_analyzer.map2d_tab.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ) as chooser:
            dialog.map_tab.export_csv()
            dialog.map_tab.export_png()
        assert chooser.call_count == 2
        assert no_modals == []

    def test_csv_is_refused_without_a_grid(self, line_dialog, no_modals):
        with patch(
            "orca_nics_analyzer.map2d_tab.QFileDialog.getSaveFileName"
        ) as chooser:
            line_dialog.map_tab.export_csv()
        assert not chooser.called  # refused before asking for a filename
        assert any("no grid" in str(call) for call in no_modals)

    def test_a_written_csv_round_trips(self, dialog, tmp_path):
        target = str(tmp_path / "map.csv")
        with patch(
            "orca_nics_analyzer.map2d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.map_tab.export_csv()
        rows = open(target, encoding="utf-8").read().splitlines()
        info = dialog.field.plane_data(dialog.map_tab._component())
        assert len(rows) == len(info["a2"]) + 1
        assert rows[0].startswith("axis2\\axis1,")


class TestScanExportFailures:
    def _inject_slice(self, dialog):
        data = load_field(dialog.field.filename).extract_line("iso", 0, 0)
        dialog.scan_tab.show_slice(data)

    def test_slice_csv_reports_an_unwritable_target(self, dialog, tmp_path, no_modals):
        self._inject_slice(dialog)
        target = _unwritable(tmp_path, "scan.csv")
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.scan_tab.export_csv()
        assert "critical" in _kinds(no_modals)
        assert any("Could not write the file" in t for t in _texts(no_modals))

    def test_native_csv_is_refused_when_the_probes_are_not_a_line(
        self, dialog, tmp_path, no_modals
    ):
        target = str(tmp_path / "scan.csv")
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.scan_tab.export_csv()
        assert "critical" in _kinds(no_modals)
        assert any("not a line" in t for t in _texts(no_modals))
        assert not os.path.exists(target)

    def test_image_reports_an_unwritable_target(self, dialog, tmp_path, no_modals):
        target = _unwritable(tmp_path, "scan.png")
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.scan_tab.export_png()
        assert "critical" in _kinds(no_modals)
        assert any("Could not write the image" in t for t in _texts(no_modals))

    def test_cancelling_writes_nothing(self, dialog, no_modals):
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ) as chooser:
            dialog.scan_tab.export_csv()
            dialog.scan_tab.export_png()
        assert chooser.call_count == 2
        assert no_modals == []

    def test_a_native_scan_csv_is_written(self, line_dialog, tmp_path):
        target = str(tmp_path / "scan.csv")
        with patch(
            "orca_nics_analyzer.scan1d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            line_dialog.scan_tab.export_csv()
        assert os.path.getsize(target) > 0


class TestCubeExportFailures:
    def test_save_as_reports_an_unwritable_target(self, dialog, tmp_path, no_modals):
        # write_cube creates missing parent directories, so the target has to
        # be something that cannot be opened at all.
        target = tmp_path / "field.cube"
        target.mkdir()
        with patch(
            "orca_nics_analyzer.icss3d_tab.QFileDialog.getSaveFileName",
            return_value=(str(target), ""),
        ):
            dialog.icss_tab.save_cube_as()
        assert "critical" in _kinds(no_modals)
        assert target.is_dir()  # untouched

    def test_save_as_writes_the_cube_it_was_given(self, dialog, tmp_path):
        target = str(tmp_path / "deep" / "field.cube")
        with patch(
            "orca_nics_analyzer.icss3d_tab.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            dialog.icss_tab.save_cube_as()
        assert os.path.getsize(target) > 0
        assert "Wrote cube" in dialog.icss_tab.status.text()

    def test_cancelling_save_as_writes_nothing(self, dialog, no_modals):
        with patch(
            "orca_nics_analyzer.icss3d_tab.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ) as chooser:
            dialog.icss_tab.save_cube_as()
        assert chooser.called
        assert no_modals == []

    def test_generate_is_refused_without_a_grid(self, line_dialog, no_modals):
        assert line_dialog.icss_tab.generate_cube() is None
        assert any("regular grid" in str(call) for call in no_modals)

    def test_save_as_is_refused_without_a_grid(self, line_dialog, no_modals):
        with patch(
            "orca_nics_analyzer.icss3d_tab.QFileDialog.getSaveFileName"
        ) as chooser:
            line_dialog.icss_tab.save_cube_as()
        assert not chooser.called
        assert any("regular grid" in str(call) for call in no_modals)

    def test_a_reported_failure_reaches_the_user(self, dialog, no_modals):
        """A cube the field itself refuses to write must not fail silently."""
        with patch.object(
            dialog.field, "ensure_cube", side_effect=OSError("disk full")
        ):
            assert dialog.icss_tab.generate_cube() is None
        assert any("disk full" in str(call) for call in no_modals)


class TestExportAllFailures:
    def test_a_file_where_the_folder_should_be_is_reported(
        self, dialog, tmp_path, no_modals
    ):
        blocker = tmp_path / "not_a_folder"
        blocker.write_text("", encoding="utf-8")
        with patch(
            "orca_nics_analyzer.gui.QFileDialog.getExistingDirectory",
            return_value=str(blocker),
        ):
            dialog.export_all()
        assert "critical" in _kinds(no_modals)

    def test_cancelling_writes_nothing(self, dialog, tmp_path, no_modals):
        with patch(
            "orca_nics_analyzer.gui.QFileDialog.getExistingDirectory", return_value=""
        ):
            dialog.export_all()
        assert no_modals == []
        assert not list(tmp_path.glob("*.csv"))

    def test_a_successful_export_lists_what_it_wrote(self, dialog, tmp_path, no_modals):
        folder = tmp_path / "out"
        with patch(
            "orca_nics_analyzer.gui.QFileDialog.getExistingDirectory",
            return_value=str(folder),
        ):
            dialog.export_all()
        assert any("file(s) to" in t for t in _texts(no_modals))
        assert list(folder.iterdir())


class TestSliceCrosshair:
    def test_it_marks_the_row_the_slice_will_cut(self, dialog):
        dialog.tabs.setCurrentWidget(dialog.map_tab)
        dialog.map_tab._slice1d_axis.setCurrentIndex(0)
        dialog.map_tab._slice1d_slider.setValue(2)
        dialog.map_tab.show_1d_line.setChecked(True)
        dialog.map_tab.refresh(force=True)

        info = dialog.field.plane_data(dialog.map_tab._component())
        ax = dialog.map_tab.figure.axes[0]
        verticals = [
            line.get_xdata()[0]
            for line in ax.lines
            if line.get_linestyle() == "--" and len(set(line.get_xdata())) == 1
        ]
        assert any(abs(x - info["a1"][2]) < 1e-6 for x in verticals)

    def test_the_other_axis_cuts_the_other_way(self, dialog):
        dialog.tabs.setCurrentWidget(dialog.map_tab)
        dialog.map_tab._slice1d_axis.setCurrentIndex(1)
        dialog.map_tab._slice1d_slider.setValue(1)
        dialog.map_tab.show_1d_line.setChecked(True)
        dialog.map_tab.refresh(force=True)

        info = dialog.field.plane_data(dialog.map_tab._component())
        ax = dialog.map_tab.figure.axes[0]
        horizontals = [
            line.get_ydata()[0]
            for line in ax.lines
            if line.get_linestyle() == "--" and len(set(line.get_ydata())) == 1
        ]
        assert any(abs(y - info["a2"][1]) < 1e-6 for y in horizontals)

    def test_nothing_is_drawn_when_the_option_is_off(self, dialog):
        dialog.tabs.setCurrentWidget(dialog.map_tab)
        dialog.map_tab.show_1d_line.setChecked(False)
        dialog.map_tab.refresh(force=True)
        ax = dialog.map_tab.figure.axes[0]
        assert [line for line in ax.lines if line.get_linestyle() == "--"] == []
