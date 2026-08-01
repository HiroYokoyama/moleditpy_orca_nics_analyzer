"""The open-a-file entry path: encodings, guards and window handling."""

from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PyQt6.QtWidgets")

import orca_nics_analyzer as plugin  # noqa: E402

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture
def opened(fake_context):
    """Open a file through the plugin and yield the dialog that was registered."""
    created = []

    def _open(path):
        with patch("orca_nics_analyzer.gui.NicsAnalyzerDialog.show"):
            plugin._open_file(path, fake_context)
        for call in fake_context.register_window.call_args_list:
            if call.args[1] is not None:
                created.append(call.args[1])
        return created[-1] if created else None

    yield _open
    for dialog in created:
        dialog.close()


class TestReadFile:
    def test_reads_utf8(self, tmp_path):
        path = tmp_path / "a.out"
        path.write_text("hello", encoding="utf-8")
        assert plugin._read_output_file(str(path), None) == "hello"

    def test_reads_utf16(self, tmp_path):
        path = tmp_path / "a.out"
        path.write_text("hello", encoding="utf-16")
        assert "hello" in plugin._read_output_file(str(path), None)

    def test_undecodable_bytes_do_not_raise(self, tmp_path):
        path = tmp_path / "a.out"
        path.write_bytes(b"ok \xff\xfe\x00 more")
        assert plugin._read_output_file(str(path), None) is not None

    def test_missing_file_reports_an_error(self, tmp_path):
        with patch("orca_nics_analyzer._warn") as warn:
            assert plugin._read_output_file(str(tmp_path / "nope.out"), None) is None
        assert warn.called


class TestOpenFile:
    def test_opens_a_dialog_and_registers_it(self, opened, volume_out, fake_context):
        dialog = opened(volume_out)
        assert dialog is not None
        fake_context.register_window.assert_any_call("nics_analyzer", dialog)

    def test_marks_the_plugin_as_used(self, opened, volume_out):
        plugin._dialog_opened = False
        opened(volume_out)
        assert plugin._dialog_opened

    def test_output_without_probes_is_refused(self, no_ghosts_out, fake_context):
        with patch("PyQt6.QtWidgets.QMessageBox.warning") as warn:
            plugin._open_file(no_ghosts_out, fake_context)
        assert warn.called
        assert "No NICS probes" in warn.call_args.args[1]
        fake_context.register_window.assert_not_called()

    def test_a_previous_window_is_closed_first(self, opened, volume_out, fake_context):
        previous = MagicMock()
        fake_context.get_window.return_value = previous
        opened(volume_out)
        previous.close.assert_called_once()

    def test_a_previous_window_that_fails_to_close_is_survived(
        self, opened, volume_out, fake_context
    ):
        """A window from an earlier document may already be a dead C++ object."""
        previous = MagicMock()
        previous.close.side_effect = RuntimeError("already deleted")
        fake_context.get_window.return_value = previous
        assert opened(volume_out) is not None

    def test_unreadable_file_opens_nothing(self, tmp_path, fake_context):
        with patch("orca_nics_analyzer._warn"):
            plugin._open_file(str(tmp_path / "nope.out"), fake_context)
        fake_context.register_window.assert_not_called()


class TestMenuAction:
    def _callback(self, context):
        plugin.initialize(context)
        return context.add_menu_action.call_args.args[1]

    def test_cancelling_the_dialog_opens_nothing(self, fake_context):
        callback = self._callback(fake_context)
        with patch(
            "PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")
        ):
            callback()
        fake_context.register_window.assert_not_called()

    def test_choosing_a_file_opens_it(self, fake_context, volume_out):
        callback = self._callback(fake_context)
        with (
            patch(
                "PyQt6.QtWidgets.QFileDialog.getOpenFileName",
                return_value=(volume_out, ""),
            ),
            patch("orca_nics_analyzer._open_file") as opener,
        ):
            callback()
        opener.assert_called_once()
        assert opener.call_args.args[0] == volume_out

    def test_filter_offers_out_and_log(self, fake_context):
        callback = self._callback(fake_context)
        with patch(
            "PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")
        ) as chooser:
            callback()
        assert "*.out" in chooser.call_args.args[3]
        assert "*.log" in chooser.call_args.args[3]
