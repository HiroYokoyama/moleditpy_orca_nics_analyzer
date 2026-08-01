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

    def test_a_previous_window_is_reused(self, opened, volume_out, fake_context):
        previous = MagicMock()
        fake_context.get_window.return_value = previous
        opened(volume_out)
        previous.load_parser.assert_called_once()

    def test_a_previous_window_that_fails_to_load_is_survived(
        self, opened, volume_out, fake_context
    ):
        """A window from an earlier document may already be a dead C++ object."""
        previous = MagicMock()
        previous.load_parser.side_effect = RuntimeError("already deleted")
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

    def test_menu_action_opens_empty_dialog(self, fake_context):
        callback = self._callback(fake_context)
        with patch("orca_nics_analyzer.gui.NicsAnalyzerDialog.show"):
            callback()
        fake_context.register_window.assert_called_once()
        assert fake_context.register_window.call_args.args[0] == "nics_analyzer"

    def test_menu_action_activates_existing_dialog(self, fake_context):
        callback = self._callback(fake_context)
        previous = MagicMock()
        fake_context.get_window.return_value = previous
        callback()
        previous.raise_.assert_called_once()
        previous.activateWindow.assert_called_once()

