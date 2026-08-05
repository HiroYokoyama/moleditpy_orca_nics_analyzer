"""The failure and edge paths: bad input, unwritable targets, cancelled dialogs.

These are the branches a user only meets when something has already gone
wrong, so they are the ones least likely to be exercised by hand.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")

from orca_nics_analyzer import settings as settings_mod  # noqa: E402
from orca_nics_analyzer.analysis import load_field  # noqa: E402
from orca_nics_analyzer.parser import NicsParser  # noqa: E402


# ---------------------------------------------------------------------------
# settings: the atomic save
# ---------------------------------------------------------------------------


class TestSettingsSaveFailures:
    def test_unwritable_target_reports_failure(self, tmp_path):
        """os.replace onto a directory fails after the temp file is written."""
        target = tmp_path / "settings.json"
        target.mkdir()  # a directory where the file should be
        assert (
            settings_mod.save_settings(dict(settings_mod.DEFAULT_SETTINGS), str(target))
            is False
        )

    def test_the_temp_file_is_not_left_behind(self, tmp_path):
        target = tmp_path / "settings.json"
        target.mkdir()
        settings_mod.save_settings(dict(settings_mod.DEFAULT_SETTINGS), str(target))
        assert not (tmp_path / "settings.json.tmp").exists()

    def test_unserialisable_values_report_failure(self, tmp_path):
        target = tmp_path / "settings.json"
        values = dict(settings_mod.DEFAULT_SETTINGS)
        values["map_levels"] = {1, 2, 3}  # a set is not JSON
        assert settings_mod.save_settings(values, str(target)) is False

    def test_a_corrupt_file_falls_back_to_defaults(self, tmp_path):
        target = tmp_path / "settings.json"
        target.write_text("{not json at all", encoding="utf-8")
        loaded = settings_mod.load_settings(str(target))
        assert loaded == settings_mod.DEFAULT_SETTINGS

    def test_a_corrupt_file_is_replaced_not_merged(self, tmp_path):
        target = tmp_path / "settings.json"
        target.write_text("{not json at all", encoding="utf-8")
        assert settings_mod.save_settings(
            dict(settings_mod.DEFAULT_SETTINGS), str(target)
        )
        assert json.loads(target.read_text(encoding="utf-8"))[settings_mod.SETTINGS_KEY]


# ---------------------------------------------------------------------------
# analysis: axis validation and cube-cache rejection
# ---------------------------------------------------------------------------


class TestAxisValidation:
    def test_custom_mode_without_a_vector_is_rejected(self, single_out):
        field = load_field(single_out)
        with pytest.raises(ValueError, match="requires a 3-vector"):
            field.set_axis_mode("custom")

    def test_a_zero_custom_axis_is_rejected(self, single_out):
        field = load_field(single_out)
        with pytest.raises(ValueError, match="non-zero"):
            field.set_axis_mode("custom", [0.0, 0.0, 0.0])

    def test_a_wrong_length_custom_axis_is_rejected(self, single_out):
        field = load_field(single_out)
        with pytest.raises(ValueError, match="finite 3-vector"):
            field.set_axis_mode("custom", [1.0, 0.0])

    def test_a_rejected_axis_leaves_the_field_untouched(self, single_out):
        """Validation runs before any state changes."""
        field = load_field(single_out)
        before = field.values("zz").copy()
        with pytest.raises(ValueError):
            field.set_axis_mode("custom", [0.0, 0.0, 0.0])
        assert field.axis_mode == "grid"
        assert np.allclose(field.values("zz"), before)


class TestCubeCacheRejection:
    def _written(self, tmp_path, volume_out, **kwargs):
        import shutil

        source = tmp_path / "v.out"
        shutil.copy(volume_out, source)
        field = load_field(str(source), **kwargs)
        field.ensure_cube("zz", plugin_version="1.2.3")
        return field, source

    def test_a_fresh_cube_is_reused(self, tmp_path, volume_out):
        field, _ = self._written(tmp_path, volume_out)
        assert field.cached_cube("zz") is not None

    def test_an_edited_source_invalidates_the_cube(self, tmp_path, volume_out):
        field, source = self._written(tmp_path, volume_out)
        with open(source, "a", encoding="utf-8") as handle:
            handle.write("\n# touched\n")
        assert field.cached_cube("zz") is None

    def test_a_vanished_source_invalidates_the_cube(self, tmp_path, volume_out):
        field, source = self._written(tmp_path, volume_out)
        os.remove(source)
        assert field.cached_cube("zz") is None

    def test_a_different_axis_invalidates_the_cube(self, tmp_path, volume_out):
        field, _ = self._written(tmp_path, volume_out)
        assert field.cached_cube("zz") is not None
        field.set_axis_mode("x")
        assert field.cached_cube("zz") is None

    def test_ring_mode_has_no_single_cube_axis(self, tmp_path, volume_out):
        """Each probe may use its own nearest-ring normal."""
        field, _ = self._written(tmp_path, volume_out, axis_mode="ring")
        assert field._cube_axis("zz") is None

    def test_the_iso_component_has_no_axis(self, tmp_path, volume_out):
        field, _ = self._written(tmp_path, volume_out)
        assert field._cube_axis("iso") is None

    def test_an_unstamped_cube_is_not_reused_for_a_rotated_axis(
        self, tmp_path, volume_out
    ):
        """An old stamp cannot prove which direction it was projected onto.

        Only ``axis_mode`` is stripped: the source fingerprint has to survive,
        or the cube would be rejected by the earlier staleness check instead
        and this branch would never run.
        """
        field, _ = self._written(tmp_path, volume_out)
        path = field.cube_path("zz")
        lines = open(path, encoding="utf-8").read().splitlines()
        stripped = " ".join(
            part
            for part in lines[1].split()
            if not part.startswith(("axis_mode=", "axis="))
        )
        assert "source_size=" in stripped and "source_mtime_ns=" in stripped
        lines[1] = stripped
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")

        from orca_nics_analyzer import cube_io

        info = cube_io.read_generation_settings(path)
        assert info["axis_mode"] is None and info["axis"] is None
        assert info["source_size"] is not None  # the earlier check still passes

        field.set_axis_mode("z")
        assert field.cached_cube("zz") is None

    def test_a_cube_with_no_recorded_axis_is_never_reused(self, tmp_path, volume_out):
        """Even in grid mode: an absent axis cannot be checked, only assumed."""
        field, _ = self._written(tmp_path, volume_out)
        path = field.cube_path("zz")
        lines = open(path, encoding="utf-8").read().splitlines()
        lines[1] = " ".join(
            part for part in lines[1].split() if not part.startswith("axis=")
        )
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        assert field.axis_mode == "grid"
        assert field.cached_cube("zz") is None


# ---------------------------------------------------------------------------
# entry point: unreadable files and reset handling
# ---------------------------------------------------------------------------


class TestEntryPointFailures:
    def test_an_unreadable_file_warns_and_returns_none(self, tmp_path):
        pytest.importorskip("PyQt6.QtWidgets")
        from orca_nics_analyzer import _read_output_file

        missing = str(tmp_path / "nope.out")
        with patch("PyQt6.QtWidgets.QMessageBox.critical") as critical:
            assert _read_output_file(missing, None) is None
        assert critical.called

    def test_a_directory_is_not_mistaken_for_an_output(self, tmp_path):
        pytest.importorskip("PyQt6.QtWidgets")
        from orca_nics_analyzer import _read_output_file

        with patch("PyQt6.QtWidgets.QMessageBox.critical") as critical:
            assert _read_output_file(str(tmp_path), None) is None
        assert critical.called

    def test_ghosts_without_shieldings_warn(self):
        pytest.importorskip("PyQt6.QtWidgets")
        from orca_nics_analyzer import _warn_missing_shieldings

        with patch("PyQt6.QtWidgets.QMessageBox.warning") as warning:
            _warn_missing_shieldings(None)
        assert warning.called
        assert "shielding" in warning.call_args.args[2].lower()

    def test_reset_is_skipped_while_the_plugin_is_loading_a_structure(self):
        """The plugin's own show_xyz_data triggers the host's document reset."""
        import orca_nics_analyzer as plugin

        context = MagicMock()
        plugin.initialize(context)
        handler = context.register_document_reset_handler.call_args.args[0]
        window = MagicMock()
        window._loading_structure = True
        context.get_window.return_value = window
        handler()
        window.close.assert_not_called()

    def test_reset_survives_an_already_deleted_window(self):
        import orca_nics_analyzer as plugin

        context = MagicMock()
        plugin.initialize(context)
        handler = context.register_document_reset_handler.call_args.args[0]
        window = MagicMock()
        window._loading_structure = False
        window.close.side_effect = RuntimeError("wrapped C/C++ object deleted")
        context.get_window.return_value = window
        handler()  # must not raise

    def test_opening_survives_an_unraisable_existing_window(self):
        import orca_nics_analyzer as plugin

        context = MagicMock()
        plugin.initialize(context)
        open_dialog = context.add_menu_action.call_args.args[1]
        window = MagicMock()
        window.raise_.side_effect = RuntimeError("wrapped C/C++ object deleted")
        context.get_window.return_value = window
        with patch("orca_nics_analyzer.gui.NicsAnalyzerDialog") as dialog:
            open_dialog()  # falls through to building a fresh dialog
        assert dialog.called


# ---------------------------------------------------------------------------
# parser: malformed output
# ---------------------------------------------------------------------------


class TestParserRobustness:
    def test_non_text_content_is_rejected(self):
        parser = NicsParser()
        with pytest.raises(TypeError, match="text"):
            parser.load_from_memory(b"\x00\x01binary")

    def test_an_unparsable_summary_row_is_skipped(self):
        parser = NicsParser()
        parser.load_from_memory(
            "CHEMICAL SHIELDING SUMMARY (ppm)\n"
            "---\n"
            "  Nucleus  Element    Isotropic     Anisotropy\n"
            "  -------  -------  ------------   ------------\n"
            "      0       C          not-a-number   1.0\n"
            "      1       C          10.0           2.0\n"
            "\n\n"
        )
        # The good row still lands; the bad one is dropped rather than fatal.
        assert 1 in parser.data["shieldings"]
        assert parser.data["shieldings"][1]["iso"] == pytest.approx(10.0)

    def test_fortran_exponents_are_understood(self):
        """ORCA can print 1.0D-01 rather than 1.0E-01."""
        parser = NicsParser()
        parser.load_from_memory(
            "CHEMICAL SHIELDING SUMMARY (ppm)\n"
            "---\n"
            "  Nucleus  Element    Isotropic     Anisotropy\n"
            "  -------  -------  ------------   ------------\n"
            "      0       C          1.0D+01        2.0D+00\n"
            "\n\n"
        )
        assert parser.data["shieldings"][0]["iso"] == pytest.approx(10.0)
        assert parser.data["shieldings"][0]["aniso"] == pytest.approx(2.0)
