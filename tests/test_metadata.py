"""Plugin metadata contract — the registry scripts read these names."""

import os
import re

import pytest

import orca_nics_analyzer as plugin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestMetadata:
    @pytest.mark.parametrize(
        "name",
        [
            "PLUGIN_NAME",
            "PLUGIN_VERSION",
            "PLUGIN_AUTHOR",
            "PLUGIN_DESCRIPTION",
            "PLUGIN_CATEGORY",
            "PLUGIN_TAGS",
            "PLUGIN_DEPENDENCIES",
            "PLUGIN_SUPPORTED_MOLEDITPY_VERSION",
        ],
    )
    def test_present(self, name):
        assert getattr(plugin, name, None) not in (None, "")

    def test_version_is_semver(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", plugin.PLUGIN_VERSION)

    def test_tags_and_dependencies_are_lists_of_strings(self):
        for value in (plugin.PLUGIN_TAGS, plugin.PLUGIN_DEPENDENCIES):
            assert isinstance(value, list)
            assert all(isinstance(v, str) and v for v in value)

    def test_declares_the_optional_dependencies_it_imports(self):
        for name in ("numpy", "matplotlib", "pyvista"):
            assert name in plugin.PLUGIN_DEPENDENCIES

    def test_supported_host_range(self):
        assert plugin.PLUGIN_SUPPORTED_MOLEDITPY_VERSION == ">=4.0.0, <5.0.0"

    def test_entry_point_exists(self):
        assert callable(plugin.initialize)

    def test_version_matches_the_release_workflow(self):
        """release.yml checks the tag against this file; it must read this file."""
        path = os.path.join(ROOT, ".github", "workflows", "release.yml")
        text = open(path, encoding="utf-8").read()
        assert "orca_nics_analyzer/__init__.py" in text


class TestInitialize:
    def test_registers_a_menu_action(self):
        from unittest.mock import MagicMock

        context = MagicMock()
        plugin.initialize(context)
        paths = [c.args[0] for c in context.add_menu_action.call_args_list]
        assert paths == ["Analysis/ORCA NICS Analyzer..."]

    def test_menu_lands_in_a_menu_the_host_already_has(self):
        """The host creates a new top-level menu for an unknown name.

        It matches on the title with '&' stripped, so "Analysis" joins the
        native &Analysis menu instead of adding a second one beside it.
        """
        from unittest.mock import MagicMock

        context = MagicMock()
        plugin.initialize(context)
        top_level = context.add_menu_action.call_args.args[0].split("/")[0]
        assert top_level in {"File", "Edit", "View", "Analysis", "Plugin", "Settings"}

    def test_menu_path_has_no_extra_separator(self):
        """The host splits on '/', so a slash in the leaf would nest a submenu."""
        from unittest.mock import MagicMock

        context = MagicMock()
        plugin.initialize(context)
        path = context.add_menu_action.call_args.args[0]
        assert path.count("/") == 1

    def test_registers_a_document_reset_handler(self):
        from unittest.mock import MagicMock

        context = MagicMock()
        plugin.initialize(context)
        assert context.register_document_reset_handler.called

    def test_reset_closes_an_open_window(self):
        from unittest.mock import MagicMock

        context = MagicMock()
        plugin.initialize(context)
        handler = context.register_document_reset_handler.call_args.args[0]
        window = MagicMock()
        context.get_window.return_value = window
        handler()
        window.close.assert_called_once()

    def test_reset_without_a_window_is_a_no_op(self):
        from unittest.mock import MagicMock

        context = MagicMock()
        plugin.initialize(context)
        handler = context.register_document_reset_handler.call_args.args[0]
        context.get_window.return_value = None
        handler()  # must not raise


class TestNoHeavyImportsAtModuleLoad:
    def test_package_root_imports_without_numpy_or_qt(self):
        """The host imports every plugin at startup, before optional deps load."""
        import ast

        path = os.path.join(ROOT, "orca_nics_analyzer", "__init__.py")
        tree = ast.parse(open(path, encoding="utf-8").read())
        top_level = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.append(node.module.split(".")[0])
        assert set(top_level) <= {"logging", "os"}
