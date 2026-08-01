"""
MoleditPy ORCA NICS Analyzer
============================
Reads an ORCA output file containing NMR shielding data on ghost atoms and
turns it into NICS numbers, 2D NICS maps and 3D ICSS isosurfaces.

Everything is derived from the ``.out`` alone: ghost probes are identified by
their zero nuclear charge in the ``CARTESIAN COORDINATES (A.U.)`` block, and
the probe layout (single point / plane / volume) is reconstructed
geometrically, so outputs prepared by any means are understood.
"""

import logging
import os

PLUGIN_NAME = "ORCA NICS Analyzer"
PLUGIN_VERSION = "0.3.9"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Analyze NICS data in ORCA output files: single-probe tables, 2D NICS maps "
    "and 3D ICSS isosurfaces, with Gaussian cube export."
)
PLUGIN_CATEGORY = "Analysis"
PLUGIN_TAGS = ["Analysis", "Visualization"]
PLUGIN_DEPENDENCIES = ["PyQt6", "numpy", "matplotlib", "pyvista"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=4.0.0, <5.0.0"

_context = None
_dialog_opened = False


def _read_output_file(path, parent_widget):
    """Read an ORCA output file, trying the encodings ORCA is seen to emit."""
    for enc in ("utf-8", "utf-16", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as fh:
                return fh.read()
        except UnicodeError:
            continue
        except OSError as e:
            _warn(parent_widget, "Error Reading File", f"Could not read file:\n{e}")
            return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as e:
        _warn(parent_widget, "Error Reading File", f"Could not read file:\n{e}")
        return None


def _warn(parent, title, text):
    from PyQt6.QtWidgets import QMessageBox

    QMessageBox.critical(parent, title, text)


def _open_file(path, context):
    """Parse an ORCA output and load its data into the analyzer window.

    If the window is already open (empty or from a previous file) the existing
    dialog is reused; otherwise a new one is created.
    """
    mw = context.get_main_window()
    content = _read_output_file(path, mw)
    if content is None:
        return

    from .parser import NicsParser

    parser = NicsParser()
    parser.load_from_memory(content, path)

    if not parser.data["ghost_indices"]:
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.warning(
            mw,
            "No NICS probes found",
            "This output has no ghost atoms with NMR shielding data.\n\n"
            "NICS requires ghost centres (e.g. 'H:' or 'Bq') in the geometry "
            "and an NMR job that includes them.",
        )
        return

    existing = context.get_window("nics_analyzer")
    if existing is not None:
        try:
            # Reuse the existing dialog — just swap in the new data.
            existing.load_parser(parser)
            existing.raise_()
            existing.activateWindow()
            return
        except (RuntimeError, AttributeError) as e:
            logging.warning("[orca_nics_analyzer] reusing previous window: %s", e)

    from .gui import NicsAnalyzerDialog

    dlg = NicsAnalyzerDialog(parser, context, parent=mw)
    context.register_window("nics_analyzer", dlg)
    global _dialog_opened
    _dialog_opened = True
    dlg.show()


def initialize(context):
    """MoleditPy plugin entry point."""
    global _context
    _context = context

    def open_dialog():
        """Open the analyzer window; let the user pick a file inside it."""
        existing = context.get_window("nics_analyzer")
        if existing is not None:
            try:
                existing.raise_()
                existing.activateWindow()
                return
            except (RuntimeError, AttributeError):
                logging.debug(
                    "[orca_nics_analyzer] failed to raise existing window",
                    exc_info=True,
                )

        mw = context.get_main_window()
        from .gui import NicsAnalyzerDialog

        dlg = NicsAnalyzerDialog(None, context, parent=mw)
        context.register_window("nics_analyzer", dlg)
        global _dialog_opened
        _dialog_opened = True
        dlg.show()

    context.add_menu_action("Analysis/ORCA NICS Analyzer...", open_dialog)

    def on_reset():
        win = context.get_window("nics_analyzer")
        if win is not None:
            if getattr(win, "_loading_structure", False) is True:
                return
            try:
                win.close()
            except (RuntimeError, AttributeError) as e:
                logging.warning("[orca_nics_analyzer] reset close: %s", e)

    if hasattr(context, "register_document_reset_handler"):
        context.register_document_reset_handler(on_reset)


def run(mw=None):  # pragma: no cover - host convenience entry point
    """Alternative entry point used when the host runs the plugin directly."""
    if _context is not None:
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            _context.get_main_window(),
            "Open ORCA Output with NICS Data",
            "",
            "ORCA Output (*.out *.log);;All Files (*)",
        )
        if path and os.path.exists(path):
            _open_file(path, _context)
