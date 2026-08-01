"""Main analyzer window."""

import logging
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import PLUGIN_VERSION
from .analysis import NicsField, export_all

#: Label -> axis_mode for the NICS_zz reference direction.
AXIS_CHOICES = (
    ("Grid normal (ICSS convention)", "grid"),
    ("Nearest ring normal", "ring"),
    ("Lab X", "x"),
    ("Lab Y", "y"),
    ("Lab Z", "z"),
)

#: File extensions accepted for drag-and-drop.
_ACCEPTED_EXTENSIONS = frozenset({".out", ".log"})


def _xyz_block(atoms, include_probes):
    """Build a headerless XYZ text block from a list of atom dicts.

    Args:
        atoms: List of atom dicts from ``NicsParser.data["atoms"]``.
        include_probes: If False, ghost atoms (``is_ghost=True``) are omitted.

    Returns:
        str: Lines of ``symbol  x  y  z`` suitable for
             ``context.show_xyz_data()``, or empty string if no atoms.
    """
    lines = []
    for atom in atoms:
        if not include_probes and atom["is_ghost"]:
            continue
        sym = atom["symbol"]
        # MoleditPy/RDKit treats "H:" as real Hydrogen (fails chemistry).
        # Emit "X" so it is parsed as a dummy atom (atomic number 0).
        if atom.get("is_ghost"):
            sym = "X"
        x, y, z = atom["xyz"]
        lines.append(f"{sym}  {x:.8f}  {y:.8f}  {z:.8f}")
    return "\n".join(lines)


class _WelcomeWidget(QWidget):
    """Placeholder shown when no file is loaded yet."""

    def __init__(self, open_callback, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hint = QLabel(
            "Drop an ORCA output file here\nor use the  Open File…  button above."
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: palette(mid); font-size: 14px; padding: 40px;")
        layout.addWidget(hint)

        btn = QPushButton("Open File…")
        btn.setFixedWidth(160)
        btn.clicked.connect(open_callback)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)


class NicsAnalyzerDialog(QDialog):
    """Tabs over one parsed ORCA output.

    ``parser`` may be ``None`` to open in an empty/welcome state; the user
    can then load a file via the  Open File…  button or by dropping a file
    onto the dialog.
    """

    def __init__(self, parser, context, parent=None):
        super().__init__(parent)
        # Do NOT set Qt.WindowType.Dialog — that would make the dialog
        # application-modal and prevent the user from interacting with the main
        # window while it is open.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.context = context
        self.field = None

        self.setWindowTitle("ORCA NICS Analyzer")
        self.resize(940, 780)
        self.setAcceptDrops(True)
        self._build_ui()

        if parser is not None:
            self._apply_parser(parser)

    # -- ui ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- header row (always visible) -----------------------------------
        header = QHBoxLayout()

        self._open_btn = QPushButton("Open File…")
        self._open_btn.setToolTip("Open an ORCA output file with NICS ghost-atom data.")
        self._open_btn.clicked.connect(self.open_file_dialog)
        header.addWidget(self._open_btn)

        header.addWidget(QLabel("NICS_zz axis:"))
        self.axis_combo = QComboBox()
        for label, mode in AXIS_CHOICES:
            self.axis_combo.addItem(label, mode)
        self.axis_combo.setToolTip(
            "The direction NICS_zz is projected onto. ICSS maps use the grid "
            "normal; single probes are usually quoted against the ring normal."
        )
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.axis_combo.setEnabled(False)
        header.addWidget(self.axis_combo)

        self._probe_chk = QCheckBox("Show probe atoms")
        self._probe_chk.setChecked(False)  # hidden by default
        self._probe_chk.setToolTip(
            "Include ghost/probe atoms when loading the molecule into the 3D viewer."
        )
        self._probe_chk.toggled.connect(self._on_probe_visibility_toggled)
        self._probe_chk.setEnabled(False)
        header.addWidget(self._probe_chk)

        header.addStretch(1)

        self._export_btn = QPushButton("Export all…")
        self._export_btn.setToolTip(
            "Write the probe CSV, the summary and every available cube into one folder."
        )
        self._export_btn.clicked.connect(self.export_all)
        self._export_btn.setEnabled(False)
        header.addWidget(self._export_btn)

        layout.addLayout(header)

        # ---- stacked body: welcome or tabs ---------------------------------
        self._stack = QStackedWidget()

        self._welcome = _WelcomeWidget(self.open_file_dialog, parent=self)
        self._stack.addWidget(self._welcome)  # index 0

        self._tabs_container = QWidget()
        tabs_layout = QVBoxLayout(self._tabs_container)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        tabs_layout.addWidget(self.tabs, 1)
        self._stack.addWidget(self._tabs_container)  # index 1

        layout.addWidget(self._stack, 1)

        # ---- footer --------------------------------------------------------
        footer = QHBoxLayout()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)
        layout.addLayout(footer)

    def _build_tabs(self):
        """Construct (or reconstruct) the tab widget for the current field."""
        # Clear any previously built tabs.
        self.tabs.clear()

        from .probe_tab import ProbeTab

        self.probe_tab = ProbeTab(self.field, self)
        self.tabs.addTab(self.probe_tab, "Probes")

        from .icss3d_tab import Icss3DTab

        self.icss_tab = Icss3DTab(
            self.field, self._plotter, plugin_version=PLUGIN_VERSION, parent=self
        )

        from .scan1d_tab import Scan1DTab

        self.scan_tab = Scan1DTab(self.field, self)
        self.tabs.addTab(self.scan_tab, "1D Scan")

        from .map2d_tab import Map2DTab

        self.map_tab = Map2DTab(
            self.field,
            self,
            show_in_3d=self._show_plane_in_3d,
            show_slice_in_1d=self._show_slice_in_1d,
        )
        self.tabs.addTab(self.map_tab, "2D Map")
        self.tabs.addTab(self.icss_tab, "3D ICSS")

        # Sync color mapping settings between the two tabs
        self.map_tab.cmap.currentTextChanged.connect(
            lambda t: self.icss_tab.cmap.setCurrentText(t)
            if self.icss_tab.cmap.currentText() != t
            else None
        )
        self.icss_tab.cmap.currentTextChanged.connect(
            lambda t: self.map_tab.cmap.setCurrentText(t)
            if self.map_tab.cmap.currentText() != t
            else None
        )

        self.map_tab.vmax.valueChanged.connect(
            lambda v: self.icss_tab.vmax.setValue(v)
            if self.icss_tab.vmax.value() != v
            else None
        )
        self.icss_tab.vmax.valueChanged.connect(
            lambda v: self.map_tab.vmax.setValue(v)
            if self.map_tab.vmax.value() != v
            else None
        )

        self.map_tab.auto_range.toggled.connect(
            lambda c: self.icss_tab.auto_range.setChecked(c)
            if self.icss_tab.auto_range.isChecked() != c
            else None
        )
        self.icss_tab.auto_range.toggled.connect(
            lambda c: self.map_tab.auto_range.setChecked(c)
            if self.map_tab.auto_range.isChecked() != c
            else None
        )

        self.icss_tab._on_slice_settings_changed = self.map_tab.refresh
        self.map_tab._get_slice_index = (
            lambda: self.icss_tab.slice_slider.value()
            if hasattr(self.icss_tab, "slice_slider")
            else 0
        )
        self.map_tab._set_slice_value_label = (
            lambda t: self.icss_tab.slice_value.setText(t)
            if hasattr(self.icss_tab, "slice_value")
            else None
        )

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(self.field.summary_text(PLUGIN_VERSION))
        self.tabs.addTab(self.summary, "Summary")

        self._select_default_tab()

    def _apply_parser(self, parser):
        """Load a parser's data into the dialog, replacing any previous state."""
        self.field = NicsField(parser)

        name = os.path.basename(parser.filename) if parser.filename else "NICS data"
        self.setWindowTitle(f"ORCA NICS Analyzer — {name}")

        # Enable controls that require loaded data.
        self.axis_combo.setEnabled(True)
        self._probe_chk.setEnabled(True)
        self._export_btn.setEnabled(True)

        # axis_combo: disable NICS_zz when there are no tensors.
        if not self.field.has_tensors:
            self.axis_combo.setEnabled(False)
            self.axis_combo.setToolTip(
                "This output has no shielding tensors, so NICS_zz cannot be computed."
            )
        else:
            self.axis_combo.setEnabled(True)
            self.axis_combo.setToolTip(
                "The direction NICS_zz is projected onto. ICSS maps use the grid "
                "normal; single probes are usually quoted against the ring normal."
            )

        self._build_tabs()
        self._stack.setCurrentIndex(1)
        self.status.setText(self._layout_hint())

        # Load the molecule into the host's 3D viewer.
        self._load_molecule(include_probes=self._probe_chk.isChecked())

    # -- public API ----------------------------------------------------------

    def load_parser(self, parser):
        """Reload the dialog with a new parser (e.g. after drag-dropping a file).

        Shuts down any 2D/3D resources from the previous session before
        replacing them.
        """
        self._shutdown_tabs()
        self._apply_parser(parser)

    def load_file(self, path):
        """Parse *path* and load it into the dialog.

        Returns True on success, False on failure (error already shown to user).
        """
        from . import _read_output_file
        from .parser import NicsParser

        mw = self.context.get_main_window()
        content = _read_output_file(path, mw)
        if content is None:
            return False

        parser = NicsParser()
        parser.load_from_memory(content, path)

        if not parser.data["ghost_indices"]:
            QMessageBox.warning(
                self,
                "No NICS probes found",
                "This output has no ghost atoms with NMR shielding data.\n\n"
                "NICS requires ghost centres (e.g. 'H:' or 'Bq') in the geometry "
                "and an NMR job that includes them.",
            )
            return False

        self.load_parser(parser)
        return True

    def open_file_dialog(self):
        """Show a file-open dialog and load the chosen file."""
        start_dir = (
            os.path.dirname(self.field.filename)
            if (self.field and self.field.filename)
            else ""
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open ORCA Output with NICS Data",
            start_dir,
            "ORCA Output (*.out *.log);;All Files (*)",
        )
        if path:
            self.load_file(path)

    # -- 3D molecule loading -------------------------------------------------

    def _load_molecule(self, include_probes):
        """Send the current molecule to the host 3D viewer.

        Args:
            include_probes: Whether ghost/probe atoms are included.
        """
        if self.field is None:
            return
        atoms = self.field.parser.data.get("atoms", [])
        if not atoms:
            return
        xyz = _xyz_block(atoms, include_probes)
        if not xyz:
            return
        name = os.path.basename(self.field.filename or "NICS molecule")
        try:
            self._loading_structure = True
            self.context.show_xyz_data(xyz, source_name=name)
            plotter = self._plotter()
            if plotter is not None:
                plotter.reset_camera()
        except Exception as e:  # noqa: BLE001
            logging.warning("[orca_nics_analyzer] show_xyz_data: %s", e)
        finally:
            self._loading_structure = False

    # -- drag-and-drop -------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):  # noqa: N802
        mime = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in _ACCEPTED_EXTENSIONS:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QDropEvent):  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in _ACCEPTED_EXTENSIONS:
                    if self.load_file(path):
                        event.acceptProposedAction()
                        return
        event.ignore()

    # -- tab selection / hints -----------------------------------------------

    def _select_default_tab(self):
        """Open on the tab that matches how the probes were laid out."""
        kind = self.field.layout["kind"]
        if kind == "volume":
            self.tabs.setCurrentWidget(self.icss_tab)
        elif kind == "plane":
            self.tabs.setCurrentWidget(self.map_tab)
        elif kind == "line":
            self.tabs.setCurrentWidget(self.scan_tab)
        else:
            self.tabs.setCurrentWidget(self.probe_tab)

    def _layout_hint(self):
        kind = self.field.layout["kind"]
        shape = self.field.layout.get("shape")
        hints = {
            "single": "One probe — see the Probes tab.",
            "line": "The probes lie on a line — the 1D Scan tab plots the profile.",
            "plane": "Planar probe grid — the 2D Map tab shows the NICS surface.",
            "volume": "3D probe grid — the 3D ICSS tab renders isosurfaces.",
            "scattered": (
                "The probes do not form a regular grid, so maps and cubes are "
                "unavailable; individual values are in the Probes tab."
            ),
            "none": "No probes found.",
        }
        text = hints.get(kind, "")
        if shape and kind in ("plane", "volume"):
            text += f"  Grid: {'x'.join(str(n) for n in shape)}."
        return text

    # -- host access ---------------------------------------------------------

    def _plotter(self):
        mw = self.context.get_main_window()
        return getattr(mw, "plotter", None)

    def _show_plane_in_3d(self, component, slice_index):
        self.icss_tab.show_plane(component, slice_index)

    def _show_slice_in_1d(self, data):
        """Route a 1D slice dict from the map tab to the scan tab."""
        self.scan_tab.show_slice(data)
        self.tabs.setCurrentWidget(self.scan_tab)

    # -- actions -------------------------------------------------------------

    def _on_axis_changed(self):
        if self.field is None:
            return
        self.field.set_axis_mode(self.axis_combo.currentData())
        self.probe_tab.refresh()
        self.scan_tab.refresh()
        self.map_tab.refresh()
        self.summary.setPlainText(self.field.summary_text(PLUGIN_VERSION))

    def _on_probe_visibility_toggled(self, checked):
        self._load_molecule(include_probes=checked)

    def export_all(self):
        default = os.path.dirname(self.field.filename or "") if self.field else ""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose a folder for the exported files", default
        )
        if not folder:
            return
        try:
            written = export_all(self.field, folder, PLUGIN_VERSION)
        except (ValueError, OSError) as e:
            logging.warning("[orca_nics_analyzer] export all: %s", e)
            QMessageBox.critical(self, "Export failed", str(e))
            return
        names = "\n".join(os.path.basename(p) for p in written)
        QMessageBox.information(
            self,
            "Export complete",
            f"Wrote {len(written)} file(s) to\n{folder}\n\n{names}",
        )

    # -- lifecycle -----------------------------------------------------------

    def _shutdown_tabs(self):
        """Release 2D/3D resources from the currently loaded tabs, if any."""
        for attr in ("map_tab", "scan_tab"):
            tab = getattr(self, attr, None)
            if tab is not None:
                tab.shutdown()
        icss = getattr(self, "icss_tab", None)
        if icss is not None:
            try:
                icss.clear_actors()
            except (RuntimeError, AttributeError) as e:
                logging.warning("[orca_nics_analyzer] clearing actors on reload: %s", e)

    def _cleanup(self):
        """Take our actors out of the host viewer and release the window slot.

        Runs at most once, from both closing paths.
        """
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True
        self._shutdown_tabs()
        try:
            # Releasing the registry slot is what lets the next open create a
            # live window instead of raising a deleted one.
            self.context.register_window("nics_analyzer", None)
        except (RuntimeError, AttributeError) as e:
            logging.warning("[orca_nics_analyzer] deregistering window: %s", e)

    def closeEvent(self, event):  # noqa: N802
        # No super() call: QDialog.closeEvent rejects, reject() calls close(),
        # and that would re-enter this handler.
        self._cleanup()
        event.accept()

    def done(self, result):
        # Esc rejects the dialog without ever raising a close event, so the
        # cleanup has to hang off done() as well.
        self._cleanup()
        super().done(result)
