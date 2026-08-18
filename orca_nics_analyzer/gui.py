"""Main analyzer window."""

import logging
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
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
from .settings import (
    DEFAULT_AXIS_MODE,
    DEFAULT_SETTINGS,
    SETTINGS_FILE,
    load_settings,
    save_settings,
)

#: Label -> axis_mode for the NICS_zz reference direction.
AXIS_CHOICES = (
    ("Grid normal (ICSS convention)", "grid"),
    ("Nearest ring normal", "ring"),
    ("Lab X", "x"),
    ("Lab Y", "y"),
    ("Lab Z", "z"),
    ("Manual vector…", "custom"),
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

    def __init__(self, parser, context, parent=None, settings_file=None):
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
        self.settings_file = settings_file or SETTINGS_FILE
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

        # ---- header settings (2 lines, always visible) ---------------------
        header_layout = QVBoxLayout()

        # Line 1: Actions & 3D view display options
        row1 = QHBoxLayout()

        self._open_btn = QPushButton("Open File…")
        self._open_btn.setToolTip("Open an ORCA output file with NICS ghost-atom data.")
        self._open_btn.clicked.connect(self.open_file_dialog)
        row1.addWidget(self._open_btn)

        self._probe_chk = QCheckBox("Show probe atoms")
        self._probe_chk.setChecked(False)  # hidden by default
        self._probe_chk.setToolTip(
            "Include ghost/probe atoms when loading the molecule into the 3D viewer."
        )
        self._probe_chk.toggled.connect(self._on_probe_visibility_toggled)
        self._probe_chk.setEnabled(False)
        row1.addWidget(self._probe_chk)

        self._vector_chk = QCheckBox("Show NICS_zz vector")
        self._vector_chk.setChecked(False)
        self._vector_chk.setToolTip(
            "Display the NICS_zz axis vector as a 3D arrow in the viewer."
        )
        self._vector_chk.toggled.connect(self._on_vector_toggled)
        self._vector_chk.setEnabled(False)
        row1.addWidget(self._vector_chk)

        row1.addStretch(1)

        self._export_btn = QPushButton("Export all…")
        self._export_btn.setToolTip(
            "Write the probe CSV, the summary and every available cube into one folder."
        )
        self._export_btn.clicked.connect(self.export_all)
        self._export_btn.setEnabled(False)
        row1.addWidget(self._export_btn)

        header_layout.addLayout(row1)

        # Line 2: Axis projection configuration
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("NICS_zz axis:"))
        self.axis_combo = QComboBox()
        for label, mode in AXIS_CHOICES:
            self.axis_combo.addItem(label, mode)
        self.axis_combo.setToolTip(
            "The direction NICS_zz is projected onto. ICSS maps use the grid "
            "normal; single probes are usually quoted against the ring normal."
        )
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        self.axis_combo.setEnabled(False)
        row2.addWidget(self.axis_combo)

        self._axis_vector_label = QLabel("Vector:")
        row2.addWidget(self._axis_vector_label)
        self._axis_vector = []
        for component in "xyz":
            spin = QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            spin.setPrefix(f"{component}=")
            spin.setValue(1.0 if component == "z" else 0.0)
            spin.valueChanged.connect(self._on_axis_changed)
            self._axis_vector.append(spin)
            row2.addWidget(spin)
        self._axis_vector_label.setVisible(False)
        for spin in self._axis_vector:
            spin.setVisible(False)

        row2.addStretch(1)

        header_layout.addLayout(row2)

        layout.addLayout(header_layout)

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
            self.field,
            self._plotter,
            plugin_version=PLUGIN_VERSION,
            parent=self,
            show_in_2d=self._show_map_tab,
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
            clear_3d=self.icss_tab.clear_actors,
        )
        self.tabs.addTab(self.map_tab, "2D Map")
        self.tabs.addTab(self.icss_tab, "3D ICSS")

        # Sync color mapping settings between the two tabs
        self.map_tab.cmap.currentTextChanged.connect(
            lambda t: (
                self.icss_tab.cmap.setCurrentText(t)
                if self.icss_tab.cmap.currentText() != t
                else None
            )
        )
        self.icss_tab.cmap.currentTextChanged.connect(
            lambda t: (
                self.map_tab.cmap.setCurrentText(t)
                if self.map_tab.cmap.currentText() != t
                else None
            )
        )

        self.map_tab.vmax.valueChanged.connect(self._set_3d_display_range)
        self.map_tab.auto_range.toggled.connect(self._set_3d_auto_display_range)
        # An auto-computed range is written into vmax with signals blocked, so
        # this is the only way the 3D plane learns the scale the 2D map used.
        self.map_tab._on_range_computed = self.icss_tab.set_display_range

        self.map_tab.component.currentIndexChanged.connect(self._sync_component_to_3d)
        self.map_tab.cmap.currentTextChanged.connect(self._refresh_3d_from_map)
        self.map_tab.vmax.valueChanged.connect(self._refresh_3d_from_map)
        self.map_tab.auto_range.toggled.connect(self._refresh_3d_from_map)
        self.map_tab.component.currentIndexChanged.connect(
            self._refresh_3d_plane_if_map_visible
        )
        self.map_tab.cmap.currentTextChanged.connect(
            self._refresh_3d_plane_if_map_visible
        )
        self.map_tab.vmax.valueChanged.connect(self._refresh_3d_plane_if_map_visible)
        self.map_tab.auto_range.toggled.connect(self._refresh_3d_plane_if_map_visible)

        self.icss_tab._on_slice_settings_changed = self._refresh_map_if_visible
        self.map_tab._get_slice_index = lambda: (
            self.icss_tab.slice_slider.value()
            if hasattr(self.icss_tab, "slice_slider")
            else 0
        )
        self.map_tab._set_slice_value_label = lambda t: (
            self.icss_tab.slice_value.setText(t)
            if hasattr(self.icss_tab, "slice_value")
            else None
        )

        self.icss_tab.show_vector.toggled.connect(self._sync_vector_chk_from_icss)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(self.field.summary_text(PLUGIN_VERSION))
        self.tabs.addTab(self.summary, "Summary")

        self.map_tab._is_tab_visible = lambda: self.tabs.currentWidget() is self.map_tab
        self.icss_tab._is_tab_visible = lambda: (
            self.tabs.currentWidget() is self.icss_tab
        )
        self.scan_tab._is_tab_visible = lambda: (
            self.tabs.currentWidget() is self.scan_tab
        )

        self._load_settings()
        self._select_default_tab()

    def _apply_parser(self, parser):
        """Load a parser's data into the dialog, replacing any previous state."""
        self.field = NicsField(parser)

        name = os.path.basename(parser.filename) if parser.filename else "NICS data"
        self.setWindowTitle(f"ORCA NICS Analyzer — {name}")

        # Enable controls that require loaded data.
        self.axis_combo.setEnabled(True)
        self._probe_chk.setEnabled(True)
        self._vector_chk.setEnabled(True)
        self._export_btn.setEnabled(True)

        # axis_combo & _vector_chk: disable NICS_zz when there are no tensors.
        if not self.field.has_tensors:
            self.axis_combo.setEnabled(False)
            self.axis_combo.setToolTip(
                "This output has no shielding tensors, so NICS_zz cannot be computed."
            )
            self._vector_chk.setEnabled(False)
            self._vector_chk.setToolTip(
                "This output has no shielding tensors, so NICS_zz vector cannot be shown."
            )
        else:
            self.axis_combo.setEnabled(True)
            self.axis_combo.setToolTip(
                "The direction NICS_zz is projected onto. ICSS maps use the grid "
                "normal; single probes are usually quoted against the ring normal."
            )
            self._vector_chk.setEnabled(True)
            self._vector_chk.setToolTip(
                "Display the NICS_zz axis vector as a 3D arrow in the viewer."
            )

        self._build_tabs()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._stack.setCurrentIndex(1)
        self.status.setText(self._layout_hint())

        # Load the molecule into the host's 3D viewer.
        self._load_molecule(include_probes=self._probe_chk.isChecked())

        # Render initial 3D surface or 2D map after molecule is in the viewer
        if self.field and self.field.is_gridded:
            if self.field.layout["kind"] == "volume":
                self.icss_tab.draw(silent=True, force=True)
            elif self.field.layout["kind"] == "plane":
                self.map_tab.refresh(force=True)
        self._refresh_3d_plane_if_map_visible()
        if self._vector_chk.isChecked():
            self.icss_tab.update_axis_vector()

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
                "NICS requires ghost centres (e.g. 'H:') in the geometry "
                "and an NMR job that includes them.",
            )
            return False
        if not parser.data.get("probe_indices"):
            from . import _warn_missing_shieldings

            _warn_missing_shieldings(self)
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

    # -- persistent user preferences -----------------------------------------

    @staticmethod
    def _set_combo_data(combo, value):
        if not combo.isEnabled():
            return
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _load_settings(self):
        settings = load_settings(self.settings_file)
        axis_index = self.axis_combo.findData(DEFAULT_AXIS_MODE)
        self.axis_combo.blockSignals(True)
        self._probe_chk.blockSignals(True)
        self._vector_chk.blockSignals(True)
        for spin in self._axis_vector:
            spin.blockSignals(True)
        try:
            # The axis belongs to one geometry, so every load resets to the
            # ICSS convention instead of inheriting the previous molecule's.
            for spin, value in zip(self._axis_vector, (0.0, 0.0, 1.0)):
                spin.setValue(value)
            if axis_index >= 0 and self.axis_combo.isEnabled():
                self.axis_combo.setCurrentIndex(axis_index)
                self._set_axis_vector_visible(False)
                if self.field is not None:
                    self.field.set_axis_mode(DEFAULT_AXIS_MODE)
            self._probe_chk.setChecked(bool(settings["show_probes"]))
            self._vector_chk.setChecked(bool(settings["icss_show_vector"]))
        finally:
            self.axis_combo.blockSignals(False)
            self._probe_chk.blockSignals(False)
            self._vector_chk.blockSignals(False)
            for spin in self._axis_vector:
                spin.blockSignals(False)

        self._set_combo_data(self.map_tab.component, settings["map_component"])
        self.map_tab.cmap.setCurrentText(settings["map_colormap"])
        self.map_tab.levels.setValue(int(settings["map_levels"]))
        # Range before the Auto flag: unticking Auto has to find the remembered
        # value already in place, or the map would redraw at the stale one.
        self.map_tab.manual_span = float(settings["map_range"])
        self.map_tab.vmax.setValue(float(settings["map_range"]))
        self.map_tab.auto_range.setChecked(bool(settings["map_auto_range"]))
        self.map_tab.show_molecule.setChecked(bool(settings["map_molecule"]))
        self.map_tab.show_contours.setChecked(bool(settings["map_contours"]))
        self.map_tab.show_probes.setChecked(bool(settings["map_probes"]))
        self.map_tab.show_1d_line.setChecked(bool(settings["map_slice_line"]))

        self._set_combo_data(self.icss_tab.component, settings["icss_component"])
        self.icss_tab.cmap.setCurrentText(settings["icss_colormap"])
        self.icss_tab.opacity.setValue(float(settings["icss_opacity"]))
        self.icss_tab.show_positive.setChecked(bool(settings["icss_positive"]))
        self.icss_tab.show_negative.setChecked(bool(settings["icss_negative"]))
        self.icss_tab.show_cut_axis.setChecked(bool(settings["icss_cut_axis"]))
        self.icss_tab.show_vector.setChecked(bool(settings["icss_show_vector"]))

    def _settings_values(self):
        return {
            "show_probes": self._probe_chk.isChecked(),
            "map_component": self.map_tab.component.currentData(),
            "map_colormap": self.map_tab.cmap.currentText(),
            "map_levels": self.map_tab.levels.value(),
            # Never vmax.value(): while Auto is on that holds a computed span.
            "map_range": (
                self.map_tab.manual_span
                if self.map_tab.manual_span is not None
                else DEFAULT_SETTINGS["map_range"]
            ),
            "map_auto_range": self.map_tab.auto_range.isChecked(),
            "map_molecule": self.map_tab.show_molecule.isChecked(),
            "map_contours": self.map_tab.show_contours.isChecked(),
            "map_probes": self.map_tab.show_probes.isChecked(),
            "map_slice_line": self.map_tab.show_1d_line.isChecked(),
            "icss_component": self.icss_tab.component.currentData(),
            "icss_colormap": self.icss_tab.cmap.currentText(),
            "icss_opacity": self.icss_tab.opacity.value(),
            "icss_positive": self.icss_tab.show_positive.isChecked(),
            "icss_negative": self.icss_tab.show_negative.isChecked(),
            "icss_cut_axis": self.icss_tab.show_cut_axis.isChecked(),
            "icss_show_vector": self._vector_chk.isChecked(),
        }

    def _save_settings(self):
        if hasattr(self, "map_tab") and hasattr(self, "icss_tab"):
            save_settings(self._settings_values(), self.settings_file)

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

    def _sync_component_to_3d(self, index):
        if self.icss_tab.component.currentIndex() != index:
            self.icss_tab.component.setCurrentIndex(index)
        self.icss_tab.draw(
            silent=True, force=self.tabs.currentWidget() is self.icss_tab
        )

    def _set_3d_display_range(self, value):
        self.icss_tab.set_display_range(value)
        self._refresh_3d_from_map()

    def _set_3d_auto_display_range(self, checked):
        self.icss_tab.set_auto_display_range(checked)
        self._refresh_3d_from_map()

    def _refresh_3d_from_map(self, *_):
        """Apply shared 2D settings when the 3D tab is visible."""
        self.icss_tab.draw(
            silent=True, force=self.tabs.currentWidget() is self.icss_tab
        )

    def _refresh_map_if_visible(self):
        if self.tabs.currentWidget() is self.map_tab:
            self.map_tab.refresh(force=True)
            self._refresh_3d_plane_if_map_visible()

    def _refresh_3d_plane_if_map_visible(self, *_):
        """Mirror the visible 2D slice as a temporary host-3D plane."""
        if self.tabs.currentWidget() is not self.map_tab:
            return
        if not self.field or not self.field.is_gridded:
            return
        try:
            self.icss_tab.show_plane(
                self.map_tab.component.currentData(),
                self.map_tab._get_slice_index(),
            )
        except (ValueError, RuntimeError) as exc:
            logging.debug("[orca_nics_analyzer] auto 2D plane: %s", exc)

    def _show_map_tab(self):
        """Switch to the 2D Map tab."""
        if hasattr(self, "map_tab"):
            self.map_tab.refresh(force=True)
        self.tabs.setCurrentWidget(self.map_tab)

    def _on_tab_changed(self, index):
        widget = self.tabs.widget(index)
        if not hasattr(self, "icss_tab"):
            return

        # Each tab owns its graphics; clear stale plugin actors before switching.
        self.icss_tab.clear_actors()
        if widget is self.icss_tab:
            self.icss_tab.draw(silent=True, force=True)
        elif hasattr(self, "map_tab") and widget is self.map_tab:
            self.map_tab.refresh(force=True)
            self._refresh_3d_plane_if_map_visible()
            if self._vector_chk.isChecked():
                self.icss_tab.update_axis_vector()
        elif hasattr(self, "scan_tab") and widget is self.scan_tab:
            self.scan_tab.refresh(force=True)
            if self._vector_chk.isChecked():
                self.icss_tab.update_axis_vector()
        elif self._vector_chk.isChecked():
            self.icss_tab.update_axis_vector()

    def _custom_axis_values(self):
        return tuple(float(spin.value()) for spin in self._axis_vector)

    def _set_axis_vector_visible(self, visible):
        self._axis_vector_label.setVisible(visible)
        for spin in self._axis_vector:
            spin.setVisible(visible)

    def _on_axis_changed(self):
        if self.field is None:
            return
        mode = self.axis_combo.currentData()
        self._set_axis_vector_visible(mode == "custom")
        try:
            self.field.set_axis_mode(
                mode, self._custom_axis_values() if mode == "custom" else None
            )
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.probe_tab.refresh()
        self.scan_tab.refresh(force=self.tabs.currentWidget() is self.scan_tab)
        self._refresh_map_if_visible()
        self.icss_tab._update_cache_label()
        self.icss_tab.draw(silent=True)
        self.icss_tab.update_axis_vector()
        self.summary.setPlainText(self.field.summary_text(PLUGIN_VERSION))

    def _on_vector_toggled(self, checked):
        if hasattr(self, "icss_tab") and hasattr(self.icss_tab, "show_vector"):
            if self.icss_tab.show_vector.isChecked() != checked:
                self.icss_tab.show_vector.blockSignals(True)
                self.icss_tab.show_vector.setChecked(checked)
                self.icss_tab.show_vector.blockSignals(False)
            self.icss_tab.update_axis_vector()

    def _sync_vector_chk_from_icss(self, checked):
        if hasattr(self, "_vector_chk") and self._vector_chk.isChecked() != checked:
            self._vector_chk.blockSignals(True)
            self._vector_chk.setChecked(checked)
            self._vector_chk.blockSignals(False)

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
        self._save_settings()
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
