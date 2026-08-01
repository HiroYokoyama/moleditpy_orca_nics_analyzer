"""Main analyzer window."""

import logging
import os

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
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


class NicsAnalyzerDialog(QDialog):
    """Tabs over one parsed ORCA output."""

    def __init__(self, parser, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.field = NicsField(parser)

        name = os.path.basename(parser.filename) if parser.filename else "NICS data"
        self.setWindowTitle(f"ORCA NICS Analyzer - {name}")
        self.resize(940, 780)
        self._build_ui()

    # -- ui --------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("NICS_zz axis:"))
        self.axis_combo = QComboBox()
        for label, mode in AXIS_CHOICES:
            self.axis_combo.addItem(label, mode)
        self.axis_combo.setToolTip(
            "The direction NICS_zz is projected onto. ICSS maps use the grid "
            "normal; single probes are usually quoted against the ring normal."
        )
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        if not self.field.has_tensors:
            self.axis_combo.setEnabled(False)
            self.axis_combo.setToolTip(
                "This output has no shielding tensors, so NICS_zz cannot be computed."
            )
        header.addWidget(self.axis_combo)
        header.addStretch(1)

        export_btn = QPushButton("Export all...")
        export_btn.setToolTip(
            "Write the probe CSV, the summary and every available cube into one folder."
        )
        export_btn.clicked.connect(self.export_all)
        header.addWidget(export_btn)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

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

        self.map_tab = Map2DTab(self.field, self, show_in_3d=self._show_plane_in_3d)
        self.tabs.addTab(self.map_tab, "2D Map")
        self.tabs.addTab(self.icss_tab, "3D ICSS")

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(self.field.summary_text(PLUGIN_VERSION))
        self.tabs.addTab(self.summary, "Summary")

        self._select_default_tab()

        footer = QHBoxLayout()
        self.status = QLabel(self._layout_hint())
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)
        layout.addLayout(footer)

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

    # -- host access -----------------------------------------------------
    def _plotter(self):
        mw = self.context.get_main_window()
        return getattr(mw, "plotter", None)

    def _show_plane_in_3d(self, component, slice_index):
        self.icss_tab.show_plane(component, slice_index)

    # -- actions ---------------------------------------------------------
    def _on_axis_changed(self):
        self.field.set_axis_mode(self.axis_combo.currentData())
        self.probe_tab.refresh()
        self.scan_tab.refresh()
        self.map_tab.refresh()
        self.summary.setPlainText(self.field.summary_text(PLUGIN_VERSION))

    def export_all(self):
        default = os.path.dirname(self.field.filename or "") or ""
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

    # -- lifecycle -------------------------------------------------------
    def _cleanup(self):
        """Take our actors out of the host viewer and release the window slot.

        Runs at most once, from both closing paths.
        """
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True
        for tab in (getattr(self, "map_tab", None), getattr(self, "scan_tab", None)):
            if tab is not None:
                tab.shutdown()
        try:
            self.icss_tab.clear_actors()
        except (RuntimeError, AttributeError) as e:
            logging.warning("[orca_nics_analyzer] clearing actors on close: %s", e)
        try:
            # Releasing the registry slot is what lets the next open create a
            # live window instead of raising a deleted one.
            self.context.register_window("nics_analyzer", None)
        except (RuntimeError, AttributeError) as e:
            logging.warning("[orca_nics_analyzer] deregistering window: %s", e)

    def closeEvent(self, event):
        # No super() call: QDialog.closeEvent rejects, reject() calls close(),
        # and that would re-enter this handler.
        self._cleanup()
        event.accept()

    def done(self, result):
        # Esc rejects the dialog without ever raising a close event, so the
        # cleanup has to hang off done() as well.
        self._cleanup()
        super().done(result)
