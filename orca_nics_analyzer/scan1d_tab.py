"""1D NICS scan: values along a line of probes, the usual NICS-scan plot."""

import logging
import os

import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
except ImportError:  # matplotlib is an optional dependency
    FigureCanvas = None
    NavigationToolbar = None
    Figure = None


class Scan1DTab(QWidget):
    """NICS(iso) and NICS_zz against distance along the probe line."""

    def __init__(self, field, parent=None):
        super().__init__(parent)
        self.field = field
        self.canvas = None
        self.figure = None
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        if FigureCanvas is None:
            layout.addWidget(
                QLabel(
                    "matplotlib is not installed.\n\n"
                    "Install it to see the scan plot:  pip install matplotlib"
                )
            )
            return

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas, 1)

        options = QGroupBox("Curves")
        row = QHBoxLayout(options)
        self.show_zz = QCheckBox("NICS_zz")
        self.show_zz.setChecked(self.field.has_tensors)
        self.show_zz.setEnabled(self.field.has_tensors)
        self.show_zz.toggled.connect(self.refresh)
        row.addWidget(self.show_zz)

        self.show_iso = QCheckBox("NICS(iso)")
        self.show_iso.setChecked(True)
        self.show_iso.toggled.connect(self.refresh)
        row.addWidget(self.show_iso)

        self.mark_extremum = QCheckBox("Mark extremum")
        self.mark_extremum.setChecked(True)
        self.mark_extremum.toggled.connect(self.refresh)
        row.addWidget(self.mark_extremum)

        self.show_points = QCheckBox("Probe markers")
        self.show_points.setChecked(True)
        self.show_points.toggled.connect(self.refresh)
        row.addWidget(self.show_points)
        row.addStretch(1)
        layout.addWidget(options)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        csv_btn = QPushButton("Export scan CSV...")
        csv_btn.clicked.connect(self.export_csv)
        buttons.addWidget(csv_btn)
        png_btn = QPushButton("Save image...")
        png_btn.clicked.connect(self.export_png)
        buttons.addWidget(png_btn)
        layout.addLayout(buttons)

    # -- drawing ---------------------------------------------------------
    def refresh(self):
        if self.canvas is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            data = self.field.line_data()
        except ValueError:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "The probes do not lie on a line.\n"
                f"Detected layout: {self.field.layout['kind']}.",
                ha="center",
                va="center",
            )
            self.canvas.draw_idle()
            self.info.setText("")
            return

        style = "o-" if self.show_points.isChecked() else "-"
        drawn = False
        for enabled, key, label, colour in (
            (self.show_zz.isChecked(), "zz", "NICS$_{zz}$", "#c8463c"),
            (self.show_iso.isChecked(), "iso", "NICS(iso)", "#3c6ec8"),
        ):
            values = data[key]
            if not enabled or not np.isfinite(values).any():
                continue
            ax.plot(data["distance"], values, style, color=colour, label=label, ms=4)
            drawn = True

        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_xlabel(data["label"])
        ax.set_ylabel("NICS / ppm")
        ax.grid(alpha=0.25)
        if drawn:
            ax.legend(fontsize=9)

        component = "zz" if self.show_zz.isChecked() else "iso"
        where, peak = self.field.scan_extremum(component)
        if self.mark_extremum.isChecked() and peak is not None:
            ax.plot([where], [peak], "k*", ms=11, zorder=5)
            ax.annotate(
                f"{peak:+.2f} ppm at {where:+.2f} A",
                (where, peak),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
            )
        ax.set_title("NICS scan", fontsize=10)
        self.canvas.draw_idle()
        self.info.setText(self._summary(data, component, where, peak))

    def _summary(self, data, component, where, peak):
        parts = [f"{len(data['distance'])} probes along the scan"]
        if peak is not None:
            name = "NICS_zz" if component == "zz" else "NICS(iso)"
            parts.append(f"largest |{name}| {peak:+.2f} ppm at {where:+.2f} A")
        offsets = data["offsets"]
        if offsets.size and float(np.max(offsets)) > 0.25:
            # A scan is meant to run along the ring normal; drifting sideways
            # means the values are not a clean height profile.
            parts.append(
                f"note: probes stray up to {float(np.max(offsets)):.2f} A off the ring axis"
            )
        return "   |   ".join(parts)

    def shutdown(self):
        """Cancel any queued redraw before the widget goes away.

        draw_idle() posts the redraw through the Qt event loop; if the dialog
        is destroyed first, the queued draw reaches a deleted canvas and
        raises from inside matplotlib.
        """
        if self.canvas is not None:
            self.canvas._draw_pending = False

    # -- export ----------------------------------------------------------
    def _default_path(self, suffix):
        if not self.field.filename:
            return ""
        return os.path.splitext(self.field.filename)[0] + suffix

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export scan",
            self._default_path("_NICS_scan.csv"),
            "CSV (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(self.field.scan_csv())
        except (OSError, ValueError) as e:
            logging.warning("[orca_nics_analyzer] scan CSV export: %s", e)
            QMessageBox.critical(
                self, "Export failed", f"Could not write the file:\n{e}"
            )

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save scan image",
            self._default_path("_NICS_scan.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=300)
        except (OSError, ValueError) as e:
            logging.warning("[orca_nics_analyzer] scan image export: %s", e)
            QMessageBox.critical(
                self, "Save failed", f"Could not write the image:\n{e}"
            )
