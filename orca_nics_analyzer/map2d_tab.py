"""2D NICS map: filled contours of the probe plane, with the molecule on top."""

import logging
import os

import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

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

from . import nics_math as nm

#: Diverging colormaps only — a NICS map is signed and must read symmetrically
#: about zero.
COLORMAPS = ("seismic", "RdBu_r", "coolwarm", "bwr", "PuOr_r")


class Map2DTab(QWidget):
    """Contour map of one slice of the probe grid."""

    def __init__(self, field, parent=None, show_in_3d=None):
        super().__init__(parent)
        self.field = field
        self._show_in_3d = show_in_3d
        self.canvas = None
        self.figure = None
        self._build_ui()
        self.refresh()

    # -- ui --------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        if FigureCanvas is None:
            layout.addWidget(
                QLabel(
                    "matplotlib is not installed.\n\n"
                    "Install it to see 2D NICS maps:  pip install matplotlib"
                )
            )
            return

        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas, 1)

        controls = QGroupBox("Map")
        grid = QGridLayout(controls)

        grid.addWidget(QLabel("Component:"), 0, 0)
        self.component = QComboBox()
        self.component.addItem("NICS_zz", "zz")
        self.component.addItem("NICS(iso)", "iso")
        if not self.field.has_tensors:
            self.component.setCurrentIndex(1)
            self.component.setEnabled(False)
            self.component.setToolTip(
                "This output has no shielding tensors, so only isotropic NICS "
                "is available."
            )
        self.component.currentIndexChanged.connect(self.refresh)
        grid.addWidget(self.component, 0, 1)

        grid.addWidget(QLabel("Colormap:"), 0, 2)
        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        self.cmap.currentIndexChanged.connect(self.refresh)
        grid.addWidget(self.cmap, 0, 3)

        grid.addWidget(QLabel("Levels:"), 0, 4)
        self.levels = QSpinBox()
        self.levels.setRange(5, 101)
        self.levels.setValue(31)
        self.levels.valueChanged.connect(self.refresh)
        grid.addWidget(self.levels, 0, 5)

        grid.addWidget(QLabel("Range +/- ppm:"), 1, 0)
        self.vmax = QDoubleSpinBox()
        self.vmax.setRange(0.1, 1000.0)
        self.vmax.setDecimals(2)
        self.vmax.setSingleStep(1.0)
        self.vmax.valueChanged.connect(self.refresh)
        grid.addWidget(self.vmax, 1, 1)

        self.auto_range = QCheckBox("Auto")
        self.auto_range.setChecked(True)
        self.auto_range.toggled.connect(self._on_auto_toggled)
        grid.addWidget(self.auto_range, 1, 2)

        self.show_molecule = QCheckBox("Molecule outline")
        self.show_molecule.setChecked(True)
        self.show_molecule.toggled.connect(self.refresh)
        grid.addWidget(self.show_molecule, 1, 3)

        self.show_contours = QCheckBox("Contour lines")
        self.show_contours.setChecked(True)
        self.show_contours.toggled.connect(self.refresh)
        grid.addWidget(self.show_contours, 1, 4)

        self.show_probes = QCheckBox("Probe dots")
        self.show_probes.setChecked(False)
        self.show_probes.toggled.connect(self.refresh)
        grid.addWidget(self.show_probes, 1, 5)

        self.slice_label = QLabel("Slice:")
        grid.addWidget(self.slice_label, 2, 0)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.valueChanged.connect(self.refresh)
        grid.addWidget(self.slice_slider, 2, 1, 1, 4)
        self.slice_value = QLabel("-")
        grid.addWidget(self.slice_value, 2, 5)

        layout.addWidget(controls)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        if self._show_in_3d is not None:
            btn3d = QPushButton("Show in 3D view")
            btn3d.setToolTip(
                "Drop this plane into the main 3D viewer next to the molecule."
            )
            btn3d.clicked.connect(self._emit_show_in_3d)
            buttons.addWidget(btn3d)
        csv_btn = QPushButton("Export grid CSV...")
        csv_btn.clicked.connect(self.export_csv)
        buttons.addWidget(csv_btn)
        png_btn = QPushButton("Save image...")
        png_btn.clicked.connect(self.export_png)
        buttons.addWidget(png_btn)
        layout.addLayout(buttons)

        self._configure_slices()

    def _configure_slices(self):
        if not self.field.is_gridded:
            return
        try:
            info = self.field.plane_data(self._component())
        except ValueError:
            return
        n = info["n_slices"]
        self.slice_slider.setMaximum(max(0, n - 1))
        self.slice_slider.setValue(info["slice_index"])
        visible = n > 1
        self.slice_slider.setVisible(visible)
        self.slice_label.setVisible(visible)
        self.slice_value.setVisible(visible)

    def _on_auto_toggled(self, checked):
        self.vmax.setEnabled(not checked)
        self.refresh()

    def _component(self):
        return self.component.currentData() if hasattr(self, "component") else "zz"

    # -- drawing ---------------------------------------------------------
    def refresh(self):
        if self.canvas is None:
            return
        self.figure.clear()

        if not self.field.is_gridded:
            ax = self.figure.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "The probes do not form a regular 2D or 3D grid.\n"
                f"Detected layout: {self.field.layout['kind']}.\n\n"
                "Use the Probes tab for individual values.",
                ha="center",
                va="center",
                wrap=True,
            )
            self.canvas.draw_idle()
            return

        component = self._component()
        info = self.field.plane_slice(component, self.slice_slider.value())
        values = info["values"]

        finite = values[np.isfinite(values)]
        if self.auto_range.isChecked():
            span = float(np.max(np.abs(finite))) if finite.size else 1.0
            span = span if span > 0 else 1.0
            self.vmax.blockSignals(True)
            self.vmax.setValue(span)
            self.vmax.blockSignals(False)
        else:
            span = self.vmax.value()

        ax = self.figure.add_subplot(111)
        levels = np.linspace(-span, span, self.levels.value())
        # values is indexed [a1, a2]; contourf wants [row=y, col=x].
        mesh = ax.contourf(
            info["a1"],
            info["a2"],
            values.T,
            levels=levels,
            cmap=self.cmap.currentText(),
            extend="both",
        )
        if self.show_contours.isChecked():
            lines = ax.contour(
                info["a1"],
                info["a2"],
                values.T,
                levels=levels[:: max(1, len(levels) // 10)],
                colors="k",
                linewidths=0.4,
                alpha=0.5,
            )
            ax.clabel(lines, inline=True, fontsize=6, fmt="%.0f")

        if self.show_probes.isChecked():
            a1, a2 = np.meshgrid(info["a1"], info["a2"], indexing="ij")
            ax.plot(a1.ravel(), a2.ravel(), "k.", markersize=2, alpha=0.4)

        if self.show_molecule.isChecked():
            self._draw_molecule(ax, info)

        label = "NICS$_{zz}$" if component == "zz" else "NICS(iso)"
        bar = self.figure.colorbar(mesh, ax=ax)
        bar.set_label(f"{label} / ppm")
        ax.set_xlabel("in-plane axis 1 / A")
        ax.set_ylabel("in-plane axis 2 / A")
        ax.set_aspect("equal")

        offset = self._slice_offset(info)
        title = f"{label} map"
        if info["n_slices"] > 1:
            title += f" — slice {info['slice_index'] + 1}/{info['n_slices']}"
        if offset is not None:
            title += f" ({offset:+.2f} A from the ring plane)"
        ax.set_title(title, fontsize=10)
        self.slice_value.setText("-" if offset is None else f"{offset:+.2f} A")

        self.canvas.draw_idle()

    def _slice_offset(self, info):
        """Signed distance from the ring plane to this slice, along the normal."""
        rings = self.field.rings
        if not rings:
            return None
        coords = self.field.layout["coords"][info["order"][2]]
        idx = min(info["slice_index"], len(coords) - 1)
        point = (
            self.field.layout["origin"]
            + float(coords[idx]) * info["normal"]
            + float(self.field.layout["coords"][info["order"][0]][0]) * info["axis1"]
            + float(self.field.layout["coords"][info["order"][1]][0]) * info["axis2"]
        )
        _, height, _ = nm.nearest_ring(point, rings)
        return height

    def _draw_molecule(self, ax, info):
        """Bonds of the real molecule, projected onto the map plane."""
        coords = self.field.real_coords
        if len(coords) == 0:
            return
        origin = self.field.layout["origin"]
        rel = coords - origin
        u = rel @ info["axis1"]
        v = rel @ info["axis2"]
        w = np.abs(rel @ info["normal"])

        for i, j in nm.bond_list(self.field.real_symbols, coords):
            # Fade atoms far from the plane so a projected 3D cage stays readable.
            alpha = float(np.clip(1.0 - max(w[i], w[j]) / 3.0, 0.15, 0.9))
            ax.plot([u[i], u[j]], [v[i], v[j]], "-", color="0.15", lw=1.2, alpha=alpha)
        heavy = [k for k, s in enumerate(self.field.real_symbols) if s != "H"]
        if heavy:
            ax.plot(u[heavy], v[heavy], "o", color="0.15", markersize=3, alpha=0.8)

    def _emit_show_in_3d(self):
        if self._show_in_3d is None:
            return
        try:
            self._show_in_3d(self._component(), self.slice_slider.value())
        except Exception as e:  # the host viewer is out of our control
            logging.warning("[orca_nics_analyzer] show in 3D: %s", e)
            QMessageBox.warning(self, "3D view", f"Could not draw the plane:\n{e}")

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
        base = os.path.splitext(self.field.filename)[0]
        return f"{base}_NICS_{self._component()}{suffix}"

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save map image",
            self._default_path("_map.png"),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=300)
        except (OSError, ValueError) as e:
            logging.warning("[orca_nics_analyzer] image export: %s", e)
            QMessageBox.critical(
                self, "Save failed", f"Could not write the image:\n{e}"
            )

    def export_csv(self):
        if not self.field.is_gridded:
            QMessageBox.information(
                self, "Export", "There is no grid to export — see the Probes tab."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export map grid",
            self._default_path("_map.csv"),
            "CSV (*.csv);;All Files (*)",
        )
        if not path:
            return
        info = self.field.plane_slice(self._component(), self.slice_slider.value())
        lines = ["axis2\\axis1," + ",".join(f"{a:.4f}" for a in info["a1"])]
        for j, b in enumerate(info["a2"]):
            cells = [
                "" if not np.isfinite(v) else f"{v:.4f}" for v in info["values"][:, j]
            ]
            lines.append(f"{b:.4f}," + ",".join(cells))
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as e:
            logging.warning("[orca_nics_analyzer] grid CSV export: %s", e)
            QMessageBox.critical(
                self, "Export failed", f"Could not write the file:\n{e}"
            )
