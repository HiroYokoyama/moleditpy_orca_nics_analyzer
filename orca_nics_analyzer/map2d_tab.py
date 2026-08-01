"""2D NICS map: filled contours of the probe plane, with the molecule on top.

The tab also hosts the 2D → 1D slicing controls: a fixed-axis selector and
a position slider let the user extract any row/column from the current 2D
plane and send it to the 1D Scan tab for display and export.
"""

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
    """Contour map of one slice of the probe grid.

    ``show_slice_in_1d`` is an optional callback ``(data_dict) -> None``
    supplied by the parent dialog to route extracted 1D slices to the scan tab.
    """

    def __init__(self, field, parent=None, show_in_3d=None, show_slice_in_1d=None):
        super().__init__(parent)
        self.field = field
        self._show_in_3d = show_in_3d
        self._show_slice_in_1d = show_slice_in_1d
        self._is_tab_visible = lambda: True
        self.canvas = None
        self.figure = None
        self._build_ui()
        self.refresh(force=True)

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

        self.figure = Figure(figsize=(6, 5), layout="constrained")
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

        # Slice controls moved to 3D tab per user request

        layout.addWidget(controls)

        # ---- 2D → 1D slice controls ------------------------------------
        slice1d_group = QGroupBox("Slice → 1D")
        slice1d_group.setToolTip(
            "Extract a 1D profile from the current 2D map by fixing one "
            "in-plane axis and sweeping the other."
        )
        s1d = QHBoxLayout(slice1d_group)

        s1d.addWidget(QLabel("Fix:"))
        self._slice1d_axis = QComboBox()
        self._slice1d_axis.addItem("Axis-1 row  (walk axis-2)", 0)
        self._slice1d_axis.addItem("Axis-2 col  (walk axis-1)", 1)
        self._slice1d_axis.currentIndexChanged.connect(self._on_slice1d_axis_changed)
        s1d.addWidget(self._slice1d_axis)

        s1d.addWidget(QLabel("Index:"))
        self._slice1d_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice1d_slider.setMinimum(0)
        self._slice1d_slider.setMaximum(0)
        self._slice1d_slider.valueChanged.connect(self._on_slice1d_changed)
        self._slice1d_slider.setToolTip("Row / column index to extract.")
        s1d.addWidget(self._slice1d_slider, 1)

        self._slice1d_spin = QSpinBox()
        self._slice1d_spin.setMinimum(0)
        self._slice1d_spin.setMaximum(0)
        self._slice1d_spin.valueChanged.connect(self._on_slice1d_spin_changed)
        s1d.addWidget(self._slice1d_spin)

        self._slice1d_btn = QPushButton("→ 1D Scan tab")
        self._slice1d_btn.setToolTip("Send this 1D profile to the 1D Scan tab.")
        self._slice1d_btn.clicked.connect(self._emit_slice_to_1d)
        s1d.addWidget(self._slice1d_btn)

        self.show_1d_line = QCheckBox("Show line in map")
        self.show_1d_line.setChecked(False)
        self.show_1d_line.toggled.connect(self.refresh)
        s1d.addWidget(self.show_1d_line)

        layout.addWidget(slice1d_group)

        # ---- bottom buttons --------------------------------------------
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

        # Set up the 2D→1D slider range based on the first in-plane axis.
        self._update_slice1d_range()

    def _update_slice1d_range(self):
        """Sync the 2D→1D index slider to the current in-plane axis choice."""
        if not self.field.is_gridded:
            return
        try:
            info = self.field.plane_data(self._component())
        except ValueError:
            return
        fixed_axis = self._slice1d_axis.currentData()
        n = len(info["a1"]) if fixed_axis == 0 else len(info["a2"])
        self._slice1d_slider.blockSignals(True)
        self._slice1d_slider.setMaximum(max(0, n - 1))
        self._slice1d_slider.setValue(max(0, n - 1) // 2)
        self._slice1d_slider.blockSignals(False)

        self._slice1d_spin.blockSignals(True)
        self._slice1d_spin.setMaximum(max(0, n - 1))
        self._slice1d_spin.setValue(self._slice1d_slider.value())
        self._slice1d_spin.blockSignals(False)

    def _on_auto_toggled(self, checked):
        self.vmax.setEnabled(not checked)
        self.refresh()

    def _on_slice1d_axis_changed(self):
        self._update_slice1d_range()
        self.refresh()

    def _on_slice1d_changed(self, value):
        self._slice1d_spin.blockSignals(True)
        self._slice1d_spin.setValue(value)
        self._slice1d_spin.blockSignals(False)
        self.refresh()

    def _on_slice1d_spin_changed(self, value):
        self._slice1d_slider.blockSignals(True)
        self._slice1d_slider.setValue(value)
        self._slice1d_slider.blockSignals(False)
        self.refresh()

    def _component(self):
        return self.component.currentData() if hasattr(self, "component") else "zz"

    # -- drawing ---------------------------------------------------------
    def refresh(self, force=False):
        if not force and not self._is_tab_visible():
            return
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

        # Get slice index from 3D tab if possible, otherwise use 0/middle
        slice_idx = 0
        if hasattr(self, "_get_slice_index"):
            slice_idx = self._get_slice_index()

        info = self.field.plane_slice(component, slice_idx)
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

        a1_offset = 0.0
        a2_offset = 0.0

        plot_a1 = info["a1"]
        plot_a2 = info["a2"]

        ax = self.figure.add_subplot(111)
        levels = np.linspace(-span, span, self.levels.value())
        # values is indexed [a1, a2]; contourf wants [row=y, col=x].
        mesh = ax.contourf(
            plot_a1,
            plot_a2,
            values.T,
            levels=levels,
            cmap=self.cmap.currentText(),
            extend="both",
        )
        if self.show_contours.isChecked():
            lines = ax.contour(
                plot_a1,
                plot_a2,
                values.T,
                levels=levels[:: max(1, len(levels) // 10)],
                colors="k",
                linewidths=0.4,
                alpha=0.5,
            )
            ax.clabel(lines, inline=True, fontsize=6, fmt="%.0f")

        if self.show_probes.isChecked():
            a1_mesh, a2_mesh = np.meshgrid(plot_a1, plot_a2, indexing="ij")
            ax.plot(a1_mesh.ravel(), a2_mesh.ravel(), "k.", markersize=2, alpha=0.4)

        if self.show_molecule.isChecked():
            self._draw_molecule(ax, info, a1_offset, a2_offset)

        # ---- 2D→1D crosshair overlay ------------------------------------
        self._draw_slice1d_crosshair(ax, info, a1_offset, a2_offset)

        label = "NICS$_{zz}$" if component == "zz" else "NICS(iso)"
        bar = self.figure.colorbar(mesh, ax=ax)
        bar.set_label(f"{label} / ppm")
        ax.set_xlabel("in-plane axis 1 / Å")
        ax.set_ylabel("in-plane axis 2 / Å")
        # Keep the map centered in the available axes beside the colorbar.
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_anchor("C")

        offset = self._slice_offset(info)
        title = f"{label} map"
        if info["n_slices"] > 1:
            title += f" — slice {info['slice_index'] + 1}/{info['n_slices']}"
        if offset is not None:
            title += f" ({offset:+.2f} Å from the ring plane)"
        ax.set_title(title, fontsize=10)

        if hasattr(self, "_set_slice_value_label"):
            self._set_slice_value_label("-" if offset is None else f"{offset:+.2f} Å")

        self.canvas.draw_idle()

    def _draw_slice1d_crosshair(self, ax, info, a1_offset, a2_offset):
        """Draw a dashed line on the map showing where the 1D slice will cut."""
        if not self.field.is_gridded:
            return
        if not getattr(self, "show_1d_line", None) or not self.show_1d_line.isChecked():
            return
        fixed_axis = self._slice1d_axis.currentData()
        idx = self._slice1d_slider.value()
        a1, a2 = info["a1"] - a1_offset, info["a2"] - a2_offset

        if fixed_axis == 0:
            # Fix a row of a1 -> x is fixed, draw a VERTICAL line at x = a1[idx]
            if idx < len(a1):
                x = a1[idx]
                ax.axvline(
                    x,
                    color="#ff9900",
                    lw=1.2,
                    linestyle="--",
                    alpha=0.85,
                    label=f"1D slice (axis-1 row {idx})",
                )
        else:
            # Fix a column of a2 -> y is fixed, draw a HORIZONTAL line at y = a2[idx]
            if idx < len(a2):
                y = a2[idx]
                ax.axhline(
                    y,
                    color="#ff9900",
                    lw=1.2,
                    linestyle="--",
                    alpha=0.85,
                    label=f"1D slice (axis-2 col {idx})",
                )

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

    def _draw_molecule(self, ax, info, a1_offset, a2_offset):
        """Bonds of the real molecule, projected onto the map plane."""
        coords = self.field.real_coords
        if len(coords) == 0:
            return
        origin = self.field.layout["origin"]
        rel = coords - origin
        u = (rel @ info["axis1"]) - a1_offset
        v = (rel @ info["axis2"]) - a2_offset
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
            slice_idx = 0
            if hasattr(self, "_get_slice_index"):
                slice_idx = self._get_slice_index()
            self._show_in_3d(self._component(), slice_idx)
        except Exception as e:  # the host viewer is out of our control
            logging.warning("[orca_nics_analyzer] show in 3D: %s", e)
            QMessageBox.warning(self, "3D view", f"Could not draw the plane:\n{e}")

    def _emit_slice_to_1d(self):
        """Extract the current 1D slice and send it to the 1D Scan tab."""
        if not self.field.is_gridded:
            QMessageBox.information(
                self,
                "Slice → 1D",
                "A regular grid is needed to extract a 1D slice.",
            )
            return
        try:
            slice_idx = 0
            if hasattr(self, "_get_slice_index"):
                slice_idx = self._get_slice_index()
            data = self.field.extract_line(
                component=self._component(),
                fixed_in_plane_axis=self._slice1d_axis.currentData(),
                fixed_index=self._slice1d_slider.value(),
                stack_index=slice_idx,
            )
        except (ValueError, IndexError) as e:
            logging.warning("[orca_nics_analyzer] extract_line: %s", e)
            QMessageBox.warning(self, "Slice → 1D", f"Could not extract slice:\n{e}")
            return

        if self._show_slice_in_1d is not None:
            self._show_slice_in_1d(data)
        else:
            QMessageBox.information(
                self,
                "Slice → 1D",
                "No 1D scan tab is connected to receive this slice.",
            )

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
        slice_idx = 0
        if hasattr(self, "_get_slice_index"):
            slice_idx = self._get_slice_index()
        info = self.field.plane_slice(self._component(), slice_idx)
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
