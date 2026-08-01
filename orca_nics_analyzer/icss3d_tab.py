"""3D ICSS tab: isosurfaces of the NICS field in the host's 3D viewer."""

import logging
import os

import numpy as np

from PyQt6.QtCore import Qt
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
    QVBoxLayout,
    QWidget,
)

try:
    import pyvista as pv
except ImportError:  # pyvista is an optional dependency
    pv = None

from . import cube_io

#: Actor names, so a redraw replaces its own actors and nothing else.
ACTOR_POSITIVE = "nics_icss_positive"
ACTOR_NEGATIVE = "nics_icss_negative"
ACTOR_PLANE = "nics_map_plane"
ALL_ACTORS = (ACTOR_POSITIVE, ACTOR_NEGATIVE, ACTOR_PLANE)


def structured_grid(data, origin, steps):
    """A PyVista grid for a field sampled on possibly non-orthogonal axes.

    A ring-frame NICS grid is not axis-aligned, so ImageData cannot represent
    it; the points are built explicitly instead.
    """
    if pv is None:
        raise RuntimeError("pyvista is not installed")
    data = np.asarray(data, dtype=float)
    n1, n2, n3 = data.shape
    i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
    pts = (
        np.asarray(origin, dtype=float)
        + i[..., None] * steps[0]
        + j[..., None] * steps[1]
        + k[..., None] * steps[2]
    )
    grid = pv.StructuredGrid()
    grid.points = pts.reshape(-1, 3, order="F")
    grid.dimensions = (n1, n2, n3)
    # order="F" above matches VTK's point ordering, so the values must be
    # flattened the same way or the field comes out transposed.
    grid["values"] = data.ravel(order="F")
    return grid


class Icss3DTab(QWidget):
    """Isovalue controls plus cube generation/caching."""

    def __init__(self, field, plotter_getter, plugin_version="0.0.0", parent=None):
        super().__init__(parent)
        self.field = field
        self._plotter_getter = plotter_getter
        self.plugin_version = plugin_version
        self._actors = set()
        self._build_ui()
        self._update_cache_label()

    # -- ui --------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        if pv is None:
            layout.addWidget(
                QLabel(
                    "pyvista is not installed, so isosurfaces cannot be drawn.\n"
                    "Cube export still works."
                )
            )

        controls = QGroupBox("Isosurface")
        grid = QGridLayout(controls)

        grid.addWidget(QLabel("Component:"), 0, 0)
        self.component = QComboBox()
        self.component.addItem("NICS_zz (ICSS)", "zz")
        self.component.addItem("NICS(iso)", "iso")
        if not self.field.has_tensors:
            self.component.setCurrentIndex(1)
            self.component.setEnabled(False)
        self.component.currentIndexChanged.connect(self._on_component_changed)
        grid.addWidget(self.component, 0, 1)

        grid.addWidget(QLabel("Isovalue / ppm:"), 0, 2)
        self.isovalue = QDoubleSpinBox()
        self.isovalue.setRange(0.01, 1000.0)
        self.isovalue.setDecimals(2)
        self.isovalue.setSingleStep(0.5)
        self.isovalue.setValue(5.0)
        self.isovalue.valueChanged.connect(self._sync_slider_from_spin)
        grid.addWidget(self.isovalue, 0, 3)

        self.iso_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_slider.setRange(1, 1000)
        self.iso_slider.valueChanged.connect(self._sync_spin_from_slider)
        grid.addWidget(self.iso_slider, 1, 0, 1, 4)

        grid.addWidget(QLabel("Opacity:"), 2, 0)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.05, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.55)
        grid.addWidget(self.opacity, 2, 1)

        self.show_positive = QCheckBox("Paratropic (+)")
        self.show_positive.setChecked(True)
        grid.addWidget(self.show_positive, 2, 2)

        self.show_negative = QCheckBox("Diatropic (-)")
        self.show_negative.setChecked(True)
        grid.addWidget(self.show_negative, 2, 3)

        layout.addWidget(controls)

        row = QHBoxLayout()
        draw = QPushButton("Draw isosurfaces")
        draw.clicked.connect(self.draw)
        row.addWidget(draw)
        clear = QPushButton("Clear from 3D view")
        clear.clicked.connect(self.clear_actors)
        row.addWidget(clear)
        row.addStretch(1)
        layout.addLayout(row)

        cube_group = QGroupBox("Cube file")
        cube_layout = QVBoxLayout(cube_group)
        self.cache_label = QLabel()
        self.cache_label.setWordWrap(True)
        cube_layout.addWidget(self.cache_label)
        cube_row = QHBoxLayout()
        gen = QPushButton("Generate / reuse cube")
        gen.setToolTip(
            "Write the cube beside the ORCA output, reusing the cached file "
            "when its grid still matches."
        )
        gen.clicked.connect(lambda: self.generate_cube(force=False))
        cube_row.addWidget(gen)
        regen = QPushButton("Regenerate")
        regen.setToolTip("Recompute and overwrite the cached cube.")
        regen.clicked.connect(lambda: self.generate_cube(force=True))
        cube_row.addWidget(regen)
        save_as = QPushButton("Save cube as...")
        save_as.clicked.connect(self.save_cube_as)
        cube_row.addWidget(save_as)
        cube_row.addStretch(1)
        cube_layout.addLayout(cube_row)
        layout.addWidget(cube_group)

        layout.addStretch(1)
        self._auto_isovalue()

    def _sync_slider_from_spin(self, value):
        self.iso_slider.blockSignals(True)
        self.iso_slider.setValue(int(max(1, min(1000, round(value * 10)))))
        self.iso_slider.blockSignals(False)

    def _sync_spin_from_slider(self, value):
        self.isovalue.blockSignals(True)
        self.isovalue.setValue(value / 10.0)
        self.isovalue.blockSignals(False)

    def _on_component_changed(self):
        self._auto_isovalue()
        self._update_cache_label()

    def _auto_isovalue(self):
        """A tenth of the peak magnitude usually frames the ring-current lobes."""
        values = self.field.values(self.component.currentData())
        finite = values[np.isfinite(values)]
        if finite.size:
            peak = float(np.max(np.abs(finite)))
            if peak > 0:
                self.isovalue.setValue(max(0.05, round(peak / 10.0, 2)))
        self._sync_slider_from_spin(self.isovalue.value())

    # -- drawing ---------------------------------------------------------
    def _plotter(self):
        try:
            return self._plotter_getter()
        except Exception as e:
            logging.warning("[orca_nics_analyzer] no plotter: %s", e)
            return None

    def _cmap_and_span(self):
        try:
            map_tab = self.parent().map_tab
            cmap = map_tab.cmap.currentText()
            span = map_tab.vmax.value()
            auto = map_tab.auto_range.isChecked()
            return cmap, span, auto
        except AttributeError:
            return "seismic", 10.0, True

    def _isosurface_colors(self):
        cmap_name, _, _ = self._cmap_and_span()
        try:
            import matplotlib.colors as mcolors
            try:
                from matplotlib import colormaps
                cmap = colormaps[cmap_name]
            except ImportError:
                from matplotlib import cm
                cmap = cm.get_cmap(cmap_name)

            color_neg = mcolors.to_hex(cmap(0.0))
            color_pos = mcolors.to_hex(cmap(1.0))
            return color_neg, color_pos
        except Exception:
            return "#3c6ec8", "#c8463c"

    def draw(self):
        if pv is None:
            QMessageBox.information(
                self,
                "3D view",
                "pyvista is not installed, so isosurfaces cannot be drawn.",
            )
            return
        if not self.field.is_gridded:
            QMessageBox.information(
                self,
                "3D view",
                "Isosurfaces need a regular 3D grid of probes.\n"
                f"The detected layout is '{self.field.layout['kind']}'.",
            )
            return
        plotter = self._plotter()
        if plotter is None:
            QMessageBox.warning(self, "3D view", "The main 3D viewer is not available.")
            return

        component = self.component.currentData()
        cube, origin, steps = self.field.grid(component)
        if np.count_nonzero(np.isfinite(cube)) < 8:
            QMessageBox.information(self, "3D view", "Not enough data to contour.")
            return

        self.clear_actors()
        grid = structured_grid(np.nan_to_num(cube, nan=0.0), origin, steps)
        level = self.isovalue.value()
        opacity = self.opacity.value()

        drawn = 0
        color_neg, color_pos = self._isosurface_colors()
        for enabled, value, colour, name in (
            (self.show_negative.isChecked(), -level, color_neg, ACTOR_NEGATIVE),
            (self.show_positive.isChecked(), level, color_pos, ACTOR_POSITIVE),
        ):
            if not enabled:
                continue
            try:
                surface = grid.contour(isosurfaces=[value], scalars="values")
            except (ValueError, RuntimeError) as e:
                logging.warning("[orca_nics_analyzer] contour %s: %s", value, e)
                continue
            if surface.n_points == 0:
                continue
            plotter.add_mesh(
                surface,
                color=colour,
                opacity=opacity,
                smooth_shading=True,
                name=name,
                show_scalar_bar=False,
            )
            self._actors.add(name)
            drawn += 1

        plotter.render()
        if drawn == 0:
            self.status.setText(
                f"No isosurface at +/-{level:.2f} ppm — try a smaller isovalue."
            )
        else:
            self.status.setText(
                f"Drew {drawn} isosurface(s) at +/-{level:.2f} ppm "
                f"on a {'x'.join(str(n) for n in cube.shape)} grid."
            )

    def show_plane(self, component, slice_index):
        """Drop one map slice into the 3D viewer as a coloured plane."""
        if pv is None:
            raise RuntimeError("pyvista is not installed")
        plotter = self._plotter()
        if plotter is None:
            raise RuntimeError("the main 3D viewer is not available")

        info = self.field.plane_slice(component, slice_index)
        values = info["values"]
        a1, a2 = info["a1"], info["a2"]
        i, j = np.meshgrid(np.arange(len(a1)), np.arange(len(a2)), indexing="ij")
        base = (
            self.field.layout["origin"]
            + float(self.field.layout["coords"][info["order"][2]][info["slice_index"]])
            * info["normal"]
        )
        pts = (
            base
            + (a1[i])[..., None] * info["axis1"]
            + (a2[j])[..., None] * info["axis2"]
        )
        plane = pv.StructuredGrid()
        plane.points = pts.reshape(-1, 3, order="F")
        plane.dimensions = (len(a1), len(a2), 1)
        plane["NICS"] = np.nan_to_num(values, nan=0.0).ravel(order="F")

        finite = values[np.isfinite(values)]
        cmap, span, auto = self._cmap_and_span()
        if auto:
            span = float(np.max(np.abs(finite))) if finite.size else 1.0

        self._remove(plotter, ACTOR_PLANE)
        plotter.add_mesh(
            plane,
            scalars="NICS",
            cmap=cmap,
            clim=(-span, span),
            opacity=0.85,
            name=ACTOR_PLANE,
            show_scalar_bar=True,
            scalar_bar_args={"title": "NICS / ppm"},
        )
        self._actors.add(ACTOR_PLANE)
        plotter.render()
        self.status.setText("Map plane added to the 3D view.")

    def _remove(self, plotter, name):
        try:
            plotter.remove_actor(name)
        except (KeyError, RuntimeError, AttributeError) as e:
            logging.debug("[orca_nics_analyzer] remove %s: %s", name, e)
        self._actors.discard(name)

    def clear_actors(self):
        """Remove everything this plugin added to the host viewer."""
        plotter = self._plotter()
        if plotter is None:
            self._actors.clear()
            return
        for name in ALL_ACTORS:
            self._remove(plotter, name)
        try:
            plotter.render()
        except (RuntimeError, AttributeError) as e:
            logging.debug("[orca_nics_analyzer] render after clear: %s", e)

    # -- cube ------------------------------------------------------------
    def _update_cache_label(self):
        component = self.component.currentData()
        path = self.field.cube_path(component)
        if path is None:
            self.cache_label.setText(
                "The source file location is unknown — use 'Save cube as...'."
            )
            return
        if os.path.exists(path):
            info = cube_io.read_generation_settings(path)
            grid = (
                "x".join(str(g) for g in info["grid"])
                if info["grid"]
                else "unknown grid"
            )
            version = f", v{info['version']}" if info["version"] else ""
            self.cache_label.setText(
                f"Cached: {os.path.basename(path)} ({grid}{version})"
            )
        else:
            self.cache_label.setText(f"Not generated yet: {os.path.basename(path)}")

    def generate_cube(self, force=False):
        if not self.field.is_gridded:
            QMessageBox.information(
                self,
                "Cube export",
                "A cube needs a regular grid of probes.\n"
                f"The detected layout is '{self.field.layout['kind']}'.",
            )
            return None
        component = self.component.currentData()
        try:
            path, cached = self.field.ensure_cube(
                component, plugin_version=self.plugin_version, force=force
            )
        except (ValueError, OSError) as e:
            logging.warning("[orca_nics_analyzer] cube generation: %s", e)
            QMessageBox.critical(self, "Cube export failed", str(e))
            return None
        self._update_cache_label()
        self.status.setText(f"{'Reused cached' if cached else 'Wrote'} cube: {path}")
        return path

    def save_cube_as(self):
        if not self.field.is_gridded:
            QMessageBox.information(
                self, "Cube export", "A cube needs a regular grid of probes."
            )
            return
        component = self.component.currentData()
        default = self.field.cube_path(component) or f"NICS_{component}.cube"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cube", default, "Gaussian cube (*.cube);;All Files (*)"
        )
        if not path:
            return
        try:
            self.field.write_cube(
                component, path=path, plugin_version=self.plugin_version
            )
        except (ValueError, OSError) as e:
            logging.warning("[orca_nics_analyzer] save cube as: %s", e)
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self.status.setText(f"Wrote cube: {path}")
