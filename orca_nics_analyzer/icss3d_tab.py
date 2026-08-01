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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    import pyvista as pv
except ImportError:  # pyvista is an optional dependency
    pv = None

from . import cube_io
from .map2d_tab import COLORMAPS

#: Actor names, so a redraw replaces its own actors and nothing else.
ACTOR_POSITIVE = "nics_icss_positive"
ACTOR_NEGATIVE = "nics_icss_negative"
ACTOR_PLANE = "nics_map_plane"
ACTOR_CUT_AXIS = "nics_cut_axis"
ALL_ACTORS = (ACTOR_POSITIVE, ACTOR_NEGATIVE, ACTOR_PLANE, ACTOR_CUT_AXIS)


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

    def __init__(
        self,
        field,
        plotter_getter,
        plugin_version="0.0.0",
        parent=None,
        show_in_2d=None,
    ):
        super().__init__(parent)
        self.field = field
        self._plotter_getter = plotter_getter
        self.plugin_version = plugin_version
        self._show_in_2d = show_in_2d
        self._actors = set()
        self._ui_ready = False
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
        self.component.currentIndexChanged.connect(self._maybe_draw)
        grid.addWidget(self.component, 0, 1)

        grid.addWidget(QLabel("Isovalue / ppm:"), 0, 2)
        self.isovalue = QDoubleSpinBox()
        self.isovalue.setRange(0.01, 1000.0)
        self.isovalue.setDecimals(2)
        self.isovalue.setSingleStep(0.5)
        self.isovalue.setValue(5.0)
        self.isovalue.valueChanged.connect(self._sync_slider_from_spin)
        self.isovalue.valueChanged.connect(self._maybe_draw)
        grid.addWidget(self.isovalue, 0, 3)

        self.iso_slider = QSlider(Qt.Orientation.Horizontal)
        self.iso_slider.setRange(1, 1000)
        self.iso_slider.valueChanged.connect(self._on_iso_slider_changed)
        grid.addWidget(self.iso_slider, 1, 0, 1, 4)

        grid.addWidget(QLabel("Opacity:"), 2, 0)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.05, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setValue(0.55)
        self.opacity.valueChanged.connect(self._maybe_draw)
        grid.addWidget(self.opacity, 2, 1)

        self.show_positive = QCheckBox("Paratropic (+)")
        self.show_positive.setChecked(True)
        self.show_positive.toggled.connect(self._maybe_draw)
        grid.addWidget(self.show_positive, 2, 2)

        self.show_negative = QCheckBox("Diatropic (-)")
        self.show_negative.setChecked(True)
        self.show_negative.toggled.connect(self._maybe_draw)
        grid.addWidget(self.show_negative, 2, 3)

        grid.addWidget(QLabel("Colormap:"), 3, 0)
        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        self.cmap.currentIndexChanged.connect(self.draw)
        grid.addWidget(self.cmap, 3, 1)

        grid.addWidget(QLabel("Range +/- ppm:"), 3, 2)
        self.vmax = QDoubleSpinBox()
        self.vmax.setRange(0.1, 1000.0)
        self.vmax.setDecimals(2)
        self.vmax.setSingleStep(1.0)
        self.vmax.setValue(10.0)
        self.vmax.setEnabled(False)
        self.vmax.valueChanged.connect(self.draw)
        grid.addWidget(self.vmax, 3, 3)

        self.auto_range = QCheckBox("Auto")
        self.auto_range.setChecked(True)
        self.auto_range.toggled.connect(self._on_auto_toggled)
        grid.addWidget(
            self.auto_range, 3, 4 if grid.columnCount() > 4 else 3
        )  # or add next to vmax

        layout.addWidget(controls)

        # ---- Slice → 2D GroupBox ----
        self.slice_group = QGroupBox("Slice → 2D")
        s2d = QHBoxLayout(self.slice_group)

        self.stack_axis_label = QLabel("Cut axis:")
        s2d.addWidget(self.stack_axis_label)
        self.stack_axis_combo = QComboBox()
        self.stack_axis_combo.addItems(
            ["Lattice axis 1", "Lattice axis 2", "Lattice axis 3"]
        )
        self.stack_axis_combo.currentIndexChanged.connect(self._on_stack_axis_changed)
        s2d.addWidget(self.stack_axis_combo)

        self.slice_label = QLabel("Slice:")
        s2d.addWidget(self.slice_label)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        s2d.addWidget(self.slice_slider, 1)

        self.slice_spin = QSpinBox()
        self.slice_spin.setMinimum(0)
        self.slice_spin.valueChanged.connect(self._on_slice_spin_changed)
        s2d.addWidget(self.slice_spin)

        self.show_cut_axis = QCheckBox("Cut axis preview")
        self.show_cut_axis.setChecked(False)
        self.show_cut_axis.toggled.connect(self._maybe_draw)
        s2d.addWidget(self.show_cut_axis)

        self.goto_2d_btn = QPushButton("→ 2D Map tab")
        self.goto_2d_btn.setToolTip("Switch to the 2D Map tab to view this slice.")
        self.goto_2d_btn.clicked.connect(self._emit_show_in_2d)
        s2d.addWidget(self.goto_2d_btn)

        layout.addWidget(self.slice_group)

        row = QHBoxLayout()
        row.addStretch(1)
        draw = QPushButton("Refresh")
        draw.clicked.connect(self.draw)
        row.addWidget(draw)

        clear = QPushButton("Clear from 3D view")
        clear.clicked.connect(self.clear_actors)
        row.addWidget(clear)
        layout.addLayout(row)

        self._configure_slices()

        self._build_cube_ui(layout)

        self._ui_ready = True
        # Initial draw
        self.draw(silent=True)

    def _on_auto_toggled(self, checked):
        self.vmax.setEnabled(not checked)
        self._maybe_draw()

    def _configure_slices(self):
        if not self.field.is_gridded:
            return

        self.stack_axis_label.setVisible(True)
        self.stack_axis_combo.setVisible(True)
        self.stack_axis_combo.blockSignals(True)
        self.stack_axis_combo.setCurrentIndex(self.field.stack_axis_index())
        self.stack_axis_combo.blockSignals(False)

        try:
            info = self.field.plane_data(
                self.component.currentData() if hasattr(self, "component") else "zz"
            )
        except ValueError:
            return
        n = info["n_slices"]
        self.slice_slider.setMaximum(max(0, n - 1))
        self.slice_slider.setValue(info["slice_index"])

        self.slice_spin.blockSignals(True)
        self.slice_spin.setMaximum(max(0, n - 1))
        self.slice_spin.setValue(info["slice_index"])
        self.slice_spin.blockSignals(False)

        visible = n > 1
        self.slice_group.setVisible(visible)

    def _emit_show_in_2d(self):
        if self._show_in_2d is not None:
            self._show_in_2d()

    def _on_stack_axis_changed(self, index):
        self.field.set_stack_axis(index)
        self._configure_slices()
        self.update_cut_axis_preview()
        self._maybe_draw()
        if hasattr(self, "_on_slice_settings_changed"):
            self._on_slice_settings_changed()

    def _on_slice_changed(self, value):
        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(value)
        self.slice_spin.blockSignals(False)
        self.update_cut_axis_preview()
        self._maybe_draw()
        if (
            hasattr(self, "_on_slice_settings_changed")
            and self._on_slice_settings_changed
        ):
            self._on_slice_settings_changed()

    def _on_slice_spin_changed(self, value):
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(value)
        self.slice_slider.blockSignals(False)
        self.update_cut_axis_preview()
        self._maybe_draw()
        if (
            hasattr(self, "_on_slice_settings_changed")
            and self._on_slice_settings_changed
        ):
            self._on_slice_settings_changed()

    def _build_cube_ui(self, layout):
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

    def _on_iso_slider_changed(self, value):
        self._sync_spin_from_slider(value)
        self._maybe_draw()

    def _on_component_changed(self):
        self._auto_isovalue()
        self._update_cache_label()

    def _maybe_draw(self, *_):
        if not self._ui_ready:
            return
        self.draw()

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
        cmap = self.cmap.currentText()
        span = self.vmax.value()
        auto = self.auto_range.isChecked()
        return cmap, span, auto

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

    def draw(self, silent=False):
        if pv is None:
            if not silent:
                QMessageBox.information(
                    self,
                    "3D view",
                    "pyvista is not installed, so isosurfaces cannot be drawn.",
                )
            return
        if self.field.layout.get("kind") != "volume":
            if not silent:
                QMessageBox.information(
                    self,
                    "3D view",
                    "Isosurfaces need a regular 3D volume of probes.\n"
                    f"The detected layout is '{self.field.layout['kind']}'.",
                )
            return
        plotter = self._plotter()
        if plotter is None:
            if not silent:
                QMessageBox.warning(
                    self, "3D view", "The main 3D viewer is not available."
                )
            return

        component = self.component.currentData()
        cube, origin, steps = self.field.grid(component)
        if np.count_nonzero(np.isfinite(cube)) < 8:
            QMessageBox.information(self, "3D view", "Not enough data to contour.")
            return

        # Persist the rendered component beside the source output. This is
        # intentionally best-effort: memory-only fields and read-only folders
        # can still be visualized, while ensure_cube reuses a valid cache.
        if self.field.filename:
            try:
                self.field.ensure_cube(
                    component, plugin_version=self.plugin_version, force=False
                )
                self._update_cache_label()
            except (ValueError, OSError) as e:
                logging.warning("[orca_nics_analyzer] auto cube save: %s", e)

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

        self.update_cut_axis_preview()

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

    def update_cut_axis_preview(self, *_):
        """Update the arrow showing the current slice axis for 3D volumes."""
        if pv is None:
            return
        plotter = self._plotter()
        if plotter is None:
            return

        self._remove(plotter, ACTOR_CUT_AXIS)
        self._remove(plotter, ACTOR_CUT_AXIS + "_edge")

        # Draw cut axis plane for volumes if checked
        if (
            self.show_cut_axis.isChecked()
            and self.field.is_gridded
            and self.field.layout["kind"] == "volume"
        ):
            stack_idx = self.field.stack_axis_index()
            if stack_idx is not None:
                axes = self.field.layout["axes"]
                coords = self.field.layout["coords"]

                # The other two axes span the plane
                idx1, idx2 = [i for i in range(3) if i != stack_idx]

                # Center of the stack axis
                stack_center = np.mean(coords[stack_idx])

                # Bounds of the other two axes
                c1_min, c1_max = coords[idx1][0], coords[idx1][-1]
                c2_min, c2_max = coords[idx2][0], coords[idx2][-1]

                origin = self.field.layout["origin"]

                slice_idx = (
                    self.slice_slider.value() if hasattr(self, "slice_slider") else 0
                )
                idx_coord = (
                    float(coords[stack_idx][slice_idx])
                    if slice_idx < len(coords[stack_idx])
                    else stack_center
                )

                # Corners of the plane
                p00 = (
                    origin
                    + axes[stack_idx] * idx_coord
                    + axes[idx1] * c1_min
                    + axes[idx2] * c2_min
                )
                p10 = (
                    origin
                    + axes[stack_idx] * idx_coord
                    + axes[idx1] * c1_max
                    + axes[idx2] * c2_min
                )
                p01 = (
                    origin
                    + axes[stack_idx] * idx_coord
                    + axes[idx1] * c1_min
                    + axes[idx2] * c2_max
                )
                p11 = (
                    origin
                    + axes[stack_idx] * idx_coord
                    + axes[idx1] * c1_max
                    + axes[idx2] * c2_max
                )

                plane = pv.StructuredGrid()
                plane.points = np.vstack((p00, p10, p01, p11))
                plane.dimensions = (2, 2, 1)

                plotter.add_mesh(
                    plane,
                    color="#ff9900",
                    opacity=0.3,
                    name=ACTOR_CUT_AXIS,
                    show_scalar_bar=False,
                )
                # also add the outline
                plotter.add_mesh(
                    plane.extract_feature_edges(),
                    color="#ff9900",
                    line_width=2,
                    name=ACTOR_CUT_AXIS + "_edge",
                )
                self._actors.add(ACTOR_CUT_AXIS)
                self._actors.add(ACTOR_CUT_AXIS + "_edge")
        try:
            plotter.render()
        except Exception as e:  # host renderers may fail after widget teardown
            logging.debug("[orca_nics_analyzer] cut-axis render: %s", e)

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
        cached = self.field.cached_cube(component)
        if cached:
            info = cube_io.read_generation_settings(cached)
            grid = (
                "x".join(str(g) for g in info["grid"])
                if info["grid"]
                else "unknown grid"
            )
            version = f", v{info['version']}" if info["version"] else ""
            self.cache_label.setText(
                f"Cached: {os.path.basename(cached)} ({grid}{version})"
            )
        elif os.path.exists(path):
            self.cache_label.setText(
                f"Stale cache: {os.path.basename(path)}; regenerate to update it."
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
        action = "Reused cached" if cached else "Wrote cube (auto-saved)"
        self.status.setText(f"{action} cube: {path}")
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
