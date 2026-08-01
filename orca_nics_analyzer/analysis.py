"""Turns a parsed ORCA output into the probe records and fields the tabs show.

All the GUI-independent logic lives here so it can be tested headlessly.
"""

import logging
import os

try:
    import numpy as np
except ImportError:  # CI installs pytest only
    np = None

from . import nics_math as nm
from . import cube_io

#: How the z of ``NICS_zz`` is chosen.
AXIS_MODES = ("grid", "ring", "x", "y", "z")

COMPONENTS = ("iso", "zz")


class NicsField:
    """Probe table, layout and grid fields for one ORCA output."""

    def __init__(self, parser, axis_mode="grid", custom_axis=None):
        self.parser = parser
        self.filename = getattr(parser, "filename", None)
        self.axis_mode = axis_mode
        self.custom_axis = custom_axis

        data = parser.data
        self.warnings = list(data.get("warnings", []))
        self.has_tensors = bool(data.get("has_tensors"))

        real = parser.real_atoms
        self.real_symbols = [a["symbol"] for a in real]
        self.real_coords = (
            np.array([a["xyz"] for a in real], dtype=float)
            if real
            else np.zeros((0, 3))
        )
        self.rings = (
            nm.ring_info(self.real_symbols, self.real_coords)
            if len(self.real_coords) >= 3
            else []
        )

        probes = parser.probes
        self.probe_indices = [p[0] for p in probes]
        self.probe_coords = (
            np.array([p[1] for p in probes], dtype=float)
            if probes
            else np.zeros((0, 3))
        )
        self.entries = [p[2] for p in probes]

        self.layout = nm.detect_layout(self.probe_coords)
        self._axes_cache = None
        self.probes = self._build_probes()

    # -- axes ------------------------------------------------------------
    def stack_axis_index(self):
        """Which lattice axis the grid is sliced along.

        For a plane it is the flat axis. For a volume it is the axis most
        parallel to the molecule's ring normal — an ICSS map is conventionally
        read as slices parallel to the ring, and picking by index or by point
        count instead would slice a cubic grid edge-on.
        """
        layout = self.layout
        shape = layout.get("shape")
        if layout["kind"] == "plane":
            return [a for a in range(3) if shape[a] <= 1][0]
        if layout["kind"] == "volume":
            ref = self.mean_ring_normal
            if ref is not None:
                return int(np.argmax(np.abs(layout["axes"] @ ref)))
            return int(np.argmin(shape))
        if layout["kind"] == "line":
            return 0
        return None

    @property
    def mean_ring_normal(self):
        """Average ring normal, sign-aligned, or None when no rings were found."""
        if not self.rings:
            return None
        ref = self.rings[0]["normal"]
        acc = np.zeros(3)
        for r in self.rings:
            n = r["normal"]
            acc += n if n @ ref >= 0 else -n
        norm = np.linalg.norm(acc)
        return acc / norm if norm > 1e-9 else ref

    @property
    def grid_normal(self):
        """Normal of the probe plane/slab, or None when the probes are not planar."""
        if self.layout["kind"] in ("plane", "volume", "line"):
            return self.layout["axes"][self.stack_axis_index()]
        return None

    def axis_for(self, probe):
        """The z direction used for this probe's NICS_zz."""
        mode = self.axis_mode
        if mode == "x":
            return np.array([1.0, 0.0, 0.0])
        if mode == "y":
            return np.array([0.0, 1.0, 0.0])
        if mode == "z":
            return np.array([0.0, 0.0, 1.0])
        if mode == "custom" and self.custom_axis is not None:
            return np.asarray(self.custom_axis, dtype=float)
        if mode == "ring":
            ring = probe.get("ring")
            if ring is not None:
                return ring["normal"]
        # "grid", and the fallbacks: the probe plane's own normal, else the
        # nearest ring's, else the lab z.
        normal = self.grid_normal
        if normal is not None:
            return normal
        ring = probe.get("ring")
        if ring is not None:
            return ring["normal"]
        return np.array([0.0, 0.0, 1.0])

    # -- probes ----------------------------------------------------------
    def _build_probes(self):
        out = []
        for n, idx in enumerate(self.probe_indices):
            entry = self.entries[n]
            xyz = self.probe_coords[n]
            ring, height, in_plane = nm.nearest_ring(xyz, self.rings)
            probe = {
                "n": n,
                "idx": idx,
                "xyz": xyz,
                "entry": entry,
                "symbol": entry.get("symbol", "Bq"),
                "ring": ring,
                "ring_id": self.rings.index(ring) if ring is not None else None,
                "height": height,
                "in_plane": in_plane,
                "iso": nm.nics_iso(entry),
            }
            probe["zz"] = nm.nics_zz(entry, self.axis_for(probe))
            probe["classification"] = nm.classify(
                probe["zz"] if probe["zz"] is not None else probe["iso"]
            )
            out.append(probe)
        return out

    def set_axis_mode(self, mode, custom_axis=None):
        self.axis_mode = mode
        self.custom_axis = custom_axis
        for probe in self.probes:
            probe["zz"] = nm.nics_zz(probe["entry"], self.axis_for(probe))
            probe["classification"] = nm.classify(
                probe["zz"] if probe["zz"] is not None else probe["iso"]
            )

    def values(self, component):
        """Flat list of NICS values in probe order; missing entries are NaN."""
        key = "iso" if component == "iso" else "zz"
        return np.array(
            [np.nan if p[key] is None else p[key] for p in self.probes], dtype=float
        )

    # -- gridded fields --------------------------------------------------
    @property
    def is_gridded(self):
        return self.layout["kind"] in ("plane", "volume") and self.layout.get("regular")

    def grid(self, component):
        """(values[i,j,k], origin, step-vectors) in the lattice frame, Angstrom."""
        if not self.is_gridded:
            raise ValueError("probe layout is not a regular grid")
        layout = self.layout
        cube = nm.layout_grid_values(layout, self.values(component))

        axes = layout["axes"]
        coords = layout["coords"]
        steps = []
        for a in range(3):
            if len(coords[a]) > 1:
                steps.append(float(coords[a][1] - coords[a][0]) * axes[a])
            else:
                # A flat axis still needs a non-zero voxel vector for the cube
                # format; 1 Angstrom keeps the slab visibly thin.
                steps.append(1.0 * axes[a])
        origin = layout["origin"] + sum(float(coords[a][0]) * axes[a] for a in range(3))
        return cube, origin, np.array(steps)

    def plane_data(self, component):
        """(2D values, axis-1 coords, axis-2 coords, in-plane axes, normal, origin).

        For a volume this returns the middle slice, so a 2D view always has
        something sane to show.
        """
        cube, origin, steps = self.grid(component)
        stack = self.stack_axis_index()
        order = [a for a in range(3) if a != stack] + [stack]

        arr = np.transpose(cube, order)
        if arr.ndim == 3 and arr.shape[2] > 1:
            arr = arr[:, :, arr.shape[2] // 2]
            slice_index = cube.shape[order[2]] // 2
        else:
            arr = arr.reshape(arr.shape[0], arr.shape[1])
            slice_index = 0

        coords = self.layout["coords"]
        a1 = np.asarray(coords[order[0]], dtype=float)
        a2 = np.asarray(coords[order[1]], dtype=float)
        axes = self.layout["axes"]
        return {
            "values": arr,
            "a1": a1,
            "a2": a2,
            "axis1": axes[order[0]],
            "axis2": axes[order[1]],
            "normal": axes[order[2]],
            "origin": origin,
            "steps": steps,
            "order": order,
            "slice_index": slice_index,
            "n_slices": cube.shape[order[2]],
        }

    def plane_slice(self, component, index):
        """The *index*-th slice of a volume, in the same shape as plane_data."""
        info = self.plane_data(component)
        cube, _, _ = self.grid(component)
        arr = np.transpose(cube, info["order"])
        index = max(0, min(int(index), arr.shape[2] - 1))
        info["values"] = arr[:, :, index]
        info["slice_index"] = index
        return info

    # -- 1D scan ---------------------------------------------------------
    @property
    def is_scan(self):
        """True when the probes lie on a line — the classic NICS scan."""
        return self.layout["kind"] == "line"

    def line_data(self, component=None):
        """A NICS scan as ordered (distance, value) pairs.

        The abscissa is the signed height above the nearest ring plane when a
        ring is available — that is what a NICS scan is read against — and
        otherwise the arc length along the probe line from its first point.
        """
        if self.layout["kind"] not in ("line", "single"):
            raise ValueError("probe layout is not a line")

        axis = self.layout["axes"][0]
        origin = self.layout["origin"]
        along = (self.probe_coords - origin) @ axis

        heights = [p["height"] for p in self.probes]
        use_height = self.rings and all(h is not None for h in heights)
        if use_height:
            distance = np.array(heights, dtype=float)
            label = "height above the ring plane / A"
            # A scan running "downwards" would otherwise plot back to front.
            if np.corrcoef(along, distance)[0, 1] < 0:
                axis = -axis
        else:
            distance = along - along.min()
            label = "distance along the scan / A"

        order = np.argsort(distance)
        result = {
            "distance": distance[order],
            "label": label,
            "axis": axis,
            "indices": [self.probes[i]["idx"] for i in order],
            "order": order,
            "offsets": np.array(
                [self.probes[i]["in_plane"] or 0.0 for i in order], dtype=float
            ),
        }
        for comp in COMPONENTS:
            result[comp] = self.values(comp)[order]
        if component is not None:
            result["values"] = result[component]
        return result

    def scan_extremum(self, component="zz"):
        """(distance, value) of the largest-magnitude point of a scan."""
        data = self.line_data()
        values = data[component]
        finite = np.isfinite(values)
        if not finite.any():
            return None, None
        idx = int(np.nanargmax(np.abs(np.where(finite, values, np.nan))))
        return float(data["distance"][idx]), float(values[idx])

    def scan_csv(self):
        data = self.line_data()
        lines = ["Index,Distance/A,Offset/A,NICS(iso)/ppm,NICS_zz/ppm"]
        for n, idx in enumerate(data["indices"]):
            iso = data["iso"][n]
            zz = data["zz"][n]
            lines.append(
                f"{idx},{data['distance'][n]:.4f},{data['offsets'][n]:.4f},"
                f"{'' if not np.isfinite(iso) else f'{iso:.4f}'},"
                f"{'' if not np.isfinite(zz) else f'{zz:.4f}'}"
            )
        return "\n".join(lines) + "\n"

    # -- cube export -----------------------------------------------------
    def cube_path(self, component, tag=None):
        return cube_io.cube_path_for(self.filename, component, tag)

    def write_cube(self, component, path=None, plugin_version="0.0.0", tag=None):
        """Write (or overwrite) the cube for *component*; returns its path."""
        cube, origin, steps = self.grid(component)
        path = path or self.cube_path(component, tag)
        if path is None:
            raise ValueError("no output path — the source file location is unknown")
        axis = self.grid_normal
        label = "NICS_iso" if component == "iso" else "NICS_zz"
        cube_io.write_cube(
            path,
            cube,
            origin,
            steps,
            symbols=self.real_symbols,
            coords=self.real_coords,
            comment=f"{label} field in ppm (NICS = -sigma)",
            stamp=cube_io.stamp_line(
                plugin_version, component, cube.shape, axis, self.filename
            ),
        )
        return path

    def cached_cube(self, component, tag=None):
        """The cached cube for *component* if one is on disk and still matches."""
        path = self.cube_path(component, tag)
        if not path or not os.path.exists(path):
            return None
        info = cube_io.read_generation_settings(path)
        if self.is_gridded:
            try:
                expected = self.grid(component)[0].shape
            except ValueError:
                expected = None
            if info.get("grid") and expected and tuple(info["grid"]) != tuple(expected):
                return None
        return path

    def ensure_cube(self, component, plugin_version="0.0.0", tag=None, force=False):
        """Reuse the cached cube, or write it. Returns (path, was_cached)."""
        if not force:
            cached = self.cached_cube(component, tag)
            if cached:
                return cached, True
        return self.write_cube(component, plugin_version=plugin_version, tag=tag), False

    # -- text export -----------------------------------------------------
    def probe_rows(self):
        """Rows for the probe table / CSV, as plain strings and floats."""
        rows = []
        for p in self.probes:
            x, y, z = p["xyz"]
            rows.append(
                {
                    "index": p["idx"],
                    "symbol": p["symbol"],
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "ring": "-" if p["ring_id"] is None else f"R{p['ring_id'] + 1}",
                    "ring_size": "-" if p["ring"] is None else p["ring"]["size"],
                    "height": p["height"],
                    "in_plane": p["in_plane"],
                    "nics_iso": p["iso"],
                    "nics_zz": p["zz"],
                    "class": p["classification"],
                }
            )
        return rows

    CSV_COLUMNS = (
        ("index", "Index"),
        ("symbol", "Symbol"),
        ("x", "X/A"),
        ("y", "Y/A"),
        ("z", "Z/A"),
        ("ring", "Nearest ring"),
        ("ring_size", "Ring size"),
        ("height", "Height/A"),
        ("in_plane", "Offset/A"),
        ("nics_iso", "NICS(iso)/ppm"),
        ("nics_zz", "NICS_zz/ppm"),
        ("class", "Character"),
    )

    def to_csv(self):
        keys = [k for k, _ in self.CSV_COLUMNS]
        out = [",".join(title for _, title in self.CSV_COLUMNS)]
        for row in self.probe_rows():
            cells = []
            for k in keys:
                v = row[k]
                if v is None:
                    cells.append("")
                elif isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            out.append(",".join(cells))
        return "\n".join(out) + "\n"

    def summary_text(self, plugin_version="0.0.0"):
        """Human-readable header shown in the GUI and written into the report."""
        lines = [
            f"ORCA NICS Analyzer v{plugin_version}",
            f"Source: {os.path.basename(self.filename) if self.filename else '(memory)'}",
            f"ORCA version: {self.parser.data.get('orca_version') or 'unknown'}",
            f"Probes: {len(self.probes)}   Real atoms: {len(self.real_coords)}"
            f"   Rings: {len(self.rings)}",
            f"Layout: {self.layout['kind']}"
            + (
                f" {'x'.join(str(n) for n in self.layout['shape'])}"
                if self.layout.get("shape") and self.layout["kind"] != "single"
                else ""
            ),
            f"Shielding tensors: {'yes' if self.has_tensors else 'no (isotropic only)'}",
            f"NICS_zz axis: {self.axis_mode}",
        ]
        for w in self.warnings:
            lines.append(f"Warning: {w}")
        return "\n".join(lines)


def load_field(path, axis_mode="grid"):
    """Convenience: parse *path* and wrap it in a :class:`NicsField`."""
    from .parser import NicsParser

    parser = NicsParser()
    parser.load(path)
    return NicsField(parser, axis_mode=axis_mode)


def export_all(field, folder=None, plugin_version="0.0.0"):
    """Write every exportable artefact for *field* into one folder.

    Returns the list of paths written. Cube export is skipped, with a note in
    the returned list, when the probes do not form a regular grid.
    """
    if folder is None:
        folder = cube_io.cube_dir_for(field.filename)
    if folder is None:
        raise ValueError("no output folder — the source file location is unknown")
    os.makedirs(folder, exist_ok=True)

    base = (
        os.path.splitext(os.path.basename(field.filename))[0]
        if field.filename
        else "nics"
    )
    written = []

    csv_path = os.path.join(folder, f"{base}_NICS_probes.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(field.to_csv())
    written.append(csv_path)

    if field.is_scan:
        scan_path = os.path.join(folder, f"{base}_NICS_scan.csv")
        with open(scan_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(field.scan_csv())
        written.append(scan_path)

    txt_path = os.path.join(folder, f"{base}_NICS_summary.txt")
    with open(txt_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(field.summary_text(plugin_version) + "\n")
    written.append(txt_path)

    if field.is_gridded:
        for component in COMPONENTS:
            if component == "zz" and not field.has_tensors:
                continue
            try:
                written.append(
                    field.write_cube(
                        component,
                        path=os.path.join(folder, f"{base}_NICS_{component}.cube"),
                        plugin_version=plugin_version,
                    )
                )
            except (ValueError, OSError) as e:
                logging.warning(
                    "[orca_nics_analyzer] cube export (%s): %s", component, e
                )
    return written
