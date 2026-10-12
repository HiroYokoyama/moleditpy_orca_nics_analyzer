"""Gaussian cube read/write for NICS and ICSS fields.

Cubes are written beside the ORCA output in a ``<base>_nics_cubes`` folder, so
the field on screen is always available to other cube viewers. The stamp on
the second line records what a cube was made from; a cube that still matches
is left alone rather than rewritten. The plugin never reads its own cubes
back: rebuilding the field from the parsed output is cheaper than parsing one.
:func:`read_cube` is the matching reader for callers and tests that do.
"""

import logging
import math
import os
import re
import tempfile
from urllib.parse import quote, unquote

try:
    import numpy as np
except ImportError:  # CI installs pytest only
    np = None

from .elements import ATOMIC_NUMBERS, BOHR_PER_ANGSTROM

logger = logging.getLogger(__name__)

CUBE_HEADER_PREFIX = "MoleditPy ORCA NICS Analyzer"


def cube_dir_for(out_path):
    """``<output>_nics_cubes`` beside the ORCA file (not created here)."""
    if not out_path:
        return None
    base = os.path.splitext(os.path.basename(out_path))[0]
    return os.path.join(
        os.path.dirname(os.path.abspath(out_path)), f"{base}_nics_cubes"
    )


def cube_path_for(out_path, component, tag=None):
    """Path of the cached cube for a component ('iso' / 'zz') and optional tag."""
    folder = cube_dir_for(out_path)
    if folder is None:
        return None
    base = os.path.splitext(os.path.basename(out_path))[0]
    suffix = f"_{_sanitize(tag)}" if tag else ""
    return os.path.join(folder, f"{base}_NICS_{_sanitize(component)}{suffix}.cube")


def _sanitize(text):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "field"


def stamp_line(version, component, shape, axis=None, source=None, axis_mode=None):
    """The second cube comment line, recording how the field was made."""
    parts = [
        f"{CUBE_HEADER_PREFIX} v{version}",
        "generation_conditions:",
        f"component={component}",
        f"grid={'x'.join(str(int(n)) for n in shape)}",
    ]
    if axis is not None:
        parts.append("axis=" + ",".join(f"{float(v):.6f}" for v in axis))
    if axis_mode is not None:
        parts.append(f"axis_mode={axis_mode}")
    if source:
        parts.append(f"source={quote(os.path.basename(source), safe='')}")
        try:
            stat = os.stat(source)
            parts.append(f"source_size={stat.st_size}")
            parts.append(f"source_mtime_ns={stat.st_mtime_ns}")
        except OSError:
            pass
    return " ".join(parts)


def read_generation_settings(filepath):
    """What a cached cube was written with; unknown fields come back None."""
    info = {
        "version": None,
        "component": None,
        "grid": None,
        "axis": None,
        "axis_mode": None,
        "source_name": None,
        "source_size": None,
        "source_mtime_ns": None,
    }
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            fh.readline()
            stamp = fh.readline()
    except OSError as e:
        logger.warning("[orca_nics_analyzer] cube header %s: %s", filepath, e)
        return info

    m = re.search(r"ORCA NICS Analyzer v(\S+)", stamp)
    if m:
        info["version"] = m.group(1)
    m = re.search(r"component=(\S+)", stamp)
    if m:
        info["component"] = m.group(1)
    m = re.search(r"grid=(\d+)x(\d+)x(\d+)", stamp)
    if m:
        info["grid"] = tuple(int(g) for g in m.groups())
    m = re.search(r"axis=([-0-9.]+),([-0-9.]+),([-0-9.]+)", stamp)
    if m:
        info["axis"] = tuple(float(g) for g in m.groups())
    m = re.search(r"axis_mode=(\S+)", stamp)
    if m:
        info["axis_mode"] = m.group(1)
    m = re.search(r"source=(\S+)", stamp)
    if m:
        info["source_name"] = unquote(m.group(1))
    m = re.search(r"source_size=(\d+)", stamp)
    if m:
        info["source_size"] = int(m.group(1))
    m = re.search(r"source_mtime_ns=(\d+)", stamp)
    if m:
        info["source_mtime_ns"] = int(m.group(1))
    return info


MAX_CUBE_BYTES = 256 * 1024 * 1024
MAX_CUBE_VALUES = 16_000_000
MAX_CUBE_ATOMS = 200_000


def _finite_number(text):
    value = float(text.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError("Cube values and coordinates must be finite")
    return value


def _validate_grid(origin, vectors):
    if origin.shape != (3,) or vectors.shape != (3, 3):
        raise ValueError("Cube origin and vectors must have shapes (3,) and (3,3)")
    if not np.isfinite(origin).all() or not np.isfinite(vectors).all():
        raise ValueError("Cube geometry must be finite")
    if np.linalg.matrix_rank(vectors) != 3:
        raise ValueError("Cube voxel vectors must be independent")


def write_cube(
    path,
    data,
    origin,
    vectors,
    symbols=(),
    coords=(),
    comment=f"{CUBE_HEADER_PREFIX}: NICS field (ppm)",
    stamp="",
):
    """Write *data* as a cube. ``origin``/``vectors``/``coords`` are Angstrom.

    ``vectors`` are the three voxel step vectors, which for a ring-frame grid
    are not axis-aligned — the cube format carries them verbatim, so no
    resampling onto a lab-aligned box is needed.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"cube data must be 3D, got shape {arr.shape}")

    if any(n <= 0 for n in arr.shape) or arr.size > MAX_CUBE_VALUES:
        raise ValueError("Unsupported cube grid size")
    if not np.isfinite(arr).all():
        raise ValueError(
            "Cube fields must contain finite values; missing data cannot be replaced with zero"
        )
    org = np.asarray(origin, dtype=float) * BOHR_PER_ANGSTROM
    vecs = np.asarray(vectors, dtype=float) * BOHR_PER_ANGSTROM
    _validate_grid(org, vecs)
    symbols, coords = list(symbols), list(coords)
    if len(symbols) != len(coords) or len(symbols) > MAX_CUBE_ATOMS:
        raise ValueError("Cube atom symbols and coordinates must match")
    atom_coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(atom_coords) != len(symbols) or not np.isfinite(atom_coords).all():
        raise ValueError("Cube atom coordinates must be finite 3-vectors")
    if any(ch in str(comment) + str(stamp) for ch in "\r\n"):
        raise ValueError("Cube comments must each be one line")
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    atoms = list(zip(symbols, coords))

    lines = [comment, stamp or f"{CUBE_HEADER_PREFIX}: generated field"]
    lines.append(f"{len(atoms):5d}{org[0]:12.6f}{org[1]:12.6f}{org[2]:12.6f}")
    for n, v in zip(arr.shape, vecs):
        lines.append(f"{n:5d}{v[0]:12.6f}{v[1]:12.6f}{v[2]:12.6f}")
    for sym, xyz in atoms:
        z = ATOMIC_NUMBERS.get(str(sym).rstrip(":"), 0)
        p = np.asarray(xyz, dtype=float) * BOHR_PER_ANGSTROM
        lines.append(f"{z:5d}{float(z):12.6f}{p[0]:12.6f}{p[1]:12.6f}{p[2]:12.6f}")

    # Gaussian's data order: x slowest, z fastest, six values per line.
    flat = arr.ravel(order="C")
    for start in range(0, flat.size, 6):
        lines.append("".join(f"{v:13.5E}" for v in flat[start : start + 6]))

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=folder or ".", delete=False
        ) as fh:
            temporary = fh.name
            fh.write("\n".join(lines) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_cube(path):
    """Read a cube written by this plugin (or any standard writer).

    Returns ``{data, origin, vectors, symbols, coords, comment, stamp}`` with
    lengths back in Angstrom.
    """
    if os.path.getsize(path) > MAX_CUBE_BYTES:
        raise ValueError("Cube file exceeds the supported 256 MiB size")
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read(MAX_CUBE_BYTES + 1)
    if len(text) > MAX_CUBE_BYTES:
        raise ValueError("Cube file exceeds the supported 256 MiB size")
    tokens = text.splitlines()
    if len(tokens) < 6:
        raise ValueError(f"cube {os.path.basename(path)} has an incomplete header")
    comment, stamp = tokens[0], tokens[1]
    header = tokens[2].split()
    if len(header) not in (4, 5):
        raise ValueError("Cube has an invalid origin header")
    natoms = int(header[0])
    has_dset = natoms < 0
    natoms = abs(natoms)
    if natoms > MAX_CUBE_ATOMS or len(tokens) < 6 + natoms:
        raise ValueError("Cube has an invalid or incomplete atom block")
    if len(header) == 5 and int(header[4]) != 1:
        raise ValueError("Multi-field cubes are not supported")
    origin = np.array([_finite_number(v) for v in header[1:4]])
    shape, vectors, signs = [], [], []
    for row in tokens[3:6]:
        parts = row.split()
        if len(parts) != 4:
            raise ValueError("Cube has an invalid voxel header")
        count = int(parts[0])
        if count == 0:
            raise ValueError("Cube voxel counts must be nonzero")
        shape.append(abs(count))
        signs.append(count < 0)
        vectors.append([_finite_number(v) for v in parts[1:4]])
    total = math.prod(shape)
    if total > MAX_CUBE_VALUES:
        raise ValueError("Cube grid exceeds the supported value count")
    if any(signs) and not all(signs):
        raise ValueError("Cube voxel units must be consistent across all axes")
    vectors = np.asarray(vectors)
    _validate_grid(origin, vectors)
    symbols, coords = [], []
    inv = {v: k for k, v in ATOMIC_NUMBERS.items()}
    for row in tokens[6 : 6 + natoms]:
        parts = row.split()
        if len(parts) != 5:
            raise ValueError("Cube has an invalid atom row")
        z = int(parts[0])
        _finite_number(parts[1])
        symbols.append(inv.get(z, "X"))
        coords.append([_finite_number(v) for v in parts[2:5]])
    rest = (v for row in tokens[6 + natoms :] for v in row.split())
    if has_dset:
        try:
            datasets = int(next(rest))
            if datasets != 1:
                raise ValueError("Multi-field cubes are not supported")
            int(next(rest))
        except StopIteration as exc:
            raise ValueError("Cube dataset identifiers are incomplete") from exc

    def values():
        for i, token in enumerate(rest):
            if i >= total:
                raise ValueError("Cube contains more values than expected")
            yield _finite_number(token)

    data = np.fromiter(values(), dtype=float)
    if data.size != total:
        raise ValueError(f"cube has {data.size} values, expected {total}")
    ang = 1.0 if all(signs) else 1.0 / BOHR_PER_ANGSTROM
    return {
        "comment": comment,
        "stamp": stamp,
        "data": data.reshape(shape),
        "origin": origin * ang,
        "vectors": vectors * ang,
        "symbols": symbols,
        "coords": np.array(coords) * ang if coords else np.zeros((0, 3)),
    }
