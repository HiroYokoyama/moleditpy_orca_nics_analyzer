"""NICS quantities, probe-layout reconstruction and ring perception.

Sign convention throughout: ``NICS = -sigma``. A negative NICS is diatropic
(aromatic), a positive one paratropic (antiaromatic).
"""

try:
    import numpy as np
except ImportError:  # CI installs pytest only
    np = None

from .elements import COVALENT_RADII
from .elements import DEFAULT_COVALENT_RADIUS as _DEFAULT_RADIUS


# ---------------------------------------------------------------------------
# Tensor -> NICS
# ---------------------------------------------------------------------------
def symmetrize(tensor):
    """(sigma + sigma^T) / 2.

    ORCA prints the unsymmetrized total shielding tensor. Only the symmetric
    part is observable, and the antisymmetric part would make the projected
    ``NICS_zz`` depend on the sign of the chosen normal, so it is dropped
    before any projection or diagonalization.
    """
    t = np.asarray(tensor, dtype=float)
    return 0.5 * (t + t.T)


def isotropic(tensor):
    t = np.asarray(tensor, dtype=float)
    return float(np.trace(t) / 3.0)


def nics_iso(entry):
    """NICS(iso) in ppm from a parser shielding entry, or None."""
    if entry is None:
        return None
    if entry.get("iso") is not None:
        return -float(entry["iso"])
    if entry.get("tensor") is not None:
        return -isotropic(entry["tensor"])
    return None


def nics_zz(entry, axis):
    """NICS_zz = -(n . sigma_sym . n) in ppm, or None when no tensor was printed."""
    if entry is None or entry.get("tensor") is None or axis is None:
        return None
    n = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    n = n / norm
    return -float(n @ symmetrize(entry["tensor"]) @ n)


def principal_components(entry):
    """(sigma_11, sigma_22, sigma_33) ascending, or None."""
    if entry is None or entry.get("tensor") is None:
        return None
    vals = np.linalg.eigvalsh(symmetrize(entry["tensor"]))
    return tuple(float(v) for v in sorted(vals))


def anisotropy(entry):
    """sigma_33 - (sigma_11 + sigma_22)/2, the usual shielding anisotropy."""
    pc = principal_components(entry)
    if pc is None:
        return entry.get("aniso") if entry else None
    s11, s22, s33 = pc
    return s33 - 0.5 * (s11 + s22)


#: +/- ppm band labelled non-aromatic, per component. The +/-2 ppm rule is
#: stated for NICS(iso); iso = (xx + yy + zz) / 3, so when the response is
#: purely out-of-plane the same ring current reads three times larger as
#: NICS_zz, and the band is scaled to match.
CLASSIFY_THRESHOLDS = {"iso": 2.0, "zz": 6.0}


def classify(value, component="iso"):
    """Coarse aromaticity label for a NICS value in ppm."""
    if value is None:
        return "-"
    threshold = CLASSIFY_THRESHOLDS[component]
    if value < -threshold:
        return "diatropic (aromatic)"
    if value > threshold:
        return "paratropic (antiaromatic)"
    return "non-aromatic"


# ---------------------------------------------------------------------------
# Ring perception on the real atoms
# ---------------------------------------------------------------------------
def bond_list(symbols, coords, tolerance=0.45):
    """Distance-based bonds: |rij| < r_i + r_j + tolerance."""
    pos = np.asarray(coords, dtype=float)
    radii = np.array([COVALENT_RADII.get(s, _DEFAULT_RADIUS) for s in symbols])
    bonds = []
    n = len(pos)
    for i in range(n):
        d = np.linalg.norm(pos[i + 1 :] - pos[i], axis=1)
        limit = radii[i] + radii[i + 1 :] + tolerance
        for offset in np.nonzero(d < limit)[0]:
            bonds.append((i, i + 1 + int(offset)))
    return bonds


def find_rings(symbols, coords, max_size=12):
    """Smallest rings, as tuples of atom indices.

    For every bond, the shortest path between its ends with that bond removed
    closes the smallest ring through it; deduplicating those gives the SSSR-like
    set NICS work needs, without an RDKit molecule to hand.
    """
    bonds = bond_list(symbols, coords)
    adj = {}
    for i, j in bonds:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    rings = {}
    for i, j in bonds:
        path = _shortest_path(adj, i, j, banned_edge=(i, j), limit=max_size - 1)
        if path is None:
            continue
        ring = tuple(path)
        key = frozenset(ring)
        if key not in rings:
            rings[key] = ring
    return sorted(rings.values(), key=lambda r: (len(r), r))


def _shortest_path(adj, start, goal, banned_edge, limit):
    banned = frozenset(banned_edge)
    queue = [(start, [start])]
    seen = {start}
    while queue:
        node, path = queue.pop(0)
        if len(path) > limit:
            continue
        for nxt in adj.get(node, ()):
            if frozenset((node, nxt)) == banned:
                continue
            if nxt == goal:
                return path + [goal]
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


def ring_centroid(positions):
    return np.asarray(positions, dtype=float).mean(axis=0)


def ring_normal(positions, reference=None):
    """Best-fit plane normal via SVD, sign-stabilized against *reference*."""
    pos = np.asarray(positions, dtype=float)
    centred = pos - pos.mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    normal = vt[2]
    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if np.dot(normal, ref) < 0:
            normal = -normal
    elif normal[np.argmax(np.abs(normal))] < 0:
        # Deterministic sign so repeated runs label faces the same way.
        normal = -normal
    return normal / np.linalg.norm(normal)


def ring_info(symbols, coords, max_size=12):
    """[{atoms, centroid, normal, size, planarity_rms}] for every ring found."""
    pos = np.asarray(coords, dtype=float)
    out = []
    for ring in find_rings(symbols, coords, max_size):
        rpos = pos[list(ring)]
        centroid = ring_centroid(rpos)
        normal = ring_normal(rpos)
        dev = (rpos - centroid) @ normal
        out.append(
            {
                "atoms": ring,
                "size": len(ring),
                "centroid": centroid,
                "normal": normal,
                "planarity_rms": float(np.sqrt(np.mean(dev**2))),
            }
        )
    return out


def nearest_ring(point, rings):
    """(ring, signed height along its normal, in-plane offset) for the nearest ring.

    "Nearest" is by 3D distance to the centroid; the height is what tells a
    NICS(0) probe from a NICS(1) one.
    """
    if not rings:
        return None, None, None
    p = np.asarray(point, dtype=float)
    best = min(rings, key=lambda r: np.linalg.norm(p - r["centroid"]))
    delta = p - best["centroid"]
    height = float(delta @ best["normal"])
    in_plane = float(np.linalg.norm(delta - height * best["normal"]))
    return best, height, in_plane


# ---------------------------------------------------------------------------
# Probe layout reconstruction
# ---------------------------------------------------------------------------
def _min_spacing(values, floor=1e-6):
    """Smallest non-zero gap between sorted values (0.0 when all coincide)."""
    s = np.sort(np.asarray(values, dtype=float))
    gaps = np.diff(s)
    gaps = gaps[gaps > floor]
    return float(gaps.min()) if len(gaps) else 0.0


def _cluster_1d(values, tol):
    """Sorted representative coordinates, merging values within *tol*."""
    order = np.argsort(values)
    reps = []
    members = []
    total = 0.0
    for idx in order:
        v = float(values[idx])
        if reps and abs(v - reps[-1]) <= tol:
            members[-1].append(idx)
            # Running mean: recomputing it over the whole group each time is
            # quadratic in the group size, which a large volume grid feels.
            total += v
            reps[-1] = total / len(members[-1])
        else:
            reps.append(v)
            members.append([idx])
            total = v
    return reps, members


def _neighbour_vectors(centred, k=6, sample=200):
    """Short difference vectors between neighbouring probes."""
    n = len(centred)
    idx = (
        np.arange(n)
        if n <= sample
        else np.unique(np.linspace(0, n - 1, sample).astype(int))
    )
    k = min(k, n - 1)
    vecs = []
    for i in idx:
        d = np.linalg.norm(centred - centred[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            vecs.append(centred[j] - centred[i])
    return np.asarray(vecs)


def _lattice_axes(centred, svd_axes, rank, ortho_tol=0.3):
    """Orthonormal frame aligned with the probe lattice.

    A square or cubic grid has degenerate principal axes, so SVD is free to
    return a frame rotated 45 degrees off the lattice — which would smear
    every grid plane across several clusters. The shortest neighbour vectors
    point along the lattice by construction, so they fix the frame; SVD is
    kept only to fill directions the probes do not span.
    """
    if rank < 1 or len(centred) < 2:
        return svd_axes
    vecs = _neighbour_vectors(centred)
    lengths = np.linalg.norm(vecs, axis=1)
    keep = lengths > 1e-8
    vecs, lengths = vecs[keep], lengths[keep]
    if len(vecs) == 0:
        return svd_axes
    order = np.argsort(lengths)

    chosen = []
    for i in order:
        v = vecs[i] / lengths[i]
        if any(abs(v @ c) > ortho_tol for c in chosen):
            continue
        chosen.append(v)
        if len(chosen) == rank:
            break
    if not chosen:
        return svd_axes

    # Gram-Schmidt the picked lattice directions, then complete the frame
    # with whatever SVD directions remain independent.
    basis = []
    for v in list(chosen) + list(svd_axes):
        w = v.astype(float).copy()
        for b in basis:
            w -= (w @ b) * b
        norm = np.linalg.norm(w)
        if norm > 1e-8:
            basis.append(w / norm)
        if len(basis) == 3:
            break
    return np.array(basis)


def detect_layout(points, rank_tol=1e-3, cluster_tol=None):
    """Reconstruct the probe arrangement from the ghost coordinates alone.

    Returns a dict with ``kind`` in {'none', 'single', 'line', 'plane',
    'volume', 'scattered'}. Regular arrangements also carry ``origin``,
    ``axes`` (3 orthonormal rows; the last is the plane/slab normal),
    ``shape`` and ``index_map`` — ``index_map[p]`` giving the grid indices of
    probe *p* — which is what the map and cube writers consume.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return {"kind": "none"}
    if len(pts) == 1:
        return {"kind": "single", "origin": pts[0], "axes": np.eye(3), "shape": (1,)}

    origin = pts.mean(axis=0)
    centred = pts - origin
    _, sv, vt = np.linalg.svd(centred, full_matrices=True)
    scale = float(sv[0]) if sv[0] > 0 else 1.0
    rank = int(np.sum(sv > rank_tol * scale))

    axes = _lattice_axes(centred, vt.copy(), rank)
    proj = centred @ axes.T  # coordinates in the lattice frame

    reps = []
    members = []
    for a in range(3):
        tol = cluster_tol
        if tol is None:
            # A quarter of the finest spacing along this axis: wide enough to
            # absorb the 6-decimal print rounding, narrow enough that adjacent
            # grid planes never merge.
            tol = max(1e-4, 0.25 * _min_spacing(proj[:, a]))
        r, m = _cluster_1d(proj[:, a], tol)
        reps.append(r)
        members.append(m)
    counts = tuple(len(r) for r in reps)

    kind = {0: "single", 1: "line", 2: "plane", 3: "volume"}.get(rank, "scattered")

    index_of = []
    for a in range(3):
        lookup = np.zeros(len(pts), dtype=int)
        for k, group in enumerate(members[a]):
            for p in group:
                lookup[p] = k
        index_of.append(lookup)
    index_map = np.stack(index_of, axis=1)

    # A matching count is not enough: two probes sharing one cell would leave
    # another cell empty yet still multiply out to the right total.
    regular = int(np.prod(counts)) == len(pts) and len(
        np.unique(index_map, axis=0)
    ) == len(pts)
    if not regular and kind in ("line", "plane", "volume"):
        kind = "scattered"

    # Maps cope with uneven spacing, but a cube carries one step per axis.
    uniform = True
    for r in reps:
        if len(r) > 2:
            gaps = np.diff(r)
            mean = float(np.mean(gaps))
            if float(np.max(np.abs(gaps - mean))) > max(1e-3, 0.02 * abs(mean)):
                uniform = False

    return {
        "kind": kind,
        "rank": rank,
        "origin": origin,
        "axes": axes,
        "coords": [np.array(r) for r in reps],
        "shape": counts,
        "index_map": index_map,
        "proj": proj,
        "regular": regular,
        "uniform": uniform,
    }


def layout_grid_values(layout, values):
    """Fill a dense array of *values* on the layout's grid; gaps become NaN."""
    shape = layout["shape"]
    grid = np.full(shape, np.nan, dtype=float)
    for p, (i, j, k) in enumerate(layout["index_map"]):
        grid[i, j, k] = values[p]
    return grid
