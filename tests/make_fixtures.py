"""Generate synthetic ORCA outputs carrying NICS ghost data.

The blocks are byte-for-byte in ORCA 5/6 layout (verified against a real
benzene NMR output), but the shielding values come from an analytic ring
current point-dipole model, so every expected NICS value in the tests can be
computed independently of the parser:

    sigma(r) = k * (3 rhat rhat^T - I) / r^3        (k > 0, diatropic)

Run ``python tests/make_fixtures.py`` to refresh ``tests/sample_outputs``.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "sample_outputs")

BOHR = 1.0 / 0.52917720859
RING_K = 30.0   # ppm * A^3, the ring-current dipole strength
ISO_K = 8.0     # ppm * A^2, scalar term so NICS(iso) is not identically zero

# Benzene, D6h, in the xy plane.
_R_CC, _R_CH = 1.3970, 1.0870


def benzene_geometry():
    symbols, coords = [], []
    for i in range(6):
        ang = np.pi / 3.0 * i
        symbols.append("C")
        coords.append([_R_CC * np.cos(ang), _R_CC * np.sin(ang), 0.0])
    for i in range(6):
        ang = np.pi / 3.0 * i
        r = _R_CC + _R_CH
        symbols.append("H")
        coords.append([r * np.cos(ang), r * np.sin(ang), 0.0])
    return symbols, np.array(coords)


def shielding_tensor(
    point, centre=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), k=RING_K, k_iso=ISO_K
):
    """Point-dipole ring-current shielding tensor at *point*, in ppm.

    The dipole part is traceless, so a scalar term is added on top; without it
    every isotropic NICS in the fixtures would be exactly zero and the
    iso-based code paths would never see real data.
    """
    r = np.asarray(point, float) - np.asarray(centre, float)
    d = np.linalg.norm(r)
    iso = np.eye(3) * (k_iso / (d**2 + 1.0))
    if d < 1e-6:
        # At the ring centre the dipole model diverges; use the analytic
        # limit of a finite loop instead so NICS(0) stays finite.
        n = np.asarray(normal, float)
        return -2.0 * k / (_R_CC**3) * (np.eye(3) - 3.0 * np.outer(n, n)) + iso
    rhat = r / d
    return k * (3.0 * np.outer(rhat, rhat) - np.eye(3)) / d**3 + iso


def nucleus_tensor(symbol):
    """A plausible local shielding for a real nucleus (constant per element)."""
    base = {"C": 60.0, "H": 24.0}.get(symbol, 100.0)
    return np.diag([base * 0.4, base * 0.9, base * 1.7])


def _fmt_tensor(t):
    return "\n".join(
        "".join(f"{v:15.3f}" for v in row) for row in np.asarray(t, dtype=float)
    )


def build_output(ghost_points, title, ghost_symbol="H"):
    symbols, coords = benzene_geometry()
    ghost_points = np.asarray(ghost_points, dtype=float).reshape(-1, 3)

    all_symbols = list(symbols) + [ghost_symbol] * len(ghost_points)
    all_coords = np.vstack([coords, ghost_points]) if len(ghost_points) else coords
    is_ghost = [False] * len(symbols) + [True] * len(ghost_points)

    tensors = []
    for sym, xyz, ghost in zip(all_symbols, all_coords, is_ghost):
        t = shielding_tensor(xyz)
        if not ghost:
            t = t + nucleus_tensor(sym)
        tensors.append(t)

    L = []
    a = L.append
    a("")
    a("                         Program Version 5.0.4 -  RELEASE  -")
    a("")
    a(f"                  * {title} *")
    a("")
    a("---------------------------------")
    a("CARTESIAN COORDINATES (ANGSTROEM)")
    a("---------------------------------")
    for sym, xyz, ghost in zip(all_symbols, all_coords, is_ghost):
        label = f"{sym}:" if ghost else sym
        a(f"  {label:<4s}{xyz[0]:12.6f}{xyz[1]:12.6f}{xyz[2]:12.6f}")
    a("")
    a("----------------------------")
    a("CARTESIAN COORDINATES (A.U.)")
    a("----------------------------")
    a("  NO LB      ZA    FRAG     MASS         X           Y           Z")
    za_of = {"C": 6.0, "H": 1.0}
    mass_of = {"C": 12.011, "H": 1.008}
    for i, (sym, xyz, ghost) in enumerate(zip(all_symbols, all_coords, is_ghost)):
        label = f"{sym}:" if ghost else sym
        za = 0.0 if ghost else za_of.get(sym, 0.0)
        mass = 0.0 if ghost else mass_of.get(sym, 0.0)
        b = np.asarray(xyz) * BOHR
        a(
            f"{i:4d} {label:<4s}{za:8.4f}    0  {mass:9.3f}"
            f"{b[0]:12.6f}{b[1]:12.6f}{b[2]:12.6f}"
        )
    a("")
    a("")
    a("-------------------------------------------------------------------")
    a("                    CHEMICAL SHIELDING (ppm)")
    a("-------------------------------------------------------------------")
    a("")
    for i, (sym, t) in enumerate(zip(all_symbols, tensors)):
        a(" --------------")
        a(f" Nucleus {i:3d}{sym} :")
        a(" --------------")
        a("")
        a("Diamagnetic contribution to the shielding tensor (ppm) : ")
        a(_fmt_tensor(t * 0.6))
        a("")
        a("Paramagnetic contribution to the shielding tensor (ppm): ")
        a(_fmt_tensor(t * 0.4))
        a("")
        a("Total shielding tensor (ppm): ")
        a(_fmt_tensor(t))
        a("")
        a("")
        eig = np.linalg.eigvalsh(0.5 * (t + t.T))
        a(" Diagonalized sT*s matrix:")
        a(" ")
        a(
            "        ---------------  ---------------  ---------------"
        )
        a(
            f" Total   {eig[0]:15.3f}  {eig[1]:14.3f}  {eig[2]:14.3f}  "
            f"iso= {np.trace(t) / 3.0:11.3f}"
        )
        a("")
    a("")
    a("--------------------------------")
    a("CHEMICAL SHIELDING SUMMARY (ppm)")
    a("--------------------------------")
    a("")
    a("")
    a("  Nucleus  Element    Isotropic     Anisotropy")
    a("  -------  -------  ------------   ------------")
    for i, (sym, t) in enumerate(zip(all_symbols, tensors)):
        eig = np.linalg.eigvalsh(0.5 * (t + t.T))
        iso = float(np.trace(t) / 3.0)
        aniso = float(eig[2] - 0.5 * (eig[0] + eig[1]))
        a(f"{i:7d}      {sym:^4s}    {iso:12.3f}  {aniso:12.3f} ")
    a("")
    a("")
    a("                             ****ORCA TERMINATED NORMALLY****")
    return "\n".join(L) + "\n"


def plane_points(n=9, half=3.0, height=1.0):
    steps = np.linspace(-half, half, n)
    return np.array([[x, y, height] for x in steps for y in steps])


def volume_points(n_xy=5, half=3.0, n_z=5, half_z=2.0):
    xy = np.linspace(-half, half, n_xy)
    z = np.linspace(-half_z, half_z, n_z)
    return np.array([[x, y, zz] for x in xy for y in xy for zz in z])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = {
        "benzene_nics_single.out": (
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            "NICS single probes",
        ),
        "benzene_nics_plane.out": (plane_points(), "NICS 2D plane scan"),
        "benzene_nics_volume.out": (volume_points(), "NICS 3D ICSS volume"),
        "benzene_no_ghosts.out": ([], "plain NMR, no NICS probes"),
    }
    for name, (points, title) in cases.items():
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(build_output(points, title))
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
