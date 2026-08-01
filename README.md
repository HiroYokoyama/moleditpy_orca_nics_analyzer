# ORCA NICS Analyzer

[![Tests](https://github.com/HiroYokoyama/moleditpy_orca_nics_analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/HiroYokoyama/moleditpy_orca_nics_analyzer/actions/workflows/tests.yml)
![coverage](https://img.shields.io/badge/coverage-%3E90%25-brightgreen)

A [MoleditPy](https://github.com/HiroYokoyama/python_molecular_editor) plugin that
reads NICS data out of an ORCA output file and turns it into numbers, maps and
isosurfaces.

It is the read side of a NICS study: place ghost probes (with
[NICS Placer](https://github.com/HiroYokoyama/moleditpy_nics_placer) or any other
means), run the ORCA NMR job, then open the `.out` here.

| Probe layout | What you get |
|---|---|
| One or a few probes | **Probes** table: NICS(0)/NICS(1), NICS_zz, distance from the ring plane, aromatic / antiaromatic character |
| A line of probes | **1D Scan**: the NICS profile against height above the ring, with the extremum marked |
| A plane of probes | **2D Map**: filled NICS contours with the molecule projected on top |
| A 3D box of probes | **3D ICSS**: ± isosurfaces drawn in MoleditPy's own 3D viewer |

Everything is exportable: probe CSV, scan CSV, map grid CSV, plot images
(PNG/PDF/SVG), Gaussian `.cube` files and a text summary — individually per tab,
or all at once with **Export all...**.

## Install

Plugin Manager → install from
`https://github.com/HiroYokoyama/moleditpy_orca_nics_analyzer`, or drop the
`orca_nics_analyzer` folder into your MoleditPy plugins directory.

Requires `PyQt6`, `numpy`, `matplotlib` and `pyvista`. The last two are optional at
runtime — without them the tabs that need them explain what is missing and
everything else keeps working.

## Use

**Analysis → ORCA NICS Analyzer...**, then pick the ORCA `.out`. The window opens
on the tab that matches how the probes were arranged.

### Preparing the ORCA job

Ghost centres carry basis functions but no nucleus, written with a trailing colon,
and they have to be named in the NMR nucleus selection:

```
! B3LYP def2-TZVP NMR

* xyz 0 1
  C   1.397000   0.000000   0.000000
  ...
  H:  0.000000   0.000000   1.000000     # the NICS(1) probe
*
```

The plugin identifies probes by their zero nuclear charge in ORCA's
`CARTESIAN COORDINATES (A.U.)` block, so it does not matter how the input was
produced or what element the ghosts borrow.

## What it computes

With `NICS = -sigma` throughout, so negative is diatropic (aromatic) and positive
paratropic (antiaromatic):

- **NICS(iso)** — from the isotropic shielding.
- **NICS_zz** — the shielding tensor projected onto a chosen axis, `-n·sigma·n`.
  ORCA prints the total tensor unsymmetrized; only the symmetric part is
  observable, so it is symmetrized before projection.
- The **axis** is selectable: the grid normal (the ICSS convention), the nearest
  ring's normal (usual for single probes), or a lab axis. Rings are perceived from
  the real atoms only, so a cloud of ghosts never invents one.

Outputs with no shielding tensors still work — the isotropic values are shown and
the NICS_zz controls are disabled rather than guessing.

### Cube files

Cubes are written to `<output>_nics_cubes/` beside the ORCA file and reused on the
next open when the stamped grid still matches; **Regenerate** overwrites. They are
ordinary Gaussian cubes in ppm, readable by Cube File Viewer, Orbital Comparator,
VMD and the rest. A ring-frame grid is not axis-aligned, and its voxel vectors are
carried through as-is rather than resampled onto a lab box.

## Development

```bash
python -m pytest tests/ -v                     # 210 tests
python -m pytest tests/ --cov=orca_nics_analyzer --cov-report=term-missing
python tests/make_fixtures.py                  # regenerate the sample outputs
```

Test fixtures are synthetic ORCA outputs in exact ORCA 5 layout whose shieldings
come from an analytic ring-current point-dipole model, so the expected NICS values
are known independently of the parser. A genuine ORCA 5.0.4 benzene NMR output is
included too, and the tensor-derived anisotropy is checked against ORCA's own
printed values.

`tests/test_api.py` statically validates every `context.*` and `mw.*` access
against the main application when both repos are checked out side by side.

## License

MIT — see [LICENSE](LICENSE).
