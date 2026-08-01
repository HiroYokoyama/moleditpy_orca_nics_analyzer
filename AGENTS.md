# Contributor Guide

## Project scope

This repository is a MoleditPy plugin for parsing ORCA NICS output, analysing probe layouts, exporting Gaussian cube fields, and rendering 2D/3D views.

## Development setup

Install the complete test stack in a virtual environment:

```bash
python -m pip install pytest pytest-cov numpy matplotlib PyQt6 pyvista
```

The dependency-free tests need only `pytest` and `numpy`.

## Required checks

Run tests through the Python module entry point:

```bash
python -m pytest tests/ -v --tb=short --cov=orca_nics_analyzer --cov-report=term-missing
```

Run formatting and lint checks when Ruff is available:

```bash
ruff format --check .
ruff check .
```

Do not use alternate test runners for validation. Keep generated cube folders and coverage artifacts out of commits.

## Change expectations

- Preserve the plugin version unless a release is explicitly requested.
- Add or update tests for behavior changes and edge cases.
- Keep cube headers identifiable as MoleditPy ORCA NICS Analyzer headers and retain generation conditions.
- Verify CI with `gh run list`/`gh run view` after pushing.
- Push a version tag only after the complete Tests workflow is green.

## Commits

Make focused commits and use the repository default Git identity. Add this trailer to each commit:

`Assisted-by: GPT 5.6 Luna`

