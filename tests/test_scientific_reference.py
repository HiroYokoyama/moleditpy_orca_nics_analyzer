"""Closed-form tensor and geometry checks independent of ORCA fixtures."""

import numpy as np
import pytest

from orca_nics_analyzer import nics_math as nm


@pytest.mark.parametrize("angle", [0.0, 0.37, 1.2, 2.8])
def test_rotated_uniaxial_shielding_has_closed_form_directional_response(angle):
    # Principal shielding (10,10,40) ppm, active rotation around y.
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    tensor = rotation @ np.diag([10.0, 10.0, 40.0]) @ rotation.T
    entry = {"tensor": tensor.tolist()}
    assert nm.nics_iso(entry) == pytest.approx(-20, abs=1e-12)
    assert nm.nics_zz(entry, [0, 0, 1]) == pytest.approx(-(10 + 30 * c * c), abs=1e-12)
    assert nm.nics_zz(entry, rotation[:, 2]) == pytest.approx(-40, abs=1e-12)
    assert nm.principal_components(entry) == pytest.approx((10, 10, 40), abs=1e-12)
    assert nm.anisotropy(entry) == pytest.approx(30, abs=1e-12)


def test_antisymmetric_response_cannot_change_observable_projection():
    entry = {"tensor": [[10, 100, -50], [-100, 20, 75], [50, -75, 30]]}
    # Direction (1,2,2)/3: diagonal contraction = (10+80+120)/9.
    assert nm.nics_zz(entry, [1, 2, 2]) == pytest.approx(-210 / 9, abs=1e-12)


def test_translated_rotated_ring_reports_one_angstrom_probe_height():
    # Plane normal (0,3,4)/5; 1.4 Angstrom regular hexagon at (7,-2,3).
    origin = np.array([7.0, -2.0, 3.0])
    normal = np.array([0.0, 0.6, 0.8])
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.8, -0.6])
    coords = np.array(
        [
            origin + 1.4 * (np.cos(k * np.pi / 3) * u + np.sin(k * np.pi / 3) * v)
            for k in range(6)
        ]
    )
    rings = nm.ring_info(["C"] * 6, coords)
    assert len(rings) == 1
    assert rings[0]["centroid"] == pytest.approx(origin, abs=1e-12)
    _, height, lateral = nm.nearest_ring(origin + normal + 0.5 * u, rings)
    assert abs(height) == pytest.approx(1.0, abs=1e-12)
    assert lateral == pytest.approx(0.5, abs=1e-12)
