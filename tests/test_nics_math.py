"""NICS math: tensor handling, ring perception and probe-layout reconstruction."""

import pytest

np = pytest.importorskip("numpy")

from orca_nics_analyzer import nics_math as nm  # noqa: E402


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class TestTensors:
    def test_symmetrize_drops_the_antisymmetric_part(self):
        t = [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]]
        s = nm.symmetrize(t)
        assert np.allclose(s, s.T)
        assert s[0][1] == pytest.approx(1.0)

    def test_nics_is_minus_sigma(self):
        entry = {"iso": 12.5, "tensor": None}
        assert nm.nics_iso(entry) == pytest.approx(-12.5)

    def test_iso_falls_back_to_the_trace(self):
        entry = {"iso": None, "tensor": np.diag([3.0, 6.0, 9.0]).tolist()}
        assert nm.nics_iso(entry) == pytest.approx(-6.0)

    def test_nics_zz_projects_onto_the_given_axis(self):
        entry = {"tensor": np.diag([1.0, 2.0, 3.0]).tolist()}
        assert nm.nics_zz(entry, [0, 0, 1]) == pytest.approx(-3.0)
        assert nm.nics_zz(entry, [1, 0, 0]) == pytest.approx(-1.0)

    def test_nics_zz_is_axis_sign_independent(self):
        entry = {"tensor": [[1.0, 0.5, 0.2], [0.4, 2.0, 0.1], [0.3, 0.7, 3.0]]}
        assert nm.nics_zz(entry, [0, 0, 1]) == pytest.approx(
            nm.nics_zz(entry, [0, 0, -1])
        )

    def test_nics_zz_normalizes_the_axis(self):
        entry = {"tensor": np.diag([1.0, 2.0, 3.0]).tolist()}
        assert nm.nics_zz(entry, [0, 0, 7]) == pytest.approx(-3.0)

    def test_nics_zz_without_a_tensor(self):
        assert nm.nics_zz({"iso": 1.0, "tensor": None}, [0, 0, 1]) is None

    def test_nics_zz_with_a_degenerate_axis(self):
        entry = {"tensor": np.eye(3).tolist()}
        assert nm.nics_zz(entry, [0, 0, 0]) is None

    def test_principal_components_are_ascending(self):
        entry = {"tensor": np.diag([9.0, 1.0, 5.0]).tolist()}
        assert nm.principal_components(entry) == pytest.approx((1.0, 5.0, 9.0))

    def test_anisotropy_matches_orca(self, real_out):
        """Computed from the tensor, it must reproduce ORCA's own summary column."""
        from orca_nics_analyzer.parser import NicsParser

        data = NicsParser().load(real_out)
        for entry in data["shieldings"].values():
            assert nm.anisotropy(entry) == pytest.approx(entry["aniso"], abs=0.01)

    @pytest.mark.parametrize(
        "value,expected",
        [(-10.0, "diatropic"), (10.0, "paratropic"), (0.5, "non-aromatic")],
    )
    def test_classification(self, value, expected):
        assert expected in nm.classify(value)

    def test_classification_of_missing_data(self):
        assert nm.classify(None) == "-"


class TestRings:
    def _benzene(self):
        symbols, coords = [], []
        for i in range(6):
            ang = np.pi / 3 * i
            symbols.append("C")
            coords.append([1.397 * np.cos(ang), 1.397 * np.sin(ang), 0.0])
        for i in range(6):
            ang = np.pi / 3 * i
            symbols.append("H")
            coords.append([2.484 * np.cos(ang), 2.484 * np.sin(ang), 0.0])
        return symbols, np.array(coords)

    def test_benzene_has_one_six_ring(self):
        rings = nm.ring_info(*self._benzene())
        assert len(rings) == 1
        assert rings[0]["size"] == 6
        assert rings[0]["planarity_rms"] == pytest.approx(0.0, abs=1e-9)

    def test_ring_centroid_and_normal(self):
        rings = nm.ring_info(*self._benzene())
        assert rings[0]["centroid"] == pytest.approx([0, 0, 0], abs=1e-9)
        assert abs(rings[0]["normal"] @ np.array([0, 0, 1.0])) == pytest.approx(1.0)

    def test_naphthalene_has_two_rings(self):
        # Two fused hexagons sharing an edge.
        symbols, coords = [], []
        for cx in (0.0, 2.42):
            for i in range(6):
                ang = np.pi / 3 * i
                symbols.append("C")
                coords.append([cx + 1.397 * np.cos(ang), 1.397 * np.sin(ang), 0.0])
        rings = nm.ring_info(symbols, np.array(coords))
        assert sum(1 for r in rings if r["size"] == 6) >= 2

    def test_acyclic_molecule_has_no_rings(self):
        symbols = ["C", "C", "C"]
        coords = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]], float)
        assert nm.ring_info(symbols, coords) == []

    def test_nearest_ring_reports_signed_height(self):
        rings = nm.ring_info(*self._benzene())
        _, height, in_plane = nm.nearest_ring([0, 0, 1.0], rings)
        assert height == pytest.approx(1.0)
        assert in_plane == pytest.approx(0.0, abs=1e-9)
        _, below, _ = nm.nearest_ring([0, 0, -1.0], rings)
        assert below == pytest.approx(-1.0)

    def test_nearest_ring_without_rings(self):
        assert nm.nearest_ring([0, 0, 0], []) == (None, None, None)

    def test_bond_list_finds_the_expected_count(self):
        symbols, coords = self._benzene()
        assert len(nm.bond_list(symbols, coords)) == 12


class TestLayoutDetection:
    def test_empty(self):
        assert nm.detect_layout(np.zeros((0, 3)))["kind"] == "none"

    def test_single_probe(self):
        assert nm.detect_layout(np.array([[1.0, 2.0, 3.0]]))["kind"] == "single"

    def test_line(self):
        pts = np.array([[0, 0, t] for t in range(7)], float)
        assert nm.detect_layout(pts)["kind"] == "line"

    def test_rectangular_plane(self):
        pts = np.array(
            [[x, y, 1.0] for x in np.linspace(-5, 5, 11) for y in np.linspace(-3, 3, 7)]
        )
        layout = nm.detect_layout(pts)
        assert layout["kind"] == "plane"
        assert sorted(layout["shape"]) == [1, 7, 11]
        assert layout["regular"]

    def test_square_plane_survives_svd_degeneracy(self):
        """Equal in-plane extents make the principal axes ambiguous.

        SVD is then free to hand back a frame rotated within the plane, which
        would smear each grid line across many clusters and lose the shape.
        """
        r = rotation([1, 2, 3], 0.7)
        steps = np.linspace(-4, 4, 9)
        pts = np.array([[x, y, 0.0] for x in steps for y in steps]) @ r.T
        pts = pts + np.array([1.0, 2.0, 3.0])
        layout = nm.detect_layout(pts)
        assert layout["kind"] == "plane"
        assert sorted(layout["shape"]) == [1, 9, 9]
        assert layout["regular"]

    def test_cubic_volume_survives_svd_degeneracy(self):
        r = rotation([0, 1, 1], 1.1)
        steps = np.linspace(-2, 2, 5)
        pts = np.array([[x, y, z] for x in steps for y in steps for z in steps]) @ r.T
        layout = nm.detect_layout(pts)
        assert layout["kind"] == "volume"
        assert layout["shape"] == (5, 5, 5)

    def test_anisotropic_volume(self):
        pts = np.array(
            [
                [x, y, z]
                for x in np.linspace(-3, 3, 7)
                for y in np.linspace(-2, 2, 5)
                for z in np.linspace(-1, 1, 3)
            ]
        )
        layout = nm.detect_layout(pts)
        assert layout["kind"] == "volume"
        assert sorted(layout["shape"]) == [3, 5, 7]

    def test_random_points_are_scattered(self):
        pts = np.random.RandomState(0).rand(25, 3) * 5
        assert nm.detect_layout(pts)["kind"] == "scattered"

    def test_incomplete_grid_is_scattered(self):
        """A grid with holes must not be reported as regular."""
        steps = np.linspace(-2, 2, 5)
        pts = np.array([[x, y, 0.0] for x in steps for y in steps])
        layout = nm.detect_layout(pts[:-3])
        assert not layout["regular"]
        assert layout["kind"] == "scattered"

    def test_index_map_reproduces_the_grid(self):
        steps = np.linspace(-2, 2, 5)
        pts = np.array([[x, y, 0.0] for x in steps for y in steps])
        layout = nm.detect_layout(pts)
        values = np.arange(len(pts), dtype=float)
        grid = nm.layout_grid_values(layout, values)
        assert grid.shape == layout["shape"]
        assert np.isfinite(grid).all()
        assert sorted(grid.ravel()) == pytest.approx(sorted(values))

    def test_grid_values_leave_gaps_as_nan(self):
        layout = {
            "shape": (2, 2, 1),
            "index_map": np.array([[0, 0, 0], [1, 1, 0]]),
        }
        grid = nm.layout_grid_values(layout, np.array([1.0, 2.0]))
        assert np.isnan(grid[0, 1, 0])
        assert grid[1, 1, 0] == pytest.approx(2.0)

    def test_axes_are_orthonormal(self):
        steps = np.linspace(-2, 2, 5)
        pts = np.array([[x, y, z] for x in steps for y in steps for z in steps])
        axes = nm.detect_layout(pts)["axes"]
        assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-9)


class TestMinSpacing:
    def test_ignores_duplicates(self):
        assert nm._min_spacing(np.array([0.0, 0.0, 1.0, 3.0])) == pytest.approx(1.0)

    def test_all_identical(self):
        assert nm._min_spacing(np.array([2.0, 2.0])) == 0.0
