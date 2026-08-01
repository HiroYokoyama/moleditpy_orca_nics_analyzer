"""NicsField: probe records, gridded fields, cube caching and exports.

Expected NICS values come from the same analytic ring-current model the
fixtures were built with, imported here rather than hard-coded, so the test
checks the parse-and-project chain rather than a copied number.
"""

import os

import pytest

np = pytest.importorskip("numpy")

from orca_nics_analyzer import cube_io  # noqa: E402
from orca_nics_analyzer.analysis import export_all, load_field  # noqa: E402
from make_fixtures import ISO_K, RING_K, shielding_tensor  # noqa: E402


def expected_nics(point, axis=(0, 0, 1)):
    """(iso, zz) the model predicts at *point*, in ppm."""
    t = shielding_tensor(point)
    n = np.asarray(axis, float)
    return -float(np.trace(t) / 3.0), -float(n @ t @ n)


class TestProbes:
    def test_single_probes(self, single_out):
        field = load_field(single_out)
        assert len(field.probes) == 3
        assert field.layout["kind"] == "line"
        assert field.has_tensors

    def test_values_match_the_ring_current_model(self, single_out):
        field = load_field(single_out)
        for probe in field.probes:
            iso, zz = expected_nics(probe["xyz"])
            assert probe["iso"] == pytest.approx(iso, abs=0.01)
            assert probe["zz"] == pytest.approx(zz, abs=0.01)

    def test_nics_one_has_the_textbook_value(self, single_out):
        """NICS(1)_zz for the shipped model parameters: -(2k/r^3 + k_iso/2)."""
        field = load_field(single_out)
        probe = next(p for p in field.probes if p["xyz"][2] == pytest.approx(1.0))
        assert probe["zz"] == pytest.approx(-(2 * RING_K + ISO_K / 2.0), abs=0.01)
        assert "diatropic" in probe["classification"]

    def test_probe_heights_are_signed(self, single_out):
        field = load_field(single_out)
        heights = sorted(p["height"] for p in field.probes)
        assert heights == pytest.approx([-1.0, 0.0, 1.0], abs=1e-6)

    def test_rings_come_from_the_real_atoms_only(self, plane_out):
        """81 ghosts in a plane must not be bonded into spurious rings."""
        field = load_field(plane_out)
        assert len(field.real_coords) == 12
        assert len(field.rings) == 1
        assert field.rings[0]["size"] == 6

    def test_no_probes(self, no_ghosts_out):
        field = load_field(no_ghosts_out)
        assert field.probes == []
        assert field.layout["kind"] == "none"
        assert not field.is_gridded


class TestAxisModes:
    def test_switching_axis_recomputes_zz(self, single_out):
        field = load_field(single_out)
        before = [p["zz"] for p in field.probes]
        field.set_axis_mode("x")
        after = [p["zz"] for p in field.probes]
        assert before != after

    def test_lab_axes(self, single_out):
        field = load_field(single_out)
        probe = next(p for p in field.probes if p["xyz"][2] == pytest.approx(1.0))
        for mode, axis in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
            field.set_axis_mode(mode)
            probe = next(p for p in field.probes if p["xyz"][2] == pytest.approx(1.0))
            _, zz = expected_nics(probe["xyz"], axis)
            assert probe["zz"] == pytest.approx(zz, abs=0.01)

    def test_ring_mode_uses_the_ring_normal(self, single_out):
        field = load_field(single_out, axis_mode="ring")
        probe = field.probes[0]
        assert abs(field.axis_for(probe) @ np.array([0, 0, 1.0])) == pytest.approx(1.0)

    def test_grid_mode_falls_back_to_the_ring_normal(self):
        """With one probe there is no grid, so the ring normal has to serve."""
        field = load_field(
            os.path.join(
                os.path.dirname(__file__), "sample_outputs", "benzene_nics_single.out"
            )
        )
        field.layout = {"kind": "scattered", "shape": (1, 1, 1)}
        assert field.grid_normal is None
        axis = field.axis_for(field.probes[0])
        assert abs(axis @ np.array([0, 0, 1.0])) == pytest.approx(1.0)


class TestGrids:
    def test_plane_is_gridded(self, plane_out):
        field = load_field(plane_out)
        assert field.layout["kind"] == "plane"
        assert field.is_gridded
        assert sorted(field.layout["shape"]) == [1, 9, 9]

    def test_volume_is_gridded(self, volume_out):
        field = load_field(volume_out)
        assert field.layout["kind"] == "volume"
        assert field.grid("zz")[0].shape == (5, 5, 5)

    def test_grid_values_land_on_the_right_nodes(self, plane_out):
        """Every grid cell must hold the value of the probe at that position."""
        field = load_field(plane_out)
        cube, origin, steps = field.grid("zz")
        for i in range(cube.shape[0]):
            for j in range(cube.shape[1]):
                point = origin + i * steps[0] + j * steps[1]
                _, zz = expected_nics(point)
                assert cube[i, j, 0] == pytest.approx(zz, abs=0.02)

    def test_slices_are_taken_parallel_to_the_ring(self, volume_out):
        field = load_field(volume_out)
        info = field.plane_data("zz")
        assert abs(info["normal"] @ np.array([0, 0, 1.0])) == pytest.approx(1.0)
        assert info["n_slices"] == 5

    def test_middle_slice_is_the_ring_plane(self, volume_out):
        field = load_field(volume_out)
        info = field.plane_data("zz")
        centre = info["values"][2, 2]
        _, zz = expected_nics([0.0, 0.0, 0.0])
        assert centre == pytest.approx(zz, abs=0.02)

    def test_slice_index_is_clamped(self, volume_out):
        field = load_field(volume_out)
        assert field.plane_slice("zz", 99)["slice_index"] == 4
        assert field.plane_slice("zz", -5)["slice_index"] == 0

    def test_grid_raises_when_not_gridded(self, single_out):
        field = load_field(single_out)
        with pytest.raises(ValueError):
            field.grid("zz")

    def test_flat_axis_still_gets_a_step_vector(self, plane_out):
        """A cube needs three non-zero voxel vectors even for a single layer."""
        field = load_field(plane_out)
        _, _, steps = field.grid("zz")
        assert all(np.linalg.norm(s) > 0 for s in steps)


class TestScan:
    @pytest.fixture
    def scan(self):
        return load_field(
            os.path.join(
                os.path.dirname(__file__), "sample_outputs", "benzene_nics_scan.out"
            )
        )

    def test_line_layout_is_a_scan(self, scan):
        assert scan.layout["kind"] == "line"
        assert scan.is_scan
        assert not scan.is_gridded

    def test_distance_is_the_height_above_the_ring(self, scan):
        data = scan.line_data()
        assert "height above the ring" in data["label"]
        assert data["distance"] == pytest.approx(np.linspace(0.0, 5.0, 11), abs=1e-6)

    def test_points_are_sorted_by_distance(self, scan):
        assert np.all(np.diff(scan.line_data()["distance"]) > 0)

    def test_values_follow_the_model(self, scan):
        data = scan.line_data()
        for distance, zz in zip(data["distance"], data["zz"]):
            _, expected = expected_nics([0.0, 0.0, float(distance)])
            assert zz == pytest.approx(expected, abs=0.02)

    def test_decays_with_distance(self, scan):
        """A ring current falls off, so the far end must be the weakest point."""
        data = scan.line_data()
        assert abs(data["zz"][-1]) < abs(data["zz"][2])

    def test_component_selection(self, scan):
        assert scan.line_data("iso")["values"] == pytest.approx(scan.line_data()["iso"])

    def test_extremum(self, scan):
        where, peak = scan.scan_extremum("zz")
        data = scan.line_data()
        assert peak == pytest.approx(data["zz"][np.argmax(np.abs(data["zz"]))])
        assert np.min(np.abs(data["distance"] - where)) == pytest.approx(0.0)

    def test_extremum_without_values(self, scan):
        for probe in scan.probes:
            probe["zz"] = None
        assert scan.scan_extremum("zz") == (None, None)

    def test_reversed_scan_plots_front_to_back(self, single_out):
        """Probes ordered downwards must still come out ascending in height."""
        field = load_field(single_out)
        data = field.line_data()
        assert np.all(np.diff(data["distance"]) > 0)
        assert data["distance"][0] == pytest.approx(-1.0)

    def test_offsets_flag_a_drifting_scan(self, scan):
        assert scan.line_data()["offsets"] == pytest.approx(np.zeros(11), abs=1e-6)

    def test_falls_back_to_arc_length_without_rings(self, scan):
        scan.rings = []
        for probe in scan.probes:
            probe["height"] = None
        data = scan.line_data()
        assert "along the scan" in data["label"]
        assert data["distance"][0] == pytest.approx(0.0)

    def test_scan_csv(self, scan):
        lines = scan.scan_csv().strip().splitlines()
        assert lines[0] == "Index,Distance/A,Offset/A,NICS(iso)/ppm,NICS_zz/ppm"
        assert len(lines) == 12

    def test_scan_csv_blanks_missing_values(self, scan):
        for probe in scan.probes:
            probe["zz"] = None
        assert scan.scan_csv().splitlines()[1].endswith(",")

    def test_line_data_rejects_a_grid(self, volume_out):
        with pytest.raises(ValueError):
            load_field(volume_out).line_data()

    def test_export_all_includes_the_scan(self, scan, tmp_path):
        written = export_all(scan, str(tmp_path), "0.1.0")
        assert any(p.endswith("_NICS_scan.csv") for p in written)


class TestCubes:
    def test_write_and_read_back(self, volume_out, tmp_path):
        field = load_field(volume_out)
        path = field.write_cube(
            "zz", path=str(tmp_path / "t.cube"), plugin_version="1.2.3"
        )
        result = cube_io.read_cube(path)
        cube, _, _ = field.grid("zz")
        assert np.allclose(result["data"], np.nan_to_num(cube))
        assert len(result["symbols"]) == 12

    def test_stamp_records_the_settings(self, volume_out, tmp_path):
        field = load_field(volume_out)
        path = field.write_cube(
            "zz", path=str(tmp_path / "t.cube"), plugin_version="1.2.3"
        )
        info = cube_io.read_generation_settings(path)
        assert info["version"] == "1.2.3"
        assert info["component"] == "zz"
        assert info["grid"] == (5, 5, 5)

    def test_cube_lands_beside_the_output(self, volume_out, tmp_path):
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        field = load_field(str(source))
        path, cached = field.ensure_cube("zz", plugin_version="0.1.0")
        assert not cached
        assert os.path.basename(os.path.dirname(path)) == "run_nics_cubes"
        assert os.path.exists(path)

    def test_second_call_reuses_the_cache(self, volume_out, tmp_path):
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        field = load_field(str(source))
        path, _ = field.ensure_cube("zz", plugin_version="0.1.0")
        stamp = os.path.getmtime(path)
        again, cached = field.ensure_cube("zz", plugin_version="0.1.0")
        assert cached
        assert again == path
        assert os.path.getmtime(again) == stamp

    def test_force_overwrites_the_cache(self, volume_out, tmp_path):
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        field = load_field(str(source))
        path, _ = field.ensure_cube("zz", plugin_version="0.1.0")
        os.remove(path)
        again, cached = field.ensure_cube("zz", plugin_version="0.1.0", force=True)
        assert not cached
        assert os.path.exists(again)

    def test_stale_cache_with_a_different_grid_is_rejected(self, volume_out, tmp_path):
        """A cube from another grid must not be served for this one."""
        source = tmp_path / "run.out"
        source.write_bytes(open(volume_out, "rb").read())
        field = load_field(str(source))
        path, _ = field.ensure_cube("zz", plugin_version="0.1.0")
        cube_io.write_cube(
            path,
            np.zeros((3, 3, 3)),
            [0, 0, 0],
            np.eye(3),
            stamp=cube_io.stamp_line("0.1.0", "zz", (3, 3, 3)),
        )
        assert field.cached_cube("zz") is None

    def test_cube_path_is_none_without_a_source_file(self, volume_out):
        field = load_field(volume_out)
        field.filename = None
        assert field.cube_path("zz") is None
        with pytest.raises(ValueError):
            field.write_cube("zz")


class TestExports:
    def test_csv_has_a_row_per_probe(self, single_out):
        field = load_field(single_out)
        lines = field.to_csv().strip().splitlines()
        assert len(lines) == 4
        assert lines[0].startswith("Index,Symbol")

    def test_csv_blanks_missing_values(self, single_out):
        field = load_field(single_out)
        for probe in field.probes:
            probe["zz"] = None
        cells = field.to_csv().splitlines()[1].split(",")
        columns = [k for k, _ in field.CSV_COLUMNS]
        assert cells[columns.index("nics_zz")] == ""
        assert cells[columns.index("nics_iso")] != ""

    def test_summary_mentions_the_layout(self, volume_out):
        text = load_field(volume_out).summary_text("9.9.9")
        assert "9.9.9" in text
        assert "volume" in text
        assert "5x5x5" in text

    def test_summary_reports_missing_tensors(self, single_out):
        field = load_field(single_out)
        field.has_tensors = False
        assert "isotropic only" in field.summary_text()

    def test_export_all_writes_every_artefact(self, volume_out, tmp_path):
        field = load_field(volume_out)
        written = export_all(field, str(tmp_path), "0.1.0")
        names = sorted(os.path.basename(p) for p in written)
        assert any(n.endswith("_probes.csv") for n in names)
        assert any(n.endswith("_summary.txt") for n in names)
        assert sum(1 for n in names if n.endswith(".cube")) == 2
        assert all(os.path.getsize(p) > 0 for p in written)

    def test_export_all_skips_cubes_without_a_grid(self, single_out, tmp_path):
        field = load_field(single_out)
        written = export_all(field, str(tmp_path), "0.1.0")
        assert not any(p.endswith(".cube") for p in written)
        # Probe CSV, summary, and — since three collinear probes are a scan —
        # the scan CSV too.
        assert sorted(os.path.basename(p).split("_NICS_")[1] for p in written) == [
            "probes.csv",
            "scan.csv",
            "summary.txt",
        ]

    def test_export_all_skips_zz_without_tensors(self, volume_out, tmp_path):
        field = load_field(volume_out)
        field.has_tensors = False
        written = export_all(field, str(tmp_path), "0.1.0")
        cubes = [p for p in written if p.endswith(".cube")]
        assert len(cubes) == 1
        assert cubes[0].endswith("_iso.cube")

    def test_export_all_without_a_folder_or_source(self, volume_out):
        field = load_field(volume_out)
        field.filename = None
        with pytest.raises(ValueError):
            export_all(field)


class TestValues:
    def test_missing_values_become_nan(self, single_out):
        field = load_field(single_out)
        field.probes[0]["zz"] = None
        values = field.values("zz")
        assert np.isnan(values[0])
        assert np.isfinite(values[1])

    def test_iso_and_zz_differ(self, single_out):
        field = load_field(single_out)
        assert not np.allclose(field.values("iso"), field.values("zz"))


class TestExtractLine:
    """NicsField.extract_line: 2D → 1D slicing."""

    def test_raises_for_non_gridded_field(self, single_out):
        field = load_field(single_out)
        assert not field.is_gridded
        with pytest.raises(ValueError, match="regular grid"):
            field.extract_line("iso", 0, 0)

    def test_returns_expected_keys(self, plane_out):
        field = load_field(plane_out)
        data = field.extract_line("iso", 0, 0)
        for key in ("distance", "label", "iso", "zz", "values", "indices", "offsets"):
            assert key in data

    def test_source_metadata_present(self, plane_out):
        field = load_field(plane_out)
        data = field.extract_line("iso", 0, 2)
        assert data["source"] == "extract_line"
        assert data["fixed_in_plane_axis"] == 0
        assert data["fixed_index"] == 2

    def test_walk_axis2_length_matches_a2(self, plane_out):
        """Fixing axis-1 (fixed_in_plane_axis=0) → profile length == len(a2)."""
        field = load_field(plane_out)
        info = field.plane_data("iso")
        n_a2 = len(info["a2"])
        data = field.extract_line("iso", fixed_in_plane_axis=0, fixed_index=0)
        assert len(data["distance"]) == n_a2
        assert len(data["values"]) == n_a2

    def test_walk_axis1_length_matches_a1(self, plane_out):
        """Fixing axis-2 (fixed_in_plane_axis=1) → profile length == len(a1)."""
        field = load_field(plane_out)
        info = field.plane_data("iso")
        n_a1 = len(info["a1"])
        data = field.extract_line("iso", fixed_in_plane_axis=1, fixed_index=0)
        assert len(data["distance"]) == n_a1
        assert len(data["values"]) == n_a1

    def test_iso_component_fills_iso_key(self, plane_out):
        field = load_field(plane_out)
        data = field.extract_line("iso", 0, 0)
        assert np.isfinite(data["iso"]).any()
        assert np.all(np.isnan(data["zz"]))

    def test_zz_component_fills_zz_key(self, plane_out):
        field = load_field(plane_out)
        data = field.extract_line("zz", 0, 0)
        assert np.isfinite(data["zz"]).any()
        assert np.all(np.isnan(data["iso"]))

    def test_values_equal_iso_for_iso_component(self, plane_out):
        field = load_field(plane_out)
        data = field.extract_line("iso", 0, 3)
        np.testing.assert_array_equal(data["values"], data["iso"])

    def test_values_equal_zz_for_zz_component(self, plane_out):
        field = load_field(plane_out)
        data = field.extract_line("zz", 1, 2)
        np.testing.assert_array_equal(data["values"], data["zz"])

    def test_index_clamped_to_zero_when_negative(self, plane_out):
        field = load_field(plane_out)
        data_neg = field.extract_line("iso", 0, -99)
        data_zero = field.extract_line("iso", 0, 0)
        np.testing.assert_array_almost_equal(data_neg["values"], data_zero["values"])

    def test_index_clamped_to_max_when_too_large(self, plane_out):
        field = load_field(plane_out)
        info = field.plane_data("iso")
        n = len(info["a1"])
        data_big = field.extract_line("iso", 0, 9999)
        data_last = field.extract_line("iso", 0, n - 1)
        np.testing.assert_array_almost_equal(data_big["values"], data_last["values"])

    def test_different_rows_give_different_values(self, plane_out):
        field = load_field(plane_out)
        d0 = field.extract_line("iso", 0, 0)
        d4 = field.extract_line("iso", 0, 4)
        # The field is not constant, so different rows differ somewhere.
        assert not np.allclose(d0["values"], d4["values"])

    def test_stack_index_accepted_for_volume(self, volume_out):
        """extract_line on a volume honours the explicit stack_index."""
        field = load_field(volume_out)
        data0 = field.extract_line("iso", 0, 0, stack_index=0)
        data2 = field.extract_line("iso", 0, 0, stack_index=2)
        # Different stack layers should yield different profiles.
        assert not np.allclose(data0["values"], data2["values"])

    def test_stack_index_in_metadata(self, volume_out):
        field = load_field(volume_out)
        data = field.extract_line("iso", 0, 0, stack_index=2)
        assert data["stack_index"] == 2

    def test_offsets_are_zero(self, plane_out):
        """extract_line profiles have no ring-axis offset (it is a synthetic line)."""
        field = load_field(plane_out)
        data = field.extract_line("iso", 0, 0)
        assert np.all(data["offsets"] == 0.0)

