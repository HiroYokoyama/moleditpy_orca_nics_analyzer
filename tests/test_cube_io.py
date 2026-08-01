"""Gaussian cube reading, writing, stamping and path conventions."""

import os

import pytest

np = pytest.importorskip("numpy")

from orca_nics_analyzer import cube_io  # noqa: E402

BOHR = cube_io.BOHR_PER_ANGSTROM


class TestRoundTrip:
    def test_data_survives(self, tmp_path):
        data = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)
        path = cube_io.write_cube(str(tmp_path / "a.cube"), data, [0, 0, 0], np.eye(3))
        assert np.allclose(cube_io.read_cube(path)["data"], data)

    def test_lengths_come_back_in_angstrom(self, tmp_path):
        path = cube_io.write_cube(
            str(tmp_path / "a.cube"),
            np.zeros((2, 2, 2)),
            [0.1, 0.2, 0.3],
            np.eye(3) * 0.5,
            symbols=["C"],
            coords=[[1.0, 2.0, 3.0]],
        )
        result = cube_io.read_cube(path)
        # The format stores Bohr to 6 decimals, so a round trip through
        # Angstrom is exact only to ~1e-6.
        assert result["origin"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
        assert result["vectors"][0] == pytest.approx([0.5, 0.0, 0.0], abs=1e-6)
        assert result["coords"][0] == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)

    def test_non_orthogonal_axes_survive(self, tmp_path):
        """A ring-frame grid is not axis-aligned; the cube must carry it verbatim."""
        vectors = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.2], [0.1, 0.0, 0.6]])
        path = cube_io.write_cube(
            str(tmp_path / "a.cube"), np.zeros((2, 2, 2)), [0, 0, 0], vectors
        )
        assert cube_io.read_cube(path)["vectors"] == pytest.approx(vectors, abs=1e-6)

    def test_atomic_numbers_map_back_to_symbols(self, tmp_path):
        path = cube_io.write_cube(
            str(tmp_path / "a.cube"),
            np.zeros((2, 2, 2)),
            [0, 0, 0],
            np.eye(3),
            symbols=["C", "H", "Fe"],
            coords=np.zeros((3, 3)),
        )
        assert cube_io.read_cube(path)["symbols"] == ["C", "H", "Fe"]

    def test_ghost_labels_are_stripped(self, tmp_path):
        path = cube_io.write_cube(
            str(tmp_path / "a.cube"),
            np.zeros((2, 2, 2)),
            [0, 0, 0],
            np.eye(3),
            symbols=["H:"],
            coords=[[0, 0, 0]],
        )
        assert cube_io.read_cube(path)["symbols"] == ["H"]

    def test_nan_is_written_as_zero(self, tmp_path):
        data = np.array([[[np.nan, 1.0]]])
        path = cube_io.write_cube(str(tmp_path / "a.cube"), data, [0, 0, 0], np.eye(3))
        assert cube_io.read_cube(path)["data"][0, 0, 0] == 0.0

    def test_directory_is_created(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "a.cube")
        cube_io.write_cube(path, np.zeros((2, 2, 2)), [0, 0, 0], np.eye(3))
        assert os.path.exists(path)

    def test_rejects_non_3d_data(self, tmp_path):
        with pytest.raises(ValueError):
            cube_io.write_cube(
                str(tmp_path / "a.cube"), np.zeros((2, 2)), [0, 0, 0], np.eye(3)
            )


class TestReading:
    def test_negative_atom_count_skips_the_dset_line(self, tmp_path):
        """An orbital cube marks itself with a negative count and an extra line."""
        path = tmp_path / "mo.cube"
        path.write_text(
            "comment\nstamp\n"
            f"   -1{0.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n"
            f"    2{1.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n"
            f"    1{0.0:12.6f}{1.0:12.6f}{0.0:12.6f}\n"
            f"    1{0.0:12.6f}{0.0:12.6f}{1.0:12.6f}\n"
            f"    6{6.0:12.6f}{0.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n"
            "    1   35\n"
            "  1.0  2.0\n",
            encoding="utf-8",
        )
        result = cube_io.read_cube(str(path))
        assert result["data"].shape == (2, 1, 1)
        assert result["data"].ravel() == pytest.approx([1.0, 2.0])

    def test_incomplete_header_raises_value_error(self, tmp_path):
        path = tmp_path / "short-header.cube"
        path.write_text("comment\nstamp\n", encoding="utf-8")
        with pytest.raises(ValueError, match="incomplete header"):
            cube_io.read_cube(str(path))
    def test_truncated_data_raises(self, tmp_path):
        path = tmp_path / "short.cube"
        path.write_text(
            "c\ns\n"
            f"    0{0.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n"
            f"    2{1.0:12.6f}{0.0:12.6f}{0.0:12.6f}\n"
            f"    2{0.0:12.6f}{1.0:12.6f}{0.0:12.6f}\n"
            f"    2{0.0:12.6f}{0.0:12.6f}{1.0:12.6f}\n"
            "  1.0 2.0 3.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="expected"):
            cube_io.read_cube(str(path))


class TestStamps:
    def test_round_trip(self, tmp_path):
        stamp = cube_io.stamp_line(
            "1.2.3", "zz", (5, 6, 7), [0, 0, 1], "/tmp/run.out", "grid"
        )
        path = cube_io.write_cube(
            str(tmp_path / "a.cube"),
            np.zeros((5, 6, 7)),
            [0, 0, 0],
            np.eye(3),
            stamp=stamp,
        )
        info = cube_io.read_generation_settings(path)
        assert info == {
            "version": "1.2.3",
            "component": "zz",
            "grid": (5, 6, 7),
            "axis": (0.0, 0.0, 1.0),
            "axis_mode": "grid",
        }

    def test_stamp_records_only_the_file_name(self):
        stamp = cube_io.stamp_line("1.0", "iso", (2, 2, 2), source="/some/dir/run.out")
        assert "run.out" in stamp
        assert "/some/dir" not in stamp

    def test_unstamped_cube_reports_unknown(self, tmp_path):
        path = cube_io.write_cube(
            str(tmp_path / "a.cube"), np.zeros((2, 2, 2)), [0, 0, 0], np.eye(3)
        )
        assert cube_io.read_generation_settings(path)["version"] is None

    def test_missing_file_reports_unknown(self, tmp_path):
        info = cube_io.read_generation_settings(str(tmp_path / "nope.cube"))
        assert all(v is None for v in info.values())


class TestPaths:
    def test_cube_dir_sits_beside_the_output(self):
        folder = cube_io.cube_dir_for(os.path.join("dir", "run.out"))
        assert os.path.basename(folder) == "run_nics_cubes"

    def test_cube_path_names_the_component(self):
        path = cube_io.cube_path_for(os.path.join("dir", "run.out"), "zz")
        assert os.path.basename(path) == "run_NICS_zz.cube"

    def test_tag_is_appended(self):
        path = cube_io.cube_path_for("run.out", "iso", "ring 2")
        assert os.path.basename(path) == "run_NICS_iso_ring_2.cube"

    def test_unsafe_characters_are_replaced(self):
        path = cube_io.cube_path_for("run.out", "zz", "a/b:c*d")
        assert os.path.basename(path) == "run_NICS_zz_a_b_c_d.cube"

    def test_no_source_file(self):
        assert cube_io.cube_dir_for(None) is None
        assert cube_io.cube_path_for(None, "zz") is None
