"""Parser tests, against both generated fixtures and a genuine ORCA 5 output."""

import pytest

from orca_nics_analyzer.parser import NicsParser, _is_rule


class TestRealOutput:
    """The parser must survive a real ORCA file it did not generate."""

    def test_geometry_and_shieldings(self, real_out):
        data = NicsParser().load(real_out)
        assert data["orca_version"] == "5.0.4"
        assert len(data["atoms"]) == 12
        assert [a["symbol"] for a in data["atoms"]].count("C") == 6
        assert data["ghost_indices"] == []
        assert len(data["shieldings"]) == 12
        assert data["has_tensors"] is True

    def test_summary_values_attach_to_the_right_nucleus(self, real_out):
        data = NicsParser().load(real_out)
        assert data["shieldings"][0]["iso"] == pytest.approx(63.950)
        assert data["shieldings"][0]["aniso"] == pytest.approx(177.035)
        assert data["shieldings"][6]["iso"] == pytest.approx(24.260)

    def test_tensor_belongs_to_its_own_nucleus(self, real_out):
        """A missing tensor block must not shift values onto the next nucleus."""
        data = NicsParser().load(real_out)
        tensor = data["shieldings"][11]["tensor"]
        assert tensor[0][0] == pytest.approx(26.861)
        assert tensor[2][2] == pytest.approx(21.251)

    def test_trace_matches_the_summary(self, real_out):
        data = NicsParser().load(real_out)
        for entry in data["shieldings"].values():
            trace = sum(entry["tensor"][i][i] for i in range(3)) / 3.0
            assert trace == pytest.approx(entry["iso"], abs=0.01)


class TestGhostDetection:
    def test_ghosts_found_by_zero_nuclear_charge(self, single_out):
        data = NicsParser().load(single_out)
        assert data["ghost_indices"] == [12, 13, 14]
        assert all(data["atoms"][i]["za"] == 0.0 for i in data["ghost_indices"])
        assert all(data["atoms"][i]["symbol"] == "H" for i in data["ghost_indices"])

    def test_real_atoms_exclude_ghosts(self, single_out):
        parser = NicsParser()
        parser.load(single_out)
        assert len(parser.real_atoms) == 12
        assert all(not a["is_ghost"] for a in parser.real_atoms)

    def test_probes_carry_coordinates_and_shieldings(self, single_out):
        parser = NicsParser()
        parser.load(single_out)
        probes = parser.probes
        assert len(probes) == 3
        idx, xyz, entry = probes[1]
        assert idx == 13
        assert xyz == pytest.approx((0.0, 0.0, 1.0))
        assert entry["tensor"] is not None

    def test_output_without_ghosts(self, no_ghosts_out):
        data = NicsParser().load(no_ghosts_out)
        assert data["ghost_indices"] == []
        assert data["probe_indices"] == []
        assert len(data["shieldings"]) == 12


class TestLargeGrids:
    def test_plane(self, plane_out):
        data = NicsParser().load(plane_out)
        assert len(data["probe_indices"]) == 81

    def test_volume(self, volume_out):
        data = NicsParser().load(volume_out)
        assert len(data["probe_indices"]) == 125
        assert len(data["atoms"]) == 12 + 125


class TestEdgeCases:
    def test_empty_input(self):
        data = NicsParser().load_from_memory("")
        assert data["atoms"] == []
        assert "No Cartesian coordinates" in data["warnings"][0]

    def test_no_nmr_section(self):
        text = (
            "CARTESIAN COORDINATES (ANGSTROEM)\n"
            "---------------------------------\n"
            "  C     0.000000    0.000000    0.000000\n"
            "  H     1.000000    0.000000    0.000000\n"
            "\n"
        )
        data = NicsParser().load_from_memory(text)
        assert len(data["atoms"]) == 2
        assert any("no NMR" in w for w in data["warnings"])

    def test_ghost_without_shielding_is_reported(self, single_out):
        with open(single_out, encoding="utf-8") as fh:
            text = fh.read()
        # Drop nucleus 14 from the summary and from the tensor blocks.
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("14 ")]
        lines = [ln for ln in lines if "Nucleus  14H" not in ln]
        data = NicsParser().load_from_memory("\n".join(lines))
        assert 14 in data["ghost_indices"]
        assert 14 not in data["probe_indices"]
        assert any("no shielding value" in w for w in data["warnings"])

    def test_last_geometry_wins(self):
        """An optimization prints many geometries; the final one is the NMR one."""
        block = (
            "CARTESIAN COORDINATES (ANGSTROEM)\n"
            "---------------------------------\n"
            "  C  {x:.6f}    0.000000    0.000000\n"
            "\n"
        )
        text = block.format(x=0.0) + block.format(x=9.0)
        data = NicsParser().load_from_memory(text)
        assert len(data["atoms"]) == 1
        assert data["atoms"][0]["xyz"][0] == pytest.approx(9.0)

    def test_colon_in_label_marks_a_ghost_without_an_au_block(self):
        text = (
            "CARTESIAN COORDINATES (ANGSTROEM)\n"
            "---------------------------------\n"
            "  C      0.000000    0.000000    0.000000\n"
            "  H:     0.000000    0.000000    1.000000\n"
            "\n"
        )
        data = NicsParser().load_from_memory(text)
        assert data["ghost_indices"] == [1]
        assert data["atoms"][1]["symbol"] == "H"

    def test_mismatched_geometry_blocks_warn(self):
        text = (
            "CARTESIAN COORDINATES (ANGSTROEM)\n"
            "---------------------------------\n"
            "  C      0.000000    0.000000    0.000000\n"
            "  H      1.000000    0.000000    0.000000\n"
            "\n"
            "CARTESIAN COORDINATES (A.U.)\n"
            "----------------------------\n"
            "  NO LB      ZA    FRAG     MASS         X           Y           Z\n"
            "   0 C     6.0000    0    12.011    0.000000    0.000000    0.000000\n"
            "\n"
        )
        data = NicsParser().load_from_memory(text)
        assert any("disagree on atom count" in w for w in data["warnings"])

    def test_truncated_tensor_block_is_ignored(self):
        text = (
            " --------------\n"
            " Nucleus   0C :\n"
            " --------------\n"
            "Total shielding tensor (ppm): \n"
            "          1.000          0.000          0.000\n"
            "          0.000          2.000          0.000\n"
            "\n"
        )
        data = NicsParser().load_from_memory(text)
        assert data["has_tensors"] is False


class TestRuleDetection:
    """The multi-column rule under the summary header broke the table reader."""

    @pytest.mark.parametrize(
        "line",
        ["-------", "  -------  -------  ------------   ------------  ", "---"],
    )
    def test_rules(self, line):
        assert _is_rule(line)

    @pytest.mark.parametrize("line", ["", "   ", "  0  C  1.0", "- 1.0"])
    def test_not_rules(self, line):
        assert not _is_rule(line)
