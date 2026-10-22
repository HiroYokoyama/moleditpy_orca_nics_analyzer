"""Result boundaries and encoding regressions found during the source audit."""

from pathlib import Path

import pytest

from orca_nics_analyzer.parser import NicsParser, read_output_text


def test_cp1252_is_not_mistaken_for_unmarked_utf16(tmp_path, single_out):
    content = Path(single_out).read_text(encoding="utf-8")
    raw = (content + "\n# café\n").encode("cp1252")
    if len(raw) % 2:
        raw += b" "
    path = tmp_path / "encoded.out"
    path.write_bytes(raw)
    assert "Program Version" in read_output_text(path)
    assert NicsParser().load(path)["probe_indices"]


def test_parser_load_accepts_bom_utf16(tmp_path, single_out):
    content = Path(single_out).read_text(encoding="utf-8")
    path = tmp_path / "encoded.out"
    path.write_text(content, encoding="utf-16")
    assert NicsParser().load(path)["probe_indices"]


@pytest.mark.parametrize(
    "boundary", ["Program Version 6.1.1", "JOB NUMBER 2", "* JOB NUMBER 2 *"]
)
def test_new_calculation_cannot_inherit_old_data(single_out, boundary):
    content = Path(single_out).read_text(encoding="utf-8")
    data = NicsParser().load_from_memory(content + "\n" + boundary + "\n")
    assert data["atoms"] == []
    assert data["shieldings"] == {}
    assert data["probe_indices"] == []


def test_new_angstrom_geometry_does_not_inherit_old_ghost_flags(single_out):
    content = Path(single_out).read_text(encoding="utf-8")
    content += "\nCARTESIAN COORDINATES (ANGSTROEM)\n---\n"
    content += "\n".join("H 0.0 0.0 1.0" for _ in range(15)) + "\n\n"
    data = NicsParser().load_from_memory(content)
    assert len(data["atoms"]) == 15
    assert data["ghost_indices"] == []


def test_entry_reports_invalid_geometry_instead_of_raising(
    qapp, monkeypatch, tmp_path, no_modals
):
    import orca_nics_analyzer as plugin

    path = tmp_path / "overflow.out"
    path.write_text("CARTESIAN COORDINATES (ANGSTROEM)\n---\nH: 1e999 0 0\n")
    assert plugin.parse_output_file(path, None) is None
    assert no_modals and no_modals[-1][0] == "critical"
