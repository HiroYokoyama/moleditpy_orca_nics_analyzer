"""ORCA output parser, restricted to what NICS analysis needs.

Three things are extracted: the final geometry, which centres are ghosts, and
the NMR shielding of every nucleus (isotropic value plus, when ORCA printed
it, the full 3x3 tensor that ``NICS_zz`` is projected out of).
"""

import logging
import re

# " Nucleus   8H :"  /  " Nucleus  12H:"  /  " Nucleus   0C  :"
_NUCLEUS_RE = re.compile(r"^\s*Nucleus\s+(\d+)\s*([A-Za-z][A-Za-z0-9]*)\s*:?\s*$")
# "   0 C     6.0000    0    12.011    1.522993   -2.161091   -0.015611"
_AU_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([-+0-9.]+)\s+(\d+)\s+([-+0-9.]+)"
    r"\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
# "  H:     0.805933   -1.143600   -0.008261"
_ANG_ROW_RE = re.compile(
    r"^\s*([A-Za-z]{1,2}\s*:?)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_TENSOR_ROW_RE = re.compile(rf"^\s*({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*$")


def _as_float(value):
    """Parse Fortran-style exponents as well as normal floats."""
    return float(value.replace("D", "E").replace("d", "e"))


def _blank(line):
    return not line.strip()


def _is_rule(line):
    """True for an ORCA rule line — including multi-column ones like
    ``  -------  -------  ------------``, whose inner spaces make a naive
    all-dashes test fail."""
    stripped = line.strip()
    return bool(stripped) and set(stripped.replace(" ", "")) <= {"-"}


class NicsParser:
    """Parse an ORCA ``.out``; results land in :attr:`data`."""

    def __init__(self):
        self.filename = None
        self.data = self._empty()

    @staticmethod
    def _empty():
        return {
            "orca_version": None,
            "atoms": [],  # dicts: idx, symbol, label, za, is_ghost, xyz
            "ghost_indices": [],
            "shieldings": {},  # idx -> {iso, aniso, tensor}
            "has_tensors": False,
            "warnings": [],
        }

    # -- entry points ----------------------------------------------------
    def load(self, path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            self.load_from_memory(fh.read(), path)
        return self.data

    def load_from_memory(self, content, path=None):
        if not isinstance(content, str):
            raise TypeError("ORCA output content must be text")
        self.filename = path
        self.data = self._empty()
        lines = content.splitlines()
        self._parse_version(lines)
        self._parse_geometry(lines)
        self._parse_shielding_summary(lines)
        self._parse_tensors(lines)
        self._finalize()
        return self.data

    # -- sections --------------------------------------------------------
    def _parse_version(self, lines):
        for line in lines[:200]:
            m = re.search(r"Program Version\s+(\S+)", line)
            if m:
                self.data["orca_version"] = m.group(1)
                return

    def _parse_geometry(self, lines):
        """Last Angstrom geometry, with ghost flags from the matching A.U. block.

        The A.U. block is authoritative for ghosts: a centre carrying basis
        functions but no nucleus prints ``ZA`` as 0.0. The Angstrom block is
        used for the coordinates themselves so no unit conversion is needed.
        """
        ang = self._last_block(
            lines, "CARTESIAN COORDINATES (ANGSTROEM)", self._read_ang
        )
        au = self._last_block(lines, "CARTESIAN COORDINATES (A.U.)", self._read_au)

        if not ang and not au:
            self.data["warnings"].append(
                "No Cartesian coordinates found in the output."
            )
            return

        atoms = []
        if ang:
            for i, (label, xyz) in enumerate(ang):
                symbol = label.rstrip(":")
                is_ghost = label.endswith(":")
                za = None
                if au and i < len(au):
                    au_label, au_za = au[i][1], au[i][2]
                    za = au_za
                    # ZA == 0 is the unambiguous ghost marker; a label colon in
                    # either block corroborates it.
                    is_ghost = abs(au_za) < 1e-6 or is_ghost or au_label.endswith(":")
                atoms.append(
                    {
                        "idx": i,
                        "symbol": symbol,
                        "label": label,
                        "za": za,
                        "is_ghost": is_ghost,
                        "xyz": xyz,
                    }
                )
        else:
            for idx, label, za, xyz_bohr in au:
                atoms.append(
                    {
                        "idx": idx,
                        "symbol": label.rstrip(":"),
                        "label": label,
                        "za": za,
                        "is_ghost": abs(za) < 1e-6 or label.endswith(":"),
                        "xyz": tuple(v * 0.52917720859 for v in xyz_bohr),
                    }
                )

        if ang and au and len(ang) != len(au):
            self.data["warnings"].append(
                f"Geometry blocks disagree on atom count ({len(ang)} vs {len(au)}); "
                "ghost detection may be incomplete."
            )

        self.data["atoms"] = atoms
        self.data["ghost_indices"] = [a["idx"] for a in atoms if a["is_ghost"]]

    @staticmethod
    def _last_block(lines, header, reader):
        """Run *reader* on the body of the last block introduced by *header*."""
        start = None
        for i, line in enumerate(lines):
            if header in line:
                start = i
        if start is None:
            return []
        i = start + 1
        while i < len(lines) and (_is_rule(lines[i]) or _blank(lines[i])):
            i += 1
        return reader(lines, i)

    @staticmethod
    def _read_ang(lines, i):
        rows = []
        while i < len(lines):
            line = lines[i]
            if _blank(line):
                break
            m = _ANG_ROW_RE.match(line)
            if not m:
                break
            label = m.group(1).replace(" ", "")
            rows.append(
                (label, (float(m.group(2)), float(m.group(3)), float(m.group(4))))
            )
            i += 1
        return rows

    @staticmethod
    def _read_au(lines, i):
        # Skip the "NO LB ZA FRAG MASS X Y Z" header line.
        if i < len(lines) and "ZA" in lines[i] and "FRAG" in lines[i]:
            i += 1
        rows = []
        while i < len(lines):
            line = lines[i]
            if _blank(line):
                break
            m = _AU_ROW_RE.match(line)
            if not m:
                break
            rows.append(
                (
                    int(m.group(1)),
                    m.group(2),
                    float(m.group(3)),
                    (float(m.group(6)), float(m.group(7)), float(m.group(8))),
                )
            )
            i += 1
        return rows

    def _parse_shielding_summary(self, lines):
        """Isotropic/anisotropy table. The last summary in the file wins."""
        start = None
        for i, line in enumerate(lines):
            if "CHEMICAL SHIELDING SUMMARY" in line.upper():
                start = i
        if start is None:
            return
        i = start + 1
        while i < len(lines):
            up = lines[i].upper()
            if "NUCLEUS" in up and "ISOTROPIC" in up:
                i += 1
                break
            i += 1
        while i < len(lines) and (_is_rule(lines[i]) or _blank(lines[i])):
            i += 1
        blanks = 0
        while i < len(lines):
            parts = lines[i].split()
            if len(parts) >= 4 and parts[0].isdigit():
                try:
                    idx = int(parts[0])
                    self.data["shieldings"][idx] = {
                        "symbol": parts[1],
                        "iso": _as_float(parts[2]),
                        "aniso": _as_float(parts[3]),
                        "tensor": None,
                    }
                except ValueError:
                    logging.debug(
                        "[orca_nics_analyzer] failed to parse summary line: %r",
                        lines[i],
                        exc_info=True,
                    )
                blanks = 0
            elif _blank(lines[i]):
                blanks += 1
                if blanks >= 2:
                    break
            else:
                break
            i += 1

    def _parse_tensors(self, lines):
        """Per-nucleus ``Total shielding tensor`` blocks, when printed."""
        i = 0
        n = len(lines)
        while i < n:
            m = _NUCLEUS_RE.match(lines[i])
            if not m:
                i += 1
                continue
            idx = int(m.group(1))
            symbol = m.group(2)
            j = i + 1
            tensor = None
            # Scan forward to this nucleus' total tensor, stopping at the next
            # nucleus header so a missing block never steals the following one.
            while j < n and not _NUCLEUS_RE.match(lines[j]):
                if "Total shielding tensor" in lines[j]:
                    tensor = self._read_tensor(lines, j + 1)
                    break
                j += 1
            if tensor is not None:
                entry = self.data["shieldings"].setdefault(
                    idx, {"symbol": symbol, "iso": None, "aniso": None, "tensor": None}
                )
                entry["tensor"] = tensor
                if entry.get("iso") is None:
                    entry["iso"] = (tensor[0][0] + tensor[1][1] + tensor[2][2]) / 3.0
                self.data["has_tensors"] = True
            i = j if j > i else i + 1

    @staticmethod
    def _read_tensor(lines, i):
        rows = []
        while i < len(lines) and len(rows) < 3:
            if _blank(lines[i]):
                i += 1
                continue
            m = _TENSOR_ROW_RE.match(lines[i])
            if not m:
                return None
            rows.append([_as_float(m.group(k)) for k in (1, 2, 3)])
            i += 1
        return rows if len(rows) == 3 else None

    def _finalize(self):
        if self.data["atoms"] and not self.data["shieldings"]:
            self.data["warnings"].append(
                "No NMR shielding data found — this output has no NMR/EPR section."
            )
        missing = [
            i for i in self.data["ghost_indices"] if i not in self.data["shieldings"]
        ]
        if missing:
            self.data["warnings"].append(
                f"{len(missing)} ghost centre(s) have no shielding value; they were "
                "probably not listed in the NMR 'Nuclei' selection."
            )
        # Ghosts with data are what every downstream tab iterates over.
        self.data["probe_indices"] = [
            i for i in self.data["ghost_indices"] if i in self.data["shieldings"]
        ]

    # -- convenience -----------------------------------------------------
    @property
    def probes(self):
        """[(index, xyz, shielding-entry)] for every ghost carrying NMR data."""
        by_idx = {a["idx"]: a for a in self.data["atoms"]}
        out = []
        for idx in self.data.get("probe_indices", []):
            atom = by_idx.get(idx)
            if atom is not None:
                out.append((idx, atom["xyz"], self.data["shieldings"][idx]))
        return out

    @property
    def real_atoms(self):
        return [a for a in self.data["atoms"] if not a["is_ghost"]]
