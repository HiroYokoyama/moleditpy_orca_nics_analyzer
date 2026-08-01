"""Probe table: every ghost centre with its NICS values."""

import logging
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QGuiApplication
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .analysis import NicsField

#: Colour ramp endpoints for the NICS column background.
_DIATROPIC = QColor(60, 110, 200)
_PARATROPIC = QColor(200, 70, 60)


class _NumericItem(QTableWidgetItem):
    """Table cell that sorts on its value, not on its formatted text."""

    def __init__(self, text, value):
        super().__init__(text)
        self.value = value

    def __lt__(self, other):
        mine = self.value
        theirs = getattr(other, "value", None)
        if mine is None:
            return theirs is not None
        if theirs is None:
            return False
        return mine < theirs


def nics_brush(value, span):
    """Blue for diatropic, red for paratropic, transparent near zero."""
    if value is None or span <= 0:
        return None
    frac = max(-1.0, min(1.0, float(value) / span))
    base = _PARATROPIC if frac > 0 else _DIATROPIC
    alpha = int(180 * abs(frac))
    return QBrush(QColor(base.red(), base.green(), base.blue(), alpha))


class ProbeTab(QWidget):
    """Sortable table of probes, with CSV copy/export."""

    HEADERS = [title for _, title in NicsField.CSV_COLUMNS]

    def __init__(self, field, parent=None):
        super().__init__(parent)
        self.field = field
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        self.table = QTableWidget(self)
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.colour_chk = QCheckBox("Colour by value")
        self.colour_chk.setChecked(True)
        self.colour_chk.toggled.connect(self.refresh)
        row.addWidget(self.colour_chk)
        row.addStretch(1)

        copy_btn = QPushButton("Copy CSV")
        copy_btn.clicked.connect(self.copy_csv)
        row.addWidget(copy_btn)

        save_btn = QPushButton("Export CSV...")
        save_btn.clicked.connect(self.export_csv)
        row.addWidget(save_btn)
        layout.addLayout(row)

    # -- data ------------------------------------------------------------
    def refresh(self):
        rows = self.field.probe_rows()
        keys = [k for k, _ in self.field.CSV_COLUMNS]

        values = [
            r["nics_zz"] if r["nics_zz"] is not None else r["nics_iso"] for r in rows
        ]
        finite = [abs(v) for v in values if v is not None]
        span = max(finite) if finite else 0.0

        # Sorting must be off while filling, or Qt re-sorts mid-population and
        # scrambles which value lands in which row.
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        colour = self.colour_chk.isChecked()
        for r, row in enumerate(rows):
            for c, key in enumerate(keys):
                value = row[key]
                if value is None:
                    item = _NumericItem("-", None)
                elif isinstance(value, float):
                    # _NumericItem so the column sorts on the value; plain text
                    # would put "-10" after "-9".
                    item = _NumericItem(f"{value:.3f}", value)
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                elif isinstance(value, int):
                    item = _NumericItem(str(value), value)
                else:
                    item = QTableWidgetItem(str(value))
                if colour and key in ("nics_iso", "nics_zz"):
                    brush = nics_brush(value, span)
                    if brush is not None:
                        item.setBackground(brush)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.info.setText(self._summary(rows))

    def _summary(self, rows):
        zz = [r["nics_zz"] for r in rows if r["nics_zz"] is not None]
        iso = [r["nics_iso"] for r in rows if r["nics_iso"] is not None]
        parts = [f"{len(rows)} probe(s)"]
        if iso:
            parts.append(f"NICS(iso) {min(iso):+.2f} to {max(iso):+.2f} ppm")
        if zz:
            parts.append(f"NICS_zz {min(zz):+.2f} to {max(zz):+.2f} ppm")
        else:
            parts.append("no shielding tensors in this output — isotropic values only")
        return "   |   ".join(parts)

    # -- export ----------------------------------------------------------
    def copy_csv(self):
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.field.to_csv())

    def export_csv(self):
        default = ""
        if self.field.filename:
            base = os.path.splitext(self.field.filename)[0]
            default = f"{base}_NICS_probes.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export probe table", default, "CSV (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(self.field.to_csv())
        except OSError as e:
            logging.warning("[orca_nics_analyzer] CSV export: %s", e)
            QMessageBox.critical(
                self, "Export failed", f"Could not write the file:\n{e}"
            )
