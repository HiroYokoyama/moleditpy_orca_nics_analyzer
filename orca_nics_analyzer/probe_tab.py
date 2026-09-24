"""Probe table: every ghost centre with its NICS values.

A model/view table rather than QTableWidget: a volume grid carries tens of
thousands of probes, and one QTableWidgetItem per cell (x 12 columns) makes
building — and rebuilding on every NICS_zz axis change — take seconds.
"""

import logging
import os

from PyQt6.QtCore import QAbstractTableModel, QSortFilterProxyModel, Qt
from PyQt6.QtGui import QBrush, QColor, QGuiApplication
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .analysis import NicsField

logger = logging.getLogger(__name__)

#: Colour ramp endpoints for the NICS column background.
_DIATROPIC = QColor(60, 110, 200)
_PARATROPIC = QColor(200, 70, 60)

#: Role carrying a cell's raw value, which the proxy sorts on.
SORT_ROLE = Qt.ItemDataRole.UserRole


def nics_brush(value, span):
    """Blue for diatropic, red for paratropic, transparent near zero."""
    if value is None or span <= 0:
        return None
    frac = max(-1.0, min(1.0, float(value) / span))
    base = _PARATROPIC if frac > 0 else _DIATROPIC
    alpha = int(180 * abs(frac))
    return QBrush(QColor(base.red(), base.green(), base.blue(), alpha))


def _sort_key(value):
    """Missing values ("-" or None) first, then numbers by value, then text."""
    if value is None or value == "-":
        return (0, 0.0, "")
    if isinstance(value, (int, float)):
        return (1, float(value), "")
    return (2, 0.0, str(value))


class ProbeTableModel(QAbstractTableModel):
    """The rows of :meth:`NicsField.probe_rows`, formatted on demand."""

    HEADERS = tuple(title for _, title in NicsField.CSV_COLUMNS)
    KEYS = tuple(key for key, _ in NicsField.CSV_COLUMNS)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._span = 0.0
        self.colour = True

    def set_rows(self, rows):
        self.beginResetModel()
        self._rows = rows
        values = [
            r["nics_zz"] if r["nics_zz"] is not None else r["nics_iso"] for r in rows
        ]
        finite = [abs(v) for v in values if v is not None]
        self._span = max(finite) if finite else 0.0
        self.endResetModel()

    def set_colour(self, enabled):
        self.colour = bool(enabled)
        if self._rows:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._rows) - 1, len(self.KEYS) - 1),
                [Qt.ItemDataRole.BackgroundRole],
            )

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self._rows)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.KEYS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return str(section + 1)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        key = self.KEYS[index.column()]
        value = self._rows[index.row()][key]
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)
        if role == SORT_ROLE:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, float):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if (
            role == Qt.ItemDataRole.BackgroundRole
            and self.colour
            and key in ("nics_iso", "nics_zz")
        ):
            return nics_brush(value, self._span)
        return None


class _ProbeSortProxy(QSortFilterProxyModel):
    """Sorts on the raw value, so "-10" lands before "-9"."""

    def lessThan(self, left, right):
        return _sort_key(left.data(SORT_ROLE)) < _sort_key(right.data(SORT_ROLE))


class ProbeTab(QWidget):
    """Sortable table of probes, with CSV copy/export."""

    HEADERS = ProbeTableModel.HEADERS

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

        self.model = ProbeTableModel(self)
        self.proxy = _ProbeSortProxy(self)
        self.proxy.setSourceModel(self.model)

        self.table = QTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        # No sort until a header is clicked: probe order is the file's order.
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.colour_chk = QCheckBox("Colour by value")
        self.colour_chk.setChecked(True)
        self.colour_chk.toggled.connect(self.model.set_colour)
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
        self.model.set_rows(rows)
        # Sizing to contents measures every row; a sample says as much.
        if len(rows) <= 2000:
            self.table.resizeColumnsToContents()
        else:
            self.table.horizontalHeader().setDefaultSectionSize(90)
        self.info.setText(self._summary(rows))

    def cell_text(self, row, column):
        """The text shown at (*row*, *column*) of the table, as sorted on screen."""
        return self.proxy.index(row, column).data(Qt.ItemDataRole.DisplayRole)

    def cell_value(self, row, column):
        """The raw value at (*row*, *column*) of the table, as sorted on screen."""
        return self.proxy.index(row, column).data(SORT_ROLE)

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
            logger.warning("[orca_nics_analyzer] CSV export: %s", e)
            QMessageBox.critical(
                self, "Export failed", f"Could not write the file:\n{e}"
            )
