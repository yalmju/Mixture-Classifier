"""page_samples.py — Samples tab: group maps into classes / batches / roles."""
from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from ui_common import *
from real_data import PEST_DEFAULT
from dataset import (discover_references, base_and_batch, load_manifest,
                     save_manifest, map_pixel_count)


# --------------------------------------------------------------------------
# Samples page
# --------------------------------------------------------------------------
class SamplingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.data_dir = PEST_DEFAULT
        self._loading = False
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(14)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Samples"); h1.setObjectName("h1")
        sub = QLabel("Group your reference maps into substance classes. Repeat "
                     "measurements of the same substance are BATCHES of one class — "
                     "not separate classes (THI, THI_2 → one 'THI'). Edit Class / "
                     "Batch / Role (train or test) per map, then Save. Model, "
                     "Predict and Real data read this from samples.csv; the Model "
                     "'manual' split trains on train-role maps and tests on test.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        browse = QPushButton("Data folder…"); browse.setObjectName("ghost")
        browse.clicked.connect(self._browse)
        self.folder_lbl = QLabel(self._short(self.data_dir)); self.folder_lbl.setObjectName("field")
        self.status = QLabel(""); self.status.setObjectName("sub")
        rescan = QPushButton("Rescan"); rescan.setObjectName("ghost")
        rescan.clicked.connect(self._reload)
        save = QPushButton("Save dataset"); save.setObjectName("primary")
        save.clicked.connect(self._save)
        ctl.addWidget(browse); ctl.addWidget(self.folder_lbl, 1)
        ctl.addWidget(self.status); ctl.addStretch(1)
        ctl.addWidget(rescan); ctl.addWidget(save)
        root.addLayout(ctl)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["File", "# px", "Class (substance)", "Batch", "Role (train/test)"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3, 4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_edit)
        root.addWidget(self.table, 1)

        self.summary = QLabel(""); self.summary.setObjectName("sub")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        self._reload()

    def _short(self, p):
        return "data: " + ("…" + p[-42:] if len(p) > 42 else p)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Data folder with your reference maps", self.data_dir)
        if d:
            self.data_dir = d; self.folder_lbl.setText(self._short(d)); self._reload()

    def _reload(self):
        self._loading = True
        self.table.setRowCount(0)
        try:
            refs = discover_references(self.data_dir)
            manifest = load_manifest(self.data_dir)
        except Exception as exc:
            refs, manifest = [], None
            self.status.setText("scan failed"); self.status.setStyleSheet(f"color:{RED};")
            print("sampling scan:", exc, file=sys.stderr)
        for name, path in refs:
            cls, batch, role = None, None, "train"
            if manifest is not None:
                hit = manifest.get(os.path.abspath(path))
                if hit and hit[0]:
                    cls, batch, role = hit
            if cls is None:
                cls, batch = base_and_batch(name)
            r = self.table.rowCount(); self.table.insertRow(r)
            fitem = QTableWidgetItem(os.path.basename(path))
            fitem.setFlags(fitem.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, fitem)
            px = QTableWidgetItem(f"{map_pixel_count(path):,}")
            px.setFlags(px.flags() & ~Qt.ItemFlag.ItemIsEditable)
            px.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 1, px)
            self.table.setItem(r, 2, QTableWidgetItem(str(cls)))
            self.table.setItem(r, 3, QTableWidgetItem(str(batch)))
            self.table.setItem(r, 4, QTableWidgetItem(role))
        self._loading = False
        if refs:
            self.status.setText(f"{len(refs)} maps"); self.status.setStyleSheet(f"color:{MUTE};")
        self._update_summary()

    def _rows(self):
        out = []
        for r in range(self.table.rowCount()):
            fn = self.table.item(r, 0).text()
            cls = (self.table.item(r, 2).text() or "").strip()
            bt = (self.table.item(r, 3).text() or "").strip()
            role = (self.table.item(r, 4).text() or "").strip()
            out.append((fn, cls, int(bt) if bt.isdigit() else 1, role))
        return out

    def _on_edit(self, _item):
        if not self._loading:
            self._update_summary()

    def _update_summary(self):
        groups = {}; n_train = n_test = 0
        for _fn, cls, _b, role in self._rows():
            groups[cls] = groups.get(cls, 0) + 1
            if role.strip().lower().startswith("te"):
                n_test += 1
            else:
                n_train += 1
        if not groups:
            self.summary.setText("no maps found in this folder"); return
        parts = [f"{c} ×{n}" if n > 1 else c for c, n in sorted(groups.items())]
        self.summary.setText(
            f"{len(groups)} classes:   " + "   ·   ".join(parts)
            + f"      |      maps: {n_train} train / {n_test} test  "
            "(set Role = test to hold maps out; Model → split = manual)")

    def _save(self):
        rows = self._rows()
        if not rows:
            self.status.setText("nothing to save"); return
        try:
            save_manifest(self.data_dir, rows)
            self.status.setText(f"saved samples.csv ({len(rows)} maps)")
            self.status.setStyleSheet(f"color:{TEAL};")
        except Exception as exc:
            self.status.setText("save failed"); self.status.setStyleSheet(f"color:{RED};")
            print("save manifest:", exc, file=sys.stderr)
