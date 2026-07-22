"""page_samples.py — Samples tab: group maps into classes / batches / roles."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QCheckBox, QSpinBox,
)

from ui_common import *
from real_data import PEST_DEFAULT
from dataset import (discover_references, base_and_batch, load_manifest,
                     save_manifest, map_pixel_count, load_preprocess,
                     save_preprocess)


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
        sub = QLabel("Step 1 — prepare the data. Group your maps into substance "
                     "classes (repeat measurements are BATCHES of one class: THI, "
                     "THI_2 → one 'THI'), choose Role (train / test / exclude), and "
                     "set the PREPROCESSING here once. Model, Predict and Real data "
                     "all reuse it — you set it once and never re-enter it.")
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

        # preprocessing — set ONCE here, reused by every downstream tab
        prep = QHBoxLayout(); prep.setSpacing(10)
        pl = QLabel("preprocessing:"); pl.setObjectName("field")
        prep.addWidget(pl)
        self.chk_base = QCheckBox("ALS baseline"); self.chk_base.setChecked(True)
        prep.addWidget(self.chk_base)
        prep.addLayout(self._combo_col("derivative", "cmb_deriv",
                                       [("none", 0), ("1st", 1), ("2nd", 2)]))
        prep.addLayout(self._combo_col("normalize", "cmb_norm",
                                       [("L2", "l2"), ("SNV", "snv"), ("none", "none")]))
        self.sp_lo = QSpinBox(); self.sp_lo.setRange(0, 4000); self.sp_lo.setSingleStep(50)
        self.sp_hi = QSpinBox(); self.sp_hi.setRange(0, 4000); self.sp_hi.setSingleStep(50)
        self.sp_hi.setValue(4000)
        prep.addLayout(self._spin_col("trim lo cm⁻¹", self.sp_lo))
        prep.addLayout(self._spin_col("trim hi cm⁻¹", self.sp_hi))
        prep.addStretch(1)
        root.addLayout(prep)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["File", "# px", "Class (substance)", "Batch", "Role (train/test/exclude)"])
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

    def _combo_col(self, label, attr, items):
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        cb = QComboBox()
        for text, data in items:
            cb.addItem(text, data)
        setattr(self, attr, cb)
        col.addWidget(lb); col.addWidget(cb)
        return col

    def _spin_col(self, label, spin):
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        col.addWidget(lb); col.addWidget(spin)
        return col

    def _gather_prep(self):
        """Current preprocessing choices as a config dict (trim None if full range)."""
        lo, hi = self.sp_lo.value(), self.sp_hi.value()
        trim = (lo, hi) if (hi > lo and (lo > 0 or hi < 4000)) else None
        return {"baseline": self.chk_base.isChecked(),
                "deriv": self.cmb_deriv.currentData(),
                "norm": self.cmb_norm.currentData(), "trim": trim}

    def _load_prep(self):
        """Set the preprocessing widgets from the folder's saved config."""
        cfg = load_preprocess(self.data_dir)
        self.chk_base.setChecked(cfg["baseline"])
        self.cmb_deriv.setCurrentIndex(max(0, self.cmb_deriv.findData(cfg["deriv"])))
        self.cmb_norm.setCurrentIndex(max(0, self.cmb_norm.findData(cfg["norm"])))
        self.sp_lo.setValue(cfg["trim"][0] if cfg["trim"] else 0)
        self.sp_hi.setValue(cfg["trim"][1] if cfg["trim"] else 4000)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Data folder with your reference maps", self.data_dir)
        if d:
            self.data_dir = d; self.folder_lbl.setText(self._short(d)); self._reload()

    def _reload(self):
        self._loading = True
        self._load_prep()                                 # folder's saved preprocessing
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
            self.table.setCellWidget(r, 4, self._role_combo(role))
        self._loading = False
        if refs:
            self.status.setText(f"{len(refs)} maps"); self.status.setStyleSheet(f"color:{MUTE};")
        self._update_summary()

    def _role_combo(self, role):
        cb = QComboBox()
        cb.addItems(["train", "test", "exclude"])
        cb.setCurrentText(role if role in ("train", "test", "exclude") else "train")
        cb.currentIndexChanged.connect(lambda _=0: self._on_edit(None))
        return cb

    def _rows(self):
        out = []
        for r in range(self.table.rowCount()):
            fn = self.table.item(r, 0).text()
            cls = (self.table.item(r, 2).text() or "").strip()
            bt = (self.table.item(r, 3).text() or "").strip()
            cb = self.table.cellWidget(r, 4)
            role = cb.currentText() if cb else "train"
            out.append((fn, cls, int(bt) if bt.isdigit() else 1, role))
        return out

    def _on_edit(self, _item):
        if not self._loading:
            self._update_summary()

    def _update_summary(self):
        groups = {}; n_train = n_test = n_excl = 0
        for _fn, cls, _b, role in self._rows():
            role = role.strip().lower()
            if role.startswith(("ex", "sk", "ig", "off")):
                n_excl += 1
                continue                                   # excluded maps aren't used
            groups[cls] = groups.get(cls, 0) + 1
            if role.startswith("te"):
                n_test += 1
            else:
                n_train += 1
        if not groups:
            self.summary.setText("no maps in use in this folder"); return
        parts = [f"{c} ×{n}" if n > 1 else c for c, n in sorted(groups.items())]
        excl = f"  ·  {n_excl} excluded" if n_excl else ""
        self.summary.setText(
            f"{len(groups)} classes:   " + "   ·   ".join(parts)
            + f"      |      maps: {n_train} train / {n_test} test{excl}  "
            "(Role = exclude drops a map; = test holds it out for Model → split = manual)")

    def _save(self):
        rows = self._rows()
        if not rows:
            self.status.setText("nothing to save"); return
        try:
            save_manifest(self.data_dir, rows)
            save_preprocess(self.data_dir, self._gather_prep())
            self.status.setText(f"saved samples.csv + preprocess.json ({len(rows)} maps)")
            self.status.setStyleSheet(f"color:{TEAL};")
        except Exception as exc:
            self.status.setText("save failed"); self.status.setStyleSheet(f"color:{RED};")
            print("save manifest:", exc, file=sys.stderr)
