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
                     save_preprocess, load_mixture_list, save_mixture_list)
from validate import parse_mixture_label


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
        self.cls_tgl = QPushButton(); self.cls_tgl.setObjectName("ghost")
        self.cls_tgl.setCheckable(True); self.cls_tgl.setChecked(True)
        self.cls_tgl.setStyleSheet("text-align:left; padding:4px 8px;")
        self.cls_tgl.toggled.connect(self._toggle_cls)
        self.table.setMaximumHeight(320)
        root.addWidget(self.cls_tgl)
        root.addWidget(self.table)
        self._toggle_cls(True)

        # ---- known-ratio mixtures (shared with Model / Recovery) ----
        mhead = QHBoxLayout(); mhead.setSpacing(8)
        self.mix_tgl = QPushButton(); self.mix_tgl.setObjectName("ghost")
        self.mix_tgl.setCheckable(True); self.mix_tgl.setChecked(True)
        self.mix_tgl.setStyleSheet("text-align:left; padding:4px 8px;")
        self.mix_tgl.toggled.connect(self._toggle_mix)
        self.chk_mix_uM = QCheckBox("ratios are µM"); self.chk_mix_uM.setObjectName("field")
        self.chk_mix_uM.setToolTip("treat the ratio numbers as absolute µM (e.g. DQ1000 → "
                                   "1000 µM) so the concentration head can be trained")
        self.chk_mix_uM.stateChanged.connect(lambda _=0: self._save_mix())
        mix_add = QPushButton("Add mixtures…"); mix_add.setObjectName("ghost")
        mix_add.clicked.connect(self._add_mix)
        mix_clr = QPushButton("Clear"); mix_clr.setObjectName("ghost")
        mix_clr.clicked.connect(self._clear_mix)
        mhead.addWidget(self.mix_tgl, 1); mhead.addStretch(1)
        mhead.addWidget(self.chk_mix_uM); mhead.addWidget(mix_add); mhead.addWidget(mix_clr)
        root.addLayout(mhead)

        self.mix_files = []
        self.mix_table = QTableWidget(0, 2)
        self.mix_table.setHorizontalHeaderLabels(["mixture file", "true ratio  (e.g. DQ:1, TBZ:3)"])
        mh = self.mix_table.horizontalHeader()
        mh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        mh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.mix_table.verticalHeader().setVisible(False)
        self.mix_table.setMaximumHeight(150)
        self.mix_table.itemChanged.connect(self._on_mix_edit)
        root.addWidget(self.mix_table)
        self._toggle_mix(True)

        root.addStretch(1)                                # absorb slack so collapsing stays tidy
        self.summary = QLabel(""); self.summary.setObjectName("sub")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        self._reload()
        self._load_mix()

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

    def _toggle_cls(self, on):
        self.table.setVisible(on)
        self.cls_tgl.setText(("▾  " if on else "▸  ") + "Classes  (substance · batch · role)")

    def _toggle_mix(self, on):
        self.mix_table.setVisible(on)
        self.mix_tgl.setText(("▾  " if on else "▸  ")
                             + "Mixtures  (known-ratio, shared with Model / Recovery)")

    def _gather_prep(self):
        """Current preprocessing choices as a config dict (trim None if full range)."""
        lo, hi = self.sp_lo.value(), self.sp_hi.value()
        trim = (lo, hi) if (hi > lo and (lo > 0 or hi < 4000)) else None
        return {"baseline": self.chk_base.isChecked(),
                "deriv": self.cmb_deriv.currentData(),
                "norm": self.cmb_norm.currentData(), "trim": trim}

    # ---- known-ratio mixtures (Step 1 prepares them; Model / Recovery share them) ----
    def _mix_ref_names(self):
        """Substance names to parse ratios against — the classes assigned above."""
        from dataset import is_blank
        names = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 2)
            n = it.text().strip() if it else ""
            if n and not is_blank(n) and n not in names:
                names.append(n)
        return names

    def _add_mix(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Known-ratio mixture maps", self.data_dir,
                                                "maps (*.csv *.txt);;all files (*)")
        if not paths:
            return
        refs = self._mix_ref_names()
        self._loading = True
        for p in paths:
            if p in self.mix_files:
                continue
            self.mix_files.append(p)
            row = self.mix_table.rowCount(); self.mix_table.insertRow(row)
            it0 = QTableWidgetItem(os.path.basename(p))
            it0.setFlags(it0.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mix_table.setItem(row, 0, it0)
            g = parse_mixture_label(os.path.splitext(os.path.basename(p))[0], refs)
            txt = ", ".join(f"{k}:{v:.3g}" for k, v in g.items()) if g else ""
            self.mix_table.setItem(row, 1, QTableWidgetItem(txt))
        self._loading = False
        self._save_mix()

    def _clear_mix(self):
        self._loading = True
        self.mix_files = []; self.mix_table.setRowCount(0)
        self._loading = False
        self._save_mix()

    def _on_mix_edit(self, _item):
        if not self._loading:
            self._save_mix()

    def _mix_items(self):
        items = []
        for row in range(self.mix_table.rowCount()):
            cell = self.mix_table.item(row, 1); ratio = {}
            for tok in (cell.text() if cell else "").replace(";", ",").split(","):
                if ":" not in tok:
                    continue
                k, v = tok.split(":", 1)
                try:
                    ratio[k.strip()] = float(v)
                except ValueError:
                    pass
            if len(ratio) >= 2:
                if self.chk_mix_uM.isChecked():
                    items.append((self.mix_files[row], ratio, {k: v * 1e-6 for k, v in ratio.items()}))
                else:
                    items.append((self.mix_files[row], ratio))
        return items

    def _save_mix(self):
        if self._loading:                                 # don't persist mid-(re)load — that
            return                                        # once wiped mixtures.json to []
        try:
            save_mixture_list(self.data_dir, self._mix_items())
            MIXTURE_BUS.changed.emit()
            n = self.mix_table.rowCount()
            self.summary.setText(self.summary.text().split("  ·  mixtures")[0] +
                                 (f"  ·  mixtures: {n}" if n else ""))
        except Exception as e:
            print(e, file=sys.stderr)

    def _load_mix(self):
        self._loading = True
        self.mix_table.setRowCount(0); self.mix_files = []
        items = load_mixture_list(self.data_dir)
        self.chk_mix_uM.setChecked(any(len(it) > 2 and it[2] for it in items))
        for it in items:
            self.mix_files.append(it[0])
            row = self.mix_table.rowCount(); self.mix_table.insertRow(row)
            it0 = QTableWidgetItem(os.path.basename(it[0]))
            it0.setFlags(it0.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.mix_table.setItem(row, 0, it0)
            self.mix_table.setItem(row, 1, QTableWidgetItem(
                ", ".join(f"{k}:{v:.3g}" for k, v in it[1].items())))
        self._loading = False

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
            self._load_mix()

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
