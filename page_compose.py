"""page_compose.py — the composition-model half of the Model tab.

Load pure references + known-ratio mixtures, pick a composition method (MLP / PLS /
RF / 1D-CNN — the best one is data-dependent), train ONCE, and save a portable .dlm
that the Recovery report and Real-data prediction just APPLY (no retraining downstream).
This is Step 2 of the workflow: Pure → train → recovery → calibration → sample predict.
"""
from __future__ import annotations

import os
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox,
    QCheckBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QProgressBar,
)

from ui_common import *
from real_data import PEST_DEFAULT
from validate import parse_mixture_label


class TrainComposeWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from dl_model import train_model, apply_model
            from real_data import load_map
            params = dict(self.params); items = params.pop("items")
            model = train_model(items=items, progress=self.progress.emit, **params)
            errs = []                                        # train-set composition recovery
            for k, it in enumerate(items):
                self.progress.emit(f"scoring {k + 1}/{len(items)}")
                wn, cube, _m, _c = load_map(it[0])
                comp = apply_model(model, wn, cube)["composition"]
                tr = it[1]; s = sum(float(tr.get(n, 0)) for n in comp)
                if s <= 0:
                    continue
                errs.append(0.5 * sum(abs(comp[n] - float(tr.get(n, 0)) / s) for n in comp))
            self.done.emit((model, float(np.mean(errs)) if errs else float("nan")))
        except Exception:
            self.fail.emit(traceback.format_exc())


class ComposePanel(QWidget):
    METHODS = [("MLP (deep)", "mlp"), ("PLS", "pls"),
               ("Random Forest", "rf"), ("1D-CNN", "cnn")]

    def __init__(self):
        super().__init__()
        self.data_dir = PEST_DEFAULT
        self.calib_path = None
        self._files = []
        self._model = None
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(12)

        sub = QLabel("Load your pure references and known-ratio mixtures, pick a method "
                     "(the best one is data-dependent), and train once. Save the model so "
                     "Recovery and Real-data just apply it — no retraining.")
        sub.setObjectName("sub"); sub.setWordWrap(True); root.addWidget(sub)

        # ---- files: pure refs + mixtures ----
        frow = QHBoxLayout(); frow.setSpacing(8)
        pure_b = QPushButton("Pure refs…"); pure_b.setObjectName("ghost")
        pure_b.setToolTip("folder of pure reference maps (one class per substance)")
        pure_b.clicked.connect(self._browse_pure)
        self.ref_lbl = QLabel(self._short(self.data_dir)); self.ref_lbl.setObjectName("field")
        add_b = QPushButton("Add mixtures…"); add_b.setObjectName("ghost")
        add_b.clicked.connect(self._add)
        cal_b = QPushButton("Calibration… (opt)"); cal_b.setObjectName("ghost")
        cal_b.setToolTip("optional dilution-series CSV — enables the MLP physics pretrain")
        cal_b.clicked.connect(self._browse_calib)
        self.cal_lbl = QLabel("no calib"); self.cal_lbl.setObjectName("field")
        clr_b = QPushButton("Clear"); clr_b.setObjectName("ghost"); clr_b.clicked.connect(self._clear)
        frow.addWidget(pure_b); frow.addWidget(self.ref_lbl, 1)
        frow.addWidget(cal_b); frow.addWidget(self.cal_lbl)
        frow.addWidget(add_b); frow.addWidget(clr_b)
        root.addLayout(frow)

        # ---- method + per-method sub-parameters (auto-shown) ----
        mrow = QHBoxLayout(); mrow.setSpacing(8)
        mlbl = QLabel("method:"); mlbl.setObjectName("field")
        self.cmb = QComboBox(); self.cmb.setObjectName("field")
        for text, data in self.METHODS:
            self.cmb.addItem(text, data)
        self.cmb.currentIndexChanged.connect(self._update_params)
        self.sp_ep = QSpinBox(); self.sp_ep.setRange(20, 3000); self.sp_ep.setSingleStep(50)
        self.sp_ep.setValue(350); self.sp_ep.setPrefix("epochs "); self.sp_ep.setObjectName("field")
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 999); self.sp_seed.setPrefix("seed ")
        self.sp_seed.setObjectName("field")
        self.chk_pre = QCheckBox("physics pretrain"); self.chk_pre.setChecked(True)
        self.chk_pre.setObjectName("field")
        self.sp_nc = QSpinBox(); self.sp_nc.setRange(1, 20); self.sp_nc.setValue(8)
        self.sp_nc.setPrefix("components "); self.sp_nc.setObjectName("field")
        self.sp_nt = QSpinBox(); self.sp_nt.setRange(20, 1000); self.sp_nt.setSingleStep(20)
        self.sp_nt.setValue(300); self.sp_nt.setPrefix("trees "); self.sp_nt.setObjectName("field")
        mrow.addWidget(mlbl); mrow.addWidget(self.cmb)
        for w in (self.sp_ep, self.sp_seed, self.chk_pre, self.sp_nc, self.sp_nt):
            mrow.addWidget(w)
        mrow.addStretch(1)
        self.train_b = QPushButton("Train"); self.train_b.setObjectName("primary")
        self.train_b.clicked.connect(self._train)
        self.save_b = QPushButton("Save model…"); self.save_b.setObjectName("ghost")
        self.save_b.setEnabled(False); self.save_b.clicked.connect(self._save)
        mrow.addWidget(self.save_b); mrow.addWidget(self.train_b)
        root.addLayout(mrow)
        self._update_params()

        self.pbar = QProgressBar(); self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(6); self.pbar.setVisible(False); root.addWidget(self.pbar)

        kp = QHBoxLayout(); kp.setSpacing(12)
        self.k_err = Kpi("train composition error"); self.k_n = Kpi("mixtures")
        self.k_m = Kpi("method")
        for k in (self.k_err, self.k_n, self.k_m):
            kp.addWidget(k)
        root.addLayout(kp)

        # editable table: file | true ratio
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["mixture file", "true ratio  (e.g. DQ:1, TBZ:3)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        root.addWidget(self.table, 1)

        self.status = QLabel(""); self.status.setObjectName("sub"); root.addWidget(self.status)

    # ---- data dir shared with the Model tab / Samples ----
    def set_data_dir(self, path):
        self.data_dir = path; self.ref_lbl.setText(self._short(path))

    def _short(self, p):
        tail = "…" + p[-38:] if len(p) > 38 else p
        return f"pure refs: {tail}"

    def _browse_pure(self):
        d = QFileDialog.getExistingDirectory(self, "Pure references folder", self.data_dir)
        if d:
            self.set_data_dir(d)

    def _browse_calib(self):
        p, _ = QFileDialog.getOpenFileName(self, "Calibration spectra CSV", "", "CSV (*.csv)")
        if p:
            self.calib_path = p; self.cal_lbl.setText("calib: " + os.path.basename(p))

    def _ref_names(self):
        try:
            from dataset import discover_dataset, is_blank
            return [c for c, _m in discover_dataset(self.data_dir) if not is_blank(c)]
        except Exception:
            return []

    def _add(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Known-ratio mixture maps", "",
                                                "maps (*.csv *.txt);;all files (*)")
        if not paths:
            return
        refs = self._ref_names()
        for p in paths:
            if p in self._files:
                continue
            self._files.append(p)
            row = self.table.rowCount(); self.table.insertRow(row)
            it0 = QTableWidgetItem(os.path.basename(p))
            it0.setFlags(it0.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, it0)
            g = parse_mixture_label(os.path.splitext(os.path.basename(p))[0], refs)
            txt = ", ".join(f"{k}:{v:.3g}" for k, v in g.items()) if g else ""
            self.table.setItem(row, 1, QTableWidgetItem(txt))
        self.status.setText(f"{len(self._files)} mixtures — edit any ratio, then Train")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _clear(self):
        self._files = []; self.table.setRowCount(0); self._model = None
        self.save_b.setEnabled(False)
        self.status.setText("cleared"); self.status.setStyleSheet(f"color:{MUTE};")

    def _items(self):
        items = []
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 1)
            ratio = {}
            for tok in (cell.text() if cell else "").replace(";", ",").split(","):
                if ":" not in tok:
                    continue
                k, v = tok.split(":", 1)
                try:
                    ratio[k.strip()] = float(v)
                except ValueError:
                    pass
            if len(ratio) >= 2:
                items.append((self._files[row], ratio))
        return items

    def _update_params(self):
        m = self.cmb.currentData()
        self.sp_ep.setVisible(m in ("mlp", "cnn"))
        self.sp_seed.setVisible(m in ("mlp", "cnn", "rf"))
        self.chk_pre.setVisible(m == "mlp")
        self.sp_nc.setVisible(m == "pls")
        self.sp_nt.setVisible(m == "rf")

    def _opts(self):
        from dataset import load_preprocess
        cfg = load_preprocess(self.data_dir)
        return dict(data_dir=self.data_dir, calib_path=self.calib_path,
                    baseline=cfg["baseline"], trim=cfg["trim"],
                    method=self.cmb.currentData(), epochs=self.sp_ep.value(),
                    seed=self.sp_seed.value(), use_pretrain=self.chk_pre.isChecked(),
                    n_components=self.sp_nc.value(), n_trees=self.sp_nt.value())

    def _train(self):
        items = self._items()
        if len(items) < 3:
            self.status.setText("add ≥3 mixtures with true ratios to train")
            self.status.setStyleSheet(f"color:{RED};"); return
        params = self._opts(); params["items"] = items
        self.train_b.setEnabled(False); self.train_b.setText("Training…")
        self.save_b.setEnabled(False)
        self.pbar.setRange(0, 0); self.pbar.setVisible(True)
        self.status.setText("● training…"); self.status.setStyleSheet(f"color:{MUTE};")
        self._thread = QThread(); self._worker = TrainComposeWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda m: self.status.setText("● " + m))
        self._worker.done.connect(self._done)
        self._worker.fail.connect(self._fail)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _done(self, res):
        model, err = res
        self._model = model
        self.train_b.setEnabled(True); self.train_b.setText("Train")
        self.save_b.setEnabled(True); self.pbar.setVisible(False)
        self.k_err.set(f"{err:.0%}" if err == err else "—")
        self.k_n.set(str(model.get("n_train", 0)))
        self.k_m.set(model.get("method", "mlp").upper())
        self.status.setText("done — trained; Save model to use it in Recovery / Real data")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _fail(self, tb):
        import sys
        self.train_b.setEnabled(True); self.train_b.setText("Train")
        self.pbar.setVisible(False)
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _save(self):
        if self._model is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save composition model", "composition.dlm",
                                           "DL model (*.dlm)")
        if not p:
            return
        from dl_model import save_model
        save_model(self._model, p)
        self.status.setText("saved → " + os.path.basename(p))
        self.status.setStyleSheet(f"color:{MUTE};")
