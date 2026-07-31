"""page_compose.py — the composition-model half of the Model tab.

Pick a composition method (MLP / PLS / RF / 1D-CNN — the best one is data-dependent),
train ONCE on the known-ratio mixtures you prepared in Samples (Step 1), and save a
portable .dlm that Recovery and Real-data just APPLY (no retraining downstream).
The mixture list is shared from Samples via MIXTURE_BUS — not re-entered here.
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
from dataset import load_mixture_list


class TrainComposeWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from dl_model import train_model
            from composition import SUBSTANCES
            params = dict(self.params); items = params.pop("items")
            model = train_model(items=items, progress=self.progress.emit, **params)
            # train-set recovery is computed IN-MEMORY inside train_model (no map reload,
            # which was holding the GIL and freezing the GUI). Reorder to SUBSTANCES.
            subs = model["subs"]; sidx = {s: j for j, s in enumerate(subs)}
            te = model.get("train_eval", {}); tvs = te.get("true", []); pvs = te.get("pred", [])
            errs = []; rows = []
            for i in range(len(tvs)):
                tv = [float(tvs[i][sidx[s]]) if s in sidx else 0.0 for s in SUBSTANCES]
                pv = [float(pvs[i][sidx[s]]) if s in sidx else 0.0 for s in SUBSTANCES]
                errs.append(0.5 * sum(abs(pv[j] - tv[j]) for j in range(len(SUBSTANCES))))
                rows.append((f"mix {i + 1}", tv, pv))
            self.done.emit((model, float(np.mean(errs)) if errs else float("nan"), rows))
        except Exception:
            self.fail.emit(traceback.format_exc())


class ComposePanel(QWidget):
    METHODS = [("MLP (deep)", "mlp"), ("PLS", "pls"),
               ("Random Forest", "rf"), ("1D-CNN", "cnn")]

    def __init__(self):
        super().__init__()
        self.data_dir = PEST_DEFAULT
        self.calib_path = None
        self._items_cache = []
        self._model = None
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(12)

        sub = QLabel("Pick a method (the best one is data-dependent) and train once on the "
                     "mixtures prepared in Samples. Save the model so Recovery and Real-data "
                     "just apply it — no retraining.")
        sub.setObjectName("sub"); sub.setWordWrap(True); root.addWidget(sub)

        # ---- files: pure refs + (mixtures come from Samples) ----
        frow = QHBoxLayout(); frow.setSpacing(8)
        pure_b = QPushButton("Pure refs…"); pure_b.setObjectName("ghost")
        pure_b.setToolTip("folder of pure reference maps (one class per substance)")
        pure_b.clicked.connect(self._browse_pure)
        self.ref_lbl = QLabel(self._short(self.data_dir)); self.ref_lbl.setObjectName("field")
        self.mix_lbl = QLabel("mixtures: from Samples"); self.mix_lbl.setObjectName("field")
        reload_b = QPushButton("Reload"); reload_b.setObjectName("ghost")
        reload_b.setToolTip("re-read the known-ratio mixtures prepared in the Samples tab")
        reload_b.clicked.connect(self._load_from_samples)
        frow.addWidget(pure_b); frow.addWidget(self.ref_lbl, 1)
        frow.addWidget(self.mix_lbl); frow.addWidget(reload_b)
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
        self.cancel_b = QPushButton("Cancel"); self.cancel_b.setObjectName("ghost")
        self.cancel_b.setVisible(False); self.cancel_b.clicked.connect(self._cancel)
        self.save_b = QPushButton("Save model…"); self.save_b.setObjectName("ghost")
        self.save_b.setEnabled(False); self.save_b.clicked.connect(self._save)
        mrow.addWidget(self.save_b); mrow.addWidget(self.cancel_b); mrow.addWidget(self.train_b)
        root.addLayout(mrow)
        self._update_params()

        self.pbar = QProgressBar(); self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(6); self.pbar.setVisible(False); root.addWidget(self.pbar)

        # read-only view of the shared mixtures (managed in Samples) — collapsible
        self.mix_tgl = QPushButton(); self.mix_tgl.setObjectName("ghost")
        self.mix_tgl.setCheckable(True); self.mix_tgl.setChecked(True)
        self.mix_tgl.setStyleSheet("text-align:left; padding:4px 8px;")
        self.mix_tgl.toggled.connect(self._toggle_table)
        root.addWidget(self.mix_tgl)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["mixture file (from Samples)", "true ratio"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(140)
        root.addWidget(self.table)
        self._toggle_table(True)

        card, lay = _card("Train-set recovery — true (○) vs predicted (●, colour = accuracy)")
        self.c_tri = Canvas(); self.c_tri.setMinimumHeight(300)
        self.c_tri.placeholder("Train to see composition recovery on the simplex")
        lay.addWidget(self.c_tri); root.addWidget(card, 1)

        self.status = QLabel(""); self.status.setObjectName("sub"); root.addWidget(self.status)

        MIXTURE_BUS.changed.connect(self._load_from_samples)   # Samples edits → refresh here
        self._load_from_samples()

    # ---- data dir shared with the Model tab / Samples ----
    def set_data_dir(self, path):
        self.data_dir = path; self.ref_lbl.setText(self._short(path))
        self._load_from_samples()

    def _short(self, p):
        tail = "…" + p[-38:] if len(p) > 38 else p
        return f"pure refs: {tail}"

    def _browse_pure(self):
        d = QFileDialog.getExistingDirectory(self, "Pure references folder", self.data_dir)
        if d:
            self.set_data_dir(d)

    def _load_from_samples(self):
        """Pull the known-ratio mixtures prepared in Samples (Step 1)."""
        self._items_cache = load_mixture_list(self.data_dir)
        self.table.setRowCount(0)
        for it in self._items_cache:
            row = self.table.rowCount(); self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(it[0])))
            self.table.setItem(row, 1, QTableWidgetItem(
                ", ".join(f"{k}:{v:.3g}" for k, v in it[1].items())))
        n = len(self._items_cache)
        has_uM = any(len(it) > 2 and it[2] for it in self._items_cache)
        self.mix_lbl.setText(f"mixtures: {n} from Samples" + (" · µM" if has_uM else ""))
        if hasattr(self, "mix_tgl"):
            self._toggle_table(self.mix_tgl.isChecked())
        if not self.status.text().startswith("●"):
            self.status.setText(f"{n} mixtures from Samples" if n
                                else "no mixtures yet — add them in the Samples tab")
            self.status.setStyleSheet(f"color:{MUTE};")

    def _toggle_table(self, on):
        self.table.setVisible(on)
        n = self.table.rowCount()
        self.mix_tgl.setText(("▾  " if on else "▸  ") + f"Mixtures from Samples ({n})")

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
        items = self._items_cache
        if len(items) < 3:
            self.status.setText("prepare ≥3 known-ratio mixtures in the Samples tab first")
            self.status.setStyleSheet(f"color:{RED};"); return
        params = self._opts(); params["items"] = items
        self.train_b.setEnabled(False); self.train_b.setText("Training…")
        self.save_b.setEnabled(False); self._cancelled = False; self.cancel_b.setVisible(True)
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

    def _cancel(self):
        th = getattr(self, "_thread", None)
        self._cancelled = True
        if th is not None and th.isRunning():
            th.quit()
            if not th.wait(300):
                th.terminate(); th.wait()
        self.cancel_b.setVisible(False); self.pbar.setVisible(False)
        self.train_b.setEnabled(True); self.train_b.setText("Train")
        self.status.setText("cancelled"); self.status.setStyleSheet(f"color:{MUTE};")

    def _done(self, res):
        if getattr(self, "_cancelled", False):
            return
        model, err, rows = res
        self._model = model
        self.train_b.setEnabled(True); self.train_b.setText("Train")
        self.save_b.setEnabled(True); self.pbar.setVisible(False); self.cancel_b.setVisible(False)
        self._plot_triangle(rows)
        MODEL_BUS.set(model, origin=f"Model tab · {model.get('method', 'mlp').upper()}")
        m = model.get("method", "mlp").upper() + ("  +µM" if model.get("has_uM") else "")
        errtxt = f"{err:.0%}" if err == err else "—"
        self.status.setText(f"done — {m} · {model.get('n_train', 0)} mixtures · train error {errtxt}. "
                            "Recovery & Real-data now use this model; Save to keep it.")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _fail(self, tb):
        import sys
        self.train_b.setEnabled(True); self.train_b.setText("Train")
        self.pbar.setVisible(False); self.cancel_b.setVisible(False)
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _plot_triangle(self, rows):
        """Ternary simplex: true (open) → predicted (filled, green=accurate) per mixture."""
        from composition import bary
        from matplotlib import cm
        fig = self.c_tri.fig; fig.clear(); ax = fig.add_subplot(111)
        A, B, C = bary([1, 0, 0]), bary([0, 0, 1]), bary([0, 1, 0])   # DQ · THI · TBZ corners
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color=INK, lw=1.0, zorder=1)
        for f, s, col in [([1, 0, 0], "DQ", substance_color("DQ", 0)),
                          ([0, 0, 1], "THI", substance_color("THI", 2)),
                          ([0, 1, 0], "TBZ", substance_color("TBZ", 1))]:
            p = bary(f); ax.text(p[0], p[1], s, ha="center", va="center", fontsize=10,
                                 fontweight="bold", color=col)
        cmap = cm.get_cmap("RdYlGn")
        for _name, tv, pv in rows:
            p0 = bary(tv); p1 = bary(pv); e = 0.5 * sum(abs(pv[j] - tv[j]) for j in range(len(tv)))
            if (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 > 1e-5:
                ax.annotate("", xy=p1, xytext=p0,
                            arrowprops=dict(arrowstyle="-|>", color="#b6bcc4", lw=0.8))
            ax.scatter(*p0, s=26, facecolors="none", edgecolors=MUTE, linewidths=1.0, zorder=3)
            ax.scatter(*p1, s=42, color=cmap(1 - min(e / 0.6, 1.0)), edgecolors="white",
                       linewidths=0.5, zorder=4)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.1, 1.05)
        fig.tight_layout(); self.c_tri.draw_idle()

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
