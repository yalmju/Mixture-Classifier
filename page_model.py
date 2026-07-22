"""page_model.py — Model tab: train a classifier on the reference maps."""
from __future__ import annotations

import re
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSpinBox, QComboBox, QCheckBox, QFileDialog, QProgressBar,
)

from ui_common import *
from model_training import train_model, TrainResult
from real_data import PEST_DEFAULT
from io_utils import write_csv


class TrainWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.done.emit(train_model(progress=self.progress.emit, **self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


# --------------------------------------------------------------------------
# Model page
# --------------------------------------------------------------------------
class ModelPage(QWidget):
    ALGOS = [("RandomForest", "rf"), ("ResNet1D (torch)", "resnet"),
             ("SVM (RBF)", "svm"), ("k-NN", "knn"),
             ("Logistic Reg.", "logreg"), ("Gradient Boosting", "gbm")]

    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self.pest_dir = PEST_DEFAULT
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(14)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Model training"); h1.setObjectName("h1")
        sub = QLabel("Train on a set of reference SERS maps — one pure substance "
                     "per map. Spatial split: the left half of each map trains, the "
                     "right half tests. Learning curve · confusion matrix · "
                     "per-class F1 · PCA.  (example data: DQ / THI / TBZ / BLK)")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        # ---- controls: row 1 = model + data + actions ----
        ctl = QHBoxLayout(); ctl.setSpacing(10)
        ctl.addLayout(self._combo_col("algorithm", "cmb", self.ALGOS))
        self.sp_ep = self._spin(QSpinBox(), 2, 100, 25, "epochs (ResNet)")
        self.sp_tr = self._spin(QSpinBox(), 60, 600, 300, "trees (RF)", step=20)
        self.sp_seed = self._spin(QSpinBox(), 0, 999, 0, "seed")
        for w in (self.sp_ep, self.sp_tr, self.sp_seed):
            ctl.addLayout(w)
        self.src = QLabel(self._short(self.pest_dir)); self.src.setObjectName("field")
        ctl.addWidget(self.src, 1)
        browse = QPushButton("Training data…"); browse.setObjectName("ghost")
        browse.clicked.connect(self._browse)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Train + evaluate"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._train)
        ctl.addWidget(browse); ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        # ---- controls: row 2 = features + split (the experimentation knobs) ----
        ctl2 = QHBoxLayout(); ctl2.setSpacing(10)
        bcol = QVBoxLayout(); bcol.setSpacing(2)
        lb = QLabel("baseline"); lb.setObjectName("field")
        self.chk_base = QCheckBox("ALS on"); self.chk_base.setChecked(True)
        bcol.addWidget(lb); bcol.addWidget(self.chk_base)
        ctl2.addLayout(bcol)
        ctl2.addLayout(self._combo_col("derivative", "cmb_deriv",
                                       [("none", 0), ("1st", 1), ("2nd", 2)]))
        ctl2.addLayout(self._combo_col("normalize", "cmb_norm",
                                       [("L2", "l2"), ("SNV", "snv"), ("none", "none")]))
        self.sp_lo = self._spin(QSpinBox(), 0, 4000, 0, "trim lo cm⁻¹", step=50)
        self.sp_hi = self._spin(QSpinBox(), 0, 4000, 4000, "trim hi cm⁻¹", step=50)
        for w in (self.sp_lo, self.sp_hi):
            ctl2.addLayout(w)
        ctl2.addLayout(self._combo_col("split", "cmb_split",
                                       [("spatial (honest)", "spatial"),
                                        ("random (leaky)", "random"),
                                        ("batch (leave-1-out)", "batch"),
                                        ("batch-CV (mean±SD)", "batch-cv"),
                                        ("manual (Samples)", "manual")]))
        self.sp_test = self._spin(QSpinBox(), 5, 90, 50, "test %", step=5)
        ctl2.addLayout(self.sp_test)
        ctl2.addStretch(1)
        root.addLayout(ctl2)

        # progress bar — visible (busy) only while training, so it never reads as frozen
        self.pbar = QProgressBar(); self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(6); self.pbar.setVisible(False)
        root.addWidget(self.pbar)

        # KPI row
        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_acc = Kpi("test accuracy"); self.k_f1 = Kpi("macro F1")
        self.k_tr = Kpi("train pixels"); self.k_te = Kpi("test pixels")
        for k in (self.k_acc, self.k_f1, self.k_tr, self.k_te):
            kpis.addWidget(k)
        root.addLayout(kpis)

        # plot grid: 2x2 + a full-width discriminative-band row
        grid = QGridLayout(); grid.setSpacing(12)
        self.c_curve = Canvas(); self.c_cm = Canvas()
        self.c_pca = Canvas(); self.c_bar = Canvas(); self.c_bands = Canvas()
        for (cv, title, r, c) in [
            (self.c_curve, "Learning curve", 0, 0),
            (self.c_cm, "Confusion matrix (held-out test)", 0, 1),
            (self.c_pca, "PCA — real per-pixel spectra by class", 1, 0),
            (self.c_bar, "Per-class precision / recall / F1", 1, 1),
        ]:
            card, lay = _card(title)
            lay.addWidget(cv)
            grid.addWidget(card, r, c)
        bcard, blay = _card("Discriminative bands — ANOVA F per wavenumber "
                            "(which bands separate the substances)")
        blay.addWidget(self.c_bands); grid.addWidget(bcard, 2, 0, 1, 2)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1); grid.setRowStretch(2, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)

        for cv, msg in [(self.c_curve, "Train to watch the learning curve"),
                        (self.c_cm, "Train to compute confusion matrix"),
                        (self.c_pca, "Train to compute PCA"),
                        (self.c_bar, "Train to compute per-class F1"),
                        (self.c_bands, "Train to rank discriminative bands (ANOVA F)")]:
            cv.placeholder(msg)

    def _short(self, p):
        tail = "…" + p[-40:] if len(p) > 40 else p
        return f"data: {tail}"

    def _spin(self, spin, lo, hi, val, label, step=1):
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        spin.setSingleStep(step)
        spin.setRange(lo, hi); spin.setValue(val)
        col.addWidget(lb); col.addWidget(spin)
        return col

    def _combo_col(self, label, attr, items):
        """A labelled combo box; stores the widget on self.<attr>."""
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        cb = QComboBox()
        for text, data in items:
            cb.addItem(text, data)
        setattr(self, attr, cb)
        col.addWidget(lb); col.addWidget(cb)
        return col

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Training data — folder with your reference maps "
            "(the data root or its Reference/ subfolder)",
            self.pest_dir)
        if d:
            self.pest_dir = d; self.src.setText(self._short(d))

    def _export(self):
        if self._res is None:
            self.src.setText("run first, then export"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res
        rows = [[nm, f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}", v[3]]
                for nm, v in r.per_component.items()]
        rows.append(["accuracy", f"{r.acc:.4f}", "", f"{r.macro_f1:.4f}", ""])
        write_csv(os.path.join(d, "model_metrics.csv"),
                  ["class", "precision", "recall", "f1", "support"], rows)
        write_csv(os.path.join(d, "model_confusion.csv"),
                  [""] + list(r.classes),
                  [[r.classes[i]] + list(map(int, r.confusion[i]))
                   for i in range(len(r.classes))])
        write_csv(os.path.join(d, "model_learning_curve.csv"),
                  [r.curve_xlabel, r.curve_label],
                  [[f"{x:g}", f"{y:.6f}"] for x, y in zip(r.curve_x, r.curve_y)])
        n = _save_figs([("model_learning_curve", self.c_curve),
                        ("model_confusion", self.c_cm),
                        ("model_pca", self.c_pca), ("model_prf", self.c_bar),
                        ("model_bands", self.c_bands)], d)
        self.src.setText(f"exported CSV + {n} PNG → {os.path.basename(d)}")

    # ---- training ----
    def _train(self):
        lo = self.sp_lo.itemAt(1).widget().value()
        hi = self.sp_hi.itemAt(1).widget().value()
        trim = (lo, hi) if (hi > lo and (lo > 0 or hi < 4000)) else None
        params = dict(pest_dir=self.pest_dir, backend=self.cmb.currentData(),
                      epochs=self.sp_ep.itemAt(1).widget().value(),
                      n_estimators=self.sp_tr.itemAt(1).widget().value(),
                      seed=self.sp_seed.itemAt(1).widget().value(),
                      baseline=self.chk_base.isChecked(), trim=trim,
                      deriv=self.cmb_deriv.currentData(),
                      norm=self.cmb_norm.currentData(),
                      split=self.cmb_split.currentData(),
                      test_frac=self.sp_test.itemAt(1).widget().value() / 100.0)
        self.btn.setEnabled(False); self.btn.setText("Training…")
        self.c_curve.placeholder("Training…")
        self.pbar.setVisible(True); self.pbar.setRange(0, 0)   # busy until first step
        self._thread = QThread()
        self._worker = TrainWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _progress(self, msg):
        # live status so a long run reads as working, not frozen
        self.btn.setText("Training…  " + msg.split("  ")[0])
        self.c_curve.placeholder("● " + msg)
        m = re.search(r"(\d+)\s*/\s*(\d+)", msg)          # "epoch 12/25" -> determinate
        if m:
            self.pbar.setRange(0, int(m.group(2))); self.pbar.setValue(int(m.group(1)))
        else:
            self.pbar.setRange(0, 0)                       # loading / finalising -> busy

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Train + evaluate")
        self.pbar.setVisible(False)
        first = tb.strip().splitlines()[-1][:90]
        self.c_curve.placeholder("Training failed — " + first)
        print(tb, file=sys.stderr)

    def _apply(self, res: TrainResult):
        self._res = res
        self.btn.setEnabled(True); self.btn.setText("Train + evaluate")
        self.pbar.setVisible(False)
        # random split is leaky; batch-CV reports cross-fold mean ± SD
        leaky = getattr(res, "split", "spatial") == "random"
        if getattr(res, "split", "") == "batch-cv" and getattr(res, "acc_std", 0):
            self.k_acc.set(f"{res.acc:.0%} ±{res.acc_std:.0%}", TEAL)
        else:
            self.k_acc.set(f"{res.acc:.0%}" + ("  ⚠leaky" if leaky else ""),
                           CORAL if leaky else TEAL)
        self.k_f1.set(f"{res.macro_f1:.3f}", BLUE)
        self.k_tr.set(f"{res.n_train:,}", AMBER)
        self.k_te.set(f"{res.n_test:,}", PURPLE)
        self._plot_curve(res); self._plot_cm(res)
        self._plot_pca(res); self._plot_bar(res); self._plot_bands(res)

    # ---- plots ----
    def _plot_curve(self, res):
        ax = self.c_curve.new_ax()
        col = TEAL if res.backend == "resnet" else BLUE
        ax.plot(res.curve_x, res.curve_y, marker="o", ms=4, lw=1.4, color=col)
        ax.set_xlabel(res.curve_xlabel); ax.set_ylabel(res.curve_label)
        ax.set_ylim(bottom=0)
        self.c_curve.fig.tight_layout(); self.c_curve.draw_idle()

    def _plot_cm(self, res):
        ax = self.c_cm.new_ax()
        cm = res.confusion; names = res.classes
        ax.imshow(cm, cmap=CM_CMAP, aspect="auto", vmin=0)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
        ax.grid(which="minor", color=PANEL, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        thr = cm.max() / 2 if cm.max() else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="#ffffff" if cm[i, j] > thr else INK, fontsize=8)
        self.c_cm.fig.tight_layout(); self.c_cm.draw_idle()

    def _plot_pca(self, res):
        ax = self.c_pca.new_ax()
        for i, nm in enumerate(res.classes):
            m = res.pca_lab == i
            if m.any():
                ax.scatter(res.pca_emb[m, 0], res.pca_emb[m, 1], s=12,
                           color=SERIES[i % len(SERIES)], alpha=0.6,
                           edgecolors="none", label=nm)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="best", framealpha=0.0, labelcolor=MUTE)
        self.c_pca.fig.tight_layout(); self.c_pca.draw_idle()

    def _plot_bar(self, res):
        ax = self.c_bar.new_ax()
        names = res.classes
        x = np.arange(len(names)); w = 0.26
        P = [res.per_component[n][0] for n in names]
        R = [res.per_component[n][1] for n in names]
        F = [res.per_component[n][2] for n in names]
        ax.bar(x - w, P, w, color=BLUE, label="precision")
        ax.bar(x, R, w, color=AMBER, label="recall")
        ax.bar(x + w, F, w, color=TEAL, label="F1")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_ylabel("score")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=3)
        self.c_bar.fig.tight_layout(); self.c_bar.draw_idle()

    def _plot_bands(self, res):
        ax = self.c_bands.new_ax()
        f = getattr(res, "band_f", None)
        wn = res.wn
        if f is None or wn is None or len(f) != len(wn):
            self.c_bands.placeholder("no band statistics"); return
        ax.plot(wn, f, lw=1.0, color=PURPLE)
        ax.fill_between(wn, f, color=PURPLE, alpha=0.15)
        # label the strongest discriminative bands with their cm⁻¹
        k = min(6, len(f))
        top = np.argsort(f)[-k:]
        for idx in top:
            ax.annotate(f"{wn[idx]:.0f}", (wn[idx], f[idx]), fontsize=7, color=INK,
                        ha="center", va="bottom",
                        xytext=(0, 2), textcoords="offset points")
            ax.plot([wn[idx]], [f[idx]], "o", ms=3, color=CORAL)
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_ylabel("ANOVA F")
        ax.margins(y=0.15)
        self.c_bands.fig.tight_layout(); self.c_bands.draw_idle()
