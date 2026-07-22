"""UNMIXR — SERS mixture analysis suite (PyQt6).

A dark, instrument-style desktop app. Not a sidebar dashboard — a top command
bar with pill navigation over a stacked content area:

One window, native PyQt6 tabs (the earlier customtkinter tools are retired —
their jobs are covered natively here):

    Samples       group raw maps into substance classes (batches of one class);
                  saves samples.csv that Model / Predict / Real data read
    Model         train a classifier on a set of reference SERS maps
                  (RandomForest or ResNet1D): learning curve · confusion · F1 · PCA
    Predict       load one unknown sample → its component ratio (per-pixel NNLS)
    Quantify      ratio → M calibration + Langmuir competition
    Real data     map analysis: single-component / mixture / composition / calib

    python unmixr.py

All pages are PyQt6 with embedded matplotlib. Rename the app by editing APP_NAME.
"""
from __future__ import annotations

import os
import re
import sys
import traceback

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QHBoxLayout, QVBoxLayout, QGridLayout, QStackedWidget, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QSizePolicy, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
)
from matplotlib.patches import Patch, Rectangle

from model_training import train_model, TrainResult
from predict import predict_sample
from calibration import build_synthetic_lab, calibrate, quantify
from real_data import compute_real, PEST_DEFAULT
from dataset import (discover_references, base_and_batch, load_manifest,
                     save_manifest)
from io_utils import load_calibration_csv, write_csv


def _save_figs(named_canvases, folder):
    """Save each (name, Canvas) to folder/<name>.png. Returns count."""
    n = 0
    for name, cv in named_canvases:
        cv.fig.savefig(os.path.join(folder, name + ".png"), dpi=150,
                       facecolor=CARD, bbox_inches="tight")
        n += 1
    return n

APP_NAME = "UNMIXR"
VERSION = "1.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# App icon: drop a PNG at assets/icon.png and it is picked up automatically.
ICON_PATH = os.path.join(BASE_DIR, "assets", "icon.png")

# ---- light palette -------------------------------------------------------
PAGE   = "#fafbfc"
PANEL  = "#ffffff"
CARD   = "#ffffff"
LINE   = "#e3e8ee"
INK    = "#1c2430"
MUTE   = "#5b6673"
FAINT  = "#98a1ac"
TEAL   = "#0f9d6b"
BLUE   = "#1a73e8"
AMBER  = "#c98a15"
CORAL  = "#d8542a"
PURPLE = "#6b5fd6"
PINK   = "#c85a8f"
GREEN  = "#4a9e2a"
RED    = "#d64545"   # detection "miss" (outcome grid)
TNGRAY = "#eef1f4"   # detection "correct absent"
# discrete label palette for scatter / bars
SERIES = [TEAL, BLUE, AMBER, CORAL, PURPLE, PINK, GREEN, "#546170"]
# single-hue teal ramp for the confusion matrix (matches the light theme):
# near-white for empty cells → teal for the strong diagonal
CM_CMAP = LinearSegmentedColormap.from_list(
    "unmixr_teal", ["#f1f7f4", "#a9ddc7", "#3fb488", TEAL, "#0a6b49"])

QSS = f"""
QMainWindow, QWidget {{ background: {PAGE}; color: {INK};
    font-family: 'Segoe UI', Arial; font-size: 14px; }}
#topbar {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
#wordmark {{ font-size: 18px; font-weight: 600; color: {INK}; }}
#logo {{ background: {TEAL}; color: #ffffff; font-size: 15px; font-weight: 700;
    border-radius: 8px; }}
#status {{ color: {FAINT}; font-family: 'Consolas', monospace; font-size: 12px; }}
QPushButton#nav {{ background: transparent; color: {MUTE}; border: none;
    padding: 8px 18px; border-radius: 8px; font-size: 14px; }}
QPushButton#nav:hover {{ background: {CARD}; color: {INK}; }}
QPushButton#nav:checked {{ background: {PAGE}; color: {INK}; font-weight: 600;
    border: 1px solid {LINE}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#kpi {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; }}
#kpiLabel {{ color: {MUTE}; font-size: 12px; }}
#kpiValue {{ color: {INK}; font-size: 26px; font-weight: 600; }}
#cardTitle {{ color: {MUTE}; font-size: 12px; font-weight: 600; }}
#h1 {{ font-size: 22px; font-weight: 600; color: {INK}; }}
#sub {{ color: {MUTE}; font-size: 13px; }}
QPushButton#primary {{ background: {TEAL}; color: #ffffff; border: none;
    border-radius: 8px; padding: 9px 20px; font-size: 14px; font-weight: 600; }}
QPushButton#primary:hover {{ background: #0c855a; }}
QPushButton#primary:disabled {{ background: {LINE}; color: {FAINT}; }}
QPushButton#ghost {{ background: transparent; color: {INK}; border: 1px solid {LINE};
    border-radius: 8px; padding: 9px 20px; font-size: 14px; }}
QPushButton#ghost:hover {{ border-color: {TEAL}; }}
QSpinBox, QDoubleSpinBox {{ background: {PANEL}; color: {INK};
    border: 1px solid {LINE}; border-radius: 6px; padding: 4px 6px; min-width: 64px; }}
QComboBox {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
    border-radius: 6px; padding: 4px 8px; min-width: 128px; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; color: {INK};
    border: 1px solid {LINE}; selection-background-color: {PAGE};
    selection-color: {INK}; outline: none; }}
QLabel#field {{ color: {MUTE}; font-size: 13px; }}
QProgressBar {{ background: {PAGE}; border: 1px solid {LINE}; border-radius: 3px; }}
QProgressBar::chunk {{ background: {TEAL}; border-radius: 3px; }}
"""


# --------------------------------------------------------------------------
# dark matplotlib canvas
# --------------------------------------------------------------------------
class Canvas(FigureCanvasQTAgg):
    def __init__(self, w=4.0, h=3.0):
        self.fig = Figure(figsize=(w, h), dpi=100)
        self.fig.patch.set_facecolor(CARD)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def style(self, ax):
        ax.set_facecolor(CARD)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.tick_params(colors=MUTE, labelsize=8)
        ax.xaxis.label.set_color(MUTE)
        ax.yaxis.label.set_color(MUTE)
        ax.title.set_color(INK)
        return ax

    def new_ax(self):
        self.fig.clear()
        return self.style(self.fig.add_subplot(111))

    def placeholder(self, text):
        ax = self.new_ax()
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color=FAINT,
                fontsize=10, transform=ax.transAxes)
        self.draw_idle()


# --------------------------------------------------------------------------
# training worker (keeps the UI responsive)
# --------------------------------------------------------------------------
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
# KPI tile
# --------------------------------------------------------------------------
class Kpi(QFrame):
    def __init__(self, label):
        super().__init__()
        self.setObjectName("kpi")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        self._l = QLabel(label); self._l.setObjectName("kpiLabel")
        self._v = QLabel("—"); self._v.setObjectName("kpiValue")
        lay.addWidget(self._l); lay.addWidget(self._v)

    def set(self, value, color=INK):
        self._v.setText(value)
        self._v.setStyleSheet(f"color:{color};")


def _card(title):
    frame = QFrame(); frame.setObjectName("card")
    lay = QVBoxLayout(frame); lay.setContentsMargins(12, 10, 12, 12); lay.setSpacing(6)
    t = QLabel(title); t.setObjectName("cardTitle")
    lay.addWidget(t)
    return frame, lay


# --------------------------------------------------------------------------
# Model page — train a single-component classifier on a set of reference SERS maps
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
                                        ("random (leaky)", "random")]))
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

        # 2x2 plot grid
        grid = QGridLayout(); grid.setSpacing(12)
        self.c_curve = Canvas(); self.c_cm = Canvas()
        self.c_pca = Canvas(); self.c_bar = Canvas()
        for (cv, title, r, c) in [
            (self.c_curve, "Learning curve", 0, 0),
            (self.c_cm, "Confusion matrix (held-out test)", 0, 1),
            (self.c_pca, "PCA — real per-pixel spectra by class", 1, 0),
            (self.c_bar, "Per-class precision / recall / F1", 1, 1),
        ]:
            card, lay = _card(title)
            lay.addWidget(cv)
            grid.addWidget(card, r, c)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)

        for cv, msg in [(self.c_curve, "Train to watch the learning curve"),
                        (self.c_cm, "Train to compute confusion matrix"),
                        (self.c_pca, "Train to compute PCA"),
                        (self.c_bar, "Train to compute per-class F1")]:
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
                        ("model_pca", self.c_pca), ("model_prf", self.c_bar)], d)
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
                      split=self.cmb_split.currentData())
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
        # random split is leaky — flag the (inflated) accuracy in warning colour
        leaky = getattr(res, "split", "spatial") == "random"
        self.k_acc.set(f"{res.acc:.0%}" + ("  ⚠leaky" if leaky else ""),
                       CORAL if leaky else TEAL)
        self.k_f1.set(f"{res.macro_f1:.3f}", BLUE)
        self.k_tr.set(f"{res.n_train:,}", AMBER)
        self.k_te.set(f"{res.n_test:,}", PURPLE)
        self._plot_curve(res); self._plot_cm(res)
        self._plot_pca(res); self._plot_bar(res)

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


# --------------------------------------------------------------------------
# Quantify page — ratio→M calibration + Langmuir competition (real compute)
# --------------------------------------------------------------------------
def _real_lab(cal, seed=0, n_validation=6):
    """Build a lab dict from a loaded calibration CSV (real dilution series).
    Validation mixtures are synthesized from the real templates + fitted physics
    to demonstrate recovery (clearly a synthetic check on real calibration)."""
    from competitive import forward_spectrum
    axis, names, dilutions = cal
    Praw = np.array([sp[int(np.argmax(c))] for c, sp in dilutions])
    P = Praw / (np.linalg.norm(Praw, axis=1, keepdims=True) + 1e-12)
    tmp = calibrate(dilutions, P, names)
    rng = np.random.default_rng(seed)
    K = tmp.K; A = tmp.gA / (tmp.gA.max() or 1.0)
    n = len(names); val_specs, val_true = [], []
    for _ in range(n_validation):
        k = int(rng.integers(2, n + 1)); idx = rng.choice(n, k, replace=False)
        C = np.zeros(n); C[idx] = 10 ** rng.uniform(-6, -3.3, k)
        y = forward_spectrum(C, K, A, P)
        y = np.clip(y + rng.normal(0, 0.01 * (y.max() or 1.0), len(axis)), 0, None)
        val_specs.append(y); val_true.append(C)
    return {"axis": axis, "names": names, "P": P, "dilutions": dilutions,
            "val_specs": np.array(val_specs), "val_true": np.array(val_true),
            "K_true": None}


def _run_quant(n_components=3, seed=0, cal=None):
    from calibration import _langmuir_B
    lab = (_real_lab(cal, seed) if cal is not None
           else build_synthetic_lab(n_components=n_components, seed=seed))
    calib = calibrate(lab["dilutions"], lab["P"], lab["names"])

    iso = []
    for i in range(calib.n):
        C = calib.C_series[i]
        dense = np.geomspace(C.min(), C.max(), 60)
        iso.append((C, calib.B_series[i], dense,
                    _langmuir_B(dense, calib.gA[i], calib.K[i])))

    quants = [quantify(y, lab["P"], calib) for y in lab["val_specs"]]
    true_flat, est_flat, col_flat = [], [], []
    for q, Ct in zip(quants, lab["val_true"]):
        for i in range(calib.n):
            if Ct[i] > 0 and q["C"][i] > 0:
                true_flat.append(Ct[i]); est_flat.append(q["C"][i]); col_flat.append(i)
    log_err = float(np.mean(np.abs(np.log10(
        np.array(est_flat) / np.array(true_flat))))) if true_flat else float("nan")

    ex = next((k for k, q in enumerate(quants)
               if q["competition"]["flipped"]), 0)
    return {
        "names": calib.names, "K_true": lab["K_true"], "K_fit": calib.K,
        "iso": iso, "parity": (np.array(true_flat), np.array(est_flat),
                               np.array(col_flat, int)),
        "log_err": log_err, "example": quants[ex],
        "example_true": lab["val_true"][ex],
        "selectivity": quants[ex]["competition"]["selectivity"],
    }


class QuantWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.done.emit(_run_quant(**self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


class QuantifyPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._cal = None       # loaded calibration (axis, names, dilutions)
        self._res = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(14)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Quantify — ratio → M + competition"); h1.setObjectName("h1")
        sub = QLabel("Calibrate each compound's Langmuir isotherm from a dilution "
                     "series (synthetic, or load your own CSV) → recover absolute M "
                     "and judge competitive adsorption.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(10)
        self.sp_k = self._spin(QSpinBox(), 2, 5, 3, "compounds")
        self.sp_seed = self._spin(QSpinBox(), 0, 999, 1, "seed")
        for w in (self.sp_k, self.sp_seed):
            ctl.addLayout(w)
        self.src = QLabel("source: synthetic"); self.src.setObjectName("field")
        ctl.addWidget(self.src); ctl.addStretch(1)
        load_b = QPushButton("Load calibration…"); load_b.setObjectName("ghost")
        load_b.clicked.connect(self._load_cal)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Calibrate + quantify"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(load_b); ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_sel = Kpi("selectivity  K_max/K_min")
        self.k_cov = Kpi("surface coverage Σθ")
        self.k_flip = Kpi("competition")
        self.k_err = Kpi("abs log₁₀ error")
        for k in (self.k_sel, self.k_cov, self.k_flip, self.k_err):
            kpis.addWidget(k)
        root.addLayout(kpis)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_iso = Canvas(); self.c_par = Canvas(); self.c_comp = Canvas()
        for (cv, title, r, c) in [
            (self.c_iso, "Langmuir isotherm fits  (B vs concentration)", 0, 0),
            (self.c_par, "Recovered vs true concentration  (M)", 0, 1),
            (self.c_comp, "Surface coverage vs solution concentration", 1, 0),
        ]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, r, c)
        # text readout card (4th cell)
        rcard, rlay = _card("Read-out  ·  example mixture")
        self.readout = QLabel("Run to compute."); self.readout.setObjectName("sub")
        self.readout.setWordWrap(True); self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignTop)
        rlay.addWidget(self.readout); rlay.addStretch(1)
        grid.addWidget(rcard, 1, 1)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)
        for cv, m in [(self.c_iso, "Calibrate to fit isotherms"),
                      (self.c_par, "Calibrate to check recovery"),
                      (self.c_comp, "Calibrate to compare coverage vs concentration")]:
            cv.placeholder(m)

    def _spin(self, spin, lo, hi, val, label):
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        spin.setRange(lo, hi); spin.setValue(val)
        col.addWidget(lb); col.addWidget(spin)
        return col

    def _load_cal(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Calibration CSV (compound, concentration_M, wavenumbers…)",
            "", "CSV (*.csv)")
        if not p:
            return
        try:
            axis, names, dilutions = load_calibration_csv(p)
            self._cal = (axis, names, dilutions)
            self.src.setText(f"source: {os.path.basename(p)} ({len(names)})")
        except Exception as exc:
            self.src.setText("load failed"); print("load cal:", exc, file=sys.stderr)

    def _export(self):
        if self._res is None:
            self.src.setText("run first, then export"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res; q = r["example"]
        rows = [[nm, f"{q['C'][i]:.3e}", f"{q['conc_ratio'][i]:.3f}",
                 f"{q['theta'][i]:.3f}", f"{r['K_fit'][i]:.3e}"]
                for i, nm in enumerate(r["names"])]
        write_csv(os.path.join(d, "quantify.csv"),
                  ["compound", "C_M", "ratio", "theta", "K_fit"], rows)
        n = _save_figs([("quantify_isotherms", self.c_iso),
                        ("quantify_parity", self.c_par),
                        ("quantify_competition", self.c_comp)], d)
        self.src.setText(f"exported CSV + {n} PNG → {os.path.basename(d)}")

    def _run(self):
        params = dict(n_components=self.sp_k.itemAt(1).widget().value(),
                      seed=self.sp_seed.itemAt(1).widget().value(), cal=self._cal)
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self._thread = QThread(); self._worker = QuantWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Calibrate + quantify")
        print(tb, file=sys.stderr)

    def _apply(self, res):
        self._res = res
        self.btn.setEnabled(True); self.btn.setText("Calibrate + quantify")
        comp = res["example"]["competition"]
        self.k_sel.set(f"{res['selectivity']:.1f}×", CORAL)
        self.k_cov.set(f"{res['example']['theta_total']:.2f}", BLUE)
        self.k_flip.set("flipped" if comp["flipped"] else "consistent",
                        CORAL if comp["flipped"] else TEAL)
        self.k_err.set(f"{res['log_err']:.2f}", PURPLE)
        self._plot_iso(res); self._plot_parity(res)
        self._plot_comp(res); self._readout(res)

    def _plot_iso(self, res):
        ax = self.c_iso.new_ax()
        for i, nm in enumerate(res["names"]):
            C, B, dc, db = res["iso"][i]
            col = SERIES[i % len(SERIES)]
            ax.scatter(C, B, s=20, color=col, zorder=3)
            ax.plot(dc, db, color=col, lw=1.4, label=nm)
        ax.set_xscale("log"); ax.set_xlabel("concentration (M)"); ax.set_ylabel("B")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE)
        self.c_iso.fig.tight_layout(); self.c_iso.draw_idle()

    def _plot_parity(self, res):
        ax = self.c_par.new_ax()
        t, e, col = res["parity"]
        if len(t):
            for i in range(len(res["names"])):
                m = col == i
                if m.any():
                    ax.scatter(t[m], e[m], s=26, color=SERIES[i % len(SERIES)],
                               label=res["names"][i], edgecolors="none")
            lo = min(t.min(), e.min()); hi = max(t.max(), e.max())
            ax.plot([lo, hi], [lo, hi], color=FAINT, lw=1, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("true (M)"); ax.set_ylabel("recovered (M)")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE)
        self.c_par.fig.tight_layout(); self.c_par.draw_idle()

    def _plot_comp(self, res):
        ax = self.c_comp.new_ax()
        q = res["example"]; names = res["names"]
        act = [i for i in range(len(names)) if q["C"][i] > 0]
        x = np.arange(len(act)); w = 0.38
        cov = [q["cov_ratio"][i] for i in act]
        con = [q["conc_ratio"][i] for i in act]
        ax.bar(x - w / 2, cov, w, color=AMBER, label="surface θ (apparent)")
        ax.bar(x + w / 2, con, w, color=TEAL, label="solution C (true)")
        ax.set_xticks(x); ax.set_xticklabels([names[i] for i in act], fontsize=8)
        ax.set_ylabel("fraction"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE)
        self.c_comp.fig.tight_layout(); self.c_comp.draw_idle()

    def _readout(self, res):
        q = res["example"]; Ct = res["example_true"]; names = res["names"]
        comp = q["competition"]
        rows = []
        for i, nm in enumerate(names):
            if q["C"][i] <= 0 and Ct[i] <= 0:
                continue
            rows.append(
                f"<tr><td style='padding-right:12px;color:{SERIES[i%len(SERIES)]};"
                f"font-weight:600'>{nm}</td>"
                f"<td style='padding-right:12px'>{q['C'][i]:.2e} M</td>"
                f"<td style='padding-right:12px;color:{MUTE}'>{q['conc_ratio'][i]*100:.0f}%"
                f"</td><td style='color:{MUTE}'>θ {q['theta'][i]:.3f}</td></tr>")
        verdict = (
            f"<b style='color:{CORAL}'>Competitive adsorption</b> — "
            f"<b>{comp['surface_dominant']}</b> dominates the surface but "
            f"<b>{comp['solution_dominant']}</b> dominates in solution "
            f"(selectivity {comp['selectivity']:.1f}×). "
            f"{comp['buried'] or '—'} is the most buried."
            if comp["flipped"] else
            f"<b style='color:{TEAL}'>Consistent</b> — surface and solution agree "
            f"(selectivity {comp['selectivity']:.1f}×).")
        self.readout.setText(
            f"<div style='color:{INK};font-size:13px'>"
            f"<table style='font-size:13px'>{''.join(rows)}</table>"
            f"<p style='margin-top:10px;font-size:13px'>{verdict}</p>"
            f"<p style='color:{FAINT};font-size:12px'>absolute M assumes the "
            f"substrate gain is stable between calibration and measurement</p></div>")


# --------------------------------------------------------------------------
# Real-data page — DQ / THI / TBZ pesticide maps
# --------------------------------------------------------------------------
class RealWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, pest_dir):
        super().__init__()
        self.pest_dir = pest_dir

    def run(self):
        try:
            self.done.emit(compute_real(self.pest_dir))
        except Exception:
            self.fail.emit(traceback.format_exc())


class RealDataPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self.pest_dir = PEST_DEFAULT
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(12)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Real-data analysis"); h1.setObjectName("h1")
        sub = QLabel("Single-component classification, mixture detection (per-pixel), "
                     "composition confusion, and response-factor correction on your "
                     "loaded maps.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        self.folder_lbl = QLabel(self._short(self.pest_dir)); self.folder_lbl.setObjectName("field")
        browse = QPushButton("Data folder…"); browse.setObjectName("ghost")
        browse.clicked.connect(self._browse)
        self.status = QLabel(""); self.status.setObjectName("sub")
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Run analysis"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(browse); ctl.addWidget(self.folder_lbl, 1)
        ctl.addWidget(self.status); ctl.addStretch(1)
        ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_pure = Kpi("single-component acc")
        self.k_f1 = Kpi("mixture detection F1")
        self.k_combo = Kpi("exact composition")
        self.k_r = Kpi("dominant response  R")
        for k in (self.k_pure, self.k_f1, self.k_combo, self.k_r):
            kpis.addWidget(k)
        root.addLayout(kpis)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_pca = Canvas(); self.c_pure = Canvas(); self.c_combo = Canvas()
        self.c_det = Canvas(); self.c_cal = Canvas(); self.c_strat = Canvas()
        for (cv, title, r, c) in [
            (self.c_pca, "PCA — real per-pixel spectra", 0, 0),
            (self.c_pure, "Single-component confusion  (spatial split)", 0, 1),
            (self.c_strat, "Detection strategy — RF vs per-pixel vs matched", 0, 2),
            (self.c_det, "Detection outcome  (per-pixel voting)", 1, 0),
            (self.c_combo, "Composition confusion  (per-pixel detector)", 1, 1),
            (self.c_cal, "Ratio: raw vs response-calibrated", 1, 2),
        ]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, r, c)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        root.addLayout(grid, 1)
        for cv, m in [(self.c_pca, "Run for PCA"),
                      (self.c_pure, "Run to classify pure substances"),
                      (self.c_strat, "Run to compare strategies"),
                      (self.c_det, "Run to show detection outcomes"),
                      (self.c_combo, "Run to build composition confusion"),
                      (self.c_cal, "Run to correct the ratios")]:
            cv.placeholder(m)

    def _short(self, p):
        return "…" + p[-42:] if len(p) > 42 else p

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Data folder (Reference/ + Ratio/ maps)",
                                             self.pest_dir)
        if d:
            self.pest_dir = d; self.folder_lbl.setText(self._short(d))

    def _export(self):
        if self._res is None:
            self.status.setText("run first, then export")
            self.status.setStyleSheet(f"color:{RED};"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res
        # per-mixture detection (true vs per-pixel predicted)
        det = []
        for k, nm in enumerate(r.mix_names):
            t = "+".join(c for i, c in enumerate(r.comps) if r.yt[k, i])
            p = "+".join(c for i, c in enumerate(r.comps) if r.yp[k, i])
            det.append([nm, t, p, "hit" if t == p else "miss"])
        write_csv(os.path.join(d, "detection.csv"),
                  ["mixture", "true", "predicted", "exact"], det)
        write_csv(os.path.join(d, "strategies.csv"),
                  ["strategy", "recall", "precision", "f1", "exact"],
                  [[s[0], f"{s[1]:.3f}", f"{s[2]:.3f}", f"{s[3]:.3f}", f"{s[4]:.3f}"]
                   for s in r.strategies])
        write_csv(os.path.join(d, "response_factors.csv"),
                  ["compound", "R"], [[c, f"{r.R[i]:.3f}"]
                                      for i, c in enumerate(r.comps)])
        n = _save_figs([("real_pca", self.c_pca), ("real_pure_confusion", self.c_pure),
                        ("real_strategy", self.c_strat), ("real_detection", self.c_det),
                        ("real_composition", self.c_combo),
                        ("real_calibration", self.c_cal)], d)
        self.status.setText(f"exported 3 CSV + {n} PNG → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _run(self):
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self.status.setText("")
        self._thread = QThread(); self._worker = RealWorker(self.pest_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Run analysis")
        first = tb.strip().splitlines()[-1][:80]
        self.status.setText("failed — " + first)
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, r):
        self._res = r
        self.btn.setEnabled(True); self.btn.setText("Run analysis")
        self.status.setText("done"); self.status.setStyleSheet(f"color:{MUTE};")
        self.k_pure.set(f"{r.acc4:.0%}", TEAL)
        self.k_f1.set(f"{r.micro['micro_f1']:.2f}", BLUE)
        self.k_combo.set(f"{r.combo_exact:.0%}", AMBER)
        di = int(np.argmax(r.R))
        self.k_r.set(f"{r.comps[di]} {r.R[di]:.1f}×", CORAL)
        self._plot_pca(r); self._plot_pure(r); self._plot_strat(r)
        self._plot_det(r); self._plot_combo(r); self._plot_cal(r)

    def _plot_pca(self, r):
        ax = self.c_pca.new_ax()
        for i, nm in enumerate(r.classes4):
            m = r.pca_lab == i
            if m.any():
                ax.scatter(r.pca_emb[m, 0], r.pca_emb[m, 1], s=10,
                           color=SERIES[i % len(SERIES)], alpha=0.6,
                           edgecolors="none", label=nm)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE)
        self.c_pca.fig.tight_layout(); self.c_pca.draw_idle()

    def _plot_strat(self, r):
        ax = self.c_strat.new_ax()
        short = ["RF\n(mean)", "per-pixel\nvote", "matched\n(mean)"]
        rec = [s[1] for s in r.strategies]
        f1 = [s[3] for s in r.strategies]
        ex = [s[4] for s in r.strategies]
        x = np.arange(len(r.strategies)); w = 0.26
        ax.bar(x - w, rec, w, color=BLUE, label="recall")
        ax.bar(x, f1, w, color=TEAL, label="F1")
        ax.bar(x + w, ex, w, color=AMBER, label="exact")
        for xi, v in zip(x, f1):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
        ax.set_ylim(0, 1.08); ax.set_ylabel("score")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=3)
        self.c_strat.fig.tight_layout(); self.c_strat.draw_idle()

    # ---- plots ----
    def _confusion(self, ax, M, xlabels, ylabels, box_diag=False):
        ax.imshow(M, cmap=CM_CMAP, aspect="auto", vmin=0)
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ylabels)), minor=True)
        ax.grid(which="minor", color=PANEL, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        thr = M.max() / 2 if M.max() else 0.5
        for (rr, cc), v in np.ndenumerate(M):
            if v:
                ax.text(cc, rr, str(v), ha="center", va="center", fontsize=9,
                        color="#ffffff" if v > thr else INK, fontweight="bold")
        if box_diag:
            xi = {c: i for i, c in enumerate(xlabels)}
            for rr, yl in enumerate(ylabels):
                if yl in xi:
                    ax.add_patch(Rectangle((xi[yl] - 0.5, rr - 0.5), 1, 1,
                                 fill=False, edgecolor=INK, lw=1.6))

    def _plot_pure(self, r):
        ax = self.c_pure.new_ax()
        self._confusion(ax, r.cm4, r.classes4, r.classes4)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"spatial split {r.acc4:.0%}   ·   random {r.acc4_random:.0%} "
                     "(leaky)", fontsize=9, color=INK)
        self.c_pure.fig.tight_layout(); self.c_pure.draw_idle()

    def _plot_combo(self, r):
        ax = self.c_combo.new_ax()
        self._confusion(ax, r.combo_M, r.combo_cols, r.combo_rows, box_diag=True)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
        self.c_combo.fig.tight_layout(); self.c_combo.draw_idle()

    def _plot_det(self, r):
        ax = self.c_det.new_ax()
        nS, nC = r.yt.shape
        for row in range(nS):
            for col in range(nC):
                t, p = r.yt[row, col], r.yp[row, col]
                if t and p:
                    fc, mark = TEAL, "O"
                elif t and not p:
                    fc, mark = RED, "X"
                elif p and not t:
                    fc, mark = AMBER, "!"
                else:
                    fc, mark = TNGRAY, ""
                ax.add_patch(Rectangle((col, nS - 1 - row), 1, 1, facecolor=fc,
                             edgecolor="white", linewidth=1.5))
                if mark:
                    ax.text(col + 0.5, nS - 1 - row + 0.5, mark, ha="center",
                            va="center", color="white", fontsize=9, fontweight="bold")
        ax.set_xlim(0, nC); ax.set_ylim(0, nS)
        ax.set_xticks(np.arange(nC) + 0.5); ax.set_xticklabels(r.comps, fontsize=9)
        ax.set_yticks(np.arange(nS) + 0.5); ax.set_yticklabels(r.mix_names[::-1], fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(handles=[Patch(facecolor=TEAL, label="hit"),
                           Patch(facecolor=RED, label="miss"),
                           Patch(facecolor=AMBER, label="false +")],
                  loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3,
                  frameon=False, fontsize=8, labelcolor=MUTE)
        self.c_det.fig.tight_layout(); self.c_det.draw_idle()

    def _plot_cal(self, r):
        self.c_cal.fig.clear()
        for idx, (key, title) in enumerate([("raw", "raw signal"),
                                            ("cal", "calibrated")]):
            ax = self.c_cal.style(self.c_cal.fig.add_subplot(1, 2, idx + 1))
            for name, present, nom, raw, cal in r.calib_rows:
                val = raw if key == "raw" else cal
                for i in present:
                    ax.scatter(nom[i], val[i], s=30, color=SERIES[i % len(SERIES)],
                               edgecolors="white", linewidths=0.5, zorder=3)
            ax.plot([0, 1], [0, 1], ls="--", color=FAINT, lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("nominal", fontsize=8); ax.set_title(title, fontsize=9)
            if idx == 0:
                ax.set_ylabel("recovered", fontsize=8)
        self.c_cal.fig.suptitle(f"mean error  {r.err_raw:.0%}  →  {r.err_cal:.0%}",
                                fontsize=9, color=INK)
        self.c_cal.fig.tight_layout(); self.c_cal.draw_idle()


# --------------------------------------------------------------------------
# Predict page — load one unknown sample, read its composition ratio
# --------------------------------------------------------------------------
class PredictWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.done.emit(predict_sample(**self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


class PredictPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self.data_dir = PEST_DEFAULT
        self.sample = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(14)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Predict — sample composition"); h1.setObjectName("h1")
        sub = QLabel("Load one unknown map and read the ratio of your reference "
                     "substances in it (mean-spectrum NNLS unmix + per-pixel vote). "
                     "Organise the references in Samples first.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(10)
        ref_b = QPushButton("Reference data…"); ref_b.setObjectName("ghost")
        ref_b.clicked.connect(self._browse_ref)
        self.ref_lbl = QLabel(self._short(self.data_dir)); self.ref_lbl.setObjectName("field")
        samp_b = QPushButton("Load sample…"); samp_b.setObjectName("ghost")
        samp_b.clicked.connect(self._browse_sample)
        self.samp_lbl = QLabel("no sample"); self.samp_lbl.setObjectName("field")
        tcol = QVBoxLayout(); tcol.setSpacing(2)
        tl = QLabel("threshold"); tl.setObjectName("field")
        self.thr = QDoubleSpinBox(); self.thr.setDecimals(2); self.thr.setSingleStep(0.05)
        self.thr.setRange(0.05, 0.9); self.thr.setValue(0.30)
        tcol.addWidget(tl); tcol.addWidget(self.thr)
        bcol = QVBoxLayout(); bcol.setSpacing(2)
        bl = QLabel("baseline"); bl.setObjectName("field")
        self.chk_base = QCheckBox("ALS on"); self.chk_base.setChecked(True)
        bcol.addWidget(bl); bcol.addWidget(self.chk_base)
        self.btn = QPushButton("Predict"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(ref_b); ctl.addWidget(self.ref_lbl)
        ctl.addWidget(samp_b); ctl.addWidget(self.samp_lbl, 1)
        ctl.addLayout(tcol); ctl.addLayout(bcol); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_dom = Kpi("dominant"); self.k_domp = Kpi("dominant %")
        self.k_n = Kpi("components"); self.k_px = Kpi("pixels")
        for k in (self.k_dom, self.k_domp, self.k_n, self.k_px):
            kpis.addWidget(k)
        root.addLayout(kpis)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_ratio = Canvas(); self.c_map = Canvas(); self.c_spec = Canvas()
        for cv, title, c in [(self.c_ratio, "Composition ratio (per-pixel NNLS)", 0),
                             (self.c_map, "Per-pixel dominant component", 1)]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, 0, c)
        card, lay = _card("Sample vs reference templates")
        lay.addWidget(self.c_spec); grid.addWidget(card, 1, 0, 1, 2)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        root.addLayout(grid, 1)
        for cv in (self.c_ratio, self.c_map, self.c_spec):
            cv.placeholder("Load a sample, then Predict")

        self.readout = QLabel(""); self.readout.setObjectName("sub")
        self.readout.setWordWrap(True)
        root.addWidget(self.readout)

    def _short(self, p):
        return "refs: " + ("…" + p[-38:] if len(p) > 38 else p)

    def _browse_ref(self):
        d = QFileDialog.getExistingDirectory(
            self, "Reference data folder (your Samples)", self.data_dir)
        if d:
            self.data_dir = d; self.ref_lbl.setText(self._short(d))

    def _browse_sample(self):
        p, _ = QFileDialog.getOpenFileName(self, "Unknown sample map CSV", "",
                                           "CSV (*.csv)")
        if p:
            self.sample = p; self.samp_lbl.setText(os.path.basename(p))

    def _run(self):
        if not self.sample:
            self.readout.setText("load a sample first"); return
        params = dict(data_dir=self.data_dir, sample_path=self.sample,
                      threshold=self.thr.value(), baseline=self.chk_base.isChecked())
        self.btn.setEnabled(False); self.btn.setText("Predicting…")
        self._thread = QThread(); self._worker = PredictWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Predict")
        self.readout.setText("failed — " + tb.strip().splitlines()[-1][:90])
        print(tb, file=sys.stderr)

    def _apply(self, res):
        self._res = res
        self.btn.setEnabled(True); self.btn.setText("Predict")
        ratio = res.ratio
        if ratio:
            dom = max(ratio, key=ratio.get)
            self.k_dom.set(dom, TEAL); self.k_domp.set(f"{ratio[dom]:.0%}", BLUE)
        else:
            self.k_dom.set("—"); self.k_domp.set("—")
        self.k_n.set(str(len(res.detected)), AMBER)
        self.k_px.set(f"{res.n_pixels:,}", PURPLE)
        self._plot_ratio(res); self._plot_map(res); self._plot_spec(res)
        parts = "  ·  ".join(f"{nm} {ratio.get(nm, 0):.0%}" for nm in res.detected)
        self.readout.setText(f"<b>detected:</b> {' + '.join(res.detected)}   "
                             f"&nbsp;&nbsp; <b>per-pixel ratio:</b> {parts}")
        self.readout.setTextFormat(Qt.TextFormat.RichText)

    def _plot_ratio(self, res):
        ax = self.c_ratio.new_ax()
        names = res.comps
        vals = [res.ratio.get(n, 0.0) for n in names]
        mvals = [res.ratio_mean.get(n, 0.0) for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, color=[SERIES[i % len(SERIES)] for i in range(len(names))],
               label="per-pixel")
        ax.scatter(x, mvals, color=INK, s=28, zorder=3, label="mean-spec")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_ylabel("proportion")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE)
        self.c_ratio.fig.tight_layout(); self.c_ratio.draw_idle()

    def _plot_map(self, res):
        ax = self.c_map.new_ax()
        x, y = res.coords[:, 0], res.coords[:, 1]
        for i, nm in enumerate(res.comps):
            m = res.pp_dominant == i
            if m.any():
                ax.scatter(x[m], y[m], s=16, color=SERIES[i % len(SERIES)],
                           edgecolors="none", label=nm)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=2)
        self.c_map.fig.tight_layout(); self.c_map.draw_idle()

    def _plot_spec(self, res):
        ax = self.c_spec.new_ax()
        axis = res.wn if res.wn is not None else np.arange(res.mean_spectrum.shape[0])
        for i, nm in enumerate(res.comps):
            ax.plot(axis, res.templates[i], lw=0.9, alpha=0.6,
                    color=SERIES[i % len(SERIES)], label=nm)
        ax.plot(axis, res.mean_spectrum, lw=1.6, color=INK, label="sample")
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_yticks([])
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=2)
        self.c_spec.fig.tight_layout(); self.c_spec.draw_idle()


# --------------------------------------------------------------------------
# Sampling page — organise raw maps into substance classes (batches)
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
                     "not separate classes (so THI and THI_2 are one 'THI'). Edit "
                     "Class / Batch, then Save — Model and Real data read this "
                     "grouping from samples.csv.")
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

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Class (substance)", "Batch"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
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
            cls, batch = None, None
            if manifest is not None:
                hit = manifest.get(os.path.abspath(path))
                if hit and hit[0]:
                    cls, batch = hit
            if cls is None:
                cls, batch = base_and_batch(name)
            r = self.table.rowCount(); self.table.insertRow(r)
            fitem = QTableWidgetItem(os.path.basename(path))
            fitem.setFlags(fitem.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, fitem)
            self.table.setItem(r, 1, QTableWidgetItem(str(cls)))
            self.table.setItem(r, 2, QTableWidgetItem(str(batch)))
        self._loading = False
        if refs:
            self.status.setText(f"{len(refs)} maps"); self.status.setStyleSheet(f"color:{MUTE};")
        self._update_summary()

    def _rows(self):
        out = []
        for r in range(self.table.rowCount()):
            fn = self.table.item(r, 0).text()
            cls = (self.table.item(r, 1).text() or "").strip()
            bt = (self.table.item(r, 2).text() or "").strip()
            out.append((fn, cls, int(bt) if bt.isdigit() else 1))
        return out

    def _on_edit(self, _item):
        if not self._loading:
            self._update_summary()

    def _update_summary(self):
        groups = {}
        for _fn, cls, _b in self._rows():
            groups.setdefault(cls, 0)
            groups[cls] += 1
        if not groups:
            self.summary.setText("no maps found in this folder"); return
        parts = [f"{c} ×{n}" if n > 1 else c for c, n in sorted(groups.items())]
        self.summary.setText(f"{len(groups)} classes:   " + "   ·   ".join(parts))

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


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    # native analysis pages — clicking switches the view in-place
    PAGES = [
        ("Samples",   "samples", "Group your maps into substance classes (batches)"),
        ("Model",     "model", "Train a classifier on your reference maps"),
        ("Predict",   "predict", "Load an unknown sample → read its component ratio"),
        ("Quantify",  "quant", "Ratio → concentration + adsorption competition"),
        ("Real data", "real",  "Analyze real maps: identify · mixtures · calibration"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1360, 880)

        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        # top command bar
        bar = QFrame(); bar.setObjectName("topbar"); bar.setFixedHeight(58)
        bl = QHBoxLayout(bar); bl.setContentsMargins(18, 0, 18, 0); bl.setSpacing(8)
        logo = QLabel()
        logo.setFixedSize(30, 30); logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(ICON_PATH):                       # use the app icon
            logo.setPixmap(QPixmap(ICON_PATH).scaled(
                28, 28, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:                                               # fallback: teal "U" badge
            logo.setObjectName("logo"); logo.setText("U")
        word = QLabel(APP_NAME); word.setObjectName("wordmark")
        bl.addWidget(logo); bl.addWidget(word); bl.addSpacing(18)

        # native page tabs — everything lives in this one window now
        self._nav_btns = {}
        for label, key, desc in self.PAGES:
            b = QPushButton(label); b.setObjectName("nav"); b.setCheckable(True)
            b.setToolTip(desc)
            b.clicked.connect(lambda _=False, k=key: self.select(k))
            bl.addWidget(b); self._nav_btns[key] = b

        bl.addStretch(1)
        self.status = QLabel(f"{APP_NAME} v{VERSION}"); self.status.setObjectName("status")
        bl.addWidget(self.status)
        outer.addWidget(bar)

        # stacked content
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.pages = {
            "samples": SamplingPage(),
            "model": ModelPage(),
            "predict": PredictPage(),
            "quant": QuantifyPage(),
            "real": RealDataPage(),
        }
        for key in ("samples", "model", "predict", "quant", "real"):
            self.stack.addWidget(self.pages[key])

        self.select("samples")

    def select(self, key):
        for k, b in self._nav_btns.items():
            b.setChecked(k == key)
        self.stack.setCurrentWidget(self.pages[key])


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    win = MainWindow()
    if os.path.exists(ICON_PATH):
        win.setWindowIcon(QIcon(ICON_PATH))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
