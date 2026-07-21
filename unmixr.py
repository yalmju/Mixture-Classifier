"""UNMIXR — SERS mixture analysis suite (PyQt6).

A dark, instrument-style desktop app. Not a sidebar dashboard — a top command
bar with pill navigation over a stacked content area:

    Model         train the mixture classifier and read its metrics live:
                  PCA scatter · confusion matrix · per-component P/R/F1 · KPI tiles
    Mixture       the mixture detector / ratio tool  (customtkinter, launched)
    Discriminator the hyperspectral map classifier   (customtkinter, launched)

    python unmixr.py

The Model page is native PyQt6 with embedded matplotlib. The other two tools are
still customtkinter apps (from the earlier suite); UNMIXR launches them for now
and they'll be ported to Qt next. Rename the app by editing APP_NAME below.
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QHBoxLayout, QVBoxLayout, QGridLayout, QStackedWidget, QSpinBox,
    QDoubleSpinBox, QSizePolicy, QFileDialog,
)
from matplotlib.patches import Patch, Rectangle

from model_metrics import compute_metrics, MetricsResult
from calibration import build_synthetic_lab, calibrate, quantify
from real_data import compute_real, PEST_DEFAULT
from io_utils import load_spectra_csv, load_calibration_csv, write_csv


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

# ---- light palette -------------------------------------------------------
PAGE   = "#f5f7fa"
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
QLabel#field {{ color: {MUTE}; font-size: 13px; }}
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

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.done.emit(compute_metrics(**self.params))
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
# Model page — the metrics dashboard
# --------------------------------------------------------------------------
class ModelPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(14)

        self._res = None
        self._pures = None     # loaded real references (raw)
        self._names = None
        self._axis = None

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Model metrics"); h1.setObjectName("h1")
        sub = QLabel("Train the mixture classifier on synthetic pure spectra "
                     "(or load your own reference CSV), then read PCA · confusion · F1.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        # controls
        ctl = QHBoxLayout(); ctl.setSpacing(10)
        self.sp_k = self._spin(QSpinBox(), 2, 8, 6, "components")
        self.sp_thr = self._spin(QDoubleSpinBox(), 0.05, 0.9, 0.30, "threshold", step=0.05)
        self.sp_aug = self._spin(QSpinBox(), 30, 400, 120, "aug / pure", step=10)
        for w in (self.sp_k, self.sp_thr, self.sp_aug):
            ctl.addLayout(w)
        self.src = QLabel("source: synthetic"); self.src.setObjectName("field")
        ctl.addWidget(self.src); ctl.addStretch(1)
        load_b = QPushButton("Load refs…"); load_b.setObjectName("ghost")
        load_b.clicked.connect(self._load_refs)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Train + evaluate"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._train)
        ctl.addWidget(load_b); ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        # KPI row
        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_f1 = Kpi("micro F1"); self.k_p = Kpi("precision")
        self.k_r = Kpi("recall"); self.k_ex = Kpi("exact match")
        for k in (self.k_f1, self.k_p, self.k_r, self.k_ex):
            kpis.addWidget(k)
        root.addLayout(kpis)

        # 2x2 plot grid
        grid = QGridLayout(); grid.setSpacing(12)
        self.c_pca = Canvas(); self.c_cm = Canvas()
        self.c_bar = Canvas(); self.c_spec = Canvas()
        for (cv, title, r, c) in [
            (self.c_pca, "PCA — spectra by component", 0, 0),
            (self.c_cm, "Confusion matrix (single-component test)", 0, 1),
            (self.c_bar, "Per-component precision / recall / F1", 1, 0),
            (self.c_spec, "Reference templates", 1, 1),
        ]:
            card, lay = _card(title)
            lay.addWidget(cv)
            grid.addWidget(card, r, c)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)

        for cv, msg in [(self.c_pca, "Train to compute PCA"),
                        (self.c_cm, "Train to compute confusion matrix"),
                        (self.c_bar, "Train to compute F1"),
                        (self.c_spec, "Train to view templates")]:
            cv.placeholder(msg)

    def _spin(self, spin, lo, hi, val, label, step=1):
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        if isinstance(spin, QDoubleSpinBox):
            spin.setDecimals(2); spin.setSingleStep(step)
        else:
            spin.setSingleStep(step)
        spin.setRange(lo, hi); spin.setValue(val)
        col.addWidget(lb); col.addWidget(spin)
        return col

    def _load_refs(self):
        p, _ = QFileDialog.getOpenFileName(self, "Reference spectra CSV "
                                           "(wavenumber + one column per component)",
                                           "", "CSV (*.csv)")
        if not p:
            return
        try:
            axis, names, spectra = load_spectra_csv(p)
            self._pures, self._names, self._axis = spectra, names, axis
            self.src.setText(f"source: {os.path.basename(p)} ({len(names)})")
        except Exception as exc:
            self.src.setText("load failed"); print("load refs:", exc, file=sys.stderr)

    def _export(self):
        if self._res is None:
            self.src.setText("run first, then export"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res
        rows = [[nm, f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}", v[3]]
                for nm, v in r.per_component.items()]
        rows.append(["micro", f"{r.micro['micro_precision']:.4f}",
                     f"{r.micro['micro_recall']:.4f}", f"{r.micro['micro_f1']:.4f}", ""])
        write_csv(os.path.join(d, "model_metrics.csv"),
                  ["component", "precision", "recall", "f1", "support"], rows)
        write_csv(os.path.join(d, "model_confusion.csv"),
                  [""] + list(r.names),
                  [[r.names[i]] + list(map(int, r.confusion[i]))
                   for i in range(len(r.names))])
        n = _save_figs([("model_pca", self.c_pca), ("model_confusion", self.c_cm),
                        ("model_prf", self.c_bar), ("model_templates", self.c_spec)], d)
        self.src.setText(f"exported CSV + {n} PNG → {os.path.basename(d)}")

    # ---- training ----
    def _train(self):
        params = dict(n_components=self.sp_k.itemAt(1).widget().value(),
                      threshold=self.sp_thr.itemAt(1).widget().value(),
                      n_per_pure=self.sp_aug.itemAt(1).widget().value(),
                      seed=0)
        if self._pures is not None:       # train on the loaded real references
            params.update(pure_raw=self._pures, names=self._names, axis=self._axis)
        self.btn.setEnabled(False); self.btn.setText("Training…")
        self._thread = QThread()
        self._worker = TrainWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Train + evaluate")
        self.c_pca.placeholder("Training failed — see console")
        print(tb, file=sys.stderr)

    def _apply(self, res: MetricsResult):
        self._res = res
        self.btn.setEnabled(True); self.btn.setText("Train + evaluate")
        m = res.micro
        self.k_f1.set(f"{m['micro_f1']:.3f}", TEAL)
        self.k_p.set(f"{m['micro_precision']:.3f}", BLUE)
        self.k_r.set(f"{m['micro_recall']:.3f}", AMBER)
        self.k_ex.set(f"{m['exact_match_ratio']:.0%}", PURPLE)
        self._plot_pca(res); self._plot_cm(res)
        self._plot_bar(res); self._plot_spec(res)

    # ---- plots ----
    def _plot_pca(self, res):
        ax = self.c_pca.new_ax()
        for i, nm in enumerate(res.names):
            pts = res.pca_points[res.pca_labels == i]
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=22, color=SERIES[i % len(SERIES)],
                           edgecolors="none", alpha=0.85, label=nm)
            ax.scatter(res.pca_pure[i, 0], res.pca_pure[i, 1], marker="*", s=210,
                       color=SERIES[i % len(SERIES)], edgecolors=INK, linewidths=0.6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        leg = ax.legend(fontsize=7, loc="best", framealpha=0.0, ncol=2,
                        labelcolor=MUTE)
        self.c_pca.fig.tight_layout(); self.c_pca.draw_idle()

    def _plot_cm(self, res):
        ax = self.c_cm.new_ax()
        cm = res.confusion
        ax.imshow(cm, cmap=CM_CMAP, aspect="auto", vmin=0)
        ax.set_xticks(range(len(res.names))); ax.set_yticks(range(len(res.names)))
        ax.set_xticklabels(res.names, fontsize=7); ax.set_yticklabels(res.names, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        # thin white gridlines between cells (clean on the light card)
        ax.set_xticks(np.arange(-0.5, len(res.names)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(res.names)), minor=True)
        ax.grid(which="minor", color=PANEL, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        thr = cm.max() / 2 if cm.max() else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="#ffffff" if cm[i, j] > thr else INK,
                            fontsize=8)
        self.c_cm.fig.tight_layout(); self.c_cm.draw_idle()

    def _plot_bar(self, res):
        ax = self.c_bar.new_ax()
        names = res.names
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

    def _plot_spec(self, res):
        ax = self.c_spec.new_ax()
        axis = res.axis if res.axis is not None else np.arange(res.templates.shape[1])
        for i, nm in enumerate(res.names):
            ax.plot(axis, res.templates[i] + i * 0.6, lw=1.0,
                    color=SERIES[i % len(SERIES)], label=nm)
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_yticks([])
        self.c_spec.fig.tight_layout(); self.c_spec.draw_idle()


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
                     "loaded maps.  Detailed tools open in their own window.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        self.folder_lbl = QLabel(self._short(self.pest_dir)); self.folder_lbl.setObjectName("field")
        browse = QPushButton("Data folder…"); browse.setObjectName("ghost")
        browse.clicked.connect(self._browse)
        self.status = QLabel(""); self.status.setObjectName("sub")
        mix_btn = QPushButton("Mixture tool"); mix_btn.setObjectName("ghost")
        mix_btn.clicked.connect(lambda: self._launch("sers_app.py"))
        disc_btn = QPushButton("Map tool"); disc_btn.setObjectName("ghost")
        disc_btn.clicked.connect(lambda: self._launch("sers_discriminator_ctk.py"))
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Run analysis"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(browse); ctl.addWidget(self.folder_lbl, 1)
        ctl.addWidget(self.status); ctl.addStretch(1)
        ctl.addWidget(mix_btn); ctl.addWidget(disc_btn)
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
            (self.c_pure, "Single-component confusion  (RF, per pixel)", 0, 1),
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
        d = QFileDialog.getExistingDirectory(self, "Pest_Discriminator folder",
                                             self.pest_dir)
        if d:
            self.pest_dir = d; self.folder_lbl.setText(self._short(d))

    def _launch(self, script):
        try:
            subprocess.Popen([sys.executable, os.path.join(BASE_DIR, script)],
                             cwd=BASE_DIR)
        except Exception as exc:
            print("launch failed:", exc, file=sys.stderr)

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

    # class colours for the 4 reference classes (DQ / THI / TBZ / BLK)
    C4 = [BLUE, TEAL, CORAL, "#98a1ac"]

    def _plot_pca(self, r):
        ax = self.c_pca.new_ax()
        for i, nm in enumerate(r.classes4):
            m = r.pca_lab == i
            if m.any():
                ax.scatter(r.pca_emb[m, 0], r.pca_emb[m, 1], s=10,
                           color=self.C4[i], alpha=0.6, edgecolors="none", label=nm)
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
# Bridge page — launch a customtkinter tool (transitional)
# --------------------------------------------------------------------------
class BridgePage(QWidget):
    def __init__(self, title, desc, script):
        super().__init__()
        self.script = script
        lay = QVBoxLayout(self); lay.setContentsMargins(24, 18, 24, 20)
        h1 = QLabel(title); h1.setObjectName("h1")
        sub = QLabel(desc); sub.setObjectName("sub"); sub.setWordWrap(True)
        lay.addWidget(h1); lay.addWidget(sub); lay.addSpacing(18)

        card, clay = _card("customtkinter tool  ·  Qt port pending")
        row = QHBoxLayout()
        btn = QPushButton(f"Launch {title}"); btn.setObjectName("primary")
        btn.clicked.connect(self._launch)
        row.addWidget(btn); row.addStretch(1)
        clay.addLayout(row)
        note = QLabel("Opens in its own window (separate process) until it is "
                      "rebuilt natively in Qt.")
        note.setObjectName("sub"); note.setWordWrap(True)
        clay.addWidget(note)
        lay.addWidget(card); lay.addStretch(1)

    def _launch(self):
        try:
            subprocess.Popen([sys.executable, os.path.join(BASE_DIR, self.script)],
                             cwd=BASE_DIR)
        except Exception as exc:
            print("launch failed:", exc, file=sys.stderr)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    NAV = [("Model", "model"), ("Quantify", "quant"), ("Discriminator", "real")]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1360, 880)

        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        # top command bar
        bar = QFrame(); bar.setObjectName("topbar"); bar.setFixedHeight(58)
        bl = QHBoxLayout(bar); bl.setContentsMargins(18, 0, 18, 0); bl.setSpacing(10)
        logo = QLabel("U"); logo.setObjectName("logo")
        logo.setFixedSize(30, 30); logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        word = QLabel(APP_NAME); word.setObjectName("wordmark")
        bl.addWidget(logo); bl.addWidget(word); bl.addSpacing(18)

        self._nav_btns = {}
        for label, key in self.NAV:
            b = QPushButton(label); b.setObjectName("nav"); b.setCheckable(True)
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
            "model": ModelPage(),
            "quant": QuantifyPage(),
            "real": RealDataPage(),
        }
        for key in ("model", "quant", "real"):
            self.stack.addWidget(self.pages[key])

        self.select("model")

    def select(self, key):
        for k, b in self._nav_btns.items():
            b.setChecked(k == key)
        self.stack.setCurrentWidget(self.pages[key])


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
