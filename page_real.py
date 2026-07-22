"""page_real.py — Real data tab: unmix one test map (NNLS / MCR-ALS) and show it
as per-substance intensity maps, a per-pixel reliability map, a measured-vs-
reconstructed validation plot, and the overall composition pie."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QComboBox, QCheckBox, QDoubleSpinBox, QFileDialog,
)

from ui_common import *
from unmix import unmix_map
from real_data import PEST_DEFAULT
from io_utils import write_csv


class RealWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.done.emit(unmix_map(progress=self.progress.emit, **self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


# --------------------------------------------------------------------------
# Real-data page
# --------------------------------------------------------------------------
class RealDataPage(QWidget):
    METHODS = [("NNLS (fixed refs)", "nnls"), ("MCR-ALS (refine)", "mcr")]

    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self.data_dir = PEST_DEFAULT
        self.test = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(12)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Real-data analysis — unmix a test map"); h1.setObjectName("h1")
        sub = QLabel("Load one test map and unmix it against your reference "
                     "substances (NNLS or MCR-ALS). See where each substance is "
                     "(intensity maps), how well the fit explains each pixel "
                     "(reliability), the measured vs reconstructed spectrum "
                     "(validation), and the overall composition (pie). Organise "
                     "references in Samples first.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        ref_b = QPushButton("Reference data…"); ref_b.setObjectName("ghost")
        ref_b.clicked.connect(self._browse_ref)
        self.ref_lbl = QLabel(self._short(self.data_dir)); self.ref_lbl.setObjectName("field")
        test_b = QPushButton("Load test map…"); test_b.setObjectName("ghost")
        test_b.clicked.connect(self._browse_test)
        self.test_lbl = QLabel("no test map"); self.test_lbl.setObjectName("field")
        self.test_x = QPushButton("✕"); self.test_x.setObjectName("ghost")
        self.test_x.setFixedWidth(30); self.test_x.setToolTip("clear")
        self.test_x.clicked.connect(self._clear_test); self.test_x.setVisible(False)
        self.cmb_method = QComboBox()
        for text, data in self.METHODS:
            self.cmb_method.addItem(text, data)
        bcol = QVBoxLayout(); bcol.setSpacing(2)
        bl = QLabel("baseline"); bl.setObjectName("field")
        self.chk_base = QCheckBox("ALS on"); self.chk_base.setChecked(True)
        bcol.addWidget(bl); bcol.addWidget(self.chk_base)
        tcol = QVBoxLayout(); tcol.setSpacing(2)
        tl = QLabel("min fraction"); tl.setObjectName("field")
        self.thr = QDoubleSpinBox(); self.thr.setDecimals(2); self.thr.setSingleStep(0.05)
        self.thr.setRange(0.01, 0.9); self.thr.setValue(0.05)
        tcol.addWidget(tl); tcol.addWidget(self.thr)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Unmix"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(ref_b); ctl.addWidget(self.ref_lbl)
        ctl.addWidget(test_b); ctl.addWidget(self.test_lbl); ctl.addWidget(self.test_x, 0)
        ctl.addWidget(self.cmb_method); ctl.addLayout(bcol); ctl.addLayout(tcol)
        ctl.addStretch(1)
        ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        self.status = QLabel(""); self.status.setObjectName("sub")
        root.addWidget(self.status)

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_dom = Kpi("dominant"); self.k_n = Kpi("components")
        self.k_r2 = Kpi("mean fit R²"); self.k_px = Kpi("pixels")
        for k in (self.k_dom, self.k_n, self.k_r2, self.k_px):
            kpis.addWidget(k)
        root.addLayout(kpis)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_maps = Canvas(); self.c_rel = Canvas()
        self.c_val = Canvas(); self.c_pie = Canvas()
        for (cv, title, r, c) in [
            (self.c_maps, "Intensity maps — where each substance is", 0, 0),
            (self.c_rel, "Reliability — per-pixel fit R²", 0, 1),
            (self.c_val, "Validation — measured vs reconstructed", 1, 0),
            (self.c_pie, "Composition (overall)", 1, 1),
        ]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, r, c)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)
        for cv, m in [(self.c_maps, "Load a test map, then Unmix"),
                      (self.c_rel, "Reliability appears here"),
                      (self.c_val, "Validation appears here"),
                      (self.c_pie, "Composition appears here")]:
            cv.placeholder(m)

    def _short(self, p):
        return "refs: " + ("…" + p[-38:] if len(p) > 38 else p)

    def _browse_ref(self):
        d = QFileDialog.getExistingDirectory(self, "Reference data folder (Samples)",
                                             self.data_dir)
        if d:
            self.data_dir = d; self.ref_lbl.setText(self._short(d))

    def _browse_test(self):
        p, _ = QFileDialog.getOpenFileName(self, "Test map CSV", "", "CSV (*.csv)")
        if p:
            self.test = p; self.test_lbl.setText(os.path.basename(p))
            self.test_x.setVisible(True)

    def _clear_test(self):
        self.test = None; self.test_lbl.setText("no test map"); self.test_x.setVisible(False)

    # ---- run ----
    def _run(self):
        if not self.test:
            self.status.setText("load a test map first")
            self.status.setStyleSheet(f"color:{RED};"); return
        params = dict(data_dir=self.data_dir, test_path=self.test,
                      method=self.cmb_method.currentData(),
                      baseline=self.chk_base.isChecked())
        self.btn.setEnabled(False); self.btn.setText("Unmixing…")
        self.status.setText(""); self.status.setStyleSheet(f"color:{MUTE};")
        self._thread = QThread(); self._worker = RealWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _progress(self, msg):
        self.btn.setText("Unmixing…"); self.status.setText("● " + msg)

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Unmix")
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, r):
        self._res = r
        self.btn.setEnabled(True); self.btn.setText("Unmix")
        self.status.setText(f"done — {r.method.upper()}")
        self.status.setStyleSheet(f"color:{MUTE};")
        thr = self.thr.value()
        n_det = int(np.sum(r.comp_frac >= thr))
        self.k_dom.set(r.dominant, TEAL)
        self.k_n.set(str(n_det), AMBER)
        self.k_r2.set(f"{r.mean_r2:.2f}", BLUE)
        self.k_px.set(f"{r.n_pixels:,}", PURPLE)
        self._plot_maps(r); self._plot_rel(r); self._plot_val(r); self._plot_pie(r)

    # ---- plots ----
    def _plot_maps(self, r):
        self.c_maps.fig.clear()
        K = len(r.comps)
        x, y = r.coords[:, 0], r.coords[:, 1]
        for k in range(K):
            ax = self.c_maps.style(self.c_maps.fig.add_subplot(1, K, k + 1))
            a = r.A[:, k]
            vmax = float(np.quantile(a, 0.99)) if a.size else 1.0
            ax.scatter(x, y, c=a, cmap=CM_CMAP, marker="s", s=14,
                       vmin=0, vmax=vmax if vmax > 0 else 1.0, edgecolors="none")
            ax.set_title(r.comps[k], fontsize=8, color=SERIES[k % len(SERIES)])
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        self.c_maps.fig.tight_layout(); self.c_maps.draw_idle()

    def _plot_rel(self, r):
        ax = self.c_rel.new_ax()
        x, y = r.coords[:, 0], r.coords[:, 1]
        sc = ax.scatter(x, y, c=r.reliab, cmap=CM_CMAP, marker="s", s=16,
                        vmin=0, vmax=1, edgecolors="none")
        self.c_rel.fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"mean R² {r.mean_r2:.2f}  ·  mean SAM {np.mean(r.sam):.1f}°",
                     fontsize=8, color=INK)
        self.c_rel.fig.tight_layout(); self.c_rel.draw_idle()

    def _plot_val(self, r):
        ax = self.c_val.new_ax()
        axis = r.wn if r.wn is not None else np.arange(r.meas_mean.shape[0])
        ax.plot(axis, r.meas_mean, lw=1.4, color=INK, label="measured")
        ax.plot(axis, r.recon_mean, lw=1.2, color=TEAL, ls="--", label="reconstructed")
        ax.fill_between(axis, r.meas_mean - r.recon_mean, color=CORAL, alpha=0.25,
                        label="residual")
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_yticks([])
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=3)
        self.c_val.fig.tight_layout(); self.c_val.draw_idle()

    def _plot_pie(self, r):
        ax = self.c_pie.new_ax()
        thr = self.thr.value()
        keep = [i for i in range(len(r.comps)) if r.comp_frac[i] >= thr]
        if not keep:
            keep = [int(r.comp_frac.argmax())]
        vals = [r.comp_frac[i] for i in keep]
        labels = [r.comps[i] for i in keep]
        cols = [SERIES[i % len(SERIES)] for i in keep]
        ax.pie(vals, labels=labels, colors=cols, autopct="%1.0f%%",
               textprops={"fontsize": 8, "color": INK})
        ax.set_aspect("equal")
        self.c_pie.fig.tight_layout(); self.c_pie.draw_idle()

    # ---- export ----
    def _export(self):
        if self._res is None:
            self.status.setText("run first, then export")
            self.status.setStyleSheet(f"color:{RED};"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res
        write_csv(os.path.join(d, "composition.csv"),
                  ["substance", "fraction"],
                  [[c, f"{r.comp_frac[i]:.4f}"] for i, c in enumerate(r.comps)])
        head = ["x", "y"] + [f"A_{c}" for c in r.comps] + ["reliability_r2", "sam_deg"]
        rows = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}"]
                + [f"{r.A[i, k]:.5f}" for k in range(len(r.comps))]
                + [f"{r.reliab[i]:.4f}", f"{r.sam[i]:.3f}"]
                for i in range(r.n_pixels)]
        write_csv(os.path.join(d, "per_pixel.csv"), head, rows)
        n = _save_figs([("real_intensity_maps", self.c_maps),
                        ("real_reliability", self.c_rel),
                        ("real_validation", self.c_val),
                        ("real_composition_pie", self.c_pie)], d)
        self.status.setText(f"exported 2 CSV + {n} PNG → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")
