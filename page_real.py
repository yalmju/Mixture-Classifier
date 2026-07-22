"""page_real.py — Real data tab: unmix one test map (background included) and show
it as an RGB intensity composite, an MCR-ALS composite, a per-pixel ratio pie-glyph
map (background greyed), and the overall composition + ratio report."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Wedge, Patch, Rectangle

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QDoubleSpinBox, QFileDialog,
)

from ui_common import *
from unmix import unmix_map
from real_data import PEST_DEFAULT
from dataset import load_preprocess
from io_utils import write_csv

BG_GREY = "#c7ccd3"


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


class RealDataPage(QWidget):
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
                     "substances (background included). Each substance gets its own "
                     "colour: an RGB intensity composite and an MCR-ALS composite "
                     "merge them, a per-pixel pie map shows every pixel's ratio "
                     "(background greyed), and the report gives the overall "
                     "composition. Preprocessing + references come from Samples.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        self.ref_lbl = QLabel(self._short(self.data_dir)); self.ref_lbl.setObjectName("field")
        test_b = QPushButton("Load test map…"); test_b.setObjectName("ghost")
        test_b.clicked.connect(self._browse_test)
        self.test_lbl = QLabel("no test map"); self.test_lbl.setObjectName("field")
        self.test_x = QPushButton("✕"); self.test_x.setObjectName("ghost")
        self.test_x.setFixedWidth(30); self.test_x.setToolTip("clear")
        self.test_x.clicked.connect(self._clear_test); self.test_x.setVisible(False)
        tcol = QVBoxLayout(); tcol.setSpacing(2)
        tl = QLabel("min substance fraction"); tl.setObjectName("field")
        self.thr = QDoubleSpinBox(); self.thr.setDecimals(2); self.thr.setSingleStep(0.05)
        self.thr.setRange(0.01, 0.9); self.thr.setValue(0.05)
        tcol.addWidget(tl); tcol.addWidget(self.thr)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Unmix"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(self.ref_lbl)
        ctl.addWidget(test_b); ctl.addWidget(self.test_lbl); ctl.addWidget(self.test_x)
        ctl.addLayout(tcol); ctl.addStretch(1)
        ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        self.status = QLabel(""); self.status.setObjectName("sub")
        root.addWidget(self.status)

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_dom = Kpi("dominant"); self.k_n = Kpi("substances")
        self.k_hit = Kpi("hit % (not background)"); self.k_px = Kpi("pixels")
        for k in (self.k_dom, self.k_n, self.k_hit, self.k_px):
            kpis.addWidget(k)
        root.addLayout(kpis)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_rgb = Canvas(); self.c_mcr = Canvas()
        self.c_pie = Canvas(); self.c_comp = Canvas()
        for (cv, title, r, c) in [
            (self.c_rgb, "Intensity map (RGB) — substances merged, NNLS", 0, 0),
            (self.c_mcr, "MCR-ALS — refined-spectra composite", 0, 1),
            (self.c_pie, "Per-pixel ratio — pie per pixel, background grey", 1, 0),
            (self.c_comp, "Composition (overall)", 1, 1),
        ]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, r, c)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)
        for cv, m in [(self.c_rgb, "Load a test map, then Unmix"),
                      (self.c_mcr, "MCR-ALS composite appears here"),
                      (self.c_pie, "Per-pixel pie map appears here"),
                      (self.c_comp, "Composition appears here")]:
            cv.placeholder(m)

        self.readout = QLabel(""); self.readout.setObjectName("sub")
        self.readout.setWordWrap(True); self.readout.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self.readout)

    def _short(self, p):
        return "refs (from Samples): " + ("…" + p[-34:] if len(p) > 34 else p)

    def set_data_dir(self, path):
        """Adopt the dataset folder chosen in Samples (single source of truth)."""
        self.data_dir = path; self.ref_lbl.setText(self._short(path))

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
        cfg = load_preprocess(self.data_dir)
        params = dict(data_dir=self.data_dir, test_path=self.test, method="nnls+mcr",
                      baseline=cfg["baseline"], trim=cfg["trim"],
                      min_frac=self.thr.value())
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

    # ---- helpers ----
    def _nb_colors(self, r):
        """One distinct colour per non-background substance."""
        return [SERIES[i % len(SERIES)] for i in range(len(r.nonbg))]

    def _apply(self, r):
        self._res = r
        self.btn.setEnabled(True); self.btn.setText("Unmix")
        self.status.setText("done"); self.status.setStyleSheet(f"color:{MUTE};")
        nb = [r.comps[i] for i in r.nonbg]
        n_det = int(np.sum(r.mean_ratio >= self.thr.value()))
        self.k_dom.set(r.dominant, TEAL)
        self.k_n.set(str(n_det), AMBER)
        self.k_hit.set(f"{r.hit_frac:.0%}", BLUE)
        self.k_px.set(f"{r.n_pixels:,}", PURPLE)
        self._plot_rgb(self.c_rgb, r, r.A, "NNLS")
        if r.A_mcr is not None:
            self._plot_rgb(self.c_mcr, r, r.A_mcr, "MCR-ALS")
        else:
            self.c_mcr.placeholder("MCR-ALS not computed")
        self._plot_pies(r); self._plot_comp(r)
        ratio = "  :  ".join(f"{nm} {r.mean_ratio[i] * 100:.0f}"
                             for i, nm in enumerate(nb))
        self.readout.setText(
            f"<b>hit:</b> {r.hit_frac:.0%} of pixels are a substance (rest background)"
            f" &nbsp;·&nbsp; <b>mean ratio</b> (over hit pixels): {ratio}"
            f" &nbsp;·&nbsp; <b>dominant:</b> {r.dominant}"
            f"<br><span style='color:{FAINT}'>load a dilution-series calibration "
            "(Quantify) to also read approximate concentration — coming from the "
            "same references.</span>")

    # ---- plots ----
    def _plot_rgb(self, canvas, r, A_source, tag):
        ax = canvas.new_ax()
        cols = np.array([to_rgb(c) for c in self._nb_colors(r)])
        Anb = A_source[:, r.nonbg]
        scale = float(np.quantile(Anb.sum(axis=1), 0.99)) or 1.0
        rgb = np.clip((Anb / scale) @ cols, 0.0, 1.0)     # merge substances by colour
        x, y = r.coords[:, 0], r.coords[:, 1]
        ax.scatter(x, y, c=rgb, marker="s", s=16, edgecolors="none")
        ax.set_facecolor("#0f141b")                        # dark = background/no signal
        ax.legend(handles=[Patch(facecolor=self._nb_colors(r)[i], label=r.comps[j])
                           for i, j in enumerate(r.nonbg)],
                  fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=2, loc="upper right")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(tag, fontsize=8, color=INK)
        canvas.fig.tight_layout(); canvas.draw_idle()

    def _plot_pies(self, r):
        ax = self.c_pie.new_ax()
        cols = self._nb_colors(r)
        x, y = r.coords[:, 0], r.coords[:, 1]
        # grid spacing → glyph radius
        ux = np.unique(x); rad = (np.median(np.diff(ux)) * 0.46) if len(ux) > 1 else 0.46
        if r.n_pixels > 2000:                              # too many pies → dominant square
            dom = r.ratio_nb.argmax(axis=1)
            fc = [cols[dom[i]] if r.hit[i] else BG_GREY for i in range(r.n_pixels)]
            ax.scatter(x, y, c=fc, marker="s", s=14, edgecolors="none")
        else:
            for i in range(r.n_pixels):
                if not r.hit[i]:
                    ax.add_patch(Wedge((x[i], y[i]), rad, 0, 360, facecolor=BG_GREY,
                                       edgecolor="none"))
                    continue
                a0 = 90.0
                for k, frac in enumerate(r.ratio_nb[i]):
                    if frac <= 0.002:
                        continue
                    a1 = a0 - frac * 360.0
                    ax.add_patch(Wedge((x[i], y[i]), rad, a1, a0,
                                       facecolor=cols[k], edgecolor="none"))
                    a0 = a1
        ax.set_xlim(x.min() - 1, x.max() + 1); ax.set_ylim(y.max() + 1, y.min() - 1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        handles = [Patch(facecolor=cols[i], label=r.comps[j])
                   for i, j in enumerate(r.nonbg)] + \
                  [Patch(facecolor=BG_GREY, label="background")]
        ax.legend(handles=handles, fontsize=7, framealpha=0.0, labelcolor=MUTE,
                  ncol=2, loc="upper right")
        self.c_pie.fig.tight_layout(); self.c_pie.draw_idle()

    def _plot_comp(self, r):
        ax = self.c_comp.new_ax()
        cols = self._nb_colors(r)
        nb = [r.comps[i] for i in r.nonbg]
        keep = [i for i in range(len(nb)) if r.mean_ratio[i] >= 0.01]
        if not keep:
            keep = [int(r.mean_ratio.argmax())]
        ax.pie([r.mean_ratio[i] for i in keep], labels=[nb[i] for i in keep],
               colors=[cols[i] for i in keep], autopct="%1.0f%%",
               textprops={"fontsize": 8, "color": INK})
        ax.set_aspect("equal")
        ax.set_title(f"hit {r.hit_frac:.0%}", fontsize=8, color=INK)
        self.c_comp.fig.tight_layout(); self.c_comp.draw_idle()

    # ---- export ----
    def _export(self):
        if self._res is None:
            self.status.setText("run first, then export")
            self.status.setStyleSheet(f"color:{RED};"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res; nb = [r.comps[i] for i in r.nonbg]
        write_csv(os.path.join(d, "composition.csv"),
                  ["substance", "mean_ratio"],
                  [[nm, f"{r.mean_ratio[i]:.4f}"] for i, nm in enumerate(nb)])
        head = (["x", "y", "hit"] + [f"ratio_{nm}" for nm in nb]
                + [f"A_{c}" for c in r.comps] + ["reliability_r2", "sam_deg"])
        rows = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}", int(r.hit[i])]
                + [f"{r.ratio_nb[i, k]:.4f}" for k in range(len(nb))]
                + [f"{r.A[i, k]:.5f}" for k in range(len(r.comps))]
                + [f"{r.reliab[i]:.4f}", f"{r.sam[i]:.3f}"]
                for i in range(r.n_pixels)]
        write_csv(os.path.join(d, "per_pixel.csv"), head, rows)
        n = _save_figs([("real_intensity_rgb", self.c_rgb),
                        ("real_mcr_als", self.c_mcr),
                        ("real_pixel_pies", self.c_pie),
                        ("real_composition", self.c_comp)], d)
        self.status.setText(f"exported 2 CSV + {n} PNG → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")
