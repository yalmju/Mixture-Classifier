"""page_real.py — Real data tab: unmix one test map (NNLS or MCR-ALS, selectable).
A band-intensity image, a per-pixel composition pie map, the spectrum of a clicked
pixel, and the overall composition. Background is unmixed as its own component."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from matplotlib.patches import Wedge, Patch

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog,
)

from ui_common import *
from unmix import unmix_map
from real_data import PEST_DEFAULT
from dataset import load_preprocess
from io_utils import write_csv

BG_GREY = "#c7ccd3"
INTEN_CMAP = "magma"


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
    METHODS = [("NNLS (fixed refs)", "nnls"), ("MCR-ALS (refine)", "mcr")]

    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self._sel = None
        self._maps_ax = {}          # axes that accept a pixel click
        self.data_dir = PEST_DEFAULT
        self.test = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(12)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Real-data analysis — unmix a test map"); h1.setObjectName("h1")
        sub = QLabel("Unmix one test map against your references (background "
                     "included) by NNLS or MCR-ALS. A band-intensity image, a "
                     "per-pixel composition pie map, and the overall composition — "
                     "click any pixel to see its spectrum. References + preprocessing "
                     "come from Samples.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        test_b = QPushButton("Load test map…"); test_b.setObjectName("ghost")
        test_b.clicked.connect(self._browse_test)
        self.test_lbl = QLabel("no test map"); self.test_lbl.setObjectName("field")
        self.test_x = QPushButton("✕"); self.test_x.setObjectName("ghost")
        self.test_x.setFixedWidth(30); self.test_x.setToolTip("clear")
        self.test_x.clicked.connect(self._clear_test); self.test_x.setVisible(False)
        self.cmb_method = self._combo("method", self.METHODS)
        self.sp_lo = QSpinBox(); self.sp_lo.setRange(0, 4000); self.sp_lo.setSingleStep(50)
        self.sp_hi = QSpinBox(); self.sp_hi.setRange(0, 4000); self.sp_hi.setSingleStep(50)
        self.sp_hi.setValue(4000)
        self.sp_lo.valueChanged.connect(self._reband)
        self.sp_hi.valueChanged.connect(self._reband)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Unmix"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(test_b); ctl.addWidget(self.test_lbl); ctl.addWidget(self.test_x)
        ctl.addLayout(self.cmb_method)
        ctl.addLayout(self._spin_col("band lo cm⁻¹", self.sp_lo))
        ctl.addLayout(self._spin_col("band hi cm⁻¹", self.sp_hi))
        ctl.addStretch(1)
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
        self.c_int = Canvas(); self.c_pie = Canvas()
        self.c_spec = Canvas(); self.c_comp = Canvas()
        for (cv, title, r, c) in [
            (self.c_int, "Band-intensity map (click a pixel)", 0, 0),
            (self.c_pie, "Per-pixel composition — pie per pixel (click a pixel)", 0, 1),
            (self.c_spec, "Selected pixel spectrum", 1, 0),
            (self.c_comp, "Composition (overall)", 1, 1),
        ]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, r, c)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)
        for cv, m in [(self.c_int, "Load a test map, then Unmix"),
                      (self.c_pie, "Composition appears here"),
                      (self.c_spec, "Click a pixel in a map to see its spectrum"),
                      (self.c_comp, "Composition appears here")]:
            cv.placeholder(m)
        self.c_int.mpl_connect("button_press_event", self._on_click)
        self.c_pie.mpl_connect("button_press_event", self._on_click)

        self.readout = QLabel(""); self.readout.setObjectName("sub")
        self.readout.setWordWrap(True); self.readout.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self.readout)

    # ---- small builders ----
    def _combo(self, label, items):
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        cb = QComboBox()
        for t, d in items:
            cb.addItem(t, d)
        col.addWidget(lb); col.addWidget(cb); self._last_combo = cb
        return col

    def _spin_col(self, label, spin):
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        col.addWidget(lb); col.addWidget(spin)
        return col

    def set_data_dir(self, path):
        self.data_dir = path                              # references come from Samples

    def _method(self):
        return self.cmb_method.itemAt(1).widget().currentData()

    def _browse_test(self):
        p, _ = QFileDialog.getOpenFileName(self, "Test map CSV", "", "CSV (*.csv)")
        if p:
            self.test = p; self.test_lbl.setText(os.path.basename(p))
            self.test_x.setVisible(True)

    def _clear_test(self):
        self.test = None; self.test_lbl.setText("no test map"); self.test_x.setVisible(False)

    def _nb_colors(self, r):
        return [SERIES[i % len(SERIES)] for i in range(len(r.nonbg))]

    # ---- run ----
    def _run(self):
        if not self.test:
            self.status.setText("load a test map first")
            self.status.setStyleSheet(f"color:{RED};"); return
        cfg = load_preprocess(self.data_dir)
        params = dict(data_dir=self.data_dir, test_path=self.test,
                      method=self._method(), baseline=cfg["baseline"],
                      trim=cfg["trim"], min_frac=self.thr_value())
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

    def thr_value(self):
        return 0.05

    def _progress(self, msg):
        self.btn.setText("Unmixing…"); self.status.setText("● " + msg)

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Unmix")
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, r):
        self._res = r; self._sel = None
        self.btn.setEnabled(True); self.btn.setText("Unmix")
        self.status.setText(f"done — {r.method.upper()}")
        self.status.setStyleSheet(f"color:{MUTE};")
        # default the band spins to the full available range on first result
        if self.sp_lo.value() == 0 and self.sp_hi.value() == 4000 and r.wn is not None:
            self.sp_lo.blockSignals(True); self.sp_hi.blockSignals(True)
            self.sp_lo.setValue(int(r.wn.min())); self.sp_hi.setValue(int(r.wn.max()))
            self.sp_lo.blockSignals(False); self.sp_hi.blockSignals(False)
        nb = [r.comps[i] for i in r.nonbg]
        self.k_dom.set(r.dominant, TEAL)
        self.k_n.set(str(int(np.sum(r.mean_ratio >= 0.05))), AMBER)
        self.k_hit.set(f"{r.hit_frac:.0%}", BLUE)
        self.k_px.set(f"{r.n_pixels:,}", PURPLE)
        self._plot_intensity(r); self._plot_pies(r); self._plot_comp(r)
        self.c_spec.placeholder("click a pixel in a map to see its spectrum")
        ratio = "  :  ".join(f"{nm} {r.mean_ratio[i] * 100:.0f}"
                             for i, nm in enumerate(nb))
        self.readout.setText(
            f"<b>hit:</b> {r.hit_frac:.0%} of pixels are a substance &nbsp;·&nbsp; "
            f"<b>mean ratio</b> (hit pixels): {ratio} &nbsp;·&nbsp; "
            f"<b>dominant:</b> {r.dominant}")

    # ---- plots ----
    def _band_mask(self, r):
        lo, hi = self.sp_lo.value(), self.sp_hi.value()
        m = (r.wn >= lo) & (r.wn <= hi)
        return m if m.sum() >= 1 else np.ones(len(r.wn), bool)

    def _plot_intensity(self, r):
        ax = self.c_int.new_ax(); self._maps_ax["int"] = ax
        inten = r.spectra[:, self._band_mask(r)].sum(axis=1)
        x, y = r.coords[:, 0], r.coords[:, 1]
        vmax = float(np.quantile(inten, 0.99)) or 1.0
        sc = ax.scatter(x, y, c=inten, cmap=INTEN_CMAP, marker="s", s=16,
                        vmin=0, vmax=vmax, edgecolors="none")
        self.c_int.fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        self._mark_sel(ax, r)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Σ intensity  {self.sp_lo.value()}–{self.sp_hi.value()} cm⁻¹",
                     fontsize=8, color=INK)
        self.c_int.fig.tight_layout(); self.c_int.draw_idle()

    def _plot_pies(self, r):
        ax = self.c_pie.new_ax(); self._maps_ax["pie"] = ax
        cols = self._nb_colors(r)
        x, y = r.coords[:, 0], r.coords[:, 1]
        ux = np.unique(x); rad = (np.median(np.diff(ux)) * 0.46) if len(ux) > 1 else 0.46
        if r.n_pixels > 2000:
            dom = r.ratio_nb.argmax(axis=1)
            fc = [cols[dom[i]] if r.hit[i] else BG_GREY for i in range(r.n_pixels)]
            ax.scatter(x, y, c=fc, marker="s", s=14, edgecolors="none")
        else:
            for i in range(r.n_pixels):
                if not r.hit[i]:
                    ax.add_patch(Wedge((x[i], y[i]), rad, 0, 360, facecolor=BG_GREY,
                                       edgecolor="none")); continue
                a0 = 90.0
                for k, frac in enumerate(r.ratio_nb[i]):
                    if frac <= 0.002:
                        continue
                    a1 = a0 - frac * 360.0
                    ax.add_patch(Wedge((x[i], y[i]), rad, a1, a0, facecolor=cols[k],
                                       edgecolor="none"))
                    a0 = a1
        self._mark_sel(ax, r)
        ax.set_xlim(x.min() - 1, x.max() + 1); ax.set_ylim(y.max() + 1, y.min() - 1)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        # legend OUTSIDE the map (below), so it never covers the pixels
        handles = [Patch(facecolor=cols[i], label=r.comps[j])
                   for i, j in enumerate(r.nonbg)] + \
                  [Patch(facecolor=BG_GREY, label="background")]
        ax.legend(handles=handles, fontsize=7, framealpha=0.0, labelcolor=MUTE,
                  loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=len(handles), frameon=False)
        self.c_pie.fig.tight_layout(); self.c_pie.draw_idle()

    def _mark_sel(self, ax, r):
        if self._sel is not None:
            ax.scatter([r.coords[self._sel, 0]], [r.coords[self._sel, 1]], s=120,
                       facecolors="none", edgecolors=BLUE, linewidths=1.8, zorder=6)

    def _plot_comp(self, r):
        ax = self.c_comp.new_ax()
        cols = self._nb_colors(r); nb = [r.comps[i] for i in r.nonbg]
        keep = [i for i in range(len(nb)) if r.mean_ratio[i] >= 0.01] or \
               [int(r.mean_ratio.argmax())]
        ax.pie([r.mean_ratio[i] for i in keep], labels=[nb[i] for i in keep],
               colors=[cols[i] for i in keep], autopct="%1.0f%%",
               textprops={"fontsize": 8, "color": INK})
        ax.set_aspect("equal")
        ax.set_title(f"hit {r.hit_frac:.0%}", fontsize=8, color=INK)
        self.c_comp.fig.tight_layout(); self.c_comp.draw_idle()

    def _plot_spec(self, r, i):
        ax = self.c_spec.new_ax()
        axis = r.wn if r.wn is not None else np.arange(r.spectra.shape[1])
        meas = np.asarray(r.spectra[i], float)
        recon = r.A[i] @ r.templates
        mm = meas.max() or 1.0; rm = recon.max() or 1.0
        ax.plot(axis, meas / mm, lw=1.3, color=INK, label="measured")
        ax.plot(axis, recon / rm, lw=1.1, color=TEAL, ls="--", label="reconstructed")
        # shade the intensity band that the map integrates
        ax.axvspan(self.sp_lo.value(), self.sp_hi.value(), color=AMBER, alpha=0.10)
        xp, yp = r.coords[i]
        rat = "  ·  ".join(f"{r.comps[j]} {r.ratio_nb[i, k] * 100:.0f}%"
                           for k, j in enumerate(r.nonbg) if r.ratio_nb[i, k] > 0.02)
        tag = rat if r.hit[i] else "background"
        ax.set_title(f"pixel ({xp:.0f}, {yp:.0f}) — {tag}", fontsize=8, color=INK)
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_yticks([])
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
        self.c_spec.fig.tight_layout(); self.c_spec.draw_idle()

    # ---- interaction ----
    def _reband(self):
        if self._res is not None:
            self._plot_intensity(self._res)
            if self._sel is not None:
                self._plot_spec(self._res, self._sel)

    def _on_click(self, event):
        r = self._res
        if r is None or event.xdata is None or event.inaxes not in self._maps_ax.values():
            return
        d = ((r.coords[:, 0] - event.xdata) ** 2
             + (r.coords[:, 1] - event.ydata) ** 2)
        self._sel = int(d.argmin())
        self._plot_spec(r, self._sel)
        self._plot_intensity(r); self._plot_pies(r)     # redraw with the highlight ring

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
        band = self._band_mask(r); inten = r.spectra[:, band].sum(axis=1)
        head = (["x", "y", "hit", f"intensity_{self.sp_lo.value()}_{self.sp_hi.value()}"]
                + [f"ratio_{nm}" for nm in nb] + [f"A_{c}" for c in r.comps]
                + ["reliability_r2"])
        rows = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}", int(r.hit[i]),
                 f"{inten[i]:.4f}"]
                + [f"{r.ratio_nb[i, k]:.4f}" for k in range(len(nb))]
                + [f"{r.A[i, k]:.5f}" for k in range(len(r.comps))]
                + [f"{r.reliab[i]:.4f}"] for i in range(r.n_pixels)]
        write_csv(os.path.join(d, "per_pixel.csv"), head, rows)
        n = _save_figs([("real_intensity", self.c_int), ("real_composition_pies", self.c_pie),
                        ("real_pixel_spectrum", self.c_spec),
                        ("real_composition", self.c_comp)], d)
        self.status.setText(f"exported 2 CSV + {n} PNG → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")
