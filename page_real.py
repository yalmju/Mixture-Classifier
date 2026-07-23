"""page_real.py — Real data tab: unmix one test map (NNLS or MCR-ALS, selectable).
A band-intensity image, a per-pixel composition pie map, the spectrum of a clicked
pixel, and the overall composition. Background is unmixed as its own component."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Wedge, Patch
from matplotlib.collections import PatchCollection

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QFileDialog, QColorDialog,
    QScrollArea, QFrame,
)

from ui_common import *
from unmix import unmix_map
from classify import classify_map
from real_data import PEST_DEFAULT
from dataset import load_preprocess, load_colors, save_colors
from io_utils import write_csv

BG_GREY = "#c7ccd3"
INTEN_CMAP = "magma"


class RealWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params, use_model=False):
        super().__init__()
        self.params = params
        self.use_model = use_model

    def run(self):
        try:
            fn = classify_map if self.use_model else unmix_map
            self.done.emit(fn(progress=self.progress.emit, **self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


class RealDataPage(QWidget):
    METHODS = [("NNLS (fixed refs)", "nnls"), ("MCR-ALS (refine)", "mcr"),
               ("Trained model", "model")]

    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self._sel = None
        self._maps_ax = {}          # axes that accept a pixel click
        self._colors = {}           # per-substance colour override {name: '#hex'}
        self.data_dir = PEST_DEFAULT
        self.test = None
        self.model_path = None      # trained model (unmixr_model.joblib) for classify
        self.calib_path = None      # optional dilution-series calibration CSV → µM
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
        model_b = QPushButton("Load model…"); model_b.setObjectName("ghost")
        model_b.setToolTip("a model exported from the Model tab (unmixr_model.joblib); "
                           "used by the 'Trained model' method")
        model_b.clicked.connect(self._browse_model)
        self.model_lbl = QLabel(""); self.model_lbl.setObjectName("field")
        cal_b = QPushButton("Load calibration…"); cal_b.setObjectName("ghost")
        cal_b.setToolTip("a dilution-series CSV → per-pixel absolute concentration (µM)")
        cal_b.clicked.connect(self._browse_calib)
        self.cal_lbl = QLabel(""); self.cal_lbl.setObjectName("field")
        self.cal_x = QPushButton("✕"); self.cal_x.setObjectName("ghost")
        self.cal_x.setFixedWidth(30); self.cal_x.setToolTip("clear calibration")
        self.cal_x.clicked.connect(self._clear_calib); self.cal_x.setVisible(False)
        self.chk_auto = QCheckBox("auto (BLK)")
        self.chk_auto.setToolTip("threshold-free: a pixel is a substance when its "
                                 "strongest component is a substance (not the learned "
                                 "blank). Unchecked = use the fraction threshold.")
        self.chk_auto.toggled.connect(self._on_auto)
        hitcol = QVBoxLayout(); hitcol.setSpacing(2)
        _hl = QLabel("hit mode"); _hl.setObjectName("field")
        hitcol.addWidget(_hl); hitcol.addWidget(self.chk_auto)
        self.thr = self._spin_col("min substance fraction", QDoubleSpinBox())
        sp = self.thr.itemAt(1).widget()
        sp.setDecimals(2); sp.setSingleStep(0.05); sp.setRange(0.01, 0.9); sp.setValue(0.15)
        sp.setToolTip("a pixel counts as a substance (not background) when the "
                      "substances make up at least this fraction of it — lower to "
                      "catch weaker signal")
        self.chk_flip = QCheckBox("flip Y")
        self.chk_flip.setToolTip("flip the map top-to-bottom if it comes out upside down")
        self.chk_flip.toggled.connect(lambda _=False: self._redraw())
        flipcol = QVBoxLayout(); flipcol.setSpacing(2)
        _fl = QLabel("orientation"); _fl.setObjectName("field")
        flipcol.addWidget(_fl); flipcol.addWidget(self.chk_flip)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Unmix"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(test_b); ctl.addWidget(self.test_lbl); ctl.addWidget(self.test_x)
        ctl.addLayout(self.cmb_method)
        ctl.addWidget(model_b); ctl.addWidget(self.model_lbl)
        ctl.addWidget(cal_b); ctl.addWidget(self.cal_lbl); ctl.addWidget(self.cal_x)
        ctl.addLayout(hitcol); ctl.addLayout(self.thr); ctl.addLayout(flipcol)
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

        # per-substance colour swatches (click to recolour), filled after a result
        self.swatches = QHBoxLayout(); self.swatches.setSpacing(6)
        self.swatches.addWidget(self._mk_lbl("colours:"))
        self.swatches.addStretch(1)
        root.addLayout(self.swatches)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_int = Canvas(); self.c_pie = Canvas()
        self.c_spec = Canvas(); self.c_comp = Canvas()
        for (cv, title, r, c) in [
            (self.c_int, "Merged composite — substances in their colours (click a pixel)", 0, 0),
            (self.c_pie, "Per-pixel composition — pie per pixel (click a pixel)", 0, 1),
            (self.c_spec, "Selected pixel spectrum", 1, 0),
            (self.c_comp, "Composition (overall)", 1, 1),
        ]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, r, c)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        for cv in (self.c_int, self.c_pie):
            cv.setMinimumHeight(300)
        for cv in (self.c_spec, self.c_comp):
            cv.setMinimumHeight(260)
        gridw = QWidget(); gridw.setLayout(grid)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidget(gridw)
        scroll.setStyleSheet("QScrollArea{background:transparent;}")
        root.addWidget(scroll, 1)
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
        self._colors = load_colors(path)                  # remembered colour choices
        if self._res is not None:
            self._rebuild_swatches(self._res); self._redraw()

    def _mk_lbl(self, text):
        lb = QLabel(text); lb.setObjectName("field"); return lb

    def _default_color(self, i):
        return SERIES[i % len(SERIES)]

    def _nb_colors(self, r):
        """Colour per non-background substance — a saved override or the default."""
        out = []
        for i, j in enumerate(r.nonbg):
            out.append(self._colors.get(r.comps[j], self._default_color(i)))
        return out

    def _rebuild_swatches(self, r):
        while self.swatches.count():
            it = self.swatches.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.swatches.addWidget(self._mk_lbl("colours:"))
        cols = self._nb_colors(r)
        for i, j in enumerate(r.nonbg):
            name = r.comps[j]
            b = QPushButton(name); b.setObjectName("ghost"); b.setFixedHeight(24)
            b.setStyleSheet(f"QPushButton{{border:2px solid {cols[i]};"
                            f"border-radius:6px;padding:2px 10px;color:{INK};}}")
            b.clicked.connect(lambda _=False, nm=name: self._pick_color(nm))
            self.swatches.addWidget(b)
        self.swatches.addStretch(1)

    def _pick_color(self, name):
        cur = QColor(self._colors.get(name, "#1a73e8"))
        c = QColorDialog.getColor(cur, self, f"Colour for {name}")
        if not c.isValid():
            return
        self._colors[name] = c.name()
        try:
            save_colors(self.data_dir, self._colors)
        except Exception as exc:
            print("save colors:", exc, file=sys.stderr)
        self._rebuild_swatches(self._res); self._redraw()

    def _redraw(self):
        r = self._res
        if r is None:
            return
        self._plot_intensity(r); self._plot_pies(r); self._plot_comp(r)
        if self._sel is not None:
            self._plot_spec(r, self._sel)

    def _method(self):
        return self.cmb_method.itemAt(1).widget().currentData()

    def _browse_test(self):
        p, _ = QFileDialog.getOpenFileName(self, "Test map", "",
                                           "maps (*.csv *.txt);;all files (*)")
        if p:
            self.test = p; self.test_lbl.setText(os.path.basename(p))
            self.test_x.setVisible(True)

    def _clear_test(self):
        self.test = None; self.test_lbl.setText("no test map"); self.test_x.setVisible(False)

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Trained model (unmixr_model.joblib)",
                                           self.data_dir, "model (*.joblib);;all (*)")
        if p:
            self.model_path = p; self.model_lbl.setText(os.path.basename(p))
            self.cmb_method.itemAt(1).widget().setCurrentIndex(2)   # switch to model

    def _browse_calib(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Calibration spectra CSV (compound, concentration_M, wavenumbers…) "
            "— e.g. calibration_spectra.csv from Quantify Export", "", "CSV (*.csv)")
        if not p:
            return
        problem = self._validate_calib(p)                 # reject fit/curve/wrong CSVs
        if problem:
            self.calib_path = None; self.cal_x.setVisible(False)
            self.cal_lbl.setText("not a calibration"); self.cal_lbl.setStyleSheet(f"color:{RED};")
            self.status.setText(
                f"{os.path.basename(p)} is not a spectra calibration — {problem}. "
                "Use calibration_spectra.csv (from Quantify → Export), not "
                "calibration_fit/curve/stats.csv.")
            self.status.setStyleSheet(f"color:{RED};")
            return
        self.calib_path = p; self.cal_lbl.setText("calib: " + os.path.basename(p))
        self.cal_lbl.setStyleSheet(""); self.cal_x.setVisible(True)

    @staticmethod
    def _validate_calib(path):
        """Return a reason string if `path` is not a per-standard spectra calibration
        (compound, concentration_M, <wavenumbers>), else None."""
        from io_utils import load_calibration_csv
        try:
            axis, names, dils = load_calibration_csv(path)
        except Exception as exc:
            return f"could not read it as spectra ({type(exc).__name__})"
        if len(axis) < 10:
            return "no wavenumber axis (needs many wavenumber columns)"

        def _isnum(s):
            try:
                float(s); return True
            except ValueError:
                return False
        if not names or all(_isnum(n) for n in names):
            return "first column isn't compound names"
        return None

    def _clear_calib(self):
        self.calib_path = None; self.cal_lbl.setText(""); self.cal_x.setVisible(False)

    # ---- run ----
    def _run(self):
        if not self.test:
            self.status.setText("load a test map first")
            self.status.setStyleSheet(f"color:{RED};"); return
        use_model = self._method() == "model"
        if use_model:
            path = self.model_path or os.path.join(self.data_dir, "unmixr_model.joblib")
            if not os.path.exists(path):
                self.status.setText("no trained model — train & Export one in Model, "
                                    "or Load model…")
                self.status.setStyleSheet(f"color:{RED};"); return
            params = dict(model_path=path, test_path=self.test, min_conf=0.0)
        else:
            cfg = load_preprocess(self.data_dir)
            params = dict(data_dir=self.data_dir, test_path=self.test,
                          method=self._method(), baseline=cfg["baseline"],
                          trim=cfg["trim"], min_frac=self.thr_value(),
                          hit_mode="auto" if self.chk_auto.isChecked() else "threshold",
                          calib_path=self.calib_path)
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self.status.setText(""); self.status.setStyleSheet(f"color:{MUTE};")
        self._thread = QThread(); self._worker = RealWorker(params, use_model=use_model)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def thr_value(self):
        return float(self.thr.itemAt(1).widget().value())

    def _on_auto(self, checked):
        self.thr.itemAt(1).widget().setEnabled(not checked)   # threshold unused in auto

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
        nb = [r.comps[i] for i in r.nonbg]
        self.k_dom.set(r.dominant, TEAL)
        self.k_n.set(str(int(np.sum(r.mean_ratio >= 0.05))), AMBER)
        self.k_hit.set(f"{r.hit_frac:.0%}", BLUE)
        self.k_px.set(f"{r.n_pixels:,}", PURPLE)
        self._rebuild_swatches(r)
        self._plot_intensity(r); self._plot_pies(r); self._plot_comp(r)
        self.c_spec.placeholder("click a pixel in a map to see its spectrum")
        ratio = "  :  ".join(f"{nm} {r.mean_ratio[i] * 100:.0f}"
                             for i, nm in enumerate(nb))
        txt = (f"<b>hit:</b> {r.hit_frac:.0%} of pixels are a substance &nbsp;·&nbsp; "
               f"<b>mean ratio</b> (hit pixels): {ratio} &nbsp;·&nbsp; "
               f"<b>dominant:</b> {r.dominant}")
        if getattr(r, "calibrated", False) and r.conc_avg is not None:
            um = r.conc_avg * 1e6
            cs = "  ·  ".join(f"{nm} {um[i]:.3g} µM" for i, nm in enumerate(nb)
                              if np.isfinite(um[i]) and um[i] > 0)
            txt += f"<br><b>mean concentration</b> (hit pixels): {cs}"
            if r.calib_r2 is not None:
                r2s = "  ·  ".join(f"{nm} R²={r.calib_r2[i]:.2f}" for i, nm in enumerate(nb))
                tag = ("  ⚠ low-quality calibration — µM approximate"
                       if float(np.min(r.calib_r2)) < 0.7 else "")
                txt += (f"<br><span style='color:{FAINT}'>calibration fit: {r2s}{tag}"
                        "  ·  click a pixel for its µM</span>")
        self.readout.setText(txt)

    # ---- plots ----
    def _flip(self):
        return self.chk_flip.isChecked()

    def _grid_rc(self, r):
        """Grid row/col index per pixel (rows by ascending Y) + the unique axes."""
        x, y = r.coords[:, 0], r.coords[:, 1]
        ux, uy = np.unique(x), np.unique(y)
        xi = {v: i for i, v in enumerate(ux)}; yi = {v: i for i, v in enumerate(uy)}
        rows = np.array([yi[v] for v in y]); cols = np.array([xi[v] for v in x])
        return rows, cols, len(uy), len(ux), ux, uy

    def _plot_intensity(self, r):
        """Merged false-colour composite: each substance's abundance painted in its
        chosen colour and added together (background = dark), as a pixel image."""
        ax = self.c_int.new_ax(); self._maps_ax["int"] = ax
        cols = np.array([to_rgb(c) for c in self._nb_colors(r)])   # (Knb, 3)
        Anb = r.A[:, r.nonbg]
        scale = float(np.quantile(Anb.sum(axis=1), 0.99)) or 1.0
        rgb = np.clip((Anb / scale) @ cols, 0.0, 1.0)              # (n, 3)
        rows, cc, ny, nx, ux, uy = self._grid_rc(r)
        img = np.zeros((ny, nx, 3))                                # dark background
        img[rows, cc] = rgb
        if self._flip():                                           # Y downwards
            origin, extent = "upper", [ux.min() - .5, ux.max() + .5,
                                       uy.max() + .5, uy.min() - .5]
        else:                                                      # Y upwards (default)
            origin, extent = "lower", [ux.min() - .5, ux.max() + .5,
                                       uy.min() - .5, uy.max() + .5]
        ax.imshow(img, extent=extent, origin=origin, aspect="equal",
                  interpolation="nearest")
        ax.legend(handles=[Patch(facecolor=self._nb_colors(r)[i], label=r.comps[j])
                           for i, j in enumerate(r.nonbg)],
                  fontsize=7, framealpha=0.0, labelcolor=MUTE,
                  loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=len(r.nonbg), frameon=False)
        self._mark_sel(ax, r)
        ax.set_xticks([]); ax.set_yticks([])
        self.c_int.fig.tight_layout(); self.c_int.draw_idle()

    def _plot_pies(self, r):
        ax = self.c_pie.new_ax(); self._maps_ax["pie"] = ax
        cols = self._nb_colors(r)
        x, y = r.coords[:, 0], r.coords[:, 1]
        ux = np.unique(x); rad = (np.median(np.diff(ux)) * 0.46) if len(ux) > 1 else 0.46
        hit = r.hit
        # background / non-hit pixels: one fast scatter (not one patch each)
        if (~hit).any():
            ax.scatter(x[~hit], y[~hit], c=BG_GREY, marker="s", s=16, edgecolors="none")
        if r.method == "model":                           # classifier → one class/pixel
            dom = r.ratio_nb.argmax(axis=1)
            if hit.any():
                ax.scatter(x[hit], y[hit], c=[cols[dom[i]] for i in np.where(hit)[0]],
                           marker="s", s=16, edgecolors="none")
            ax.set_title("predicted class per pixel (not a mixture ratio)",
                         fontsize=8, color=INK)
        else:                                             # per-pixel pie for hit pixels
            wedges, wcols = [], []
            for i in np.where(hit)[0]:
                a0 = 90.0
                for k, frac in enumerate(r.ratio_nb[i]):
                    if frac <= 0.002:
                        continue
                    a1 = a0 - frac * 360.0
                    wedges.append(Wedge((x[i], y[i]), rad, a1, a0)); wcols.append(cols[k])
                    a0 = a1
            if wedges:
                ax.add_collection(PatchCollection(wedges, facecolors=wcols,
                                                  edgecolors="none"))
        self._mark_sel(ax, r)
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.set_ylim(*((y.max() + 1, y.min() - 1) if self._flip()
                      else (y.min() - 1, y.max() + 1)))
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
        mm = meas.max() or 1.0
        ax.plot(axis, meas / mm, lw=1.3, color=INK, label="measured")
        if r.templates is not None:                       # NNLS/MCR: overlay the fit
            recon = r.A[i] @ r.templates
            ax.plot(axis, recon / (recon.max() or 1.0), lw=1.1, color=TEAL,
                    ls="--", label="reconstructed")
        xp, yp = r.coords[i]
        rat = "  ·  ".join(f"{r.comps[j]} {r.ratio_nb[i, k] * 100:.0f}%"
                           for k, j in enumerate(r.nonbg) if r.ratio_nb[i, k] > 0.02)
        tag = rat if r.hit[i] else "background"
        if getattr(r, "conc", None) is not None and r.hit[i]:   # absolute µM per pixel
            um = r.conc[i] * 1e6
            cs = "  ·  ".join(f"{r.comps[j]} {um[k]:.3g}µM" for k, j in enumerate(r.nonbg)
                              if np.isfinite(um[k]) and um[k] > 0)
            sat = ("  ⚠sat" if r.pp_theta is not None and r.pp_theta[i] > 0.85 else "")
            if cs:
                tag += f"  |  {cs}{sat}"
        ax.set_title(f"pixel ({xp:.0f}, {yp:.0f}) — {tag}", fontsize=8, color=INK)
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_yticks([])
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE,
                  loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
        self.c_spec.fig.tight_layout(); self.c_spec.draw_idle()

    # ---- interaction ----
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
        inten = r.spectra.sum(axis=1)                      # total baseline-removed signal
        cal = getattr(r, "conc", None) is not None
        head = (["x", "y", "hit", "total_intensity"]
                + [f"ratio_{nm}" for nm in nb] + [f"A_{c}" for c in r.comps]
                + ([f"conc_uM_{nm}" for nm in nb] if cal else []) + ["reliability_r2"])
        rows = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}", int(r.hit[i]),
                 f"{inten[i]:.4f}"]
                + [f"{r.ratio_nb[i, k]:.4f}" for k in range(len(nb))]
                + [f"{r.A[i, k]:.5f}" for k in range(len(r.comps))]
                + ([f"{r.conc[i, k] * 1e6:.4g}" for k in range(len(nb))] if cal else [])
                + [f"{r.reliab[i]:.4f}"] for i in range(r.n_pixels)]
        write_csv(os.path.join(d, "per_pixel.csv"), head, rows)
        n = _save_figs([("real_intensity", self.c_int), ("real_composition_pies", self.c_pie),
                        ("real_pixel_spectrum", self.c_spec),
                        ("real_composition", self.c_comp)], d)
        self.status.setText(f"exported 2 CSV + {n} PNG → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")
