"""page_predict.py — Predict tab: an unknown sample -> its component ratio."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QDoubleSpinBox, QCheckBox, QFileDialog,
)

from ui_common import *
from predict import predict_sample
from real_data import PEST_DEFAULT
from dataset import load_preprocess
from io_utils import load_calibration_csv


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


# --------------------------------------------------------------------------
# Predict page
# --------------------------------------------------------------------------
class PredictPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self._map_ax = None        # the RGB-map axes (for click hit-testing)
        self._sel = None           # index of the clicked pixel, if any
        self.data_dir = PEST_DEFAULT
        self.sample = None
        self.calib_path = None     # optional dilution-series calibration CSV
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(14)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Predict — sample composition"); h1.setObjectName("h1")
        sub = QLabel("Load one unknown map and read the per-pixel ratio of your "
                     "reference substances (RGB composite — click a pixel for its "
                     "pie). Add a dilution-series calibration to also read absolute "
                     "concentration (µM) per pixel. Organise references in Samples first.")
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
        self.samp_x = self._clear_btn(self._clear_sample)
        cal_b = QPushButton("Load calibration…"); cal_b.setObjectName("ghost")
        cal_b.clicked.connect(self._browse_calib)
        self.cal_lbl = QLabel("no calib (ratio only)"); self.cal_lbl.setObjectName("field")
        self.cal_x = self._clear_btn(self._clear_calib)
        tcol = QVBoxLayout(); tcol.setSpacing(2)
        tl = QLabel("threshold"); tl.setObjectName("field")
        self.thr = QDoubleSpinBox(); self.thr.setDecimals(2); self.thr.setSingleStep(0.05)
        self.thr.setRange(0.05, 0.9); self.thr.setValue(0.30)
        tcol.addWidget(tl); tcol.addWidget(self.thr)
        self.btn = QPushButton("Predict"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(ref_b); ctl.addWidget(self.ref_lbl)
        ctl.addWidget(samp_b); ctl.addWidget(self.samp_lbl); ctl.addWidget(self.samp_x)
        ctl.addWidget(cal_b); ctl.addWidget(self.cal_lbl, 1); ctl.addWidget(self.cal_x)
        ctl.addLayout(tcol); ctl.addWidget(self.btn)
        root.addLayout(ctl)
        self._sync_clear()

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_dom = Kpi("dominant"); self.k_domp = Kpi("dominant %")
        self.k_n = Kpi("components"); self.k_px = Kpi("pixels")
        for k in (self.k_dom, self.k_domp, self.k_n, self.k_px):
            kpis.addWidget(k)
        root.addLayout(kpis)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_ratio = Canvas(); self.c_map = Canvas(); self.c_spec = Canvas()
        for cv, title, c in [(self.c_ratio, "Composition ratio (per-pixel NNLS)", 0),
                             (self.c_map, "RGB composite — click a pixel for its ratio", 1)]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, 0, c)
        card, lay = _card("Sample vs reference templates")
        lay.addWidget(self.c_spec); grid.addWidget(card, 1, 0, 1, 2)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        root.addLayout(grid, 1)
        for cv in (self.c_ratio, self.c_map, self.c_spec):
            cv.placeholder("Load a sample, then Predict")
        self.c_map.mpl_connect("button_press_event", self._on_click)

        self.readout = QLabel(""); self.readout.setObjectName("sub")
        self.readout.setWordWrap(True)
        root.addWidget(self.readout)

    def _short(self, p):
        return "refs: " + ("…" + p[-38:] if len(p) > 38 else p)

    def _clear_btn(self, on_click):
        """A small ✕ button that clears a loaded file (undo a wrong pick)."""
        b = QPushButton("✕"); b.setObjectName("ghost")
        b.setFixedWidth(30); b.setToolTip("clear")
        b.clicked.connect(on_click)
        return b

    def _sync_clear(self):
        """Show a ✕ only when there is something to clear."""
        self.samp_x.setVisible(self.sample is not None)
        self.cal_x.setVisible(self.calib_path is not None)

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
            self._sync_clear()

    def _clear_sample(self):
        self.sample = None; self.samp_lbl.setText("no sample"); self._sync_clear()

    def _browse_calib(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Calibration CSV (compound, concentration_M, wavenumbers…)", "",
            "CSV (*.csv)")
        if not p:
            return
        problem = self._validate_calib(p)                 # reject a wrong file early
        if problem:
            self.calib_path = None
            self.cal_lbl.setText("not a calibration CSV")
            self.cal_lbl.setStyleSheet(f"color:{CORAL};")
            self.readout.setText(
                f"<b>calibration not loaded</b> — {os.path.basename(p)}: {problem}"
                "<br>expected format: <code>compound, concentration_M, &lt;wn1&gt;, "
                "&lt;wn2&gt;, …</code> (one row per measured standard). A map CSV is "
                "not a calibration file.")
            self.readout.setTextFormat(Qt.TextFormat.RichText)
            self._sync_clear()
            return
        self.calib_path = p
        self.cal_lbl.setText("calib: " + os.path.basename(p))
        self.cal_lbl.setStyleSheet("")
        self._sync_clear()

    def _clear_calib(self):
        self.calib_path = None
        self.cal_lbl.setText("no calib (ratio only)"); self.cal_lbl.setStyleSheet("")
        self._sync_clear()

    @staticmethod
    def _validate_calib(path):
        """Return a reason string if `path` is not a usable calibration CSV, else
        None. Guards against loading a map file (or any wrong CSV) by mistake."""
        try:
            axis, names, dils = load_calibration_csv(path)
        except Exception as exc:
            return f"could not read it ({type(exc).__name__})"
        if len(axis) < 10:
            return "no wavenumber axis found in the header"

        def _isnum(s):
            try:
                float(s); return True
            except ValueError:
                return False
        if not names or all(_isnum(n) for n in names):
            return "first column has no compound names (looks like a map, not a calibration)"
        return None

    def _run(self):
        if not self.sample:
            self.readout.setText("load a sample first"); return
        cfg = load_preprocess(self.data_dir)              # preprocessing set in Samples
        params = dict(data_dir=self.data_dir, sample_path=self.sample,
                      threshold=self.thr.value(), baseline=cfg["baseline"],
                      trim=cfg["trim"], calib_path=self.calib_path)
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
        self._res = res; self._sel = None
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
        txt = (f"<b>detected:</b> {' + '.join(res.detected)}   "
               f"&nbsp;&nbsp; <b>per-pixel ratio:</b> {parts}")
        if getattr(res, "calibrated", False) and res.conc_avg is not None:
            um = res.conc_avg * 1e6
            cs = "  ·  ".join(f"{nm} {um[j]:.3g} µM" for j, nm in enumerate(res.comps)
                              if np.isfinite(um[j]) and um[j] > 0)
            txt += f"<br><b>map-avg quantified:</b> {cs}   (click a pixel for its µM)"
            if res.calib_r2 is not None:
                r2s = "  ·  ".join(f"{nm} R²={res.calib_r2[j]:.2f}"
                                   for j, nm in enumerate(res.comps))
                tag = ("  ⚠ low-quality calibration — µM approximate"
                       if float(np.min(res.calib_r2)) < 0.7 else "")
                txt += (f"<br><span style='color:{MUTE}'>calibration fit: "
                        f"{r2s}{tag}</span>")
        self.readout.setText(txt)
        self.readout.setTextFormat(Qt.TextFormat.RichText)

    def _plot_ratio(self, res, pp_vec=None, title=None):
        ax = self.c_ratio.new_ax()
        names = res.comps
        cols = [SERIES[i % len(SERIES)] for i in range(len(names))]
        if pp_vec is not None:                            # clicked pixel -> pie
            vals = np.asarray(pp_vec, float)
            keep = [i for i in range(len(names)) if vals[i] > 0.005]
            ax.pie([vals[i] for i in keep], labels=[names[i] for i in keep],
                   colors=[cols[i] for i in keep], autopct="%1.0f%%",
                   textprops={"fontsize": 8, "color": INK})
            ax.set_aspect("equal")
        else:                                             # map average -> bar
            vals = [res.ratio.get(n, 0.0) for n in names]
            mvals = [res.ratio_mean.get(n, 0.0) for n in names]
            x = np.arange(len(names))
            ax.bar(x, vals, color=cols, label="map average")
            ax.scatter(x, mvals, color=INK, s=28, zorder=3, label="mean-spec")
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, color=INK)
            ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
            ax.set_ylim(0, 1.1); ax.set_ylabel("proportion")
            ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE)
        if title:
            ax.set_title(title, fontsize=9, color=INK)
        self.c_ratio.fig.tight_layout(); self.c_ratio.draw_idle()

    def _plot_map(self, res):
        ax = self.c_map.new_ax(); self._map_ax = ax
        cols = np.array([to_rgb(SERIES[i % len(SERIES)]) for i in range(len(res.comps))])
        rgb = np.clip(res.pp @ cols, 0.0, 1.0)            # per-pixel convex colour blend
        x, y = res.coords[:, 0], res.coords[:, 1]
        ax.scatter(x, y, c=rgb, marker="s", s=26, edgecolors="none")
        if self._sel is not None:                         # ring the clicked pixel
            ax.scatter([x[self._sel]], [y[self._sel]], s=110, facecolors="none",
                       edgecolors=INK, linewidths=1.6, zorder=5)
        ax.legend(handles=[Patch(facecolor=SERIES[i % len(SERIES)], label=nm)
                           for i, nm in enumerate(res.comps)],
                  fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=2, loc="upper right")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        self.c_map.fig.tight_layout(); self.c_map.draw_idle()

    # ---- interactive per-pixel readout ----
    def _on_click(self, event):
        r = self._res
        if r is None or event.xdata is None or event.inaxes is not self._map_ax:
            return
        d = (r.coords[:, 0] - event.xdata) ** 2 + (r.coords[:, 1] - event.ydata) ** 2
        self._show_pixel(int(d.argmin()))

    def _show_pixel(self, i):
        r = self._res; self._sel = i
        vec = r.pp[i]; xp, yp = r.coords[i]
        self._plot_ratio(r, pp_vec=vec, title=f"pixel @ ({xp:.0f}, {yp:.0f})")
        self._plot_map(r)                                 # redraw with the highlight ring
        ratio_s = "  ·  ".join(f"{nm} {vec[j]:.0%}" for j, nm in enumerate(r.comps)
                               if vec[j] > 0.005)
        txt = f"<b>pixel @ ({xp:.0f}, {yp:.0f})</b> — ratio: {ratio_s}"
        if getattr(r, "conc", None) is not None:          # absolute concentration
            um = r.conc[i] * 1e6                           # M -> µM
            conc_s = "  ·  ".join(f"{nm} {um[j]:.3g} µM" for j, nm in enumerate(r.comps)
                                  if np.isfinite(um[j]) and um[j] > 0)
            th = float(r.pp_theta[i]) if r.pp_theta is not None else 0.0
            warn = (f"  ⚠ near-saturation (Σθ={th:.2f}) — dilute for reliable µM"
                    if th > 0.85 else "")
            txt += f"<br><b>quantified:</b> {conc_s or '—'}{warn}"
        self.readout.setText(txt)
        self.readout.setTextFormat(Qt.TextFormat.RichText)

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
