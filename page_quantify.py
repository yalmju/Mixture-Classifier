"""page_quantify.py — Quantify tab: ratio -> M calibration + Langmuir competition."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSpinBox, QFileDialog,
)

from ui_common import *
from calibration import build_synthetic_lab, calibrate, quantify
from io_utils import load_calibration_csv, load_calibration_folder, write_csv


def _real_lab(cal, seed=0, n_validation=6):
    """Build a lab dict from a loaded calibration CSV (real dilution series).
    Validation mixtures are synthesized from the real templates + fitted physics
    to demonstrate recovery (clearly a synthetic check on real calibration)."""
    from competitive import forward_spectrum
    axis, names, dilutions = cal
    Praw = np.array([sp[int(np.argmax(c))] for c, sp in dilutions])
    P = Praw / (np.linalg.norm(Praw, axis=1, keepdims=True) + 1e-12)
    n = len(names)
    if n < 2:                                            # single compound: curve only
        return {"axis": axis, "names": names, "P": P, "dilutions": dilutions,
                "val_specs": np.empty((0, len(axis))),
                "val_true": np.empty((0, n)), "K_true": None}
    tmp = calibrate(dilutions, P, names)
    rng = np.random.default_rng(seed)
    K = tmp.K; A = tmp.gA / (tmp.gA.max() or 1.0)
    val_specs, val_true = [], []
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

    iso, r2 = [], []
    for i in range(calib.n):
        C = calib.C_series[i]
        dense = np.geomspace(C.min(), C.max(), 60)
        fit = _langmuir_B(dense, calib.gA[i], calib.K[i])
        iso.append((C, calib.B_series[i], dense, fit))
        B = np.asarray(calib.B_series[i], float)
        pred = _langmuir_B(C, calib.gA[i], calib.K[i])
        sst = float(np.sum((B - B.mean()) ** 2))
        r2.append(1.0 - float(np.sum((B - pred) ** 2)) / sst if sst > 0 else 0.0)

    # single-compound calibration → fit the curve only (no competition / recovery)
    if len(lab["val_specs"]) == 0 or calib.n < 2:
        return {"names": calib.names, "K_true": lab["K_true"], "K_fit": calib.K,
                "gA_fit": calib.gA, "iso": iso, "r2": r2,
                "parity": (np.array([]), np.array([]), np.array([], int)),
                "log_err": float("nan"), "example": None,
                "example_true": None, "selectivity": float("nan")}

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
        "gA_fit": calib.gA, "iso": iso, "r2": r2,
        "parity": (np.array(true_flat), np.array(est_flat), np.array(col_flat, int)),
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


# --------------------------------------------------------------------------
# Quantify page
# --------------------------------------------------------------------------
class QuantifyPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._cal = None       # loaded calibration (axis, names, dilutions)
        self._acc = {}         # accumulated per-compound folders {name: (concs, specs)}
        self._axis = None      # shared wavenumber axis of the accumulated folders
        self._res = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(14)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Quantify — calibration curve"); h1.setObjectName("h1")
        sub = QLabel("Load a dilution series (a folder of per-concentration map CSVs, "
                     "or one calibration CSV) and draw the calibration curve — signal "
                     "vs concentration with a Langmuir fit and R² per compound. With "
                     "≥2 compounds it also reports competitive adsorption.")
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
        fold_b = QPushButton("Load conc. folder…"); fold_b.setObjectName("ghost")
        fold_b.clicked.connect(self._load_cal_folder)
        fold_b.setToolTip("A folder of per-concentration map CSVs (1nM/10uM/1mM…) "
                          "for ONE compound; load several to build a multi-compound set")
        load_b = QPushButton("Load calibration…"); load_b.setObjectName("ghost")
        load_b.clicked.connect(self._load_cal)
        clr_b = QPushButton("✕"); clr_b.setObjectName("ghost"); clr_b.setFixedWidth(30)
        clr_b.setToolTip("clear loaded calibration"); clr_b.clicked.connect(self._clear_cal)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Calibrate + quantify"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(fold_b); ctl.addWidget(load_b); ctl.addWidget(clr_b)
        ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_ncmp = Kpi("compounds"); self.k_npts = Kpi("points / compound")
        self.k_r2 = Kpi("mean fit R²"); self.k_range = Kpi("concentration range")
        for k in (self.k_ncmp, self.k_npts, self.k_r2, self.k_range):
            kpis.addWidget(k)
        root.addLayout(kpis)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_iso = Canvas()
        icard, ilay = _card("Calibration curve — signal (B) vs concentration + fit")
        ilay.addWidget(self.c_iso); grid.addWidget(icard, 0, 0)
        rcard, rlay = _card("Fit parameters + read-out")
        self.readout = QLabel("Load a dilution series, then Calibrate.")
        self.readout.setObjectName("sub")
        self.readout.setWordWrap(True); self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignTop)
        rlay.addWidget(self.readout); rlay.addStretch(1)
        grid.addWidget(rcard, 0, 1)
        grid.setColumnStretch(0, 3); grid.setColumnStretch(1, 2); grid.setRowStretch(0, 1)
        root.addLayout(grid, 1)
        self.c_iso.placeholder("Load a dilution series, then Calibrate")

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
            self._acc = {}; self._axis = None            # a single CSV replaces the set
            self._cal = (axis, names, dilutions)
            self.src.setText(f"source: {os.path.basename(p)} ({len(names)})")
            self.src.setStyleSheet("")
        except Exception as exc:
            self.src.setText("load failed"); self.src.setStyleSheet(f"color:{RED};")
            print("load cal:", exc, file=sys.stderr)

    def _load_cal_folder(self):
        """Add ONE compound's dilution series from a folder of per-concentration
        map CSVs (concentration parsed from each filename). Load several folders to
        assemble a multi-compound calibration."""
        d = QFileDialog.getExistingDirectory(
            self, "Concentration folder for ONE compound (files named 1nM/10uM/1mM…)")
        if not d:
            return
        try:
            axis, name, concs, specs = load_calibration_folder(d)
            if self._axis is not None and len(axis) != len(self._axis):
                raise ValueError(f"{name}: axis length {len(axis)} ≠ {len(self._axis)} "
                                 "— all compounds must share one wavenumber axis.")
            self._axis = axis
            self._acc[name] = (concs, specs)
            self._rebuild_cal()
            pts = "  ·  ".join(f"{n} ({len(c)})" for n, (c, _s) in self._acc.items())
            self.src.setText(f"source: {pts}"); self.src.setStyleSheet("")
        except Exception as exc:
            self.src.setText(f"folder load failed — {exc}")
            self.src.setStyleSheet(f"color:{RED};")
            print("load cal folder:", exc, file=sys.stderr)

    def _rebuild_cal(self):
        names = list(self._acc)
        dilutions = [self._acc[n] for n in names]
        self._cal = (self._axis, names, dilutions)

    def _clear_cal(self):
        self._cal = None; self._acc = {}; self._axis = None
        self.src.setText("source: synthetic"); self.src.setStyleSheet("")

    def _export(self):
        if self._res is None:
            self.src.setText("run first, then export"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res
        # calibration curve — every measured point (compound, concentration, B) so
        # the isotherm can be re-plotted or re-fit outside the app
        crows = []
        for i, nm in enumerate(r["names"]):
            C, B, _dc, _db = r["iso"][i]
            for c, b in zip(C, B):
                crows.append([nm, f"{c:.4e}", f"{b:.6f}"])
        write_csv(os.path.join(d, "calibration_curve.csv"),
                  ["compound", "concentration_M", "B"], crows)
        # fitted isotherm parameters (Langmuir K and gA) per compound
        gA = r.get("gA_fit")
        write_csv(os.path.join(d, "calibration_fit.csv"),
                  ["compound", "K_fit", "gA_fit"],
                  [[nm, f"{r['K_fit'][i]:.4e}",
                    f"{gA[i]:.4e}" if gA is not None else ""]
                   for i, nm in enumerate(r["names"])])
        # per-mixture quantification (only when a ≥2-compound example exists)
        if r["example"] is not None:
            q = r["example"]
            rows = [[nm, f"{q['C'][i]:.3e}", f"{q['conc_ratio'][i]:.3f}",
                     f"{q['theta'][i]:.3f}", f"{r['K_fit'][i]:.3e}"]
                    for i, nm in enumerate(r["names"])]
            write_csv(os.path.join(d, "quantify.csv"),
                      ["compound", "C_M", "ratio", "theta", "K_fit"], rows)
        n = _save_figs([("calibration_curve", self.c_iso)], d)
        self.src.setText(f"exported CSV + {n} PNG → {os.path.basename(d)}")
        self.src.setStyleSheet("")

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
        names = res["names"]; r2 = res.get("r2", [])
        npts = res["iso"][0][0].shape[0] if res["iso"] else 0
        allC = (np.concatenate([iso[0] for iso in res["iso"]])
                if res["iso"] else np.array([1.0]))
        self.k_ncmp.set(str(len(names)), TEAL)
        self.k_npts.set(str(npts), AMBER)
        self.k_r2.set(f"{np.mean(r2):.3f}" if r2 else "—", BLUE)
        self.k_range.set(f"{allC.min():.0e}–{allC.max():.0e} M", PURPLE)
        self._plot_iso(res); self._readout(res)

    def _plot_iso(self, res):
        ax = self.c_iso.new_ax()
        r2 = res.get("r2", [None] * len(res["names"]))
        for i, nm in enumerate(res["names"]):
            C, B, dc, db = res["iso"][i]
            col = SERIES[i % len(SERIES)]
            ax.scatter(C, B, s=26, color=col, zorder=3, edgecolors="white", linewidths=0.5)
            lab = nm if r2[i] is None else f"{nm}  (R²={r2[i]:.2f})"
            ax.plot(dc, db, color=col, lw=1.6, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("concentration (M)")
        ax.set_ylabel("signal  B")
        ax.legend(fontsize=8, framealpha=0.0, labelcolor=MUTE)
        self.c_iso.fig.tight_layout(); self.c_iso.draw_idle()

    def _readout(self, res):
        names = res["names"]; K = res["K_fit"]; gA = res.get("gA_fit")
        r2 = res.get("r2", [0.0] * len(names))
        rows = []
        for i, nm in enumerate(names):
            rows.append(
                f"<tr><td style='padding-right:12px;color:{SERIES[i%len(SERIES)]};"
                f"font-weight:600'>{nm}</td>"
                f"<td style='padding-right:12px'>K={K[i]:.2e}</td>"
                f"<td style='padding-right:12px'>gA="
                f"{(gA[i] if gA is not None else float('nan')):.2e}</td>"
                f"<td style='color:{MUTE}'>R²={r2[i]:.2f}</td></tr>")
        html = (f"<div style='color:{INK};font-size:13px'>"
                f"<b>Langmuir fit</b>  (B = gA·K·C / (1+K·C))"
                f"<table style='font-size:13px;margin-top:6px'>{''.join(rows)}</table>")
        if res["example"] is not None:                    # ≥2 compounds → competition
            comp = res["example"]["competition"]
            html += (f"<p style='margin-top:10px'><b style='color:{CORAL}'>"
                     "Competitive adsorption</b> — " +
                     (f"<b>{comp['surface_dominant']}</b> dominates the surface but "
                      f"<b>{comp['solution_dominant']}</b> dominates in solution "
                      f"(selectivity {comp['selectivity']:.1f}×)."
                      if comp["flipped"] else
                      f"surface and solution agree (selectivity "
                      f"{comp['selectivity']:.1f}×).") + "</p>")
        else:
            html += (f"<p style='color:{FAINT};margin-top:10px'>load another "
                     "compound's concentration folder to add competition analysis.</p>")
        html += "</div>"
        self.readout.setText(html)
