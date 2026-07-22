"""page_quantify.py — Quantify tab: ratio -> M calibration + Langmuir competition."""
from __future__ import annotations

import os
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSpinBox, QFileDialog,
)

from ui_common import *
from calibration import build_synthetic_lab, calibrate, quantify
from io_utils import load_calibration_csv, write_csv


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


# --------------------------------------------------------------------------
# Quantify page
# --------------------------------------------------------------------------
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
