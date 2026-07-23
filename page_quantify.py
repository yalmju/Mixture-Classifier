"""page_quantify.py — Quantify tab: ratio -> M calibration + Langmuir competition."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSpinBox, QCheckBox, QLineEdit, QFileDialog, QDialog, QDialogButtonBox,
)

from ui_common import *
from calibration import build_synthetic_lab, calibrate, quantify
from io_utils import load_calibration_csv, load_calibration_folder, write_csv


def _prep_specs(specs, baseline=True):
    """ALS-baseline-remove each spectrum (keep magnitude, no normalisation) then clip
    negatives. With baseline=False the data is assumed already baseline-corrected, so
    only the clip is applied — no second ALS pass."""
    specs = np.asarray(specs, float)
    if baseline:
        from sers_mixture import als_baseline
        specs = np.stack([y - als_baseline(y) for y in specs])
    return np.clip(specs, 0.0, None)


def _real_lab(cal, seed=0, n_validation=6, baseline=True):
    """Build a lab dict from a loaded calibration CSV (real dilution series).
    Validation mixtures are synthesized from the real templates + fitted physics
    to demonstrate recovery (clearly a synthetic check on real calibration)."""
    from competitive import forward_spectrum
    axis, names, dilutions = cal
    # baseline-remove each standard (keep magnitude, no normalisation) so the fitted
    # signal B tracks the peak vs concentration, not the fluorescence background
    dilutions = [(np.asarray(c, float), _prep_specs(sp, baseline))
                 for c, sp in dilutions]
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


def _r2_on_means(C, B, gA, K):
    """R² of the Langmuir fit against the MEAN response per concentration — the
    calibration-curve quality, kept separate from replicate scatter (which is the
    precision, reported as CV%). Computing R² on every noisy replicate instead
    would just measure spot-to-spot variability, not whether the model fits."""
    from calibration import _langmuir_B
    C = np.asarray(C, float); B = np.asarray(B, float)
    uc = np.unique(C)
    mean_B = np.array([B[C == c].mean() for c in uc])
    pred = _langmuir_B(uc, gA, K)
    sst = float(np.sum((mean_B - mean_B.mean()) ** 2))
    return 1.0 - float(np.sum((mean_B - pred) ** 2)) / sst if sst > 0 else 0.0


def _langmuir_fit(C, B):
    """Fit B = gA·K·C/(1+K·C) to (C, B); returns (gA, K) with a safe fallback."""
    from scipy.optimize import curve_fit
    from calibration import _langmuir_B
    C = np.asarray(C, float); B = np.asarray(B, float)
    try:
        p0 = [max(B.max(), 1e-9), 1.0 / max(np.median(C), 1e-12)]
        (gA, K), _ = curve_fit(lambda c, gA, K: _langmuir_B(c, gA, K), C, B,
                               p0=p0, maxfev=10000, bounds=(0, np.inf))
    except Exception:
        gA, K = float(B.max()), 1.0 / max(float(np.median(C)), 1e-12)
    return float(gA), float(K)


def _fmt_conc(c):
    """Human concentration: 3.3e-9 M → '3.3 nM', 1.2e-6 → '1.2 µM', etc."""
    if c is None or not np.isfinite(c) or c <= 0:
        return "—"
    for scale, unit in ((1e-3, "mM"), (1e-6, "µM"), (1e-9, "nM"), (1e-12, "pM")):
        if c >= scale:
            return f"{c / scale:.3g} {unit}"
    return f"{c:.3g} M"


def _linear_fit(C, B):
    """Ordinary least-squares line B = m·C + b."""
    C = np.asarray(C, float); B = np.asarray(B, float)
    m, b = np.polyfit(C, B, 1)
    return float(m), float(b)


def _r2_lin_on_means(C, B, m, b):
    """R² of a linear fit against the mean response per concentration."""
    C = np.asarray(C, float); B = np.asarray(B, float)
    uc = np.unique(C)
    mean_B = np.array([B[C == c].mean() for c in uc])
    pred = m * uc + b
    sst = float(np.sum((mean_B - mean_B.mean()) ** 2))
    return 1.0 - float(np.sum((mean_B - pred) ** 2)) / sst if sst > 0 else 0.0


def _lod_loq(C, B):
    """IUPAC calibration-curve limit of detection / quantification. Fit a line to the
    low-concentration (still-linear) region, take σ = residual standard deviation and
    m = slope, then LOD = 3.3·σ/m and LOQ = 10·σ/m (both in M). Returns (lod, loq),
    NaN when it can't be estimated (too few points, flat/negative slope)."""
    C = np.asarray(C, float); B = np.asarray(B, float)
    uc = np.unique(C)
    if len(uc) < 3:
        return float("nan"), float("nan")
    lin_uc = uc[:max(3, min(len(uc), 5))]          # lowest ~5 concs ≈ linear region
    mask = np.isin(C, lin_uc)
    Cl, Bl = C[mask], B[mask]
    if len(np.unique(Cl)) < 2:
        return float("nan"), float("nan")
    m, b = np.polyfit(Cl, Bl, 1)
    if m <= 0:
        return float("nan"), float("nan")
    resid = Bl - (m * Cl + b)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else float(np.std(resid))
    if sigma <= 0:
        return float("nan"), float("nan")
    return 3.3 * sigma / m, 10.0 * sigma / m


def _lod_from_blank(C, B, blank_vals):
    """Blank-based LOD/LOQ (the SERS-appropriate form): slope from the low-range
    linear calibration, σ from the ACTUAL blank replicates measured the same way as B
    (the paper/substrate BLK). LOD = 3.3·σ_blank/slope, LOQ = 10·σ_blank/slope."""
    C = np.asarray(C, float); B = np.asarray(B, float)
    blank_vals = np.asarray(blank_vals, float)
    uc = np.unique(C)
    lin_uc = uc[:max(3, min(len(uc), 5))]
    mask = np.isin(C, lin_uc); Cl, Bl = C[mask], B[mask]
    if len(np.unique(Cl)) < 2 or len(blank_vals) < 2:
        return float("nan"), float("nan")
    m, b = np.polyfit(Cl, Bl, 1)
    if m <= 0:
        return float("nan"), float("nan")
    sigma = float(np.std(blank_vals, ddof=1))
    if sigma <= 0:
        return float("nan"), float("nan")
    return 3.3 * sigma / m, 10.0 * sigma / m


def _peak_quant(cal, peak, window=10.0, model="langmuir", baseline=True, blank=None):
    """Calibration from a marker band: B = baseline-removed intensity summed over
    peak ± window, Langmuir-fit per compound. ``peak`` is one wavenumber (same band
    for every compound) or a {compound: wavenumber} map (each compound at its own
    band). Curve only (no competition)."""
    from calibration import _langmuir_B
    axis, names, dilutions = cal
    blk = (_prep_specs(blank[1], baseline)                  # aligned BLK spectra, or None
           if blank is not None and len(blank[0]) == len(axis) else None)
    lod_method = "blank" if blk is not None else "residual"
    iso, r2, K_fit, gA_fit, peaks_used, lods, loqs = [], [], [], [], [], [], []
    for name, (C, specs) in zip(names, dilutions):
        pk = peak.get(name) if isinstance(peak, dict) else peak
        peaks_used.append(pk)
        m = (axis >= pk - window) & (axis <= pk + window)
        if m.sum() < 1:
            m = np.abs(axis - pk).argmin() == np.arange(len(axis))
        C = np.asarray(C, float)
        bl = _prep_specs(specs, baseline)
        B = bl[:, m].max(axis=1)                        # peak HEIGHT (matches the
        #  measured intensity), not the integrated band area
        dense = np.geomspace(C.min(), C.max(), 200)
        if model == "linear":
            slope, b0 = _linear_fit(C, B)
            iso.append((C, B, dense, slope * dense + b0))
            r2.append(_r2_lin_on_means(C, B, slope, b0))
            K_fit.append(slope); gA_fit.append(b0)     # slope, intercept
        else:
            gA, K = _langmuir_fit(C, B)
            iso.append((C, B, dense, _langmuir_B(dense, gA, K)))
            r2.append(_r2_on_means(C, B, gA, K))
            K_fit.append(K); gA_fit.append(gA)
        if blk is not None:                            # blank-based on the same band
            lod, loq = _lod_from_blank(C, B, blk[:, m].max(axis=1))
        else:
            lod, loq = _lod_loq(C, B)
        lods.append(lod); loqs.append(loq)
    per_cmpd = isinstance(peak, dict)
    return {"names": names, "K_true": None, "K_fit": np.array(K_fit),
            "gA_fit": np.array(gA_fit), "iso": iso, "r2": r2, "model": model,
            "lod": lods, "loq": loqs, "lod_method": lod_method,
            "parity": (np.array([]), np.array([]), np.array([], int)),
            "log_err": float("nan"), "example": None, "example_true": None,
            "selectivity": float("nan"),
            "peak_wn": None if per_cmpd else peak, "peaks_used": peaks_used}


def _run_quant(n_components=3, seed=0, cal=None, peak_wn=0.0, peak_map=None,
               model="langmuir", baseline=True, blank=None):
    from calibration import _langmuir_B
    if cal is not None and peak_map:
        return _peak_quant(cal, peak_map, model=model, baseline=baseline, blank=blank)
    if cal is not None and peak_wn and peak_wn > 0:
        return _peak_quant(cal, peak_wn, model=model, baseline=baseline, blank=blank)
    lab = (_real_lab(cal, seed, baseline=baseline) if cal is not None
           else build_synthetic_lab(n_components=n_components, seed=seed))
    calib = calibrate(lab["dilutions"], lab["P"], lab["names"])

    # blank projected onto the same templates (whole-spectrum B), for blank-based LOD
    blank_B = None
    if blank is not None and len(blank[0]) == len(lab["axis"]):
        from competitive import fit_B
        blk = _prep_specs(blank[1], baseline)
        blank_B = np.array([fit_B(y, lab["P"])[0] for y in blk])   # (n_blank, n_comp)
    lod_method = "blank" if blank_B is not None else "residual"

    iso, r2, K_out, gA_out, lods, loqs = [], [], [], [], [], []
    for i in range(calib.n):
        C = np.asarray(calib.C_series[i], float); B = np.asarray(calib.B_series[i], float)
        dense = np.geomspace(C.min(), C.max(), 200)
        if model == "linear":
            slope, b0 = _linear_fit(C, B)
            iso.append((C, B, dense, slope * dense + b0))
            r2.append(_r2_lin_on_means(C, B, slope, b0))
            K_out.append(slope); gA_out.append(b0)     # slope, intercept
        else:
            iso.append((C, B, dense, _langmuir_B(dense, calib.gA[i], calib.K[i])))
            r2.append(_r2_on_means(C, B, calib.gA[i], calib.K[i]))
            K_out.append(calib.K[i]); gA_out.append(calib.gA[i])
        if blank_B is not None:
            lod, loq = _lod_from_blank(C, B, blank_B[:, i])
        else:
            lod, loq = _lod_loq(C, B)
        lods.append(lod); loqs.append(loq)
    K_out = np.array(K_out); gA_out = np.array(gA_out)

    # single-compound calibration → fit the curve only (no competition / recovery)
    if len(lab["val_specs"]) == 0 or calib.n < 2:
        return {"names": calib.names, "K_true": lab["K_true"], "K_fit": K_out,
                "gA_fit": gA_out, "iso": iso, "r2": r2, "model": model,
                "lod": lods, "loq": loqs, "lod_method": lod_method,
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
        "names": calib.names, "K_true": lab["K_true"], "K_fit": K_out,
        "gA_fit": gA_out, "iso": iso, "r2": r2, "model": model,
        "lod": lods, "loq": loqs, "lod_method": lod_method,
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


class PeakPickerDialog(QDialog):
    """Pick each compound's calibration peak by CLICKING on its learned spectrum,
    instead of typing wavenumbers. One stacked panel per compound; a click snaps to
    the strongest band near the cursor. Returns {compound: wavenumber}."""

    def __init__(self, cal, initial=None, baseline=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick calibration peaks — click each spectrum")
        self.resize(860, 560)
        axis, names, dils = cal
        self.axis = np.asarray(axis, float); self.names = list(names)
        self.means = []
        for C, specs in dils:                              # top-conc mean, baselined
            C = np.asarray(C, float); specs = np.asarray(specs, float)
            top = specs[C == C.max()].mean(axis=0)
            self.means.append(_prep_specs(top[None, :], baseline)[0])
        band = (self.axis >= 400) & (self.axis <= 1800)
        self._band = band if band.sum() >= 5 else np.ones(len(self.axis), bool)
        self.peaks = dict(initial or {})
        for i, nm in enumerate(self.names):                # default = strongest band
            if nm not in self.peaks:
                self.peaks[nm] = float(self.axis[int(np.argmax(
                    np.where(self._band, self.means[i], -np.inf)))])

        lay = QVBoxLayout(self)
        hint = QLabel("Click on a compound's spectrum to set its peak (snaps to the "
                      "nearest band). One peak per compound — this fills the "
                      "per-compound peaks box.")
        hint.setObjectName("sub"); hint.setWordWrap(True); lay.addWidget(hint)
        self.canvas = Canvas(); self.canvas.setMinimumHeight(380)
        lay.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._axmap = {}
        self._draw()

    def _draw(self):
        self.canvas.fig.clear(); self._axmap = {}
        n = len(self.names)
        for i, nm in enumerate(self.names):
            ax = self.canvas.style(self.canvas.fig.add_subplot(n, 1, i + 1))
            y = self.means[i]; ym = y.max() or 1.0
            ax.plot(self.axis, y / ym, lw=1.1, color=SERIES[i % len(SERIES)])
            ax.axvline(self.peaks[nm], color=INK, ls="--", lw=1.1)
            ax.annotate(f"{nm} @ {self.peaks[nm]:.0f} cm⁻¹", xy=(0.99, 0.8),
                        xycoords="axes fraction", ha="right", fontsize=9, color=INK)
            ax.set_yticks([])
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("wavenumber (cm⁻¹)")
            self._axmap[ax] = i
        self.canvas.fig.tight_layout(); self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes not in self._axmap or event.xdata is None:
            return
        i = self._axmap[event.inaxes]; nm = self.names[i]
        win = np.abs(self.axis - event.xdata) <= 15.0     # snap to nearest strong band
        if win.any():
            self.peaks[nm] = float(self.axis[int(np.argmax(
                np.where(win, self.means[i], -np.inf)))])
        else:
            self.peaks[nm] = float(self.axis[int(np.abs(self.axis - event.xdata).argmin())])
        self._draw()

    def get_peaks(self):
        return dict(self.peaks)


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
        self.data_dir = None   # Samples folder (for the BLK class → blank-based LOD)
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
        self.sp_peak = self._spin(QSpinBox(), 0, 4000, 0, "peak cm⁻¹ (0=whole)")
        self.sp_peak.itemAt(1).widget().setSingleStep(10)
        self.sp_peak.itemAt(1).widget().setToolTip(
            "0 = signal is the whole-fingerprint projection (robust). Set a "
            "wavenumber to calibrate on that single marker band's intensity instead.")
        for w in (self.sp_k, self.sp_seed, self.sp_peak):
            ctl.addLayout(w)
        acol = QVBoxLayout(); acol.setSpacing(2)
        _al = QLabel("per-cmpd marker"); _al.setObjectName("field"); acol.addWidget(_al)
        self.chk_autopeak = QCheckBox("auto peak")
        self.chk_autopeak.setToolTip("calibrate each compound on ITS OWN marker band "
                                     "(the wavenumber where it most exceeds the others) "
                                     "— fills the per-compound peaks box, which you can edit")
        self.chk_autopeak.toggled.connect(self._on_autopeak)
        acol.addWidget(self.chk_autopeak); ctl.addLayout(acol)
        lcol = QVBoxLayout(); lcol.setSpacing(2)
        _ll = QLabel("fit"); _ll.setObjectName("field"); lcol.addWidget(_ll)
        self.chk_linear = QCheckBox("linear")
        self.chk_linear.setToolTip("fit a straight line B = m·C + b instead of the "
                                   "Langmuir isotherm (LOD/LOQ use a linear low-range "
                                   "fit either way)")
        lcol.addWidget(self.chk_linear); ctl.addLayout(lcol)
        bcol = QVBoxLayout(); bcol.setSpacing(2)
        _bl = QLabel("baseline"); _bl.setObjectName("field"); bcol.addWidget(_bl)
        self.chk_baselined = QCheckBox("already corrected")
        self.chk_baselined.setToolTip("your CSVs are already baseline-corrected — skip "
                                      "the app's internal ALS baseline so it isn't "
                                      "applied twice")
        bcol.addWidget(self.chk_baselined); ctl.addLayout(bcol)
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

        # per-compound marker peaks — auto-fills from 'auto peak', fully editable
        ctl2 = QHBoxLayout(); ctl2.setSpacing(8)
        pl = QLabel("per-compound peaks (cm⁻¹):"); pl.setObjectName("field")
        self.peaks_txt = QLineEdit()
        self.peaks_txt.setPlaceholderText("e.g.  DQ:610, TBZ:1000, THI:1370   "
                                          "(leave blank to use the single peak / whole)")
        self.peaks_txt.setToolTip("one band per compound as name:wavenumber, "
                                  "comma-separated. Overrides the single peak box. "
                                  "'auto peak' fills this in for you to correct.")
        pick_b = QPushButton("Pick from spectrum…"); pick_b.setObjectName("ghost")
        pick_b.setToolTip("open the learned calibration spectra and click each "
                          "compound's peak instead of typing it")
        pick_b.clicked.connect(self._pick_peaks)
        ctl2.addWidget(pl); ctl2.addWidget(self.peaks_txt, 1); ctl2.addWidget(pick_b)
        root.addLayout(ctl2)

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
        """Load a dilution series from either ONE compound's concentration folder,
        or a PARENT folder that holds a per-compound subfolder each (conc_tbz/,
        conc_dq/, …) — every subfolder becomes a compound. Concentration is parsed
        from each filename (1nM/10uM/1_mM…). Loads accumulate across calls."""
        d = QFileDialog.getExistingDirectory(
            self, "Concentration folder (one compound), or a parent of per-compound folders")
        if not d:
            return
        subdirs = [os.path.join(d, x) for x in sorted(os.listdir(d))
                   if os.path.isdir(os.path.join(d, x))]
        added, errs = [], []
        for f in [d] + subdirs:                            # the folder itself, then children
            try:
                axis, name, concs, specs = load_calibration_folder(f)
            except Exception as exc:
                errs.append(str(exc)); continue
            if self._axis is not None and len(axis) != len(self._axis):
                errs.append(f"{name}: axis {len(axis)} ≠ {len(self._axis)}"); continue
            self._axis = axis; self._acc[name] = (concs, specs); added.append(name)
        if added:
            self._rebuild_cal()
            pts = "  ·  ".join(f"{n} ({len(c)})" for n, (c, _s) in self._acc.items())
            note = ""
            pk = self._suggest_peak()                      # recommend the marker band
            if pk is not None:
                self.sp_peak.itemAt(1).widget().setValue(int(round(pk)))
                note = f"   ·   suggested peak {pk:.0f} cm⁻¹ (set to 0 for whole spectrum)"
            self.src.setText(f"source: {pts}{note}"); self.src.setStyleSheet("")
        else:
            self.src.setText("folder load failed — " + (errs[-1] if errs else "no data"))
            self.src.setStyleSheet(f"color:{RED};")
            print("load cal folder:", errs, file=sys.stderr)

    def _suggest_peak(self):
        """The strongest baseline-removed band of the loaded compound (from its
        highest-concentration spectrum) — the best band to calibrate on. Only when a
        single compound is loaded (a single wavenumber can't suit several)."""
        if len(self._acc) != 1 or self._axis is None:
            return None
        from sers_mixture import als_baseline
        (concs, specs), = self._acc.values()
        concs = np.asarray(concs, float); specs = np.asarray(specs, float)
        top = specs[concs == concs.max()].mean(axis=0)
        bl = np.clip(top - als_baseline(top), 0.0, None)
        band = (self._axis >= 400) & (self._axis <= 1800)      # fingerprint region
        if band.sum() < 5:
            band = np.ones(len(self._axis), bool)
        idx = np.where(band)[0][int(np.argmax(bl[band]))]
        return float(self._axis[idx])

    def _marker_peaks(self):
        """Per-compound discriminative band: for each loaded compound, the wavenumber
        where its (top-concentration, baseline-removed) spectrum most exceeds every
        other loaded compound — its VIP-style marker band. {compound: wavenumber}."""
        if self._cal is None:
            return None
        from sers_mixture import als_baseline
        axis, names, dils = self._cal
        means = []
        for C, specs in dils:
            C = np.asarray(C, float); specs = np.asarray(specs, float)
            top = specs[C == C.max()].mean(axis=0)
            means.append(np.clip(top - als_baseline(top), 0.0, None))
        means = np.array(means)
        band = (axis >= 400) & (axis <= 1800)
        if band.sum() < 5:
            band = np.ones(len(axis), bool)
        peaks = {}
        for i, nm in enumerate(names):
            others = np.delete(means, i, axis=0)
            score = means[i] - (others.max(axis=0) if len(others) else 0.0)
            score = np.where(band, score, -np.inf)
            peaks[nm] = float(axis[int(np.argmax(score))])
        return peaks

    def _on_autopeak(self, checked):
        """Fill the editable per-compound peaks box from the auto marker bands, so the
        user sees them and can correct any that landed on the wrong band."""
        if not checked:
            return
        peaks = self._marker_peaks()
        if not peaks:
            self.src.setText("load a calibration first to auto-fill peaks")
            self.src.setStyleSheet(f"color:{RED};")
            self.chk_autopeak.setChecked(False); return
        self.peaks_txt.setText(", ".join(f"{n}:{v:.0f}" for n, v in peaks.items()))

    def _pick_peaks(self):
        """Open the peak-picker on the learned calibration spectra; the chosen bands
        fill the per-compound peaks box."""
        if self._cal is None:
            self.src.setText("load a calibration first, then pick peaks")
            self.src.setStyleSheet(f"color:{RED};"); return
        initial = self._peaks_from_text() or self._marker_peaks()
        dlg = PeakPickerDialog(self._cal, initial=initial,
                               baseline=not self.chk_baselined.isChecked(), parent=self)
        if dlg.exec():
            peaks = dlg.get_peaks()
            self.peaks_txt.setText(", ".join(f"{n}:{v:.0f}" for n, v in peaks.items()))

    def _peaks_from_text(self):
        """Parse the per-compound peaks box → {name: wavenumber}. Only names that are
        actually in the loaded calibration are kept; empty/blank → None."""
        txt = self.peaks_txt.text().strip()
        if not txt:
            return None
        valid = set(self._cal[1]) if self._cal is not None else None
        out = {}
        for tok in txt.replace(";", ",").split(","):
            if ":" not in tok:
                continue
            k, v = tok.split(":", 1); k = k.strip()
            try:
                wn = float(v)
            except ValueError:
                continue
            if valid is None or k in valid:
                out[k] = wn
        return out or None

    def _rebuild_cal(self):
        names = list(self._acc)
        dilutions = [self._acc[n] for n in names]
        self._cal = (self._axis, names, dilutions)

    def _clear_cal(self):
        self._cal = None; self._acc = {}; self._axis = None
        self.src.setText("source: synthetic"); self.src.setStyleSheet("")

    def set_data_dir(self, path):
        self.data_dir = path       # so LOD can use the Samples BLK class as the blank

    def _load_blank(self):
        """BLK spectra from the Samples dataset (the paper/substrate blank) for a
        blank-based LOD: (axis, spectra) or None if there is no blank class or its
        axis doesn't match the loaded calibration."""
        if not self.data_dir or self._cal is None:
            return None
        try:
            from dataset import discover_dataset, is_blank
            from real_data import load_map
            groups = discover_dataset(self.data_dir)
            cubes, wn = [], None
            for c, maps in groups:
                if not is_blank(c):
                    continue
                for _b, p, _r in maps:
                    wn, cube, _m, _c = load_map(p); cubes.append(cube)
            if not cubes:
                return None
            specs = np.vstack(cubes)
            if len(wn) != len(self._cal[0]):     # must share the calibration axis
                return None
            return (np.asarray(wn, float), specs)
        except Exception as exc:
            print("load blank:", exc, file=sys.stderr)
            return None

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
        # per-concentration replicate statistics (mean ± SD, CV%)
        srows = []
        for i, nm in enumerate(r["names"]):
            C, B, _dc, _db = r["iso"][i]
            C = np.asarray(C, float); B = np.asarray(B, float)
            for c in np.unique(C):
                b = B[C == c]; n = len(b); mu = b.mean()
                sd = b.std(ddof=1) if n > 1 else 0.0
                sem = sd / np.sqrt(n) if n else 0.0
                srows.append([nm, f"{c:.4e}", n, f"{mu:.6f}", f"{sd:.6f}",
                              f"{sem:.6f}", f"{100 * sd / mu:.1f}" if mu else ""])
        write_csv(os.path.join(d, "calibration_stats.csv"),
                  ["compound", "concentration_M", "n", "B_mean", "B_std", "B_SE", "CV_%"],
                  srows)
        # calibration SPECTRA (mean spectrum per concentration) in the format Real
        # data / Predict 'Load calibration…' reads — bridges the folder calibration
        # to per-pixel µM quantification
        if self._cal is not None:
            axis, names, dils = self._cal
            head = ["compound", "concentration_M"] + [f"{v:.2f}" for v in axis]
            srows2 = []
            for nm, (C, specs) in zip(names, dils):
                C = np.asarray(C, float); specs = np.asarray(specs, float)
                for c in np.unique(C):
                    mean_sp = specs[C == c].mean(axis=0)
                    srows2.append([nm, f"{c:.4e}"] + [f"{v:.4f}" for v in mean_sp])
            write_csv(os.path.join(d, "calibration_spectra.csv"), head, srows2)
        # fitted parameters + detection limits per compound. For Langmuir K/gA are the
        # isotherm; for a linear fit K_fit holds the slope and gA_fit the intercept.
        gA = r.get("gA_fit"); lod = r.get("lod", []); loq = r.get("loq", [])
        model = r.get("model", "langmuir")
        p1, p2 = ("slope", "intercept") if model == "linear" else ("K_fit", "gA_fit")
        lmeth = r.get("lod_method", "residual")
        write_csv(os.path.join(d, "calibration_fit.csv"),
                  ["compound", "model", p1, p2, "R2", "LOD_M", "LOQ_M", "LOD_basis"],
                  [[nm, model, f"{r['K_fit'][i]:.4e}",
                    f"{gA[i]:.4e}" if gA is not None else "",
                    f"{r['r2'][i]:.4f}" if r.get('r2') else "",
                    f"{lod[i]:.4e}" if i < len(lod) and np.isfinite(lod[i]) else "",
                    f"{loq[i]:.4e}" if i < len(loq) and np.isfinite(loq[i]) else "",
                    lmeth]
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
        # priority: an edited/auto per-compound peaks box wins over the single peak
        peak_map = self._peaks_from_text()
        if peak_map is None and self.chk_autopeak.isChecked():
            peak_map = self._marker_peaks()
        params = dict(n_components=self.sp_k.itemAt(1).widget().value(),
                      seed=self.sp_seed.itemAt(1).widget().value(), cal=self._cal,
                      peak_wn=float(self.sp_peak.itemAt(1).widget().value()),
                      peak_map=peak_map,
                      model="linear" if self.chk_linear.isChecked() else "langmuir",
                      baseline=not self.chk_baselined.isChecked(),
                      blank=self._load_blank())     # Samples BLK → blank-based LOD
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
        pks = res.get("peaks_used"); lod = res.get("lod"); loq = res.get("loq")
        for i, nm in enumerate(res["names"]):
            C, B, dc, db = res["iso"][i]
            col = SERIES[i % len(SERIES)]
            C = np.asarray(C, float); B = np.asarray(B, float)
            uc = np.unique(C)
            means = np.array([B[C == c].mean() for c in uc])
            # standard ERROR of the mean (SD/√n), not SD — the uncertainty of the
            # per-concentration mean the curve is fit to
            ns = np.array([max(np.sum(C == c), 1) for c in uc])
            sems = np.array([B[C == c].std(ddof=1) if np.sum(C == c) > 1 else 0.0
                             for c in uc]) / np.sqrt(ns)
            reps = int(np.median(ns))
            if reps > 1:                                   # replicates → mean ± SE error bars
                ax.scatter(C, B, s=12, color=col, alpha=0.25, edgecolors="none", zorder=2)
                ax.errorbar(uc, means, yerr=sems, fmt="o", ms=5, color=col,
                            capsize=3, elinewidth=1.0, zorder=3)
            else:
                ax.scatter(C, B, s=26, color=col, zorder=3,
                           edgecolors="white", linewidths=0.5)
            lab = nm if r2[i] is None else f"{nm}  (R²={r2[i]:.2f})"
            if pks is not None and i < len(pks) and pks[i]:
                lab += f"  @{pks[i]:.0f}"                   # per-compound marker band
            if lod is not None and i < len(lod) and np.isfinite(lod[i]):
                lab += f"  LOD {_fmt_conc(lod[i])}"
                ax.axvline(lod[i], color=col, ls=":", lw=1.0, alpha=0.45, zorder=1)
            ax.plot(dc, db, color=col, lw=1.6, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("concentration (M)")
        pk = res.get("peak_wn")
        ax.set_ylabel(f"peak height @ {pk:.0f} cm⁻¹  (mean ± SE)" if pk
                      else ("marker-peak height (mean ± SE)" if pks
                            else "signal  B  (mean ± SE)"))
        ax.legend(fontsize=8, framealpha=0.0, labelcolor=MUTE)
        self.c_iso.fig.tight_layout(); self.c_iso.draw_idle()

    def _readout(self, res):
        names = res["names"]; K = res["K_fit"]; gA = res.get("gA_fit")
        r2 = res.get("r2", [0.0] * len(names))
        lod = res.get("lod", [float("nan")] * len(names))
        loq = res.get("loq", [float("nan")] * len(names))
        linear = res.get("model") == "linear"
        rows = []
        for i, nm in enumerate(names):
            if linear:                                     # K_fit=slope, gA_fit=intercept
                p1 = f"<td style='padding-right:12px'>slope={K[i]:.2e}</td>"
                p2 = (f"<td style='padding-right:12px'>b="
                      f"{(gA[i] if gA is not None else float('nan')):.2e}</td>")
            else:
                p1 = f"<td style='padding-right:12px'>K={K[i]:.2e}</td>"
                p2 = (f"<td style='padding-right:12px'>gA="
                      f"{(gA[i] if gA is not None else float('nan')):.2e}</td>")
            rows.append(
                f"<tr><td style='padding-right:12px;color:{SERIES[i%len(SERIES)]};"
                f"font-weight:600'>{nm}</td>{p1}{p2}"
                f"<td style='padding-right:12px;color:{MUTE}'>R²={r2[i]:.2f}</td>"
                f"<td style='padding-right:12px;color:{TEAL}'>LOD "
                f"{_fmt_conc(lod[i] if i < len(lod) else None)}</td>"
                f"<td style='color:{BLUE}'>LOQ "
                f"{_fmt_conc(loq[i] if i < len(loq) else None)}</td></tr>")
        title = ("<b>Linear fit</b>  (B = m·C + b)" if linear
                 else "<b>Langmuir fit</b>  (B = gA·K·C / (1+K·C))")
        note = ("σ = the Samples BLK (paper blank) noise" if res.get("lod_method") == "blank"
                else "σ = calibration residual — load a BLK class in Samples for a "
                     "blank-based limit")
        html = (f"<div style='color:{INK};font-size:13px'>{title}"
                f"<table style='font-size:13px;margin-top:6px'>{''.join(rows)}</table>"
                f"<p style='color:{FAINT};margin-top:4px'>LOD = 3.3σ/slope, "
                f"LOQ = 10σ/slope · {note}.</p>")
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
