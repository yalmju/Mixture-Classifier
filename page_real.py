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

from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QFileDialog, QColorDialog,
    QScrollArea, QFrame, QProgressBar, QLineEdit,
)

from ui_common import *
from unmix import unmix_map, vip_bands
from classify import classify_map
from real_data import PEST_DEFAULT
from dataset import load_preprocess, load_colors, save_colors
from io_utils import write_csv, write_readme

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
    EX_COLS = ["#d7dde6", "#a06bff", "#ff9f40", "#4dd2c0"]   # user-added band hues
    _GS_L, _GS_R, _GS_WS, _GS_HS = 0.015, 0.985, 0.06, 0.06
    METHODS = [("Composition model (per pixel)", "dlpx"), ("NNLS (fixed refs)", "nnls"),
               ("MCR-ALS (refine)", "mcr"), ("ResNet1D (DL)", "dl"),
               ("Trained model", "model")]

    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self.dl_model = None
        self._bl_override = None    # set when the model's baseline flag overrides the folder's
        self._sel = None
        self._click_axes = []       # axes that accept a pixel click
        self._colors = {}           # per-substance colour override {name: '#hex'}
        self._bands = {}            # per-substance mapped wavenumber {name: cm⁻¹}
        self._band_spins = {}
        self._extra_bands = []      # user-added band panels (cm⁻¹); + adds, − removes
        self._scale_ui = {}         # key -> (manual chk, min spin, max spin)
        self._chan_scale = {}       # band panel key -> (min spin, max spin)
        self.chk_chan = None        # 'manual scale' for the band card
        COLOR_BUS.changed.connect(self._on_colors_changed)   # top-bar picker sync
        CALIB_BUS.changed.connect(lambda: self._refresh_calib_label())
        self.data_dir = PEST_DEFAULT
        self.test = None
        self.model_path = None      # trained model (unmixr_model.joblib) for classify
        self.calib_path = None      # optional dilution-series calibration CSV → µM
        self.bg_paths = []          # measured background map(s) → direct bg judgment
        self.rf = {}                # response factors {name: ×} from Validate (correction)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(12)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Real-data analysis — unmix a test map"); h1.setObjectName("h1")
        sub = QLabel("Read one measured map: raw band maps, stage-1 unmixed "
                     "abundance maps (NNLS/MCR — or per-pixel probabilities under a "
                     "composition model), the per-pixel composition pie map, apparent "
                     "µM maps, and the overall composition as bars with a µM line. "
                     "Click any pixel for its spectrum. References + preprocessing "
                     "come from Samples.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        # ── left control rail | right results ─────────────────────────────
        # Everything you LOAD or SET lives in a fixed 300px rail on the left;
        # the right side is nothing but results. One screen, no hunting.
        outer = QHBoxLayout(); outer.setSpacing(14)
        root.addLayout(outer, 1)
        leftw = QWidget(); leftw.setFixedWidth(300)
        left = QVBoxLayout(leftw)
        left.setContentsMargins(0, 0, 0, 0); left.setSpacing(10)
        outer.addWidget(leftw)

        ctl = QVBoxLayout(); ctl.setSpacing(6)
        test_b = QPushButton("Load test map…"); test_b.setObjectName("ghost")
        test_b.clicked.connect(self._browse_test)
        self.test_lbl = QLabel("no test map"); self.test_lbl.setObjectName("field")
        self.test_x = QPushButton("✕"); self.test_x.setObjectName("ghost")
        self._compact_x(self.test_x, "clear")
        self.test_x.clicked.connect(self._clear_test); self.test_x.setVisible(False)
        self.cmb_method = self._combo("method", self.METHODS)
        model_b = QPushButton("Load model…"); model_b.setObjectName("ghost")
        model_b.setToolTip("a model exported from the Model tab (unmixr_model.joblib); "
                           "used by the 'Trained model' method")
        model_b.clicked.connect(self._browse_model)
        self.model_lbl = QLabel(""); self.model_lbl.setObjectName("field")
        dlm_b = QPushButton("Load DL model…"); dlm_b.setObjectName("ghost")
        dlm_b.setToolTip("a DL model saved from the Recovery tab (.dlm) → physics-informed "
                         "DL composition (+ approximate µM) for this map, on top of the NNLS unmix")
        dlm_b.clicked.connect(self._browse_dl)
        self.dlm_lbl = QLabel(""); self.dlm_lbl.setObjectName("field")
        MODEL_BUS.changed.connect(self._adopt_model)       # auto-use a model trained in the Model tab
        self._adopt_model()
        bg_b = QPushButton("Load background…"); bg_b.setObjectName("ghost")
        bg_b.setToolTip("measured blank/background map(s) (e.g. Pest/BLk) — a pixel "
                        "whose spectrum matches them is judged BACKGROUND directly, "
                        "independent of the NNLS abundances")
        bg_b.clicked.connect(self._browse_bg)
        self.bg_lbl = QLabel(""); self.bg_lbl.setObjectName("field")
        self.bg_x = QPushButton("✕"); self.bg_x.setObjectName("ghost")
        self._compact_x(self.bg_x, "clear background")
        self.bg_x.clicked.connect(self._clear_bg); self.bg_x.setVisible(False)
        cal_b = QPushButton("Load calibration…"); cal_b.setObjectName("ghost")
        cal_b.setToolTip("a dilution-series CSV → per-pixel apparent SERS-equivalent concentration (µM)")
        cal_b.clicked.connect(self._browse_calib)
        self.cal_lbl = QLabel(""); self.cal_lbl.setObjectName("field")
        self.cal_x = QPushButton("✕"); self.cal_x.setObjectName("ghost")
        self._compact_x(self.cal_x, "clear calibration")
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
        # 0.20 settled on the 260812 trio map: 100% of the lettering, 5% stray hits
        sp.setDecimals(2); sp.setSingleStep(0.05); sp.setRange(0.01, 0.9); sp.setValue(0.20)
        sp.setToolTip("a pixel counts as a substance (not background) when the "
                      "substances make up at least this fraction of it — lower to "
                      "catch weaker signal")
        self.chk_flip = QCheckBox("flip Y")
        self.chk_flip.setToolTip("flip the map top-to-bottom if it comes out upside down")
        self.chk_flip.toggled.connect(lambda _=False: self._redraw())
        flipcol = QVBoxLayout(); flipcol.setSpacing(2)
        _fl = QLabel("orientation"); _fl.setObjectName("field")
        flipcol.addWidget(_fl); flipcol.addWidget(self.chk_flip)
        self.chk_rel = QCheckBox("drop low-R²"); self.chk_rel.setChecked(False)
        self.chk_rel.setToolTip("OFF by default: low-R² pixels are MARKED (coral ✕) and "
                                "counted, but still composed — a poor fit to the pure "
                                "spectra is a warning (saturated / clipped / something "
                                "the references don't cover), not proof the pixel is "
                                "wrong. Tick this to exclude them from the composition.")
        self.chk_rel.toggled.connect(lambda _=False: self._on_corr())
        self.rel_thr = QDoubleSpinBox(); self.rel_thr.setDecimals(2)
        self.rel_thr.setSingleStep(0.05); self.rel_thr.setRange(0.0, 0.95)
        self.rel_thr.setValue(0.50); self.rel_thr.setToolTip("minimum reconstruction R²")
        self.rel_thr.valueChanged.connect(lambda _=0: self._on_corr())
        relcol = QVBoxLayout(); relcol.setSpacing(2)
        _rl = QLabel("reliability (min R²)"); _rl.setObjectName("field")
        relrow = QHBoxLayout(); relrow.setSpacing(4)
        relrow.addWidget(self.chk_rel); relrow.addWidget(self.rel_thr)
        relcol.addWidget(_rl); relcol.addLayout(relrow)
        # saturation is quarantined AS saturation — its own category, not a repair
        # you have to trust: clipped pixels leave every statistic and are painted
        # amber in the pie map so the hole is visible, never silent
        self.chk_sat = QCheckBox("quarantine saturated"); self.chk_sat.setChecked(True)
        self.chk_sat.setObjectName("field")
        self.chk_sat.setToolTip(
            "pixels with detector-clipped channels (>1% of the spectrum, flat-top "
            "or the jagged post-ALS residue) are not classified at all — they show "
            "as background in every map and leave composition and µM. The run "
            "status counts them. Untick to classify them anyway.")
        self.chk_sat.toggled.connect(lambda _=False: self._on_corr())
        satcol = QVBoxLayout(); satcol.setSpacing(2)
        _sl = QLabel("saturation"); _sl.setObjectName("field")
        satcol.addWidget(_sl); satcol.addWidget(self.chk_sat)
        corr_b = QPushButton("Load correction…"); corr_b.setObjectName("ghost")
        corr_b.setToolTip("response_factors.csv from the Validate tab → convert the "
                          "surface ratio to the solution ratio")
        corr_b.clicked.connect(self._browse_corr)
        self.chk_corr = QCheckBox("solution ratio")
        self.chk_corr.setToolTip("apply the loaded response factors so the ratio / "
                                 "composition reflect the SOLUTION, not the raw surface "
                                 "signal (which over-weights high-response substances)")
        self.chk_corr.setEnabled(False)
        self.chk_corr.toggled.connect(lambda _=False: self._on_corr())
        corrcol = QVBoxLayout(); corrcol.setSpacing(2)
        self.corr_lbl = QLabel("correction"); self.corr_lbl.setObjectName("field")
        corrcol.addWidget(self.corr_lbl); corrcol.addWidget(self.chk_corr)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Unmix"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        self.pbar = QProgressBar(); self.pbar.setRange(0, 0)   # indeterminate = busy
        self.pbar.setFixedWidth(120); self.pbar.setFixedHeight(8)
        self.pbar.setTextVisible(False); self.pbar.hide()
        # rail top: what you touch on every run
        _r1 = QHBoxLayout(); _r1.setSpacing(6)
        _r1.addWidget(test_b); _r1.addWidget(self.test_lbl, 1)
        _r1.addWidget(self.test_x)
        ctl.addLayout(_r1)
        ctl.addLayout(self.cmb_method)
        self.dlm_lbl.setWordWrap(True)
        ctl.addWidget(self.dlm_lbl)
        _r2 = QHBoxLayout(); _r2.setSpacing(6)
        _r2.addWidget(self.btn, 1); _r2.addWidget(exp_b)
        ctl.addLayout(_r2)
        ctl.addWidget(self.pbar)
        left.addLayout(ctl)

        # everything else folds away — sources (models / calibration / correction) and the
        # per-pixel thresholds are set once and then just sit there cluttering the header
        self.opt_tgl = QPushButton(); self.opt_tgl.setObjectName("ghost")
        self.opt_tgl.setCheckable(True); self.opt_tgl.setChecked(False)
        self.opt_tgl.setStyleSheet("text-align:left; padding:4px 8px;")
        self.opt_tgl.toggled.connect(self._toggle_opts)
        left.addWidget(self.opt_tgl)

        self.optbox = QWidget(); obl = QVBoxLayout(self.optbox)
        obl.setContentsMargins(0, 0, 0, 0); obl.setSpacing(8)
        def _group(title):
            lbl = QLabel(title); lbl.setObjectName("field")
            lbl.setStyleSheet("font-weight:600; margin-top:2px;")
            return lbl

        obl.addWidget(_group("sources"))
        for _lbl in (self.model_lbl, self.bg_lbl, self.cal_lbl):
            _lbl.setWordWrap(True)
        obl.addWidget(model_b); obl.addWidget(self.model_lbl)
        obl.addWidget(dlm_b)
        _bgrow = QHBoxLayout(); _bgrow.setSpacing(4)
        _bgrow.addWidget(bg_b, 1); _bgrow.addWidget(self.bg_x)
        obl.addLayout(_bgrow); obl.addWidget(self.bg_lbl)
        _calrow = QHBoxLayout(); _calrow.setSpacing(4)
        _calrow.addWidget(cal_b, 1); _calrow.addWidget(self.cal_x)
        obl.addLayout(_calrow); obl.addWidget(self.cal_lbl)
        obl.addWidget(corr_b); obl.addLayout(corrcol)
        obl.addWidget(_group("pixel gate & view"))
        obl.addLayout(hitcol); obl.addLayout(self.thr); obl.addLayout(relcol)
        obl.addLayout(satcol); obl.addLayout(flipcol)
        left.addWidget(self.optbox)
        self._toggle_opts(False)
        self.cmb_method.itemAt(1).widget().currentIndexChanged.connect(
            lambda _=0: self._sync_controls())
        self._sync_controls()

        self.status = QLabel(""); self.status.setObjectName("sub")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        kpis = QGridLayout(); kpis.setSpacing(8)
        self.k_dom = Kpi("dominant"); self.k_n = Kpi("substances")
        self.k_hit = Kpi("hit %"); self.k_px = Kpi("pixels")
        kpis.addWidget(self.k_dom, 0, 0); kpis.addWidget(self.k_n, 0, 1)
        kpis.addWidget(self.k_hit, 1, 0); kpis.addWidget(self.k_px, 1, 1)
        left.addLayout(kpis)

        # per-substance colour swatches (click to recolour), filled after a result
        self.swatches = QHBoxLayout(); self.swatches.setSpacing(6)
        self.swatches.addWidget(self._mk_lbl("colours:"))
        self.swatches.addStretch(1)
        left.addLayout(self.swatches)
        left.addStretch(1)

        # ---------- stacked result sections (scrollable) ----------
        body = QVBoxLayout(); body.setSpacing(8)

        # 1) band maps: raw intensity at one marker band per substance + their RGB merge
        self.c_maps = Canvas()
        card_maps, lay_maps = _card(
            "Band maps — raw intensity at one band per substance, and the three "
            "read as R/G/B (click a pixel)")
        self.bandrow = QHBoxLayout(); self.bandrow.setSpacing(6)
        self.bandrow.addWidget(self._mk_lbl("bands (cm⁻¹):"))
        self.bandrow.addStretch(1)
        lay_maps.addLayout(self.bandrow)
        lay_maps.addWidget(self.c_maps); self.c_maps.setMinimumHeight(300)
        # one min/max pair PER band panel (rebuilt with the band row) — the card-wide
        # pair could not stretch a weak channel without flattening a strong one
        self.scalerow = QHBoxLayout(); self.scalerow.setSpacing(6)
        self.scalerow.addWidget(self._mk_lbl("scale:")); self.scalerow.addStretch(1)
        lay_maps.addLayout(self.scalerow)
        self._add_fold(lay_maps, self.c_maps, "band maps", opened=True, key="maps")

        # 1b) unmixed abundance maps — NNLS runs FIRST in every path (the gate), so
        #     its per-substance abundances belong beside the raw band maps: bands =
        #     what the camera saw, abundances = what stage-1 unmixing made of it.
        self.c_abund = Canvas()
        card_ab, lay_ab = _card(
            "Unmixed maps — how stage-1 splits each pixel among the references. "
            "NNLS/MCR runs show abundances; a composition model shows its per-pixel "
            "probabilities. ALL panels share ONE scale (0–1 for probabilities, "
            "0–P99 for abundances) so brightness compares across components")
        lay_ab.addWidget(self.c_abund); self.c_abund.setMinimumHeight(300)
        lay_ab.addLayout(self._scale_row("abund", 1.0))
        self._add_fold(lay_ab, self.c_abund, "abundance maps", opened=True, key="abund")
        # band | abundance half-and-half on one row — raw evidence beside the
        # stage-1 split, no scrolling between them
        mrow = QHBoxLayout(); mrow.setSpacing(12)
        mrow.addWidget(card_maps, 1); mrow.addWidget(card_ab, 1)
        mrow_w = QWidget(); mrow_w.setLayout(mrow); body.addWidget(mrow_w)

        # 2) the clicked pixel's spectrum — full width but SHORT. It sits directly
        #    under the maps (a click lands there, the trace it produces should be the
        #    very next thing on screen) and a trace needs width, not half a screen of
        #    height; the pixel's own numbers ride along in the panel title.
        self.c_spec = Canvas()
        scard, slay = _card("Selected pixel spectrum — measured vs reconstructed")
        slay.addWidget(self.c_spec)
        self.c_spec.setMinimumHeight(190); self.c_spec.setMaximumHeight(220)
        body.addWidget(scard)

        # 3) per-pixel composition pie | the same composition summed over the map —
        #    the pie map and the number it adds up to belong on one row
        self.c_pie = Canvas(); self.c_comp = Canvas()
        pcard, play = _card("Per-pixel composition — pie per pixel (click a pixel)")
        play.addWidget(self.c_pie); self.c_pie.setMinimumHeight(380)
        ccard, clay = _card("Composition (overall)")
        clay.addWidget(self.c_comp); self.c_comp.setMinimumHeight(300)
        prow = QHBoxLayout(); prow.setSpacing(12)
        prow.addWidget(pcard, 3); prow.addWidget(ccard, 1)
        prow_w = QWidget(); prow_w.setLayout(prow); body.addWidget(prow_w)

        # 4) per-substance concentration (µM) maps — its own full-width row
        self.c_conc = Canvas()
        self.card_conc, lay_conc = _card(
            "Apparent SERS-equivalent concentration (µM) — from the composition model, "
            "or from a loaded calibration when one is given")
        vrow = QHBoxLayout(); vrow.setSpacing(6)
        _vl = QLabel("dispensed volume (µL) — 0 = off"); _vl.setObjectName("field")
        self.vol_spin = QDoubleSpinBox(); self.vol_spin.setDecimals(1)
        self.vol_spin.setRange(0.0, 100.0); self.vol_spin.setSingleStep(0.5)
        self.vol_spin.setValue(0.0); self.vol_spin.setFixedWidth(84)
        self.vol_spin.setToolTip(
            "volume of the droplet/ink you dispensed. When set, the summary bars "
            "also show the apparent amount = median µM × volume (pmol). APPARENT — "
            "it reads the SERS-equivalent concentration, not a mass balance.")
        self.vol_spin.valueChanged.connect(
            lambda _=0: self._plot_conc(self._res) if self._res is not None else None)
        vrow.addWidget(_vl); vrow.addWidget(self.vol_spin)
        _tl = QLabel("true µM (a,b,c) — blank = off"); _tl.setObjectName("field")
        self.true_edit = QLineEdit(); self.true_edit.setFixedWidth(110)
        self.true_edit.setPlaceholderText("12,12,12")
        self.true_edit.setToolTip("dispensed truth per substance, comma-separated in "
                                  "the panel order. Adds a red tick at each true value "
                                  "and a recovery % under the bars.")
        self.true_edit.editingFinished.connect(
            lambda: self._plot_conc(self._res) if self._res is not None else None)
        vrow.addWidget(_tl); vrow.addWidget(self.true_edit)
        # the VALIDATED quantification window, not the label range: held-out
        # recovery was demonstrated on the 3-24 uM grid, so µM claims outside the
        # window the user declares here are OOD no matter what the head says
        _rl2 = QLabel("report range µM (lo–hi) — 0 = model's"); _rl2.setObjectName("field")
        self.rep_lo = QDoubleSpinBox(); self.rep_lo.setDecimals(1)
        self.rep_lo.setRange(0.0, 1e6); self.rep_lo.setValue(0.0)
        self.rep_lo.setFixedWidth(70)
        self.rep_hi = QDoubleSpinBox(); self.rep_hi.setDecimals(1)
        self.rep_hi.setRange(0.0, 1e6); self.rep_hi.setValue(0.0)
        self.rep_hi.setFixedWidth(70)
        for _sp in (self.rep_lo, self.rep_hi):
            _sp.setToolTip("the range where held-out recovery was actually "
                           "demonstrated (e.g. 3–24 for the grid). Pixels outside "
                           "leave the medians and are counted as OOD. 0–0 falls "
                           "back to the model's own training-label range.")
            _sp.valueChanged.connect(
                lambda _=0: self._plot_conc(self._res) if self._res is not None else None)
        vrow.addWidget(_rl2); vrow.addWidget(self.rep_lo); vrow.addWidget(self.rep_hi)
        vrow.addStretch(1)
        lay_conc.addLayout(vrow)
        lay_conc.addWidget(self.c_conc); self.c_conc.setMinimumHeight(340)
        body.addWidget(self.card_conc)
        self.card_conc.setVisible(False)

        bodyw = QWidget(); bodyw.setLayout(body)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidget(bodyw)
        scroll.setStyleSheet("QScrollArea{background:transparent;}")
        outer.addWidget(scroll, 1)

        for cv, m in [(self.c_maps, "Load a test map, then Unmix"),
                      (self.c_pie, "Composition appears here"),
                      (self.c_comp, "Composition appears here"),
                      (self.c_spec, "Click a pixel in a map to see its spectrum")]:
            cv.placeholder(m)
        self.c_maps.mpl_connect("button_press_event", self._on_click)
        self.c_abund.mpl_connect("button_press_event", self._on_click)
        self.c_pie.mpl_connect("button_press_event", self._on_click)
        self.c_conc.mpl_connect("button_press_event", self._on_click)


    # ---- small builders ----
    def _compact_x(self, b, tip):
        """A small square ✕ clear button (overrides the tall ghost padding)."""
        b.setFixedSize(24, 24); b.setToolTip(tip)
        b.setStyleSheet("QPushButton{padding:0px; border:1px solid %s; border-radius:6px;"
                        "color:%s; font-size:12px;}" % (LINE, MUTE))

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
        set_substance_colors(self._colors)                # seed the shared colour state
        if self._res is not None:
            self._rebuild_swatches(self._res)
            self._rebuild_bandrow(self._res)   # the spin borders carry the colours too
            self._redraw()

    def _on_colors_changed(self):
        """Shared colours changed (e.g. from the top-bar picker) — resync + redraw."""
        self._colors = substance_colors()
        if self._res is not None:
            self._rebuild_swatches(self._res)
            self._rebuild_bandrow(self._res)   # the spin borders carry the colours too
            self._redraw()

    def _mk_lbl(self, text):
        lb = QLabel(text); lb.setObjectName("field"); return lb

    def _default_color(self, i):
        return SERIES[i % len(SERIES)]

    def _all_colors(self, r):
        """Colour per component (index in r.comps): its hue if a substance, else the
        chosen background colour (saved override or grey). Removed in the band-map
        rewrite; restored for the abundance card, which panels BLK/INK too."""
        nb = self._nb_colors(r)
        nbmap = {j: nb[i] for i, j in enumerate(r.nonbg)}
        return [nbmap.get(k, self._colors.get(r.comps[k], BG_GREY))
                for k in range(len(r.comps))]

    def _nb_colors(self, r):
        """Colour per non-background substance — a saved override or the default."""
        out = []
        for i, j in enumerate(r.nonbg):
            out.append(self._colors.get(r.comps[j], self._default_color(i)))
        return out

    def _bg_color(self, r):
        """Chosen background colour (first background component's override, or grey)."""
        for k in range(len(r.comps)):
            if r.bg_mask[k]:
                return self._colors.get(r.comps[k], BG_GREY)
        return BG_GREY

    # ---- marker bands driving the band maps ----
    BAND_HALF_WIDTH = 8.0                  # cm⁻¹ averaged either side of the band

    def _default_bands(self, r):
        """One marker band per non-background substance: its VIP band — the peak
        where its L2-normalised reference shape most exceeds every other reference,
        i.e. the least cross-talking band it has. Falls back to the plain argmax of
        the template when the references give no usable peak."""
        nb_names = [r.comps[j] for j in r.nonbg]
        if r.templates is None or r.wn is None:
            return {nm: float(np.median(r.wn)) for nm in nb_names}
        try:
            picks = vip_bands(r.wn, r.templates[r.nonbg], nb_names, k=1)
        except Exception:
            picks = {}
        out = {}
        for i, nm in enumerate(nb_names):
            wl = picks.get(nm) or []
            out[nm] = float(wl[0]) if wl else float(
                r.wn[int(np.argmax(r.templates[r.nonbg][i]))])
        return out

    def _parse_scale(self, key):
        """(lo, hi) from the card's manual min/max when ticked; else None = auto."""
        ui = self._scale_ui.get(key)
        if not ui:
            return None
        chk, mn, mx = ui
        if not chk.isChecked():
            return None
        lo, hi = float(mn.value()), float(mx.value())
        return (lo, hi) if hi > lo else None

    def _scale_row(self, key, hi_default):
        """manual min/max for EVERY panel in the card. The ramps themselves sit
        under each panel as real horizontal colour-bars in that panel's hue."""
        row = QHBoxLayout(); row.setSpacing(6)
        chk = QCheckBox("manual scale")
        chk.setToolTip("pin every panel in this card to this min–max; "
                       "unticked = auto")
        mn = QDoubleSpinBox(); mn.setDecimals(3); mn.setRange(-1e9, 1e9)
        mn.setValue(0.0); mn.setFixedWidth(92); mn.setEnabled(False)
        mx = QDoubleSpinBox(); mx.setDecimals(3); mx.setRange(-1e9, 1e9)
        mx.setValue(hi_default); mx.setFixedWidth(92); mx.setEnabled(False)
        self._scale_ui[key] = (chk, mn, mx)

        def _upd(_=None):
            mn.setEnabled(chk.isChecked()); mx.setEnabled(chk.isChecked())
            if self._res is not None:
                (self._plot_maps if key == "maps" else self._plot_abund)(self._res)

        chk.toggled.connect(_upd)
        mn.editingFinished.connect(_upd); mx.editingFinished.connect(_upd)
        row.addWidget(chk)
        row.addWidget(QLabel("min")); row.addWidget(mn)
        row.addWidget(QLabel("max")); row.addWidget(mx)
        row.addStretch(1)
        return row

    def _row_gs(self, fig, n, ny, nx):
        """The grid both map cards draw on: images across the top row, their colour
        ramps in a thin row directly under them.

        Two things this fixes. Every image cell is the SAME box, so the merged panel
        lines up with the components instead of floating above them — it used to keep
        the height the colour-bars stole from its neighbours. And the pair of rows is
        hugged to the height these maps actually need (a 60x30 map in a narrow column
        is short) and centred, instead of leaving the card two-thirds empty."""
        bot, top, cb = self._row_span(fig, n, ny, nx)
        gs = fig.add_gridspec(2, n, height_ratios=[1.0, cb], hspace=self._GS_HS,
                              wspace=self._GS_WS, left=self._GS_L, right=self._GS_R,
                              bottom=bot, top=top)
        # a resized card changes what "hugged" means; re-hug without replotting
        cv = fig.canvas
        old = getattr(cv, "_rowgs_cid", None)
        if old is not None:
            cv.mpl_disconnect(old)

        def _refit(_e=None, _gs=gs, _n=n, _ny=ny, _nx=nx, _f=fig):
            b, t, c = self._row_span(_f, _n, _ny, _nx)
            _gs.set_height_ratios([1.0, c])
            _gs.update(bottom=b, top=t)

        cv._rowgs_cid = cv.mpl_connect("resize_event", _refit)
        return gs

    def _row_span(self, fig, n, ny, nx):
        """(bottom, top, cb_ratio) for _row_gs — the two rows hugged to the height
        ny/nx maps need at this figure size and centred in what the card gives us.

        The ramp row is a FIXED thickness in inches, not a share of the image row:
        tie it to the image and a wide short map squeezes its own colour-bar into a
        hairline."""
        L, R, WS, HS = self._GS_L, self._GS_R, self._GS_WS, self._GS_HS
        BOT, TOP, BAR_IN = 0.08, 0.92, 0.11
        figw, figh = fig.get_size_inches()
        figh = max(float(figh), 1e-6)
        cell_w = (R - L) / (n + (n - 1) * WS) * float(figw)          # inches
        need = cell_w * (ny / max(nx, 1)) / figh                     # image row, frac
        need = min(max(need, 0.05), TOP - BOT)
        bar = BAR_IN / figh                                          # ramp row, frac
        # hspace is a fraction of the MEAN row height
        span = need + bar + HS * (need + bar) / 2.0
        if span > TOP - BOT:                       # too tall for the card — scale both
            k = (TOP - BOT) / span
            need *= k; bar *= k; span = TOP - BOT
        mid = (TOP + BOT) / 2.0
        return mid - span / 2.0, mid + span / 2.0, bar / need

    def _chan_lims(self, key, auto):
        """(lo, hi) for ONE band panel: its own manual pair when the band card's
        'manual scale' is ticked, else the auto P1-P99 handed in. Per panel, because
        one card-wide pair cannot stretch a weak channel without flattening a strong
        one -- and comparing channels is the whole point of these maps."""
        if not (self.chk_chan is not None and self.chk_chan.isChecked()):
            return auto
        ui = self._chan_scale.get(key)
        if not ui:
            return auto
        lo, hi = float(ui[0].value()), float(ui[1].value())
        return (lo, hi) if hi > lo else auto

    def _sync_chan_spins(self, keys, lims):
        """While manual is off the spins mirror what is drawn, so ticking the box
        starts from the picture already on screen instead of from an arbitrary 0-1500."""
        if self.chk_chan is None or self.chk_chan.isChecked():
            return
        for k, (lo, hi) in zip(keys, lims):
            ui = self._chan_scale.get(k)
            if not ui:
                continue
            for sp, v in zip(ui, (lo, hi)):
                sp.blockSignals(True); sp.setValue(float(v)); sp.blockSignals(False)

    def _rebuild_scalerow(self, r):
        """A min/max pair per band panel, coloured like the panel it drives."""
        while self.scalerow.count():
            it = self.scalerow.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._chan_scale = {}
        chk = QCheckBox("manual scale"); chk.setObjectName("field")
        chk.setToolTip("pin each band panel to its own min-max below; "
                       "unticked = that panel's own P1-P99")
        self.chk_chan = chk
        self.scalerow.addWidget(chk)
        cols = self._nb_colors(r)
        items = [(r.comps[j], cols[i]) for i, j in enumerate(r.nonbg)]
        items += [(f"@{wl:.0f}", self.EX_COLS[i % len(self.EX_COLS)])
                  for i, wl in enumerate(self._extra_bands)]
        keys = ([r.comps[j] for j in r.nonbg]
                + [f"extra{i}" for i in range(len(self._extra_bands))])
        for (nm, col), key in zip(items, keys):
            lb = self._mk_lbl(nm); lb.setStyleSheet(f"color:{col};font-weight:600;")
            self.scalerow.addWidget(lb)
            pair = []
            for what in ("min", "max"):
                sp = QDoubleSpinBox(); sp.setDecimals(1); sp.setRange(-1e9, 1e9)
                sp.setFixedWidth(76); sp.setEnabled(False)
                sp.setToolTip(f"{what} of the colour ramp for the {nm} panel")
                sp.editingFinished.connect(
                    lambda: self._res is not None and self._plot_maps(self._res))
                self.scalerow.addWidget(sp)
                pair.append(sp)
            self._chan_scale[key] = tuple(pair)

        def _tgl(on):
            for a, b in self._chan_scale.values():
                a.setEnabled(on); b.setEnabled(on)
            if self._res is not None:
                self._plot_maps(self._res)

        chk.toggled.connect(_tgl)
        self.scalerow.addStretch(1)

    def _add_fold(self, lay, canvas, name, opened, key):
        """Fold a heavy card: the button sits under the title, the canvas hides, and
        while hidden the card is NOT rendered at all — scrolling gets shorter and a
        run gets faster. Reopening replots if a result arrived meanwhile."""
        if not hasattr(self, "_fold_dirty"):
            self._fold_dirty = {}
        self._fold_dirty[key] = False
        btn = QPushButton(("▾ " if opened else "▸ ") + name)
        btn.setObjectName("ghost"); btn.setCheckable(True); btn.setChecked(opened)
        btn.setStyleSheet("text-align:left; padding:2px 6px;")
        canvas.setVisible(opened)

        def _tgl(on, c=canvas, k=key, b_=btn, nm=name):
            c.setVisible(on)
            b_.setText(("▾ " if on else "▸ ") + nm)
            if on and self._fold_dirty.get(k) and self._res is not None:
                (self._plot_maps if k == "maps" else self._plot_abund)(self._res)

        btn.toggled.connect(_tgl)
        lay.insertWidget(1, btn)                  # right under the card title

    def _band_of(self, r, name):
        """Chosen band for a substance, defaulting to its VIP band on first sight."""
        if name not in self._bands:
            self._bands.update({k: v for k, v in self._default_bands(r).items()
                                if k not in self._bands})
        return self._bands.get(name, float(np.median(r.wn)))

    def _band_image(self, r, wl):
        """Per-pixel intensity at ``wl`` — mean over the ±BAND_HALF_WIDTH window, so
        one noisy channel does not decide the pixel. Uses the measured (baseline-
        removed) spectra, NOT the unmixed abundances: this map is the raw evidence."""
        wn = np.asarray(r.wn, float)
        m = np.abs(wn - float(wl)) <= self.BAND_HALF_WIDTH
        if not m.any():                                   # window fell between channels
            m = np.zeros(len(wn), bool); m[int(np.argmin(np.abs(wn - float(wl))))] = True
        return np.asarray(r.spectra, float)[:, m].mean(axis=1)

    def _rebuild_bandrow(self, r):
        """A wavenumber spin box per substance, coloured like its panel."""
        while self.bandrow.count():
            it = self.bandrow.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.bandrow.addWidget(self._mk_lbl("bands (cm⁻¹):"))
        cols = self._nb_colors(r)
        lo, hi = float(np.min(r.wn)), float(np.max(r.wn))
        self._band_spins = {}
        for i, j in enumerate(r.nonbg):
            nm = r.comps[j]
            self.bandrow.addWidget(self._mk_lbl(nm))
            sp = QDoubleSpinBox(); sp.setDecimals(1); sp.setSingleStep(5.0)
            sp.setRange(lo, hi); sp.setValue(self._band_of(r, nm))
            sp.setFixedWidth(96)
            sp.setStyleSheet(f"QDoubleSpinBox{{border:2px solid {cols[i]};"
                             f"border-radius:6px;padding:1px 4px;}}")
            sp.setToolTip(f"wavenumber mapped for {nm} — intensity is averaged over "
                          f"±{self.BAND_HALF_WIDTH:.0f} cm⁻¹ around it")
            sp.valueChanged.connect(lambda v, name=nm: self._on_band(name, v))
            self.bandrow.addWidget(sp)
            self._band_spins[nm] = sp
        # ---- user-added extra panels: press + and type any wavenumber ----
        # (asked for when TBZ 1010 lit up OUTSIDE the leaf — an extra band beside the
        #  substance bands is how you check what that region actually is)
        for ei, wl in enumerate(self._extra_bands):
            esp = QDoubleSpinBox(); esp.setDecimals(1); esp.setSingleStep(5.0)
            esp.setRange(lo, hi); esp.setValue(float(wl)); esp.setFixedWidth(96)
            esp.setStyleSheet("QDoubleSpinBox{border:2px solid #8a94a3;"
                              "border-radius:6px;padding:1px 4px;}")

            def _moved(v, k=ei):
                self._extra_bands[k] = float(v)
                if self._res is not None:
                    self._plot_maps(self._res)

            esp.valueChanged.connect(_moved)
            self.bandrow.addWidget(esp)
        # plain buttons, fixed square, explicit padding — the ghost style's padding
        # clipped the +/− glyphs into blank squares at 28px ("지우기안된다")
        _pm_css = ("QPushButton{min-width:30px; max-width:30px; padding:1px 0; "
                   "font-size:14px; font-weight:600;}")
        plus = QPushButton("+"); plus.setObjectName("ghost")
        plus.setStyleSheet(_pm_css)
        plus.setToolTip("add a band panel at a wavenumber you type — e.g. a leaf or "
                        "substrate band, to see what a region that should be empty "
                        "actually is")
        minus = QPushButton("-"); minus.setObjectName("ghost")
        minus.setStyleSheet(_pm_css)
        minus.setEnabled(bool(self._extra_bands))
        minus.setToolTip("remove the last added band panel")

        def _plus():
            base = (self._extra_bands[-1] + 50.0 if self._extra_bands
                    else float(np.median(r.wn)))
            self._extra_bands.append(min(max(base, lo), hi))
            self._rebuild_bandrow(r)
            if self._res is not None:
                self._plot_maps(self._res)

        def _minus():
            if self._extra_bands:
                self._extra_bands.pop()
                self._rebuild_bandrow(r)
                if self._res is not None:
                    self._plot_maps(self._res)

        plus.clicked.connect(lambda _=False: _plus())
        minus.clicked.connect(lambda _=False: _minus())
        self.bandrow.addWidget(plus); self.bandrow.addWidget(minus)
        self.bandrow.addStretch(1)
        self._rebuild_scalerow(r)          # one min/max pair per panel, extras included

    def _on_band(self, name, value):
        self._bands[name] = float(value)
        if self._res is not None:
            self._plot_maps(self._res)     # only the band card depends on the choice

    def _rebuild_swatches(self, r):
        while self.swatches.count():
            it = self.swatches.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.swatches.addWidget(self._mk_lbl("colours:"))
        cols = self._nb_colors(r)
        for i, j in enumerate(r.nonbg):                    # substances
            self.swatches.addWidget(self._swatch(r.comps[j], cols[i]))
        for k in range(len(r.comps)):                      # background (recolourable too)
            if r.bg_mask[k]:
                self.swatches.addWidget(
                    self._swatch(r.comps[k], self._colors.get(r.comps[k], BG_GREY)))
        self.swatches.addStretch(1)

    def _swatch(self, name, color):
        b = QPushButton(name); b.setObjectName("ghost"); b.setFixedHeight(24)
        b.setStyleSheet(f"QPushButton{{border:2px solid {color};"
                        f"border-radius:6px;padding:2px 10px;color:{INK};}}")
        b.clicked.connect(lambda _=False, nm=name: self._pick_color(nm))
        return b

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
        set_substance_colors(self._colors)                # broadcast to every page
        COLOR_BUS.changed.emit()

    def _redraw(self):
        r = self._res
        if r is None:
            return
        self._plot_maps(r); self._plot_abund(r); self._plot_pies(r); self._plot_comp(r); self._plot_conc(r)
        if self._sel is not None:
            self._plot_spec(r, self._sel)

    def _method(self):
        return self.cmb_method.itemAt(1).widget().currentData()

    def _blank_tag(self):
        """Say on the label whether this model carries a blank class. Without one it
        cannot answer 'nothing here', so NNLS still supplies the substance mass and the
        background call — load a measured background map to take that over instead of
        retraining."""
        m = getattr(self, "dl_model", None)
        if not m:
            return ""
        blank = m.get("blank")
        if blank and blank in (m.get("subs") or []):
            return f"  ·  {blank} class ✓"
        return "  ·  no blank class"

    def _dl_judges_bg(self):
        """True when the loaded composition model carries a blank class AND the per-pixel
        model method is selected — then the model alone decides background."""
        m = getattr(self, "dl_model", None)
        return bool(self._method() == "dlpx" and m and m.get("blank")
                    and m.get("blank") in (m.get("subs") or []))

    def _sync_controls(self):
        """Grey out the gates that no longer get a vote, so the header shows at a glance
        which single rule will decide background."""
        if not hasattr(self, "thr"):        # _adopt_model can fire before the widgets exist
            return
        by_bg = bool(self.bg_paths)                    # a measured background map wins
        by_dl = self._dl_judges_bg()
        manual = not (by_bg or by_dl)
        self.chk_auto.setEnabled(manual)
        self.thr.itemAt(1).widget().setEnabled(manual and not self.chk_auto.isChecked())
        why = ("a loaded background map decides background" if by_bg else
               "the composition model's blank class decides background" if by_dl else "")
        for w in (self.chk_auto, self.thr.itemAt(1).widget()):
            w.setToolTip(f"not used — {why}" if why else "")

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

    def _toggle_opts(self, on):
        self.optbox.setVisible(on)
        self.opt_tgl.setText(("▾  " if on else "▸  ")
                             + "Options  (model · calibration · correction · thresholds)")

    def _adopt_model(self):
        """Adopt the composition model trained in the Model tab (Step 2), if any. Just
        stash it + label — do NOT re-run analysis here (that would block the GUI on the
        main thread the moment training finishes); the next Analyze picks it up."""
        if MODEL_BUS.model is None:
            return
        self.dl_model = MODEL_BUS.model
        self.dlm_lbl.setText("DL: " + (MODEL_BUS.origin or "trained model")
                             + self._blank_tag())
        self.dlm_lbl.setStyleSheet(""); self._sync_controls()
        self._refresh_calib_label()          # the model may carry its own calibration

    def _browse_dl(self):
        p, _ = QFileDialog.getOpenFileName(self, "DL model (.dlm from Recovery)", "",
                                           "DL / Pixel surface model (*.dlm *.psm);;all (*)")
        if not p:
            return
        try:
            if p.lower().endswith(".psm"):
                from pixel_surface import load_pixel_surface
                self.dl_model = load_pixel_surface(p)
            else:
                from dl_model import load_model
                self.dl_model = load_model(p)
            self.dlm_lbl.setText("DL: " + os.path.basename(p) + self._blank_tag())
            self.dlm_lbl.setStyleSheet(""); self._sync_controls()
            self._refresh_calib_label()      # the model may carry its own calibration
            if self._res is not None:
                self._apply(self._res)                    # redraw with the DL model in play
        except Exception as e:
            self.dl_model = None; self.dlm_lbl.setText("DL load failed")
            print(e, file=sys.stderr)

    def _browse_bg(self):
        """Pick one or more MEASURED background/blank maps (e.g. Pest/BLk) — they
        teach the analysis what 'nothing here' looks like on this substrate."""
        ps, _ = QFileDialog.getOpenFileNames(
            self, "Measured background map(s) (e.g. Pest/BLk)", "",
            "maps (*.csv *.txt);;all files (*)")
        if not ps:
            return
        self.bg_paths = list(ps)
        self.bg_lbl.setText("bg: " + (os.path.basename(ps[0]) if len(ps) == 1
                                      else f"{len(ps)} maps"))
        self.bg_x.setVisible(True); self._sync_controls()

    def _clear_bg(self):
        self.bg_paths = []; self.bg_lbl.setText(""); self.bg_x.setVisible(False)
        self._sync_controls()

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
        self.calib_path = None; self.cal_x.setVisible(False)
        self._refresh_calib_label()

    def _model_calib_path(self):
        """The calibration the loaded .dlm carries (train_model embeds the CSV), as a
        real file path — written to temp once per model so unmix can just read it."""
        m = self.dl_model if isinstance(self.dl_model, dict) else None
        txt = (m or {}).get("calib_csv_text")
        if not txt:
            return None
        import tempfile, hashlib
        h = hashlib.sha1(txt.encode("utf-8", "replace")).hexdigest()[:12]
        p = os.path.join(tempfile.gettempdir(), f"unmixr_model_calib_{h}.csv")
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(txt)
        return p

    def _effective_calib(self):
        """(path, origin): a CSV browsed here wins; else the calibration embedded in
        the model; else the one Quantify published. None when there is none at all.

        EXCEPT: in unmix_map a calib_path beats the model's µM head, so the AUTO
        sources must stand down when a composition model with a µM head is driving —
        otherwise loading a .dlm would silently swap its trained µM for the Langmuir
        inversion. Browsing a CSV here stays an explicit override, as before."""
        if self.calib_path:
            return self.calib_path, "loaded"
        m = self.dl_model if isinstance(self.dl_model, dict) else None
        if self._method() == "dlpx" and m and m.get("uM"):
            return None, None                  # the model's own µM head reports
        p = self._model_calib_path()
        if p:
            return p, "from model"
        if CALIB_BUS.path and os.path.exists(CALIB_BUS.path):
            return CALIB_BUS.path, "from Quantify"
        return None, None

    def _refresh_calib_label(self):
        p, org = self._effective_calib()
        if p is None:
            self.cal_lbl.setText("")
        elif org == "loaded":
            self.cal_lbl.setText("calib: " + os.path.basename(p))
        else:
            m = self.dl_model if isinstance(self.dl_model, dict) else {}
            nm = m.get("calib_csv_name") if org == "from model" else CALIB_BUS.origin
            self.cal_lbl.setText(f"calib: {org}" + (f" ({nm})" if nm else ""))
        self.cal_lbl.setStyleSheet("")

    def _browse_corr(self):
        """Load response_factors.csv (from Validate) → enable the solution-ratio toggle."""
        p, _ = QFileDialog.getOpenFileName(
            self, "response_factors.csv (from Validate → Export)", "", "CSV (*.csv)")
        if not p:
            return
        try:
            import csv as _csv
            rf = {}
            with open(p, newline="", encoding="utf-8-sig") as f:
                for row in _csv.DictReader(f):
                    rf[row["substance"]] = float(row["response_factor"])
            if not rf:
                raise ValueError("no rows")
        except Exception as exc:
            self.corr_lbl.setText("not a correction"); self.corr_lbl.setStyleSheet(f"color:{RED};")
            self.status.setText(f"{os.path.basename(p)} is not a response_factors.csv "
                                f"({type(exc).__name__}) — export it from the Validate tab.")
            self.status.setStyleSheet(f"color:{RED};"); return
        self.rf = rf
        self.corr_lbl.setText("correction ✓"); self.corr_lbl.setStyleSheet("")
        self.chk_corr.setEnabled(True); self.chk_corr.setChecked(True)

    def _on_corr(self):
        if self._res is not None:
            self._apply(self._res)              # recompute ratio/dominant + redraw

    def _rf_vec(self, r):
        """Response-factor vector aligned to the non-bg substances, or None when the
        solution-ratio correction is off / unavailable."""
        if not (self.chk_corr.isChecked() and self.rf):
            return None
        return np.array([self.rf.get(r.comps[j], 1.0) for j in r.nonbg], float)

    def _ratio_nb(self, r):
        """Per-pixel non-bg composition — response-corrected to the solution ratio when
        that toggle is on, else the raw surface ratio. Nothing is deleted: a substance
        that reads low reads low. (A 'min substance %' control used to zero out any
        substance whose MAP-WIDE mean fell below 5% and renormalise the rest — which
        erased exactly the trace levels this instrument exists to find, and erased them
        map-wide even where they were locally strong.)"""
        rf = self._rf_vec(r)
        if rf is None:
            rn = r.ratio_nb.astype(float).copy()
        else:
            rn = r.A[:, r.nonbg] / np.where(rf > 0, rf, 1.0)
        s = rn.sum(axis=1, keepdims=True)
        return np.divide(rn, s, out=np.zeros_like(rn), where=s > 0)

    def _saturated(self, r):
        """Detect ADC-like flat tops (three adjacent channels at the spectrum maximum).
        Fallback only — results carry sat_frac from the load-time detector now."""
        y = np.asarray(r.spectra, float)
        if y.ndim != 2 or y.shape[1] < 3:
            return np.zeros(r.n_pixels, bool)
        mx = np.nanmax(y, axis=1)
        tol = np.maximum(np.abs(mx) * 1e-6, 1e-12)
        top = y >= (mx[:, None] - tol[:, None])
        return np.any(top[:, :-2] & top[:, 1:-1] & top[:, 2:], axis=1)

    def _clipped(self, r):
        """Saturated pixels — the load-time detector's verdict (>1% of channels
        clipped: a flat-top run, or the jagged block an outside ALS pass leaves).
        The 1% floor keeps a 3-channel apex of a genuinely wide band out of
        quarantine. Falls back to the flat-top heuristic on old results."""
        sf = getattr(r, "sat_frac", None)
        if sf is not None:
            return np.asarray(sf, float) > 0.01
        return self._saturated(r)

    def _flagged(self, r):
        """Pixels whose reconstruction R² is below the threshold — ALWAYS computed and
        always drawn/counted. A low R² means the pure references fit this pixel badly
        (saturated, clipped, or a species the references don't cover); it is a warning
        about the pixel, not a verdict on its composition."""
        low_r2 = (np.zeros(r.n_pixels, bool) if getattr(r, "reliab", None) is None
                  else r.reliab < float(self.rel_thr.value()))
        return low_r2

    def _reliable(self, r):
        """Pixels kept in the composition. Low-R² pixels are dropped ONLY when the user
        opts in — otherwise they are merely flagged, so nothing disappears silently."""
        if not self.chk_rel.isChecked():
            return np.ones(r.n_pixels, bool)
        return ~self._flagged(r)

    def _hit(self, r):
        """Effective hit = a substance pixel (one rule decided that, see r.hit_rule)
        minus low-R² pixels the user chose to drop, minus quarantined saturation."""
        keep = r.hit & self._reliable(r)
        if self.chk_sat.isChecked():
            keep = keep & ~self._clipped(r)
        return keep

    def _mean_ratio(self, r):
        """SIGNAL-WEIGHTED mean composition over the hit pixels: each pixel's ratio
        weighted by its total baseline-removed intensity. A plain average lets the
        dim fringe pixels — where the marker evidence fades first — dilute the
        strong-ink composition (on the 260812 trio map it pulled THI 40%→32%);
        weighting by signal keeps the well-measured pixels in charge while every
        hit pixel still shows on the map."""
        rn = self._ratio_nb(r); hit = self._hit(r)
        if not hit.any():
            return rn.mean(axis=0)
        w = np.clip(np.asarray(r.spectra, float), 0.0, None).sum(axis=1)[hit]
        if w.sum() <= 0:
            return rn[hit].mean(axis=0)
        return (rn[hit] * w[:, None]).sum(axis=0) / w.sum()

    # ---- run ----
    def _run(self):
        if worker_busy(self):                             # already running — ignore
            return
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
            # When the composition model drives, IT decides the preprocessing: the pixel
            # spectra it reads must be prepared the way its training spectra were. The tab
            # used to pass the data-dir setting regardless, so a model trained on already
            # corrected references (baseline=False) got ALS applied a second time here and
            # nobody was told. The same flag also builds the NNLS templates, which is the
            # point — training used _refs(data_dir, model['baseline'], …).
            self._bl_override = None
            bl = bool(cfg["baseline"])
            if self._method() == "dlpx" and isinstance(self.dl_model, dict) \
                    and "baseline" in self.dl_model:
                mbl = bool(self.dl_model["baseline"])
                if mbl != bl:
                    self._bl_override = mbl
                bl = mbl
            params = dict(data_dir=self.data_dir, test_path=self.test,
                          method=self._method(), baseline=bl,
                          trim=cfg["trim"], min_frac=self.thr_value(),
                          hit_mode="auto" if self.chk_auto.isChecked() else "threshold",
                          calib_path=self._effective_calib()[0], dl_model=self.dl_model,
                          bg_map=self.bg_paths or None)
            if params["method"] == "dlpx" and self.dl_model is None:
                self.status.setText("no composition model — train one in the Model tab "
                                    "(or Load DL model…)")
                self.status.setStyleSheet(f"color:{RED};"); return
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self.pbar.show()
        import time as _t
        self._t0 = _t.time()
        self.status.setText("● started — the bar keeps moving while the worker runs")
        self.status.setStyleSheet(f"color:{MUTE};")
        start_worker(self, RealWorker(params, use_model=use_model),
                     done=self._apply, fail=self._error, progress=self._progress)

    def thr_value(self):
        return float(self.thr.itemAt(1).widget().value())

    def _on_auto(self, checked):
        self._sync_controls()                      # threshold unused in auto / model mode

    def _progress(self, msg):
        import time as _t
        el = _t.time() - getattr(self, "_t0", _t.time())
        self.btn.setText("Unmixing…")
        self.status.setText(f"● {msg}   ({el:.0f}s)")

    def _error(self, tb):
        self.pbar.hide()
        self.btn.setEnabled(True); self.btn.setText("Unmix")
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, r):
        self._res = r; self._sel = None
        self._click_axes = []            # one reset per run — every plot re-registers
        self.pbar.hide()
        self.btn.setEnabled(True); self.btn.setText("Unmix")
        ov = getattr(self, "_bl_override", None)
        _ns = int((self._clipped(r) & r.hit).sum())
        self.status.setText(f"done — {r.method.upper()}"
                            + (f" · {_ns} saturated px quarantined" if _ns else "")
                            + ("" if ov is None else
                            f" · baseline removal {'on' if ov else 'off'} — followed the "
                            f"model's own setting, not this folder's"))
        self.status.setStyleSheet(f"color:{MUTE};")
        nb = [r.comps[i] for i in r.nonbg]
        mr = self._mean_ratio(r)                          # corrected when toggle on
        eff_hit = self._hit(r)
        dom = nb[int(mr.argmax())] if len(nb) else r.dominant
        self.k_dom.set(dom, TEAL)
        self.k_n.set(str(int(np.sum(mr >= 0.05))), AMBER)
        self.k_hit.set(f"{eff_hit.mean():.0%}", BLUE)
        self.k_px.set(f"{r.n_pixels:,}", PURPLE)
        self._rebuild_swatches(r)
        self._rebuild_bandrow(r)     # seeds each substance's VIP band on a fresh run
        self._plot_maps(r); self._plot_abund(r); self._plot_pies(r); self._plot_comp(r); self._plot_conc(r)
        self.c_spec.placeholder("click a pixel in a map to see its spectrum")

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

    def _extent_origin(self, ux, uy):
        if self._flip():                                           # Y downwards
            return "upper", [ux.min() - .5, ux.max() + .5,
                             uy.max() + .5, uy.min() - .5]
        return "lower", [ux.min() - .5, ux.max() + .5,             # Y upwards (default)
                         uy.min() - .5, uy.max() + .5]

    def _plot_maps(self, r):
        """One RAW band-intensity map per substance, at the wavenumber picked for it,
        plus the same three channels read as R/G/B in one merged panel.

        These panels deliberately show NO unmixing: each is just the measured
        intensity in a ±window around one band. That is what a band-ratio / RGB
        readout of this map actually sees, so when the merged panel comes out one
        colour everywhere, it is showing that picking bands is not enough to tell
        these substances apart here — the unmixed pies below are the comparison.

        Every channel is stretched between its OWN P1 and P99, so a strong emitter
        cannot simply wash the merge out and the residual background floor (these
        spectra are baseline-removed, not background-free) does not grey everything
        out; what remains is genuine band overlap. The single-substance panels use
        the SAME stretch as the merge — their colour-bars carry the real intensity
        values, so nothing about the contrast is hidden."""
        from matplotlib.colors import LinearSegmentedColormap
        if not self.c_maps.isVisible():           # folded — skip the work entirely
            self._fold_dirty["maps"] = True
            return
        self._fold_dirty["maps"] = False
        self.c_maps.fig.clear()
        self._exp_maps = []            # (label, ax, cb_ax) for one-file-per-panel export
        nbcols = self._nb_colors(r)
        nb = [r.comps[j] for j in r.nonbg]
        rows, cc, ny, nx, ux, uy = self._grid_rc(r)
        origin, extent = self._extent_origin(ux, uy)

        bands = [self._band_of(r, nm) for nm in nb]
        chans = [self._band_image(r, wl) for wl in bands]
        extras = list(self._extra_bands)                  # user-added bands
        ex_chans = [self._band_image(r, wl) for wl in extras]
        n = len(nb) + 1 + len(extras)                     # merge + substances + extras
        ex_cols = [self.EX_COLS[i % len(self.EX_COLS)] for i in range(len(extras))]
        # auto = each channel over its OWN P1..P99; the scale row can override any
        # single panel without touching its neighbours
        _auto = [(float(np.quantile(v, 0.01)), float(np.quantile(v, 0.99)))
                 for v in chans + ex_chans]
        _auto = [(a, b if b > a else a + 1.0) for a, b in _auto]   # never a zero span
        _keys = ([r.comps[j] for j in r.nonbg]
                 + [f"extra{i}" for i in range(len(extras))])
        _all_lims = [self._chan_lims(k, a) for k, a in zip(_keys, _auto)]
        self._sync_chan_spins(_keys, _all_lims)
        lims = _all_lims[:len(nb)]
        ex_lims = _all_lims[len(nb):]
        gs = self._row_gs(self.c_maps.fig, n, ny, nx)

        # ---- merged R/G/B: each channel stretched over its own P1..P99 ----
        ax = self.c_maps.style(self.c_maps.fig.add_subplot(gs[0, 0]))
        ax.set_anchor("S")                       # sit on the ramp line, like the rest
        self.c_maps.fig.add_subplot(gs[1, 0]).set_axis_off()   # holds the merge's box
        cols = np.array([to_rgb(c) for c in (nbcols + ex_cols)])
        norm = np.stack([np.clip((v - va) / (vb - va), 0.0, 1.0)
                         for v, (va, vb) in zip(chans + ex_chans, lims + ex_lims)], 1)
        # additive merge blows out to white wherever several channels are strong
        # ("rgb merge 너무 밝다") - keep the hue, normalise the brightness:
        # where the channel sum exceeds 1, divide by it instead of clipping.
        wsum = np.maximum(norm.sum(axis=1, keepdims=True), 1.0)
        img = np.zeros((ny, nx, 3))
        img[rows, cc] = np.clip((norm / wsum) @ cols * np.minimum(
            norm.sum(axis=1, keepdims=True), 1.0), 0.0, 1.0)
        ax.imshow(img, extent=extent, origin=origin, aspect="equal",
                  interpolation="nearest")
        ax.set_title("merged (R/G/B)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        # no legend under the merge — the per-panel titles already carry name + band
        self._exp_maps.append(("band_merged", ax, None))
        self._click_axes.append(ax)

        # ---- one panel per substance, its own band ----
        # No colour-bars. `aspect="equal"` shrinks the image inside its axes, but a
        # colorbar sizes itself off the FULL axes box, so the bar always came out
        # taller than the map beside it. These panels are a picture of where a band
        # is strong; the numbers behind them are in the CSV export.
        for i, nm in enumerate(nb):
            ax = self.c_maps.style(self.c_maps.fig.add_subplot(gs[0, i + 1]))
            ax.set_anchor("S")
            grid = np.zeros((ny, nx)); grid[rows, cc] = chans[i]
            cmap = LinearSegmentedColormap.from_list("m", ["#0b0d10", nbcols[i]])
            _im = ax.imshow(grid, extent=extent, origin=origin, aspect="equal",
                            interpolation="nearest", cmap=cmap,
                            vmin=lims[i][0], vmax=lims[i][1])
            # mathtext, not "cm⁻¹" — Arial has no superscript-minus glyph, so the
            # literal character renders as a box in the exported PNG
            ax.set_title(f"{nm} @ {bands[i]:.0f} cm$^{{-1}}$", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            # ramp UNDER the panel — horizontal bars share the panel's width, so
            # (unlike the old vertical ones) they cannot outgrow the map
            cb = self.c_maps.fig.colorbar(
                _im, cax=self.c_maps.fig.add_subplot(gs[1, i + 1]),
                orientation="horizontal")
            cb.ax.tick_params(labelsize=7, colors="black")
            cb.outline.set_linewidth(0.5)
            self._exp_maps.append((f"band_{nm}", ax, cb.ax))
            self._click_axes.append(ax)
        for ei, wl in enumerate(extras):                   # user-added bands, own hue
            v = ex_chans[ei]
            va, vb = ex_lims[ei]
            ax = self.c_maps.style(
                self.c_maps.fig.add_subplot(gs[0, len(nb) + 1 + ei]))
            ax.set_anchor("S")
            grid = np.zeros((ny, nx)); grid[rows, cc] = v
            cmap = LinearSegmentedColormap.from_list("m", ["#0b0d10", ex_cols[ei]])
            _im = ax.imshow(grid, extent=extent, origin=origin, aspect="equal",
                            interpolation="nearest", cmap=cmap, vmin=va, vmax=vb)
            ax.set_title(f"@ {wl:.0f} cm$^{{-1}}$", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            cb = self.c_maps.fig.colorbar(
                _im, cax=self.c_maps.fig.add_subplot(gs[1, len(nb) + 1 + ei]),
                orientation="horizontal")
            cb.ax.tick_params(labelsize=7, colors="black")
            cb.outline.set_linewidth(0.5)
            self._exp_maps.append((f"band_extra_{wl:.0f}", ax, cb.ax))
            self._click_axes.append(ax)
        self.c_maps.draw_idle()          # gridspec already carries explicit margins

    def _plot_abund(self, r):
        """Merged false-colour composite PLUS one panel per component (background
        included), one row. This is the OLD band-card view, kept because stage-1
        unmixing runs in every path: the merge paints every substance's abundance
        in its colour; each single panel is that component on black→colour with a
        colour-bar. BLK/INK panels show where the gate sees non-analyte."""
        from matplotlib.colors import LinearSegmentedColormap
        if not self.c_abund.isVisible():          # folded — skip the work entirely
            self._fold_dirty["abund"] = True
            return
        self._fold_dirty["abund"] = False
        self.c_abund.fig.clear()
        self._exp_abund = []
        if getattr(r, "A", None) is None:                  # some result types carry no A
            self.c_abund.draw_idle(); return
        nbcols = self._nb_colors(r)
        allcols = self._all_colors(r)
        Anb = r.A[:, r.nonbg]
        mscale = float(np.quantile(Anb.sum(axis=1), 0.99)) or 1.0
        # ONE scale for every panel, background included — per-panel scales made the
        # row unreadable (INK 0–0.08 next to BLK 0–0.8 looked equally bright).
        # Probabilities get the natural fixed 0–1; abundances share a global P99.
        Aall = np.asarray(r.A, float)
        amax = float(np.nanmax(Aall)) if Aall.size else 1.0
        vshared = 1.0 if amax <= 1.05 else (float(np.quantile(Aall, 0.99)) or 1.0)
        _man = self._parse_scale("abund")
        vlo = _man[0] if _man is not None else 0.0
        if _man is not None:
            vshared = _man[1]
        rows, cc, ny, nx, ux, uy = self._grid_rc(r)
        origin, extent = self._extent_origin(ux, uy)
        panels = [("merged", None)] + [(r.comps[k], k) for k in range(len(r.comps))]
        n = len(panels)
        gs = self._row_gs(self.c_abund.fig, n, ny, nx)
        for idx, (title, k) in enumerate(panels):
            ax = self.c_abund.style(self.c_abund.fig.add_subplot(gs[0, idx]))
            ax.set_anchor("S")
            cb = None
            if k is None:
                cols = np.array([to_rgb(c) for c in nbcols])
                _nrm = np.clip(Anb / mscale, 0.0, 1.0)
                _ws = np.maximum(_nrm.sum(axis=1, keepdims=True), 1.0)
                img = np.zeros((ny, nx, 3))
                img[rows, cc] = np.clip((_nrm / _ws) @ cols * np.minimum(
                    _nrm.sum(axis=1, keepdims=True), 1.0), 0.0, 1.0)
                ax.imshow(img, extent=extent, origin=origin, aspect="equal",
                          interpolation="nearest")
                # empty ramp cell — it holds the merge's image box the same size
                self.c_abund.fig.add_subplot(gs[1, idx]).set_axis_off()
                # short — a long title runs into the neighbouring panel
                title = f"merged ({vlo:.3g}–{vshared:.3g})"
            else:
                sc = vshared
                grid = np.zeros((ny, nx)); grid[rows, cc] = r.A[:, k]
                cmap = LinearSegmentedColormap.from_list("m", ["#0b0d10", allcols[k]])
                # no colour-bars — they size off the full axes box and shrink the
                # map beside the merge (same reason the band card dropped them);
                # the numbers live in per_pixel.csv (A_ columns)
                _im = ax.imshow(grid, extent=extent, origin=origin, aspect="equal",
                                interpolation="nearest", cmap=cmap, vmin=vlo, vmax=sc)
                cb = self.c_abund.fig.colorbar(
                    _im, cax=self.c_abund.fig.add_subplot(gs[1, idx]),
                    orientation="horizontal")
                cb.ax.tick_params(labelsize=7, colors="black")
                cb.outline.set_linewidth(0.5)
                title = title + (" (bkg)" if r.bg_mask[k] else "")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            self._exp_abund.append((f"abund_{title.split(' ')[0]}", ax,
                                    cb.ax if cb is not None else None))
            self._click_axes.append(ax)
        self.c_abund.draw_idle()         # gridspec already carries explicit margins

    def _plot_conc(self, r):
        """Per-substance apparent SERS-equivalent concentration (µM) heat-maps — only when a
        dilution-series calibration has been applied."""
        if not getattr(r, "calibrated", False) or r.conc is None:
            self.card_conc.setVisible(False)
            return
        self.card_conc.setVisible(True)
        self.c_conc.fig.clear()
        self._exp_conc = []
        nb = [r.comps[i] for i in r.nonbg]; nbcols = self._nb_colors(r)
        rows, cc, ny, nx, ux, uy = self._grid_rc(r)
        origin, extent = self._extent_origin(ux, uy)
        n = (len(nb) or 1) + 1                             # + summary bars at the end
        hit = self._hit(r)                                 # exclude saturated/low-R² px
        # SHARED µM colour axis across substances, so the maps are directly comparable
        um_all = r.conc * 1e6
        # the model's own out-of-range judgment: per-pixel OOD flags plus the stored
        # reportable range. A weak binder (DQ) has a nearly flat response, so its
        # inversion EXPLODES on spurious signal — thousands of µM on a 3–500 µM
        # training range. Those pixels may not silently drive the medians.
        ood = getattr(r, "conc_ood", None)
        rngs = getattr(r, "conc_ranges", None)
        hi_um = None
        lo_um = None
        if rngs is not None:
            _h = np.asarray(rngs, float)[:, 1] * 1e6
            hi_um = np.where(np.isfinite(_h), _h, np.inf)
        # user-declared VALIDATED window beats the label range
        _ulo, _uhi = float(self.rep_lo.value()), float(self.rep_hi.value())
        if _uhi > _ulo > 0 or (_ulo == 0 and _uhi > 0):
            hi_um = np.full(len(nb), _uhi)
            lo_um = np.full(len(nb), _ulo) if _ulo > 0 else None
        vmask = hit[:, None] & np.isfinite(um_all) & (um_all > 0)
        vals = um_all[vmask]
        vmax = float(np.quantile(vals, 0.90)) if vals.size else 1.0
        if hi_um is not None and np.isfinite(hi_um).any():
            # never stretch the shared ramp past the calibrated range — one exploding
            # substance was blanking every other panel
            vmax = min(vmax, float(np.nanmax(hi_um[np.isfinite(hi_um)])))
        vmax = vmax or 1.0
        from matplotlib.colors import LinearSegmentedColormap
        for i, nm in enumerate(nb):
            ax = self.c_conc.style(self.c_conc.fig.add_subplot(1, n, i + 1))
            um = np.where(hit & np.isfinite(um_all[:, i]) & (um_all[:, i] > 0),
                          um_all[:, i], np.nan)
            grid = np.full((ny, nx), np.nan); grid[rows, cc] = um
            cmap = LinearSegmentedColormap.from_list("m", ["#0b0d10", nbcols[i]])
            cmap.set_bad("#0b0d10")
            ax.set_facecolor("#0b0d10")
            im = ax.imshow(grid, extent=extent, origin=origin, aspect="equal",
                           interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
            cb = self.c_conc.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.48, aspect=14)
            cb.ax.tick_params(labelsize=8, colors="black")       # same 0..vmax on every panel
            self._exp_conc.append((f"uM_{nm}", ax, cb.ax))
            ax.set_title(f"{nm} (µM; scale capped at hit-pixel P90)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            self._click_axes.append(ax)
        # ---- summary bars: the maps show WHERE, this shows HOW MUCH ----
        axb = self.c_conc.style(self.c_conc.fig.add_subplot(1, n, n))
        med = np.full(len(nb), np.nan); q1 = med.copy(); q3 = med.copy()
        bad_frac = np.zeros(len(nb))
        for i in range(len(nb)):
            sel = hit if hit.any() else np.ones(r.n_pixels, bool)
            v = um_all[sel, i]
            fin = np.isfinite(v) & (v > 0)
            bad = np.zeros(len(v), bool)
            if ood is not None:
                bad |= np.asarray(ood, bool)[sel, i]
            if hi_um is not None and np.isfinite(hi_um[i]):
                bad |= fin & (v > hi_um[i])
            if lo_um is not None:
                bad |= fin & (v < lo_um[i])
            keep = fin & ~bad
            bad_frac[i] = float(bad[fin].mean()) if fin.any() else 0.0
            vv = v[keep]
            if vv.size:
                med[i], q1[i], q3[i] = (float(np.median(vv)),
                                        float(np.quantile(vv, 0.25)),
                                        float(np.quantile(vv, 0.75)))
        xs = np.arange(len(nb))
        ok = np.isfinite(med)
        axb.bar(xs[ok], med[ok], width=0.6, color=[nbcols[i] for i in np.where(ok)[0]],
                edgecolor="none", zorder=2)
        axb.errorbar(xs[ok], med[ok],
                     yerr=[(med - q1)[ok], (q3 - med)[ok]],
                     fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=3)
        vol = float(self.vol_spin.value()) if hasattr(self, "vol_spin") else 0.0
        tv = None
        txt = self.true_edit.text().strip() if hasattr(self, "true_edit") else ""
        if txt:
            try:
                parts = [float(t) for t in txt.replace(" ", "").split(",")]
                if len(parts) == len(nb) and all(v > 0 for v in parts):
                    tv = parts
            except ValueError:
                tv = None
        if tv is not None:                                 # red tick = dispensed truth
            axb.plot(xs, tv, ls="none", marker="_", ms=16, mew=1.8, color=RED,
                     zorder=4)
        for i in np.where(ok)[0]:
            lab = f"{med[i]:.1f}"
            if tv is not None:
                lab += chr(10) + f"rec {100 * med[i] / tv[i]:.0f}%"
            if vol > 0:                                    # µM × µL = pmol
                lab += chr(10) + f"≈{med[i] * vol:.0f} pmol"
            if bad_frac[i] >= 0.05:                        # the hole must be visible
                lab += chr(10) + f"({bad_frac[i] * 100:.0f}% OOD excl.)"
            axb.annotate(lab, (xs[i], q3[i]), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8,
                         color=INK)
        # a substance whose hit pixels are ALL out of range gets a verdict, not a bar
        for i in np.where(~ok)[0]:
            if bad_frac[i] > 0:
                axb.annotate("not reportable" + chr(10)
                             + f"({bad_frac[i] * 100:.0f}% OOD)",
                             (xs[i], 0), xytext=(0, 8), textcoords="offset points",
                             ha="center", fontsize=8, color=RED)
        axb.set_xticks(xs); axb.set_xticklabels(nb, fontsize=8)
        axb.set_ylabel("median µM (in-range hit px, IQR)"
                       + (f" · {vol:g} µL" if vol > 0 else ""), fontsize=8)
        axb.tick_params(labelsize=8)
        axb.set_ylim(bottom=0)
        self._exp_conc.append(("uM_summary_bars", axb, None))
        self.c_conc.fig.tight_layout(); self.c_conc.draw_idle()

    # Final pie-map style (settled with the 260812 trio map): pure black ground,
    # the full measurement grid in white so the map reads as MAP DATA, cell-filling
    # pies, and a heavier white outline tracing the hit region. Low-R² ✕ marks are
    # gone for good — on a badly-fit map they covered every pixel and said nothing;
    # the R² still reaches the user via the pixel click-out and the CSV export.
    PIE_BG = "#000000"
    PIE_GRID = "#ffffff"

    def _plot_pies(self, r):
        from matplotlib.collections import LineCollection
        ax = self.c_pie.new_ax(); self._click_axes.append(ax)
        cols = self._nb_colors(r)
        x, y = r.coords[:, 0], r.coords[:, 1]
        ux, uy = np.unique(x), np.unique(y)
        sx = float(np.median(np.diff(ux))) if len(ux) > 1 else 1.0
        sy = float(np.median(np.diff(uy))) if len(uy) > 1 else 1.0
        rad = sx * 0.5                                    # pies fill their cell
        hit = self._hit(r)                                # substance pixels kept
        ax.set_facecolor(self.PIE_BG)
        # every cell boundary — the empty cells are measured background, not canvas
        gsegs = [[(ux[0] - sx / 2 + j * sx, uy[0] - sy / 2),
                  (ux[0] - sx / 2 + j * sx, uy[-1] + sy / 2)]
                 for j in range(len(ux) + 1)]
        gsegs += [[(ux[0] - sx / 2, uy[0] - sy / 2 + i * sy),
                   (ux[-1] + sx / 2, uy[0] - sy / 2 + i * sy)]
                  for i in range(len(uy) + 1)]
        ax.add_collection(LineCollection(gsegs, colors=self.PIE_GRID,
                                         linewidths=0.5, zorder=1))
        # saturated pixels are simply NOT classified — they render as empty cells
        # like background ("분류하지 말고 넘기는게 낫지 않나"); the run status and
        # per_pixel.csv carry the count, so the hole is documented, just not loud
        if r.method == "model":                           # classifier → one class/pixel
            dom = r.ratio_nb.argmax(axis=1)
            if hit.any():
                ax.scatter(x[hit], y[hit], c=[cols[dom[i]] for i in np.where(hit)[0]],
                           marker="s", s=16, edgecolors="none")
        else:                                             # per-pixel pie for hit pixels
            ratio_nb = self._ratio_nb(r)                  # corrected when toggle on
            wedges, wcols = [], []
            for i in np.where(hit)[0]:
                a0 = 90.0
                for k, frac in enumerate(ratio_nb[i]):
                    if frac <= 0.002:
                        continue
                    a1 = a0 - frac * 360.0
                    wedges.append(Wedge((x[i], y[i]), rad, a1, a0)); wcols.append(cols[k])
                    a0 = a1
            if wedges:
                ax.add_collection(PatchCollection(wedges, facecolors=wcols,
                                                  edgecolors="none"))
        # hit-region outline: cell-edge segments, twice the grid weight
        xi = {v: j for j, v in enumerate(ux)}; yi = {v: i for i, v in enumerate(uy)}
        H = np.zeros((len(uy), len(ux)), bool)
        H[[yi[v] for v in y[hit]], [xi[v] for v in x[hit]]] = True
        segs = []
        for i in range(len(uy)):
            for j in range(len(ux)):
                if not H[i, j]:
                    continue
                x0, y0 = ux[j] - sx / 2, uy[i] - sy / 2
                x1, y1 = ux[j] + sx / 2, uy[i] + sy / 2
                if i == 0 or not H[i - 1, j]:
                    segs.append([(x0, y0), (x1, y0)])
                if i == len(uy) - 1 or not H[i + 1, j]:
                    segs.append([(x0, y1), (x1, y1)])
                if j == 0 or not H[i, j - 1]:
                    segs.append([(x0, y0), (x0, y1)])
                if j == len(ux) - 1 or not H[i, j + 1]:
                    segs.append([(x1, y0), (x1, y1)])
        if segs:
            ax.add_collection(LineCollection(segs, colors=self.PIE_GRID,
                                             linewidths=1.6, zorder=6))
        self._sel_arts = []                      # figs were cleared — old rings gone
        self._pie_ax = ax
        self._exp_pie = [("composition_pies", ax, None)]
        ax.set_xlim(x.min() - sx, x.max() + sx)
        ax.set_ylim(*((y.max() + sy, y.min() - sy) if self._flip()
                      else (y.min() - sy, y.max() + sy)))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        # legend OUTSIDE the map (below), so it never covers the pixels
        handles = [Patch(facecolor=cols[i], label=r.comps[j])
                   for i, j in enumerate(r.nonbg)] + \
                  [Patch(facecolor=self.PIE_BG, label="background")]
        ax.legend(handles=handles, fontsize=9, framealpha=0.0, labelcolor="black",
                  loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=len(handles), frameon=False)
        self.c_pie.fig.tight_layout(); self.c_pie.draw_idle()

    def _update_sel_rings(self, r):
        """One ring on EVERY map (band, abundance, pie) at the clicked pixel — the
        click lands far from the pie, so the ring must appear where you clicked."""
        for art in getattr(self, "_sel_arts", []):
            try:
                art.remove()
            except Exception:
                pass
        self._sel_arts = []
        if self._sel is None:
            return
        px = float(r.coords[self._sel, 0]); py = float(r.coords[self._sel, 1])
        for ax in self._click_axes:
            try:
                self._sel_arts.append(
                    ax.scatter([px], [py], s=120, facecolors="none",
                               edgecolors=BLUE, linewidths=1.8, zorder=7))
            except Exception:
                pass
        for c in (self.c_maps, self.c_abund, self.c_pie, self.c_conc):
            c.draw_idle()

    def _mark_sel(self, ax, r):
        """Draw (or move) the selection ring as ONE artist — a click must not
        rebuild 2400 wedges; that is what made clicking feel dead on big maps."""
        art = getattr(self, "_sel_art", None)
        if art is not None:
            try:
                art.remove()
            except Exception:
                pass
            self._sel_art = None
        if self._sel is not None:
            self._sel_art = ax.scatter(
                [r.coords[self._sel, 0]], [r.coords[self._sel, 1]], s=120,
                facecolors="none", edgecolors=BLUE, linewidths=1.8, zorder=7)

    def _plot_comp(self, r):
        ax = self.c_comp.new_ax()
        cols = self._nb_colors(r); nb = [r.comps[i] for i in r.nonbg]
        mr = self._mean_ratio(r)                          # corrected when toggle on
        keep = [i for i in range(len(nb)) if mr[i] >= 0.01] or [int(mr.argmax())]
        ax.pie([mr[i] for i in keep], labels=[nb[i] for i in keep],
               colors=[cols[i] for i in keep], autopct="%1.0f%%",
               textprops={"fontsize": 10, "color": INK})
        ax.set_aspect("equal")
        self.c_comp.fig.tight_layout(); self.c_comp.draw_idle()

    def _plot_spec(self, r, i):
        ax = self.c_spec.new_ax()
        axis = r.wn if r.wn is not None else np.arange(r.spectra.shape[1])
        meas = np.asarray(r.spectra[i], float)
        mm = meas.max() or 1.0
        ax.plot(axis, meas / mm, lw=1.3, color=INK, label="measured")
        ratio_nb = self._ratio_nb(r)                      # corrected when toggle on
        if r.templates is not None:
            # ONE curve per substance — A_k x template_k in the substance's colour —
            # instead of a single summed "reference mix" that matched nothing visibly
            # ("green이 뭘 의미하는지 모르겠음"). Which peak belongs to whom, and how
            # much, is what this panel is for. The thin grey dashed curve is their sum:
            # the actual fit under NNLS/MCR; under the composition model A holds
            # probabilities, so it only shows the references the model leaned on,
            # not a goodness-of-fit.
            recon = r.A[i] @ r.templates
            rmax = float(recon.max()) or 1.0
            cols = self._nb_colors(r)
            for k, j in enumerate(r.nonbg):
                contrib = r.A[i, j] * r.templates[j]
                if float(contrib.max()) / rmax < 0.02:
                    continue                       # absent substance — no clutter
                ax.plot(axis, contrib / rmax, lw=1.0, color=cols[k], alpha=0.9,
                        label=f"{r.comps[j]} {ratio_nb[i, k] * 100:.0f}%")
            lab = ("references the model picked" if r.method == "dlpx"
                   else "fit (sum)")
            ax.plot(axis, recon / rmax, lw=0.9, color=FAINT, ls="--", label=lab)
        xp, yp = r.coords[i]
        rat = "  ·  ".join(f"{r.comps[j]} {ratio_nb[i, k] * 100:.0f}%"
                           for k, j in enumerate(r.nonbg) if ratio_nb[i, k] > 0.02)
        tag = rat if r.hit[i] else "background"
        if (not r.hit[i] and getattr(r, "bg_score", None) is not None
                and r.bg_score[i] >= r.bg_thr):
            tag = f"background (matches measured bg, score {r.bg_score[i]:.2f})"
        if getattr(r, "sat_frac", None) is not None and r.sat_frac[i] > 0:
            tag += (f"  (!) clipped — {r.sat_frac[i] * 100:.0f}% of channels were "
                    "saturated and bridged")
        if self._flagged(r)[i]:                           # low-R² warning
            r2 = float(r.reliab[i]) if getattr(r, "reliab", None) is not None else 0.0
            tag += (f"  (!) low R²={r2:.2f} — references fit this pixel badly"
                    + (" (excluded)" if self.chk_rel.isChecked() else ""))
        if getattr(r, "conc", None) is not None and r.hit[i]:   # absolute µM per pixel
            um = r.conc[i] * 1e6
            cs = "  ·  ".join(f"{r.comps[j]} {um[k]:.3g}µM" for k, j in enumerate(r.nonbg)
                              if np.isfinite(um[k]) and um[k] > 0)
            sat = ("  (!) saturated" if r.pp_theta is not None
                   and r.pp_theta[i] > 0.85 else "")
            ood_names = [r.comps[j] for k, j in enumerate(r.nonbg) if not np.isfinite(um[k])]
            if cs:
                tag += f"  |  {cs}{sat}"
            if ood_names:
                tag += "  |  OOD: " + ", ".join(ood_names)
        # the pixel readout goes in the TITLE. The card is short now, and this text
        # was being built and then thrown away — composition, µM, the low-R² warning
        # and the OOD list never reached the screen at all.
        ax.set_title(f"({xp:g}, {yp:g})  ·  {tag}", fontsize=9, pad=4)
        ax.set_xlabel("Raman shift (cm$^{-1}$)", labelpad=1); ax.set_yticks([])
        # legend inside the axes — a short card cannot spare a strip under the plot
        ax.legend(fontsize=8, framealpha=0.0, labelcolor="black",
                  loc="upper right", ncol=6, frameon=False,
                  handlelength=1.4, columnspacing=0.9)
        self.c_spec.fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.20)
        self.c_spec.draw_idle()

    # ---- interaction ----
    def _on_click(self, event):
        r = self._res
        if r is None or event.xdata is None or event.inaxes not in self._click_axes:
            return
        d = ((r.coords[:, 0] - event.xdata) ** 2
             + (r.coords[:, 1] - event.ydata) ** 2)
        self._sel = int(d.argmin())
        self._plot_spec(r, self._sel)
        self._update_sel_rings(r)                # rings on every map, never a rebuild

    # ---- export ----
    def _export(self):
        if self._res is None:
            self.status.setText("run first, then export")
            self.status.setStyleSheet(f"color:{RED};"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res; nb = [r.comps[i] for i in r.nonbg]
        # both aggregates, so the pie can be redrawn either way: the plain average
        # and the signal-weighted one the app now displays (see _mean_ratio)
        mr_un = (r.ratio_nb[self._hit(r)].mean(axis=0) if self._hit(r).any()
                 else r.ratio_nb.mean(axis=0))
        mr_w = self._mean_ratio(r)
        write_csv(os.path.join(d, "composition.csv"),
                  ["substance", "mean_ratio_unweighted", "mean_ratio_signal_weighted"],
                  [[nm, f"{mr_un[i]:.4f}", f"{mr_w[i]:.4f}"]
                   for i, nm in enumerate(nb)])
        inten = r.spectra.sum(axis=1)                      # total baseline-removed signal
        cal = getattr(r, "conc", None) is not None
        has_bg = getattr(r, "bg_score", None) is not None
        has_sat = getattr(r, "sat_frac", None) is not None
        head = (["x", "y", "hit", "total_intensity"]
                + [f"ratio_{nm}" for nm in nb] + [f"A_{c}" for c in r.comps]
                + ([f"conc_uM_{nm}" for nm in nb] if cal else []) + ["reliability_r2"]
                + (["clipped_frac"] if has_sat else [])
                + (["bg_match"] if has_bg else []))
        rows = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}", int(r.hit[i]),
                 f"{inten[i]:.4f}"]
                + [f"{r.ratio_nb[i, k]:.4f}" for k in range(len(nb))]
                + [f"{r.A[i, k]:.5f}" for k in range(len(r.comps))]
                + ([f"{r.conc[i, k] * 1e6:.4g}" if np.isfinite(r.conc[i, k]) else "OOD"
                    for k in range(len(nb))] if cal else [])
                + [f"{r.reliab[i]:.4f}"]
                + ([f"{r.sat_frac[i]:.4f}"] if has_sat else [])
                + ([f"{r.bg_score[i]:.4f}"] if has_bg else [])
                for i in range(r.n_pixels)]
        write_csv(os.path.join(d, "per_pixel.csv"), head, rows)
        ncsv = 2
        # band-map values per pixel (extras included) — per_pixel.csv carries the
        # unmixed numbers; this carries what the band panels actually display
        _bands = [(nm, self._band_of(r, nm)) for nm in nb]
        _extras = list(self._extra_bands)
        _chans = ([self._band_image(r, wl) for _nm, wl in _bands]
                  + [self._band_image(r, wl) for wl in _extras])
        _bh = (["x", "y"] + [f"band_{nm}_{wl:.0f}" for nm, wl in _bands]
               + [f"extra_{wl:.0f}" for wl in _extras])
        _br = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}"]
               + [f"{c[i]:.6g}" for c in _chans] for i in range(r.n_pixels)]
        write_csv(os.path.join(d, "band_maps.csv"), _bh, _br)
        ncsv += 1
        # the µM summary-bar table, numbers identical to the drawn bars
        if cal:
            _hit = self._hit(r)
            _um = r.conc * 1e6
            _tv = None
            _txt = self.true_edit.text().strip() if hasattr(self, "true_edit") else ""
            if _txt:
                try:
                    _pp = [float(t) for t in _txt.replace(" ", "").split(",")]
                    if len(_pp) == len(nb) and all(v > 0 for v in _pp):
                        _tv = _pp
                except ValueError:
                    _tv = None
            _vol = float(self.vol_spin.value()) if hasattr(self, "vol_spin") else 0.0
            _sr = []
            for i, nm in enumerate(nb):
                v = _um[_hit, i] if _hit.any() else _um[:, i]
                v = v[np.isfinite(v) & (v > 0)]
                if not v.size:
                    _sr.append([nm, "0", "", "", "", "", "", ""])
                    continue
                med = float(np.median(v))
                _sr.append([nm, str(int(v.size)), f"{med:.4f}",
                            f"{float(np.quantile(v, 0.25)):.4f}",
                            f"{float(np.quantile(v, 0.75)):.4f}",
                            f"{_tv[i]:g}" if _tv else "",
                            f"{100 * med / _tv[i]:.1f}" if _tv else "",
                            f"{med * _vol:.2f}" if _vol > 0 else ""])
            write_csv(os.path.join(d, "um_summary.csv"),
                      ["substance", "n_hit_px", "median_uM", "q1_uM", "q3_uM",
                       "true_uM", "recovery_pct", "apparent_amount_pmol"], _sr)
            ncsv += 1
        # figures export WITHOUT the selection ring — the clicked-pixel highlight
        # is a working aid, not figure content. Redraw clean, save, then restore.
        _sel = self._sel
        if _sel is not None:
            self._sel = None; self._plot_pies(r)
        figs = [("real_band_maps", self.c_maps),
                ("real_abundance_maps", self.c_abund),
                ("real_composition_pies", self.c_pie),
                ("real_composition", self.c_comp),
                ("real_pixel_spectrum", self.c_spec)]
        if getattr(r, "calibrated", False) and r.conc is not None:
            figs.append(("real_concentration_maps", self.c_conc))
        n = _save_figs(figs, d)
        # one file PER PANEL as well — figures are for the screen, panels are what
        # actually lands in a slide. Each crop includes its own title and ramp.
        import matplotlib.transforms as _mt
        pdir = os.path.join(d, "panels"); os.makedirs(pdir, exist_ok=True)
        np_ = 0
        groups = [(self.c_maps, getattr(self, "_exp_maps", [])),
                  (self.c_abund, getattr(self, "_exp_abund", [])),
                  (self.c_pie, getattr(self, "_exp_pie", [])),
                  (self.c_conc, getattr(self, "_exp_conc", []))]
        for cv, entries in groups:
            if not entries or not cv.isVisible():
                continue
            cv.draw()                                    # renderer must be current
            ren = cv.get_renderer() if hasattr(cv, "get_renderer") else None
            for label, ax, cbax in entries:
                try:
                    bb = ax.get_tightbbox(ren)
                    if cbax is not None:
                        bb = _mt.Bbox.union([bb, cbax.get_tightbbox(ren)])
                    bb = bb.transformed(cv.fig.dpi_scale_trans.inverted()).padded(0.05)
                    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_"
                                   for ch in label)
                    cv.fig.savefig(os.path.join(pdir, f"{safe}.png"),
                                   dpi=300, bbox_inches=bb, transparent=True)
                    np_ += 1
                except Exception:
                    pass                                 # one bad crop must not kill export
        if _sel is not None:
            self._sel = _sel; self._plot_pies(r)         # put the highlight back
        self._export_readme(d, r, nb, [f[0] for f in figs])
        self.status.setText(f"exported README + {ncsv} CSV + {n} PNG + {np_} panel PNG "
                            f"→ {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _export_readme(self, d, r, nb, fig_names):
        """WHAT / HOW / RESULT for a Real-data export (readable without the app)."""
        from dataset import load_preprocess
        cfg = load_preprocess(self.data_dir)
        trim = cfg.get("trim")
        window = f"{trim[0]:.0f}–{trim[1]:.0f} cm⁻¹" if trim else "full range"
        mrw = self._mean_ratio(r)                          # signal-weighted, as displayed
        ratio_str = " · ".join(f"{nm} {mrw[i]:.0%}" for i, nm in enumerate(nb)) \
                    + " (signal-weighted over hit pixels)"
        cal = getattr(r, "calibrated", False) and getattr(r, "conc_median", None) is not None
        se_arr = (r.conc_se if getattr(r, "conc_se", None) is not None
                  else np.zeros_like(r.conc_median)) if cal else None
        conc_str = (" · ".join(
            f"{nm} {r.conc_median[i] * 1e6:.3g} ± {se_arr[i] * 1e6:.2g} µM "
            f"(median ± spatial-bootstrap SE; P10–P90 "
            f"{r.conc_p10[i] * 1e6:.3g}–{r.conc_p90[i] * 1e6:.3g}; "
            f"95% CI {r.conc_ci_low[i] * 1e6:.3g}–{r.conc_ci_high[i] * 1e6:.3g})"
            for i, nm in enumerate(nb)) if cal else "not computed (no calibration loaded)")
        fig_docs = {
            "real_band_maps": ("raw baseline-removed intensity at one marker band per "
                               "substance, and those channels read as R/G/B — no "
                               "unmixing, so it shows what a band/RGB readout alone can "
                               "separate."),
            "real_composition_pies": "mean composition (pie) over the hit pixels.",
            "real_composition": "per-pixel dominant-substance / composition map.",
            "real_pixel_spectrum": "measured spectrum of the selected pixel.",
            "real_concentration_maps": "per-pixel apparent SERS-equivalent concentration (µM).",
        }
        sections = {
            "What this is": [
                "A real SERS map unmixed against the pure references for per-pixel "
                "composition (which substance, how much) and — if a calibration is loaded — "
                "apparent SERS-equivalent concentration (µM). Background pixels are excluded; "
                "component values are not constrained to sum to the applied concentration."],
            "How it was produced": [
                f"- References: {self.data_dir}",
                f"- Map: {os.path.basename(self.test) if getattr(self, 'test', None) else '(current)'}",
                f"- Unmixing: {r.method.upper()} against pure reference templates",
                f"- Baseline removal: {'on' if cfg.get('baseline') else 'off'}; "
                f"spectral window: {window}",
                f"- Calibration: {os.path.basename(self.calib_path) if self.calib_path else 'none (ratio only)'}"],
            "Results": [
                f"- Dominant substance: {r.dominant}",
                f"- Substance pixels (hit fraction): {r.hit_frac:.0%}",
                f"- Mean composition over hit pixels: {ratio_str}",
                f"- Mean reconstruction R²: {r.mean_r2:.2f}",
                f"- Median concentration across hit pixels: {conc_str}"],
        }
        figures = [(fn, fig_docs[fn]) for fn in fig_names if fn in fig_docs]
        write_readme(d, "UNMIXR — Real-data export", sections, figures)
