"""page_real.py — Real data tab: unmix one test map (NNLS or MCR-ALS, selectable).
A band-intensity image, a per-pixel composition pie map, the spectrum of a clicked
pixel, and the overall composition. Background is unmixed as its own component."""
from __future__ import annotations

import os
import sys
import traceback
import textwrap

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Wedge, Patch
from matplotlib.collections import PatchCollection

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QFileDialog, QColorDialog,
    QScrollArea, QFrame, QProgressBar, QLineEdit, QSizePolicy, QSplitter,
    QDialog, QDialogButtonBox, QTreeWidget, QTreeWidgetItem,
)

from ui_common import *
from unmix import unmix_map, vip_bands
from classify import classify_map
from real_data import PEST_DEFAULT
from dataset import (load_preprocess, load_colors, save_colors,
                     parse_mixture_label, BLANK_ALIASES)
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



class _ExportPicker(QDialog):
    """Export에서 내보낼 항목 선택 — 표(CSV)·합성 그림·패널(이미지만) 트리.
    부모 항목을 체크하면 자식 전부가 따라간다. 선택은 세션 안에서 기억된다."""

    def __init__(self, parent, groups, state):
        super().__init__(parent)
        self.setWindowTitle("Export — choose items")
        self.setMinimumSize(560, 600)
        lay = QVBoxLayout(self)
        self.tree = QTreeWidget(); self.tree.setHeaderHidden(True)
        self._items = {}
        for gname, entries in groups:
            top = QTreeWidgetItem(self.tree, [gname])
            top.setFlags(top.flags() | Qt.ItemFlag.ItemIsAutoTristate
                         | Qt.ItemFlag.ItemIsUserCheckable)
            for key, label in entries:
                it = QTreeWidgetItem(top, [label])
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(0, Qt.CheckState.Checked if state.get(key, True)
                                 else Qt.CheckState.Unchecked)
                self._items[key] = it
            top.setExpanded(True)
        lay.addWidget(self.tree)
        row = QHBoxLayout()
        b_all = QPushButton("All"); b_none = QPushButton("None")
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none.clicked.connect(lambda: self._set_all(False))
        row.addWidget(b_all); row.addWidget(b_none); row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        row.addWidget(bb)
        lay.addLayout(row)

    def _set_all(self, on):
        st = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for it in self._items.values():
            it.setCheckState(0, st)

    def selection(self):
        return {k: it.checkState(0) == Qt.CheckState.Checked
                for k, it in self._items.items()}
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
        self._roi = None            # (x0, x1, y0, y1) map coords — drag-selected region
        self._roi_drag = None       # press start while dragging
        self._roi_rubber = None     # live rectangle patch during the drag
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
        root.setContentsMargins(10, 8, 10, 8); root.setSpacing(5)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Real-data analysis — unmix a test map"); h1.setObjectName("h1")
        sub = QLabel("Read one measured map: raw band maps, reconstructed component "
                     "maps (NNLS/MCR — or per-pixel probabilities under a "
                     "composition model), the per-pixel composition pie map, apparent "
                     "µM maps, and the overall composition as bars with a µM line. "
                     "Click any pixel for its spectrum. References + preprocessing "
                     "come from Samples.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        title_row = QHBoxLayout(); title_row.setSpacing(8)
        title_row.addWidget(h1); title_row.addStretch(1)
        self.focus_b = QPushButton("Results focus")
        self.focus_b.setObjectName("ghost"); self.focus_b.setCheckable(True)
        self.focus_b.setToolTip("Hide the control rail and give the complete width to results")
        self.focus_b.toggled.connect(self._toggle_results_focus)
        title_row.addWidget(self.focus_b)
        head.addLayout(title_row); head.addWidget(sub)
        root.addLayout(head)

        # ── left control rail | right results ─────────────────────────────
        # Everything you LOAD or SET lives in a compact fixed rail on the left;
        # the right side is nothing but results. One screen, no hunting.
        outer = QHBoxLayout(); outer.setSpacing(10)
        root.addLayout(outer, 1)
        # Keep the control rail's content height independent from the dashboard.
        # Opening Options may scroll this rail, but must never stretch result rows.
        leftw = QWidget()
        left = QVBoxLayout(leftw)
        left.setContentsMargins(0, 0, 0, 0); left.setSpacing(10)
        left_scroll = QScrollArea()
        left_scroll.setObjectName("controlRail")
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # The rail is USER-RESIZABLE (splitter handle on its right edge): a fixed
        # 280 px clipped wide controls with no way to reach them.
        left_scroll.setMinimumWidth(220)
        self._left_scroll = left_scroll
        left_scroll.setWidget(leftw)
        # 폭을 넘는 자식이 포커스를 받으면 Qt 가 레일을 수평으로 밀어버리는데,
        # 가로 스크롤바가 숨겨져 있어 사용자가 되돌릴 수 없다(라벨 앞글자가 잘린 채
        # "찌그러져" 보이는 증상). 수평 스크롤을 항상 0 에 고정한다.
        _hbar = left_scroll.horizontalScrollBar()
        _hbar.rangeChanged.connect(lambda *_: _hbar.setValue(0))
        _hbar.valueChanged.connect(lambda v: v and _hbar.setValue(0))
        self._split = QSplitter(Qt.Orientation.Horizontal)
        self._split.setChildrenCollapsible(False)
        self._split.addWidget(left_scroll)
        outer.addWidget(self._split, 1)

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
        self.chk_auto = QCheckBox("auto background gate")
        # Default OFF: the settled recipe is the 0.15 fraction gate, and with auto
        # on the threshold spin is (correctly) greyed out — which read as "broken".
        self.chk_auto.setChecked(False)
        self.chk_auto.setToolTip(
            "Uses the full spectrum: signal only when summed analyte evidence "
            "exceeds summed BLK + INK evidence. Unchecked = fraction threshold.")
        self.chk_auto.toggled.connect(self._on_auto)
        hitcol = QVBoxLayout(); hitcol.setSpacing(2)
        _hl = QLabel("hit mode"); _hl.setObjectName("field")
        hitcol.addWidget(_hl); hitcol.addWidget(self.chk_auto)
        self.thr = self._spin_col("min substance fraction", QDoubleSpinBox())
        sp = self.thr.itemAt(1).widget()
        # 0.15 — the settled letter-map recipe (NNLS 0.15 gate + model composition):
        # keeps all of the lettering while the ink/background stays dropped.
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
        # 잎 시료: @1000 cm⁻¹ 밝기(Otsu)로 잉크 도포 영역(밝음)/무신호 외부를
        # 가르고 외부 픽셀은 전부 null, 잉크 영역 경계는 흰 윤곽선 (2026-09-04).
        self.chk_leaf = QCheckBox("ink area only"); self.chk_leaf.setChecked(False)
        self.chk_leaf.setToolTip("잎 시료: 1000 cm-1 밴드가 밝은 영역 = SERS 잉크가 도포된 "
                                 "곳(잎 자체는 신호 없음). 그 바깥 픽셀을 분석에서 제외하고 "
                                 "잉크 영역 경계를 흰 윤곽선으로 그린다. 액적 맵에서는 끄세요.")
        self.chk_leaf.toggled.connect(
            lambda _=False: self._apply(self._res) if self._res is not None else None)
        relrow.addWidget(self.chk_leaf)
        # ROI: 아무 맵에서나 드래그하면 그 사각 영역만 리포트(조성·µM·KPI·export).
        self.btn_roi = QPushButton("clear ROI"); self.btn_roi.setObjectName("ghost")
        self.btn_roi.setToolTip("drag a rectangle on any map to report only that region "
                                "(composition, µM, KPIs, export). Click = pixel spectrum "
                                "as before. This button clears the region.")
        self.btn_roi.setEnabled(False)
        self.btn_roi.clicked.connect(self._clear_roi)
        relrow.addWidget(self.btn_roi)
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
        corr_b.setVisible(False); self.corr_lbl.setVisible(False); self.chk_corr.setVisible(False)
        exp_b = QPushButton("Export results…"); exp_b.setObjectName("ghost")
        exp_b.setMinimumHeight(34)
        exp_b.setStyleSheet(
            f"QPushButton{{border:1.5px solid {BLUE}; color:{BLUE}; font-weight:700; "
            "padding:6px 10px; border-radius:7px; text-align:center;}"
            f"QPushButton:hover{{background:{BLUE}; color:white;}}")
        exp_b.setToolTip("Export CSV data and each result panel as a separate PNG")
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
        _r2.addWidget(self.btn, 1)
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
        # 경쟁왜곡 크기 Δ — 참값 없이 계산: NNLS 표면 THI% − 복원(MLP) THI%.
        # 92맵 검증에서 µM 판독 오차와 무상관(r=0.04) — 왜곡이 커도 판독은 유효.
        self.k_dthi = Kpi("surface ΔTHI")
        self.k_dthi.setToolTip(
            "NNLS가 읽은 표면 THI 조성 − 복원된 THI 조성 (%p).\n"
            "경쟁흡착이 THI를 얼마나 비대로 보이게 했는지 — 클수록 표면 왜곡이 큰 맵.")
        kpis.addWidget(self.k_dom, 0, 0); kpis.addWidget(self.k_n, 0, 1)
        kpis.addWidget(self.k_hit, 1, 0); kpis.addWidget(self.k_px, 1, 1)
        kpis.addWidget(self.k_dthi, 2, 0, 1, 2)
        left.addLayout(kpis)

        # per-substance colour swatches (click to recolour), filled after a result
        left.addWidget(exp_b)
        self.swatches = QHBoxLayout(); self.swatches.setSpacing(6)
        self.swatches.addWidget(self._mk_lbl("colours:"))
        self.swatches.addStretch(1)
        left.addLayout(self.swatches)
        left.addStretch(1)

        # ---------- result dashboard: compact hierarchy, no page scroll ----------
        body = QGridLayout(); body.setSpacing(5)
        self._fold_dirty = {"maps": False, "abund": False}

        # 1) band maps: raw intensity at one marker band per substance + their RGB merge
        self.c_maps = Canvas()
        card_maps, lay_maps = _card("Raw band maps — marker intensity + RGB merge")
        self.bandrow = QHBoxLayout(); self.bandrow.setSpacing(6)
        self.bandrow.addWidget(self._mk_lbl("bands (cm⁻¹):"))
        self.bandrow.addStretch(1)
        lay_maps.addLayout(self.bandrow)
        # Two map rows need real vertical pixels.  A short canvas leaves large card
        # gutters while ``aspect='equal'`` silently shrinks the actual data image.
        # Keep the maps readable even when the controls or titles are edited.
        lay_maps.addWidget(self.c_maps); self.c_maps.setMinimumHeight(220)
        # one min/max pair PER band panel (rebuilt with the band row) — the card-wide
        # pair could not stretch a weak channel without flattening a strong one
        self.scalerow = QHBoxLayout(); self.scalerow.setSpacing(6)
        self.scalerow.addWidget(self._mk_lbl("scale:")); self.scalerow.addStretch(1)
        lay_maps.addLayout(self.scalerow)


        # 1b) unmixed abundance maps — NNLS runs FIRST in every path (the gate), so
        #     its per-substance abundances belong beside the raw band maps: bands =
        #     what the camera saw, abundances = what NNLS unmixing made of it.
        self.c_abund = Canvas()
        card_ab, lay_ab = _card(
            "MLP reconstruction — composition + background gate")
        lay_ab.addWidget(self.c_abund); self.c_abund.setMinimumHeight(220)
        _abrow = self._scale_row("abund", 1.0)
        # 명암: 모델 조성에 픽셀의 상대 마커밴드 신호(99퍼센타일 정규화)를 곱해
        # raw 밴드맵의 밝고 어두움을 재구성 맵에도 싣는다 (사용자 요청 2026-09-04).
        self.chk_abund_shade = QCheckBox("shade by signal")
        self.chk_abund_shade.setChecked(True)
        self.chk_abund_shade.setToolTip(
            "multiply each pixel's model composition by its relative marker-band "
            "signal (sum of the three VIP bands, p5–p95 mapped to 0.45–1.0 brightness) so the "
            "reconstruction carries the raw map's light and shade. Off = flat composition.")
        self.chk_abund_shade.toggled.connect(
            lambda _=False: self._res is not None and self._plot_abund(self._res))
        _abrow.addWidget(self.chk_abund_shade)
        # 픽셀 격자: 검정 배경 위 각 픽셀의 흰 테두리 — composition·concentration 카드
        # (raw 밴드맵은 제외). 사용자 2026-09-11.
        self.chk_grid = QCheckBox("pixel grid"); self.chk_grid.setChecked(True)
        self.chk_grid.setToolTip("draw a thin white border around every pixel on the "
                                 "composition and concentration maps")
        self.chk_grid.toggled.connect(
            lambda _=False: (self._plot_abund(self._res), self._plot_conc(self._res))
            if self._res is not None else None)
        _abrow.addWidget(self.chk_grid)
        lay_ab.addLayout(_abrow)

        # Nine equal map slots across the result area. Raw uses four (merge + 3),
        # NNLS evidence uses five (merge + 3 + background), so every spatial map has the
        # same physical width instead of shrinking because its card has more panels.
        # 2) the clicked pixel's spectrum — a short readout beside the pie map.
        #    The pixel's own numbers ride along in the panel title.
        self.c_spec = Canvas()
        scard, slay = _card("Selected pixel spectrum — measured vs reconstructed")
        slay.addWidget(self.c_spec)
        self.c_spec.setMinimumHeight(210)

        # 3) per-pixel composition pie | the same composition summed over the map —
        #    the pie map and the number it adds up to belong on one row
        self.c_pie = Canvas(); self.c_comp = Canvas()
        pcard, play = _card("Primary comparison — NNLS raw vs MLP + pixel-wise change")
        play.addWidget(self.c_pie); self.c_pie.setMinimumHeight(120)
        play.addLayout(self._scale_row("delta", 50.0))
        pcard.setMaximumHeight(270)
        ccard, clay = _card("Overall fractions — NNLS vs MLP")
        clay.addWidget(self.c_comp); self.c_comp.setMinimumHeight(175)

        # Two balanced columns: raw bands over NNLS evidence on the left; the
        # primary NNLS↔MLP comparison over summary + spectrum on the right.
        # Both sides occupy exactly two equal-height rows.
        body.addWidget(card_maps, 0, 0, 1, 10)
        body.addWidget(card_ab,   1, 0, 1, 10)
        body.addWidget(pcard,     0, 10, 1, 8)
        body.addWidget(ccard,     1, 10, 1, 4)
        body.addWidget(scard,     1, 14, 1, 4)

        # 4) per-substance concentration (µM) maps — its own full-width row
        self.c_conc = Canvas()
        self.card_conc, lay_conc = _card(
            "Apparent concentration (µM) — spatial maps + pixel distribution")
        vrow = QHBoxLayout(); vrow.setSpacing(6)
        # Reading order = reporting priority: declared total drives the main
        # (constrained) numbers, truth ticks are for validation runs, volume is a
        # convenience conversion. The old order buried the main input last.
        _ktl = QLabel("declared total µM (main route)"); _ktl.setObjectName("field")
        self.total_edit = QLineEdit(); self.total_edit.setFixedWidth(64)
        self.total_edit.setPlaceholderText("e.g. 36")
        self.total_edit.setToolTip(
            "sample-prep metadata: the KNOWN summed analyte concentration (e.g. "
            "12+12+12 = 36). When set, the panel adds the known-total reconstruction "
            "(composition × total, held-out ≤100 µM: within-2× 78→91%) and caps the "
            "spectrum-only µM at the total. Both are CONSTRAINED numbers — the "
            "improvement comes from the added information, not the model — and the "
            "export flags them so. Unknown field samples: leave blank, spectrum-only "
            "(semi-quantitative) reporting stands.")
        self.total_edit.editingFinished.connect(
            lambda: self._plot_conc(self._res) if self._res is not None else None)
        vrow.addWidget(_ktl); vrow.addWidget(self.total_edit)
        _tl = QLabel("   true µM (a,b,c) for validation"); _tl.setObjectName("field")
        self.true_edit = QLineEdit(); self.true_edit.setFixedWidth(110)
        self.true_edit.setPlaceholderText("e.g. 12,12,12")
        self.true_edit.setToolTip("dispensed truth per substance, comma-separated in "
                                  "the panel order. Adds a red tick at each true value "
                                  "and a red truth tick beneath each substance.")
        self.true_edit.editingFinished.connect(
            lambda: self._plot_conc(self._res) if self._res is not None else None)
        vrow.addWidget(_tl); vrow.addWidget(self.true_edit)
        _vl = QLabel("   dispensed volume µL (0 = off)"); _vl.setObjectName("field")
        self.vol_spin = QDoubleSpinBox(); self.vol_spin.setDecimals(1)
        self.vol_spin.setRange(0.0, 100.0); self.vol_spin.setSingleStep(0.5)
        self.vol_spin.setValue(0.0); self.vol_spin.setFixedWidth(84)
        self.vol_spin.setToolTip(
            "volume of the droplet/ink you dispensed. When set, the distribution label "
            "also show the apparent amount = median µM × volume (pmol). APPARENT — "
            "it reads the SERS-equivalent concentration, not a mass balance.")
        self.vol_spin.valueChanged.connect(
            lambda _=0: self._plot_conc(self._res) if self._res is not None else None)
        vrow.addWidget(_vl); vrow.addWidget(self.vol_spin)
        # One-point batch recalibration: sessions measured far from the calibration
        # batch shift the whole intensity axis (the 260812 cross-batch smoke: ×8 on
        # absolute µM while composition held). Anchoring on ONE map with known truth
        # rescales apparent µM per substance for the rest of the session — standard
        # one-point recalibration, honestly labelled on the panel.
        self._anchor = None
        self.anchor_b = QPushButton("Set batch anchor"); self.anchor_b.setObjectName("ghost")
        self.anchor_b.setToolTip(
            "Use the CURRENT unmixed map as the session's batch anchor: enter its "
            "true µM (a,b,c) first, then click. Per-substance factors = truth / "
            "median apparent µM are applied to apparent µM on every following map. "
            "A substance sitting at its validated ceiling (saturated, e.g. THI at "
            "high µM) is skipped — its factor stays 1. Clear with ✕.")
        self.anchor_b.clicked.connect(self._set_anchor)
        self.anchor_lbl = QLabel(""); self.anchor_lbl.setObjectName("field")
        self.anchor_x = QPushButton("✕"); self.anchor_x.setObjectName("ghost")
        self._compact_x(self.anchor_x, "clear batch anchor")
        self.anchor_x.setVisible(False)
        self.anchor_x.clicked.connect(self._clear_anchor)
        vrow.addSpacing(10)
        vrow.addWidget(self.anchor_b); vrow.addWidget(self.anchor_lbl)
        vrow.addWidget(self.anchor_x)
        # µM 판독 경로 선택: model head(정확도 우선) vs library k-NN(측정값 조회 —
        # 반환이 학습 맵 실측치의 내분점이라 외삽 불가). 무응답 규칙은 공통.
        _rl = QLabel("   µM readout"); _rl.setObjectName("field")
        self.cmb_umroute = QComboBox()
        self.cmb_umroute.addItem("auto", "auto")
        self.cmb_umroute.addItem("model head", "model")
        self.cmb_umroute.addItem("library k-NN", "knn")
        self.cmb_umroute.addItem("pixel k-NN", "pxknn")
        # 잎/잉크 시료용: µM-등가 대신 픽셀별 VIP 밴드 세기(counts)를 같은 배선에
        # 흘린다 — "raw VIP가 더 잘 보여준다"(2026-09-04) 비교용.
        self.cmb_umroute.addItem("raw VIP band signal", "raw")
        # raw(측정) × MLP(누구 몫인가): 픽셀 총 밴드 신호를 MLP 조성으로 배분,
        # 게이트 밖(배경/잉크)은 0 — raw와의 차이가 곧 MLP가 한 일 (2026-09-04).
        self.cmb_umroute.addItem("MLP-corrected signal", "mlpsig")
        # 시료 유형 스위치 (2026-09-04 확정): droplet = auto µM(검증창·라이브러리),
        # leaf/ink = MLP-corrected signal을 얼굴로, raw는 근거, µM은 KT 있을 때만.
        # LOO 스위치: 라이브러리 맵을 다시 열면 자기 항목을 빼고 조회(검증 정직성).
        # 끄면 배포 동작 — 같은 조건이 라이브러리에 있으면 그 실측 농도를 그대로.
        self.chk_loo = QCheckBox("LOO (exclude this map)"); self.chk_loo.setChecked(False)
        self.chk_loo.setToolTip("ON: treat the loaded map as unknown (its own library entry "
                                "is excluded). OFF: deployment behaviour - an identical "
                                "library condition returns its measured concentration.")
        self.chk_loo.toggled.connect(
            lambda _=False: self._plot_conc(self._res) if self._res is not None else None)
        vrow.addWidget(self.chk_loo)
        self.cmb_umroute.setToolTip(
            "model head: residual-net estimate (validated 7.5 µM RMSE in-window).\n"
            "library k-NN: distance-weighted lookup of the 3 nearest TRAINING maps' "
            "measured concentrations (LOO 21.8 µM RMSE, but every value is an "
            "interpolation of measured maps — no extrapolation possible).\n"
            "Both refuse to answer when the map is outside the library (distance > 3).")
        self.cmb_umroute.currentIndexChanged.connect(
            lambda _=0: self._plot_conc(self._res) if self._res is not None else None)
        vrow.addWidget(_rl); vrow.addWidget(self.cmb_umroute)
        # 스파이럴 점 표시 비율 — 0% = percentile 곡선·밴드만, 100% = 픽셀 점
        # 전부. 드래그 중엔 다시 그리지 않고 놓을 때 한 번 (pxknn 조회가 무겁다).
        _sl = QLabel("radial detail"); _sl.setObjectName("field")
        from PyQt6.QtWidgets import QSlider
        self.sl_spiral = QSlider(Qt.Orientation.Horizontal)
        self.sl_spiral.setRange(18, 48); self.sl_spiral.setValue(36)
        self.sl_spiral.setFixedWidth(110)
        self.sl_spiral.setToolTip(
            "원형 heatmap을 나누는 각도 구간 수입니다.\n"
            "낮으면 큰 경향이, 높으면 픽셀 순위의 세부 변화가 잘 보입니다.")
        self.sl_spiral.sliderReleased.connect(
            lambda: self._plot_conc(self._res) if self._res is not None else None)
        self.sl_spiral.valueChanged.connect(
            lambda v: self.sl_spiral.setToolTip(f"radial bins: {v}"))
        _msl = QLabel("map scale"); _msl.setObjectName("field")
        self.sl_map_scale = QSlider(Qt.Orientation.Horizontal)
        self.sl_map_scale.setRange(40, 200); self.sl_map_scale.setValue(100)
        self.sl_map_scale.setFixedWidth(95)
        self.sl_map_scale.setToolTip("concentration-map colour maximum: auto × 1.00")
        self.sl_map_scale.sliderReleased.connect(
            lambda: self._plot_conc(self._res) if self._res is not None else None)
        self.sl_map_scale.valueChanged.connect(lambda v: self.sl_map_scale.setToolTip(
            f"concentration-map colour maximum: auto × {v / 100:.2f}"))
        _ysl = QLabel("distribution Y"); _ysl.setObjectName("field")
        self.sl_dist_y = QSlider(Qt.Orientation.Horizontal)
        self.sl_dist_y.setRange(40, 200); self.sl_dist_y.setValue(100)
        self.sl_dist_y.setFixedWidth(95)
        self.sl_dist_y.setToolTip("distribution Y-axis maximum: auto × 1.00")
        self.sl_dist_y.sliderReleased.connect(
            lambda: self._plot_conc(self._res) if self._res is not None else None)
        self.sl_dist_y.valueChanged.connect(lambda v: self.sl_dist_y.setToolTip(
            f"distribution Y-axis maximum: auto × {v / 100:.2f}"))
        # signal floor: 게이트를 통과했지만 VIP 밴드 신호가 약한 픽셀(잎 바깥·
        # 기판에 깔린 것들)을 hit에서 뺀다. 기준 = hit 픽셀 신호합 p99의 n %.
        # 맵·파이·분포·KPI 모두 같은 hit을 쓰므로 놓을 때 전체를 다시 그린다.
        _fl = QLabel("signal floor"); _fl.setObjectName("field")
        self.sl_floor = QSlider(Qt.Orientation.Horizontal)
        self.sl_floor.setRange(0, 80); self.sl_floor.setValue(0)
        self.sl_floor.setFixedWidth(110)
        self.lbl_floor = QLabel("off"); self.lbl_floor.setObjectName("field")
        self.lbl_floor.setMinimumWidth(120)
        self.sl_floor.setToolTip(
            "drop gate-positive pixels whose summed VIP-band signal is below this "
            "fraction of the hit pixels' 99th percentile. 0 = off. Applies to maps, "
            "pies, distribution and the hit % alike.")
        self.sl_floor.valueChanged.connect(self._floor_label)
        self.sl_floor.sliderReleased.connect(
            lambda: self._apply(self._res) if self._res is not None else None)
        vrow.addWidget(_fl); vrow.addWidget(self.sl_floor); vrow.addWidget(self.lbl_floor)
        # the reportable window is NOT typed here — the model file carries it
        # (validated_ranges_M: levels recovered within 2-fold on a held-out split),
        # and the summary shows which window it used
        vrow.addStretch(1)
        self.conc_optbox = QWidget(); self.conc_optbox.setLayout(vrow)
        self.conc_optbox.setVisible(False)
        self.conc_opt_tgl = QPushButton("▸ concentration options")
        self.conc_opt_tgl.setObjectName("ghost"); self.conc_opt_tgl.setCheckable(True)
        self.conc_opt_tgl.setStyleSheet("text-align:left; padding:2px 6px;")
        def _toggle_conc_opts(on):
            self.conc_optbox.setVisible(on)
            self.conc_opt_tgl.setText(("▾ " if on else "▸ ") + "concentration options")
        self.conc_opt_tgl.toggled.connect(_toggle_conc_opts)
        lay_conc.addWidget(self.conc_opt_tgl)
        lay_conc.addWidget(self.conc_optbox)
        lay_conc.addWidget(self.c_conc)
        # Controls belong directly under the result they change: 5 maps | 6 plot | 7 radial.
        slider_row = QHBoxLayout(); slider_row.setSpacing(10)
        self._conc_scale_names = ()
        self._conc_scale_spins = {}
        self.conc_scale_box = QWidget()
        self.conc_scale_lay = QHBoxLayout(self.conc_scale_box)
        self.conc_scale_lay.setContentsMargins(0, 0, 0, 0)
        self.conc_scale_lay.setSpacing(5)
        self.conc_scale_lay.addStretch(1)
        slider_row.addWidget(self.conc_scale_box, 1)
        for label, slider in ((_ysl, self.sl_dist_y),
                              (_sl, self.sl_spiral)):
            cell = QWidget(); cell_lay = QHBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0); cell_lay.setSpacing(7)
            cell_lay.addStretch(1); cell_lay.addWidget(label)
            slider.setFixedWidth(145); cell_lay.addWidget(slider); cell_lay.addStretch(1)
            slider_row.addWidget(cell, 1)
        lay_conc.addLayout(slider_row)
        # Fixed height: letting this canvas expand smeared the row across a tall
        # card — tiny maps floating in whitespace. Tall enough to use the row.
        # 고정 높이는 창 공간이 남아도 아래가 잘린다(2026-09-02) — 최소만 잡고
        # 남는 세로 공간은 이 행이 흡수해 늘어난다 (result_grid rowStretch=1).
        self.c_conc.setMinimumHeight(255)
        self.conc_opt_tgl.setChecked(True)     # declared total is the main input
        # Long headings used to become hard minimum widths (over 2,100 px for the
        # whole page). Wrap them inside their cards so a normal laptop window can
        # show the complete wording without a horizontal scrollbar.
        for _lay in (lay_maps, lay_ab, slay, play, clay, lay_conc):
            _title = _lay.itemAt(0).widget()
            _title.setWordWrap(True)
            _title.setMinimumWidth(0)
            _title.setSizePolicy(QSizePolicy.Policy.Ignored,
                                 QSizePolicy.Policy.Fixed)
            _lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        body.addWidget(self.card_conc, 2, 0, 1, 18)
        self.card_conc.setVisible(False)

        self.result_grid = body
        for col in range(18):
            body.setColumnStretch(col, 1)
        # The two important spatial-map cards receive matching height.  The right
        # summary cards may retain whitespace, but can no longer compress maps 1/2.
        body.setRowStretch(0, 1)
        body.setRowStretch(1, 1)
        body.setRowStretch(2, 0)  # enabled only while concentration is visible

        # Matplotlib advertises a large preferred width. Ignore that hint so three
        # canvases can share one row instead of growing the page sideways.
        for cv in (self.c_maps, self.c_abund, self.c_spec, self.c_pie,
                   self.c_comp, self.c_conc):
            cv.setMinimumWidth(0)
            cv.setSizePolicy(QSizePolicy.Policy.Ignored,
                             QSizePolicy.Policy.Expanding)

        bodyw = QWidget(); bodyw.setLayout(body)
        bodyw.setMinimumWidth(0)
        bodyw.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Expanding)
        # 결과가 창보다 커지면 잘리는 대신 세로 스크롤이 생겨야 한다
        # (2026-09-02 — 농도 행을 키우자 하단이 잘리고 스크롤도 없던 문제).
        # 가로는 캔버스가 폭에 맞춰 줄어드니 스크롤바를 끈다.
        body_scroll = QScrollArea()
        body_scroll.setFrameShape(QFrame.Shape.NoFrame)
        body_scroll.setWidgetResizable(True)
        body_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        body_scroll.setWidget(bodyw)
        self._split.addWidget(body_scroll)
        self._split.setStretchFactor(0, 0)
        self._split.setStretchFactor(1, 1)
        self._split.setSizes([235, 1125])

        for cv, m in [(self.c_maps, "Load a test map, then Unmix"),
                      (self.c_pie, "Composition appears here"),
                      (self.c_comp, "Composition appears here"),
                      (self.c_spec, "Click a pixel in a map to see its spectrum")]:
            cv.placeholder(m)
        for _cv in (self.c_maps, self.c_abund, self.c_pie, self.c_conc):
            _cv.mpl_connect("button_press_event", self._on_click)
            _cv.mpl_connect("motion_notify_event", self._on_drag)
            _cv.mpl_connect("button_release_event", self._on_release)
        self._autoload_default_model()


    # ---- small builders ----
    def _toggle_results_focus(self, on):
        """Give the dashboard the full window without losing any result panel."""
        self._left_scroll.setVisible(not on)
        self.focus_b.setText("Show controls" if on else "Results focus")

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
        cb.setMinimumWidth(0)
        cb.setSizePolicy(QSizePolicy.Policy.Ignored,
                         QSizePolicy.Policy.Fixed)
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
                draw = ({"maps": self._plot_maps,
                         "abund": self._plot_abund,
                         "delta": self._plot_pies}.get(key))
                if draw is not None:
                    draw(self._res)

        chk.toggled.connect(_upd)
        mn.editingFinished.connect(_upd); mx.editingFinished.connect(_upd)
        row.addWidget(chk)
        row.addWidget(QLabel("min")); row.addWidget(mn)
        row.addWidget(QLabel("max")); row.addWidget(mx)
        row.addStretch(1)
        return row

    def _common_map_lr(self, fig, n):
        """Centred left/right bounds giving every spatial map one common width.

        The band-control row makes its card wider than its grid span alone would;
        derive the target from the two top map cards instead of trusting card width.
        """
        # Fixed-height canvases leave slightly different usable horizontal fractions:
        # the five-panel colour-ramp row and the one-panel pie legend consume a little
        # more than GridSpec reports. These are layout factors, not data scaling.
        def _fit(slots):
            return 0.918 if slots == 5 else (0.977 if slots == 3 else 1.0)

        caps = []
        for source, slots in ((self.c_maps.fig, 4), (self.c_abund.fig, 5)):
            fw = float(source.bbox.width)
            if fw > 1:
                caps.append(_fit(slots) * (self._GS_R - self._GS_L) * fw /
                            (slots + (slots - 1) * self._GS_WS))
        if not caps:
            return self._GS_L, self._GS_R
        # Leave a small common gutter: Qt card margins and Matplotlib's colour-ramp
        # row consume a few pixels that the nominal GridSpec capacity omits. Without
        # it the five-panel stage card alone becomes ~8% narrower.
        target = min(caps) * 0.95
        fw = max(float(fig.bbox.width), 1.0)
        cap = (self._GS_R - self._GS_L) * fw / (n + (n - 1) * self._GS_WS)
        requested = target / _fit(n)
        requested = min(requested, cap)
        span = requested * (n + (n - 1) * self._GS_WS) / fw
        return (1.0 - span) / 2.0, (1.0 + span) / 2.0

    def _row_gs(self, fig, n, ny, nx):
        """The grid both map cards draw on: images across the top row, their colour
        ramps in a thin row directly under them.

        Two things this fixes. Every image cell is the SAME box, so the merged panel
        lines up with the components instead of floating above them — it used to keep
        the height the colour-bars stole from its neighbours. And the pair of rows is
        hugged to the height these maps actually need (a 60x30 map in a narrow column
        is short) and centred, instead of leaving the card two-thirds empty."""
        L, R = self._common_map_lr(fig, n)
        bot, top, cb = self._row_span(fig, n, ny, nx, (L, R))
        gs = fig.add_gridspec(2, n, height_ratios=[1.0, cb], hspace=self._GS_HS,
                              wspace=self._GS_WS, left=L, right=R,
                              bottom=bot, top=top)
        # a resized card changes what "hugged" means; re-hug without replotting
        cv = fig.canvas
        old = getattr(cv, "_rowgs_cid", None)
        if old is not None:
            cv.mpl_disconnect(old)

        def _refit(_e=None, _gs=gs, _n=n, _ny=ny, _nx=nx, _f=fig):
            l, r = self._common_map_lr(_f, _n)
            b, t, c = self._row_span(_f, _n, _ny, _nx, (l, r))
            _gs.set_height_ratios([1.0, c])
            _gs.update(left=l, right=r, bottom=b, top=t)

        cv._rowgs_cid = cv.mpl_connect("resize_event", _refit)
        return gs

    def _tile_specs(self, fig, n, cols):
        """Return (image, colourbar) slots in two rows to avoid wide empty cards."""
        rows = int(np.ceil(n / cols))
        gs = fig.add_gridspec(rows * 2, cols,
                              height_ratios=sum(([1.0, 0.07] for _ in range(rows)), []),
                              hspace=0.24, wspace=0.055,
                              left=0.015, right=0.985, bottom=0.025, top=0.965)
        specs = []
        for i in range(n):
            rr, cc = divmod(i, cols)
            specs.append((gs[rr * 2, cc], gs[rr * 2 + 1, cc]))
        return specs

    def _short_colorbar(self, fig, ax, image, holder_spec):
        """A thin bar locked to the rendered map width, never the wide grid cell."""
        fig.add_subplot(holder_spec).set_axis_off()  # reserve vertical breathing room
        cax = ax.inset_axes([0.0, -0.13, 1.0, 0.055])
        cb = fig.colorbar(image, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7, colors="black", length=2, pad=1)
        cb.outline.set_linewidth(0.5)
        return cb

    def _side_colorbar(self, fig, ax, image, ticks=None, labels=None):
        """Compact vertical scale immediately to the right of its map.

        An inset follows the rendered map axes, so resizing or typing in controls
        cannot steal image height or create a card-wide horizontal ramp.
        """
        cax = ax.inset_axes([1.025, 0.04, 0.035, 0.92])
        cb = fig.colorbar(image, cax=cax, orientation="vertical")
        if ticks is not None:
            cb.set_ticks(ticks)
        if labels is not None:
            cb.set_ticklabels(labels)
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
        cb.outline.set_linewidth(0.45)
        return cb

    def _single_row_specs(self, fig, n):
        """One map row with space between panels for right-hand colour scales."""
        gs = fig.add_gridspec(1, n, left=0.018, right=0.982,
                              bottom=0.08, top=0.91, wspace=0.32)
        return [gs[0, i] for i in range(n)]

    def _row_span(self, fig, n, ny, nx, lr=None):
        """(bottom, top, cb_ratio) for _row_gs — the two rows hugged to the height
        ny/nx maps need at this figure size and centred in what the card gives us.

        The ramp row is a FIXED thickness in inches, not a share of the image row:
        tie it to the image and a wide short map squeezes its own colour-bar into a
        hairline."""
        L, R = lr if lr is not None else (self._GS_L, self._GS_R)
        WS, HS = self._GS_WS, self._GS_HS
        BOT, TOP, BAR_IN = 0.08, 0.92, 0.11
        _, figh = fig.get_size_inches()
        figh = max(float(figh), 1e-6)
        bar = BAR_IN / figh                                          # ramp row, frac
        # A common row height keeps the same measurement equally tall in the raw
        # four-panel and abundance five-panel cards. Data aspect is kept inside it.
        need = min(0.75, TOP - BOT - bar)
        need = max(need, 0.05)
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
        mode = QComboBox(); mode.setObjectName("field")
        mode.addItem("Auto contrast", "auto")
        mode.addItem("Shared scale", "shared")
        mode.addItem("Full range", "full")
        mode.setToolTip(
            "Auto contrast: robust P1–P99 for each band (recommended).\n"
            "Shared scale: one robust numeric scale for direct band comparison.\n"
            "Full range: each band's measured minimum–maximum.")
        old_mode = getattr(self, "cmb_band_scale", None)
        if old_mode is not None:
            wanted = old_mode.currentData()
            idx = mode.findData(wanted)
            mode.setCurrentIndex(max(idx, 0))
        self.cmb_band_scale = mode
        self.scalerow.addWidget(self._mk_lbl("display:"))
        self.scalerow.addWidget(mode)
        chk = QCheckBox("manual"); chk.setObjectName("field")
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
                sp.setFixedWidth(72)
                sp.setToolTip(f"{what} of the colour ramp for the {nm} panel")
                sp.editingFinished.connect(
                    lambda: self._res is not None and self._plot_maps(self._res))
                self.scalerow.addWidget(sp)
                pair.append(sp)
            self._chan_scale[key] = tuple(pair)

        def _tgl(on):
            if self._res is not None:
                self._plot_maps(self._res)

        chk.toggled.connect(_tgl)
        mode.currentIndexChanged.connect(_tgl)
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
            sp.setFixedWidth(72)
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
            esp.setRange(lo, hi); esp.setValue(float(wl)); esp.setFixedWidth(72)
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
        self._plot_maps(r); self._plot_abund(r); self._plot_comp(r); self._plot_conc(r); self._plot_pies(r)
        if self._sel is not None:
            self._plot_spec(r, self._sel)

    def _method(self):
        return self.cmb_method.itemAt(1).widget().currentData()

    def _activate_dl_method(self):
        """A loaded/published .dlm is actionable immediately. In particular its µM
        head only runs under dlpx, so leaving a previous NNLS/MCR selection in place
        made concentration silently disappear."""
        cb = self.cmb_method.itemAt(1).widget()
        idx = cb.findData("dlpx")
        if idx >= 0:
            cb.setCurrentIndex(idx)

    def _um_tag(self):
        m = getattr(self, "dl_model", None)
        if not isinstance(m, dict):
            return ""
        return "  ·  µM head ✓" if m.get("uM") else "  ·  composition only (no µM head)"

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

    def _sync_controls(self):
        """Grey out the gates that no longer get a vote, so the header shows at a glance
        which single rule will decide background."""
        if not hasattr(self, "thr"):        # _adopt_model can fire before the widgets exist
            return
        by_bg = bool(self.bg_paths)
        manual = not by_bg
        self.chk_auto.setEnabled(manual)
        # The fraction gate remains editable even when a measured background
        # is loaded; this lets users tune the 0.15 default before re-running.
        self.thr.itemAt(1).widget().setEnabled(not self.chk_auto.isChecked())
        if by_bg:
            tip = "not used: the loaded measured background map decides background"
            self.chk_auto.setToolTip(tip)
            self.thr.itemAt(1).widget().setToolTip(tip)
        else:
            self.chk_auto.setToolTip("Full spectrum: summed analytes > summed BLK + INK.")
            self.thr.itemAt(1).widget().setToolTip("Used only when auto gate is unchecked.")

    def _browse_test(self):
        p, _ = QFileDialog.getOpenFileName(self, "Test map", "",
                                           "maps (*.csv *.txt);;all files (*)")
        if p:
            self.test = p; self.test_lbl.setText(os.path.basename(p))
            self.test_x.setVisible(True)
            self._set_truth_from_filename(p)

    def _set_truth_from_filename(self, path):
        """Use a filename such as DQ24-TBZ24-THI3 as the comparison truth.

        The filename is measurement metadata. Requiring it to be typed again here
        invites order and transcription errors, especially for a trace component.
        """
        if not hasattr(self, "true_edit"):
            return
        subs = []
        model = getattr(self, "dl_model", None)
        if isinstance(model, dict):
            subs = [str(s) for s in (model.get("subs") or [])
                    if str(s).lower() not in BLANK_ALIASES]
        if not subs:
            subs = ["DQ", "TBZ", "THI"]
        stem = os.path.splitext(os.path.basename(path))[0]
        amounts = parse_mixture_label(stem, subs)
        if amounts and all(name in amounts for name in subs):
            self.true_edit.setText(",".join(f"{amounts[name]:g}" for name in subs))
            self.true_edit.setToolTip(
                "auto-read from the test-map filename in panel order: "
                + ", ".join(f"{name}={amounts[name]:g} µM" for name in subs))
            # The declared total follows for free — no re-typing what the
            # filename already states.
            if hasattr(self, "total_edit"):
                self.total_edit.setText(f"{sum(amounts[n] for n in subs):g}")
        else:
            self.true_edit.clear()

    def _clear_test(self):
        self.test = None; self.test_lbl.setText("no test map"); self.test_x.setVisible(False)
        if hasattr(self, "true_edit"):
            self.true_edit.clear()

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Trained model (unmixr_model.joblib)",
                                           self.data_dir, "model (*.joblib);;all (*)")
        if p:
            self.model_path = p; self.model_lbl.setText(os.path.basename(p))
            self.cmb_method.itemAt(1).widget().setCurrentIndex(2)   # switch to model

    def _toggle_opts(self, on):
        self.optbox.setVisible(on)
        self.opt_tgl.setText(("▾  " if on else "▸  ")
                             + "Options — sources & pixel gate")

    def _adopt_model(self):
        """Adopt the composition model trained in the Model tab (Step 2), if any. Just
        stash it + label — do NOT re-run analysis here (that would block the GUI on the
        main thread the moment training finishes); the next Analyze picks it up."""
        if MODEL_BUS.model is None:
            return
        self.dl_model = MODEL_BUS.model
        self._activate_dl_method()
        self.dlm_lbl.setText("DL: " + (MODEL_BUS.origin or "trained model")
                             + self._blank_tag() + self._um_tag())
        self.dlm_lbl.setStyleSheet(""); self._sync_controls()
        self._refresh_calib_label()          # the model may carry its own calibration

    # Real is the day-to-day tab; it must stand alone. The deployed FINAL bundle is
    # adopted at startup so an operator can open the app, load a map, and Unmix —
    # no trip through Model/Recovery first. $UNMIXR_DLM overrides the location.
    DEFAULT_DLM = os.path.join(r"S:\Google Drive\내 드라이브\ACF_PEST_DB",
                               "260831_Model_FINAL", "mlp_composition_260831_final.dlm")

    def _autoload_default_model(self):
        path = os.environ.get("UNMIXR_DLM") or self.DEFAULT_DLM
        if self.dl_model is not None or not os.path.exists(path):
            return
        try:
            from dl_model import load_model
            self.dl_model = load_model(path)
        except Exception as e:
            print("default DL model load:", e, file=sys.stderr)
            return
        self._activate_dl_method()
        self.dlm_lbl.setText("DL: " + os.path.basename(path) + self._blank_tag()
                             + self._um_tag())
        self.dlm_lbl.setStyleSheet("")
        self._sync_controls()
        self._refresh_calib_label()
        self.status.setText("deployed model ready — load a test map and Unmix")
        self.status.setStyleSheet(f"color:{MUTE};")

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
            self._activate_dl_method()
            self.dlm_lbl.setText("DL: " + os.path.basename(p) + self._blank_tag()
                                 + self._um_tag())
            self.dlm_lbl.setStyleSheet(""); self._sync_controls()
            self._refresh_calib_label()      # the model may carry its own calibration
            # Existing results were computed by the previous method/model; merely
            # redrawing them made a freshly loaded µM model look as though it returned
            # no concentration. Keep the plots as historical context, but require a run.
            self.status.setText("DL model loaded — click Unmix to compute composition"
                                + (" + concentration" if self.dl_model.get("uM") else ""))
            self.status.setStyleSheet(f"color:{MUTE};")
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
            self, "Calibration spectra CSV or summary curve XLSX", "",
            "Calibration (*.csv *.xlsx *.xlsm);;CSV (*.csv);;Excel (*.xlsx *.xlsm)")
        if not p:
            return
        problem = self._validate_calib(p)                 # reject fit/curve/wrong CSVs
        if problem:
            self.calib_path = None; self.cal_x.setVisible(False)
            self.cal_lbl.setText("not a calibration"); self.cal_lbl.setStyleSheet(f"color:{RED};")
            self.status.setText(
                f"{os.path.basename(p)} is not a usable calibration — {problem}. "
                "Use calibration_spectra.csv or a concentration/Mean/std XLSX, not "
                "calibration_fit/curve/stats.csv.")
            self.status.setStyleSheet(f"color:{RED};")
            return
        self.calib_path = p; self.cal_lbl.setText("calib: " + os.path.basename(p))
        self.cal_lbl.setStyleSheet(""); self.cal_x.setVisible(True)

    @staticmethod
    def _validate_calib(path):
        """Return a reason string if `path` is not a per-standard spectra calibration
        (compound, concentration_M, <wavenumbers>), else None."""
        if os.path.splitext(path)[1].lower() in (".xlsx", ".xlsm"):
            try:
                from summary_calibration import load_summary_calibration
                curves, _corrections = load_summary_calibration(path)
            except Exception as exc:
                return f"could not read summary curves ({type(exc).__name__})"
            return None if curves else "no summary curves"
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

    def _known_total_uM(self):
        """User-declared summed analyte concentration (µM), or None when blank.
        Sample-prep metadata, NOT read from any label file — real unknowns leave it
        blank and the spectrum-only numbers stand."""
        txt = self.total_edit.text().strip() if hasattr(self, "total_edit") else ""
        if not txt:
            return None
        try:
            v = float(txt.replace(",", ""))
        except ValueError:
            return None
        return v if v > 0 else None

    def _known_total_vec(self, r):
        """Known-total reconstruction Ci = pi × Ctotal over the non-bg substances,
        using the SAME pooled composition the pie reports. None when no total set."""
        total = self._known_total_uM()
        if total is None:
            return None
        ratio = self._ratio_nb(r)
        sel = self._hit(r)
        if not sel.any():
            sel = np.ones(r.n_pixels, bool)
        comp_mean = np.nanmean(ratio[sel], axis=0)
        s = float(np.nansum(comp_mean))
        if not np.isfinite(s) or s <= 0:
            return None
        return comp_mean / s * total

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

    def _spectral_ratio_nb(self, r):
        """NNLS full-spectrum composition before the learned model."""
        rn = np.asarray(getattr(r, "A_evidence", r.A), float)[:, r.nonbg]
        rf = self._rf_vec(r)
        if rf is not None:
            rn = rn / np.where(rf > 0, rf, 1.0)
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
        keep = keep & self._floor_mask(r)
        keep = keep & self._roi_mask(r)
        return keep

    def _roi_mask(self, r):
        roi = getattr(self, "_roi", None)
        if roi is None:
            return np.ones(r.n_pixels, bool)
        x0, x1, y0, y1 = roi
        xy = np.asarray(r.coords, float)
        return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)

    def _clear_roi(self):
        self._roi = None
        self.btn_roi.setEnabled(False)
        if self._res is not None:
            self._apply(self._res)

    def _roi_report(self, r):
        """상태줄용: ROI 안 조성 vs 전체 맵 조성 (신호가중 평균)."""
        if self._roi is None:
            return ""
        nb = [r.comps[i] for i in r.nonbg]
        inside = self._mean_ratio(r)
        n_in = int(self._hit(r).sum())
        roi, self._roi = self._roi, None
        try:
            whole = self._mean_ratio(r); n_all = int(self._hit(r).sum())
        finally:
            self._roi = roi
        f = lambda v: "/".join(f"{x * 100:.0f}" for x in v)
        return (f" · ROI {n_in}/{n_all} px · composition {'/'.join(nb)} "
                f"{f(inside)} % (whole map {f(whole)} %)")

    def _sig_sum(self, r):
        """픽셀별 VIP 밴드 신호합(비-배경 성분 밴드, 음수 클립). 밴드 설정으로 캐시."""
        nb = [r.comps[i] for i in r.nonbg]
        bands = tuple(float(self._band_of(r, nm)) for nm in nb)
        cache = getattr(r, "_sig_sum_cache", None)
        if cache is not None and cache[0] == bands:
            return cache[1]
        S = np.sum([np.clip(np.asarray(self._band_image(r, wl), float), 0, None)
                    for wl in bands], axis=0) if bands else np.zeros(r.n_pixels)
        r._sig_sum_cache = (bands, S)
        return S

    def _floor_mask(self, r):
        v = (self.sl_floor.value()
             if getattr(self, "sl_floor", None) is not None else 0)
        if v <= 0:
            return np.ones(r.n_pixels, bool)
        S = self._sig_sum(r)
        base = np.asarray(r.hit, bool)
        p99 = float(np.quantile(S[base], 0.99)) if base.any() else 0.0
        return S >= (v / 100.0) * p99

    def _floor_label(self, v=None):
        """슬라이더 옆 라벨: 기준 % 와 제외되는 픽셀 수(드래그 중에도 갱신)."""
        if v is None:
            v = self.sl_floor.value()
        r = self._res
        if v <= 0:
            self.lbl_floor.setText("off"); return
        if r is None:
            self.lbl_floor.setText(f"{v} % of p99"); return
        base = np.asarray(r.hit, bool)
        S = self._sig_sum(r)
        p99 = float(np.quantile(S[base], 0.99)) if base.any() else 0.0
        drop = int((base & (S < (v / 100.0) * p99)).sum())
        self.lbl_floor.setText(f"{v} % of p99 · −{drop} px")

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
                          bg_map=self.bg_paths or None,
                          calib_bands=dict(self._bands))
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

    def _leaf_mask(self, r):
        """잉크 도포 영역(True)/외부(False) 마스크. @1000 cm⁻¹ 밴드 Otsu로 밝은
        쪽 = SERS 잉크가 도포된 곳(사용자 확정: 잎 자체는 신호가 없다)."""
        from scipy import ndimage
        # baseline-제거된 r.spectra는 기판의 넓은 배경이 깎여 잎/외부 분리가
        # 흐려진다(외부 7px로 오판). 원본 맵의 1000±8 최대값은 134/134 완벽 분리
        # (Sample1 검증, 2026-09-04) — 원본을 직접 읽는다.
        try:
            from unmix import load_map
            wn0, cube, _m, _c = load_map(self.test)
            wn0 = np.asarray(wn0, float); cube = np.asarray(cube, float)
            win = np.abs(wn0 - 1000.0) <= 8.0
            if cube.shape[0] == r.n_pixels and win.any():
                b = cube[:, win].max(axis=1)
            else:
                b = self._band_image(r, 1000.0)
        except Exception:
            b = self._band_image(r, 1000.0)
        fin = np.isfinite(b)
        if fin.sum() < 10:
            return np.ones(r.n_pixels, bool)
        hist, edges = np.histogram(b[fin], bins=128)
        mids = (edges[:-1] + edges[1:]) / 2
        w = hist.astype(float); csum = np.cumsum(w); cmean = np.cumsum(w * mids)
        best, thr = -1.0, float(np.median(b[fin]))
        for i in range(1, len(mids) - 1):
            w0, w1 = csum[i], csum[-1] - csum[i]
            if w0 <= 0 or w1 <= 0:
                continue
            m0, m1 = cmean[i] / w0, (cmean[-1] - cmean[i]) / w1
            var = w0 * w1 * (m0 - m1) ** 2
            if var > best:
                best, thr = var, mids[i]
        # 사용자 확정(2026-09-04): @1000 cm⁻¹ **밝은 곳이 잎**, 어두운 곳은
        # 아무것도 없는 바깥. 잎 = 밝은 연결영역(8px 이상), 잎 안의 어두운
        # 구멍(잎맥·틈)은 잎으로 메우고, 바깥의 고립된 밝은 점은 바깥으로 둔다.
        leaf = b > thr
        rows, cc, ny, nx, _ux, _uy = self._grid_rc(r)
        g = np.zeros((ny, nx), bool); g[rows, cc] = leaf
        lab, n = ndimage.label(g)
        if n:
            sizes = ndimage.sum(g, lab, index=np.arange(1, n + 1))
            small = np.isin(lab, np.where(sizes < 8)[0] + 1)
            g[small] = False
        g = ndimage.binary_fill_holes(g)
        return g[rows, cc]

    def _leaf_outline(self, ax, r, extent, origin):
        """잉크 도포 영역 경계 윤곽선 — 마스크가 있을 때만. ROI 사각형도 여기서."""
        roi = getattr(self, "_roi", None)
        if roi is not None:
            from matplotlib.patches import Rectangle
            x0, x1, y0, y1 = roi
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="white", linestyle="--", linewidth=1.0,
                                   zorder=7))
        m = getattr(r, "leaf_mask", None)
        if m is None:
            return
        rows, cc, ny, nx, _ux, _uy = self._grid_rc(r)
        g = np.zeros((ny, nx)); g[rows, cc] = m.astype(float)
        try:
            ax.contour(g, levels=[0.5], colors="white", linewidths=0.9,
                       extent=extent, origin=origin, zorder=6)
        except Exception:
            pass

    def _pixel_grid(self, ax, r, extent, origin):
        """픽셀 경계선(흰색, 가늘게) — chk_grid 가 켜져 있을 때만."""
        if getattr(self, "chk_grid", None) is None or not self.chk_grid.isChecked():
            return
        _ri, _ci, _ny, _nx, ux, uy = self._grid_rc(r)
        if len(ux) > 400 or len(uy) > 400:
            return                                  # 너무 촘촘하면 격자가 면이 된다
        px = float(np.min(np.diff(ux))) if len(ux) > 1 else 1.0
        py = float(np.min(np.diff(uy))) if len(uy) > 1 else 1.0
        xs = np.concatenate([ux - px / 2, [ux[-1] + px / 2]])
        ys = np.concatenate([uy - py / 2, [uy[-1] + py / 2]])
        x0, x1, y0, y1 = extent
        ax.vlines(xs, min(y0, y1), max(y0, y1), colors="white", linewidths=0.25,
                  alpha=0.45, zorder=5)
        ax.hlines(ys, min(x0, x1), max(x0, x1), colors="white", linewidths=0.25,
                  alpha=0.45, zorder=5)

    def _apply(self, r):
        self._res = r; self._sel = None
        # 잎 마스크: 원본 hit을 보관해 두고 토글에 따라 외부 픽셀을 null 처리
        if not hasattr(r, "hit_orig"):
            r.hit_orig = np.asarray(r.hit, bool).copy()
        if getattr(self, "chk_leaf", None) is not None and self.chk_leaf.isChecked():
            r.leaf_mask = self._leaf_mask(r)
            r.hit = r.hit_orig & r.leaf_mask
        else:
            r.leaf_mask = None
            r.hit = r.hit_orig.copy()
        if getattr(self, "lbl_floor", None) is not None:
            self._floor_label()          # 라벨의 제외 픽셀 수를 이 맵 기준으로
        self._click_axes = []            # one reset per run — every plot re-registers
        self.pbar.hide()
        self.btn.setEnabled(True); self.btn.setText("Unmix")
        ov = getattr(self, "_bl_override", None)
        _ns = int((self._clipped(r) & r.hit).sum())
        _conc_note = ""
        if not (getattr(r, "calibrated", False) and getattr(r, "conc", None) is not None):
            if r.method == "dlpx" and isinstance(self.dl_model, dict):
                _conc_note = (" · concentration unavailable: this model has no µM head"
                              if not self.dl_model.get("uM") else
                              " · concentration unavailable: the µM head returned no values")
            else:
                _conc_note = " · concentration unavailable: no µM model/calibration applied"
        self.status.setText(f"done — {r.method.upper()}"
                            + (f" · {_ns} saturated px quarantined" if _ns else "")
                            + ("" if ov is None else
                            f" · baseline removal {'on' if ov else 'off'} — followed the "
                            f"model's own setting, not this folder's")
                            + _conc_note + self._roi_report(r))
        # The dynamic band/scale controls settle their card widths on the next Qt
        # layout pass. Refit once then so all four map groups use the same slot width.
        QTimer.singleShot(0, self._redraw)

        self.status.setStyleSheet(f"color:{MUTE};")
        nb = [r.comps[i] for i in r.nonbg]
        mr = self._mean_ratio(r)                          # corrected when toggle on
        eff_hit = self._hit(r)
        dom = nb[int(mr.argmax())] if len(nb) else r.dominant
        self.k_dom.set(dom, TEAL)
        self.k_n.set(str(int(np.sum(mr >= 0.05))), AMBER)
        if "THI" in nb and eff_hit.any():
            s1 = self._spectral_ratio_nb(r)[eff_hit].mean(0)
            dthi = (float(s1[nb.index("THI")])
                    - float(mr[nb.index("THI")])) * 100.0
            self.k_dthi.set(f"{dthi:+.0f} %p", AMBER)
        else:
            self.k_dthi.set("—", MUTE)
        self.k_hit.set(f"{eff_hit.mean():.0%}", BLUE)
        self.k_px.set(f"{r.n_pixels:,}", PURPLE)
        self._rebuild_swatches(r)
        self._rebuild_bandrow(r)     # seeds each substance's VIP band on a fresh run
        self._plot_maps(r); self._plot_abund(r); self._plot_comp(r); self._plot_conc(r); self._plot_pies(r)
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
        scale_mode = (self.cmb_band_scale.currentData()
                      if getattr(self, "cmb_band_scale", None) is not None else "auto")
        all_channels = chans + ex_chans
        if scale_mode == "shared" and all_channels:
            pooled = np.concatenate([np.asarray(v, float).ravel() for v in all_channels])
            shared = (float(np.quantile(pooled, 0.01)),
                      float(np.quantile(pooled, 0.99)))
            if shared[1] <= shared[0]:
                shared = (shared[0], shared[0] + 1.0)
            _auto = [shared] * len(all_channels)
        elif scale_mode == "full":
            _auto = [(float(np.nanmin(v)), float(np.nanmax(v))) for v in all_channels]
            _auto = [(a, b if b > a else a + 1.0) for a, b in _auto]
        _keys = ([r.comps[j] for j in r.nonbg]
                 + [f"extra{i}" for i in range(len(extras))])
        _all_lims = [self._chan_lims(k, a) for k, a in zip(_keys, _auto)]
        self._sync_chan_spins(_keys, _all_lims)
        lims = _all_lims[:len(nb)]
        ex_lims = _all_lims[len(nb):]
        specs = self._single_row_specs(self.c_maps.fig, n)

        # ---- merged R/G/B: each channel stretched over its own P1..P99 ----
        ax = self.c_maps.style(self.c_maps.fig.add_subplot(specs[0]))
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
        self._leaf_outline(ax, r, extent, origin)
        ax.set_title("merged (R/G/B)", fontsize=8)
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
            ax = self.c_maps.style(self.c_maps.fig.add_subplot(specs[i + 1]))
            grid = np.zeros((ny, nx)); grid[rows, cc] = chans[i]
            cmap = LinearSegmentedColormap.from_list("m", ["#0b0d10", nbcols[i]])
            _im = ax.imshow(grid, extent=extent, origin=origin, aspect="equal",
                            interpolation="nearest", cmap=cmap,
                            vmin=lims[i][0], vmax=lims[i][1])
            self._leaf_outline(ax, r, extent, origin)
            # mathtext, not "cm⁻¹" — Arial has no superscript-minus glyph, so the
            # literal character renders as a box in the exported PNG
            ax.set_title(f"{nm} @ {bands[i]:.0f} cm$^{{-1}}$", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            # ramp UNDER the panel — horizontal bars share the panel's width, so
            # (unlike the old vertical ones) they cannot outgrow the map
            cb = self._side_colorbar(self.c_maps.fig, ax, _im,
                                     ticks=[lims[i][0], lims[i][1]],
                                     labels=[f"{lims[i][0]:.0f}", f"{lims[i][1]:.0f}"])
            self._exp_maps.append((f"band_{nm}", ax, cb.ax))
            self._click_axes.append(ax)
        for ei, wl in enumerate(extras):                   # user-added bands, own hue
            v = ex_chans[ei]
            va, vb = ex_lims[ei]
            slot = specs[len(nb) + 1 + ei]
            ax = self.c_maps.style(self.c_maps.fig.add_subplot(slot))
            grid = np.zeros((ny, nx)); grid[rows, cc] = v
            cmap = LinearSegmentedColormap.from_list("m", ["#0b0d10", ex_cols[ei]])
            _im = ax.imshow(grid, extent=extent, origin=origin, aspect="equal",
                            interpolation="nearest", cmap=cmap, vmin=va, vmax=vb)
            self._leaf_outline(ax, r, extent, origin)
            ax.set_title(f"@ {wl:.0f} cm$^{{-1}}$", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            cb = self._side_colorbar(self.c_maps.fig, ax, _im,
                                     ticks=[va, vb], labels=[f"{va:.0f}", f"{vb:.0f}"])
            self._exp_maps.append((f"band_extra_{wl:.0f}", ax, cb.ax))
            self._click_axes.append(ax)
        self.c_maps.draw_idle()          # gridspec already carries explicit margins

    def _plot_abund(self, r):
        """NNLS analyte maps plus one combined background map.

        BLK and INK are both nuisance/background classes for interpretation. Showing
        them as separate full panels repeated the same answer and squeezed the analyte
        maps, so they are summed into one background probability/abundance panel.
        Every single-component panel uses the same scale; the numeric arrays remain in
        the per-pixel export.
        """
        from matplotlib.colors import LinearSegmentedColormap
        if not self.c_abund.isVisible():
            self._fold_dirty["abund"] = True
            return
        self._fold_dirty["abund"] = False
        self.c_abund.fig.clear()
        self._exp_abund = []
        if getattr(r, "A", None) is None:
            self.c_abund.draw_idle()
            return

        Aall = np.asarray(getattr(r, "A_evidence", r.A), float)
        nb_idx = list(r.nonbg)
        nbcols = self._nb_colors(r)
        # dlpx 기본 표시: NNLS 증거 대신 모델 조성(분석물 간 재정규화)을
        # 그린다. raw 확률은 BLK 몫이 희석해 채널이 흐려진다(260812 SERS 글씨맵
        # 사건) — ratio_nb 는 hit 픽셀에서 항상 분석물 합=1 이라 글자가 살아난다.
        model_view = (getattr(r, "method", "") == "dlpx"
                      and getattr(r, "ratio_nb", None) is not None)
        Anb = (np.asarray(r.ratio_nb, float) if model_view else Aall[:, nb_idx])
        hit = self._hit(r)
        shade = (model_view and getattr(self, "chk_abund_shade", None) is not None
                 and self.chk_abund_shade.isChecked())
        if shade:
            _S = np.sum([np.clip(self._band_image(r, self._band_of(r, r.comps[k])),
                                 0, None) for k in nb_idx], axis=0)
            # raw 밴드맵의 auto contrast 와 같은 관례: hit 픽셀 p5–p95 를
            # 밝기 0.3–1.0 으로 (바닥 0.3 은 조성 자체가 지워지지 않게).
            if hit.any():
                _lo, _hi = np.quantile(_S[hit], [0.05, 0.95])
            else:
                _lo, _hi = 0.0, 0.0
            _b = (0.45 + 0.55 * np.clip((_S - _lo) / (_hi - _lo), 0.0, 1.0)
                  if _hi > _lo else np.ones_like(_S))
            Anb = np.asarray(Anb, float) * _b[:, None]
        # Values remain available in the export for every measured pixel, but an
        # analyte prediction has no meaning after the gate called that pixel
        # background. Do not paint those nuisance responses as analyte signal.
        Anb_draw = np.where(hit[:, None], Anb, np.nan)
        _mass = Anb[hit].sum(axis=1) if hit.any() else np.array([], float)
        mscale = float(np.quantile(_mass, 0.99)) if _mass.size else 1.0
        mscale = mscale or 1.0

        panels = [("merged" + (" · model comp" if model_view else "")
                   + (" · shaded" if shade else ""), None, None)]
        panels.extend((r.comps[k], np.where(hit, Anb[:, i], np.nan), nbcols[i])
                      for i, k in enumerate(nb_idx))
        bg_idx = np.flatnonzero(np.asarray(r.bg_mask, bool))
        if bg_idx.size:
            # This panel exactly complements the masked analyte maps.
            panels.append(("background (gate)", (~hit).astype(float), "#6b7280"))

        # Scale from analyte hits only; the binary gate panel must not flatten them.
        scale_values = Anb_draw
        finite_scale = scale_values[np.isfinite(scale_values)]
        amax = float(np.max(finite_scale)) if finite_scale.size else 1.0
        vshared = 1.0 if amax <= 1.05 else (
            float(np.quantile(finite_scale, 0.99)) or 1.0)
        manual = self._parse_scale("abund")
        vlo = manual[0] if manual is not None else 0.0
        if manual is not None:
            vshared = manual[1]
        elif model_view:
            # 조성 0-1 에서 등몰 글씨(~0.33)가 절반 이상의 채도로 오도록 — 29c 렌더와
            # 같은 관례. manual scale 이 있으면 그쪽이 이긴다.
            vshared = 0.6

        rows, cc, ny, nx, ux, uy = self._grid_rc(r)
        origin, extent = self._extent_origin(ux, uy)
        # 문맥 밑그림: 모든 픽셀의 총 신호를 어두운 회색(0.06–0.30)으로 깐다 —
        # raw 밴드맵이 보여주는 잎·잉크 형태가 복원 맵에도 남고, hit 픽셀은 그
        # 위에 조성 색(밝기 ≥ 0.45)으로 얹힌다 (사용자 2026-09-11).
        _tot = np.clip(np.asarray(r.spectra, float), 0, None).sum(axis=1)
        _g_lo, _g_hi = np.percentile(_tot, [2, 98])
        _gray = (0.06 + 0.24 * np.clip((_tot - _g_lo) / max(_g_hi - _g_lo, 1e-9), 0, 1))
        under = np.zeros((ny, nx, 3))
        under[rows, cc] = _gray[:, None]
        specs = self._single_row_specs(self.c_abund.fig, len(panels))
        for idx, (title, values, panel_color) in enumerate(panels):
            ax = self.c_abund.style(self.c_abund.fig.add_subplot(specs[idx]))
            panel_im = None
            if values is None:
                cols = np.array([to_rgb(c) for c in nbcols])
                norm = np.nan_to_num(np.clip(Anb_draw / mscale, 0.0, 1.0), nan=0.0)
                weights = np.maximum(norm.sum(axis=1, keepdims=True), 1.0)
                col_px = np.clip((norm / weights) @ cols * np.minimum(
                    norm.sum(axis=1, keepdims=True), 1.0), 0.0, 1.0)
                img = under.copy()
                img[rows[hit], cc[hit]] = col_px[hit]
                ax.imshow(img, extent=extent, origin=origin, aspect="equal",
                          interpolation="nearest")
                self._pixel_grid(ax, r, extent, origin)
                self._leaf_outline(ax, r, extent, origin)
                title = f"merged ({vlo:.3g}–{vshared:.3g})"
            else:
                grid = np.full((ny, nx), np.nan); grid[rows, cc] = values
                cmap = LinearSegmentedColormap.from_list(
                    "m", ["#0b0d10", panel_color])
                ax.set_facecolor("#0b0d10")
                is_bg = title.startswith("background")
                if is_bg:
                    cmap.set_bad("#0b0d10")
                else:
                    # 비-hit 픽셀은 투명 → 아래 회색 밑그림이 비친다
                    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
                    ax.imshow(under, extent=extent, origin=origin, aspect="equal",
                              interpolation="nearest", zorder=0)
                _vmax = 1.0 if is_bg else vshared
                panel_im = ax.imshow(grid, extent=extent, origin=origin, aspect="equal",
                                     interpolation="nearest", cmap=cmap,
                                     vmin=0.0 if is_bg else vlo, vmax=_vmax)
                if not is_bg:
                    self._pixel_grid(ax, r, extent, origin)
                self._leaf_outline(ax, r, extent, origin)
                self._side_colorbar(
                    self.c_abund.fig, ax, panel_im,
                    ticks=[0.0 if is_bg else vlo, _vmax],
                    labels=[f"{0.0 if is_bg else vlo:.2g}", f"{_vmax:.2g}"])
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            self._exp_abund.append((f"comp_{title.split(' ')[0]}", ax, None))
            self._click_axes.append(ax)
        self.c_abund.draw_idle()

    KNN_MAX_DIST = 3.0        # 이 z-거리 밖이면 라이브러리에 닮은 맵이 없다 — 무응답

    def _knn_lookup(self, r):
        """학습 맵 라이브러리(k=3, 거리가중) 조회. 반환: dict(dmin, uM, names) 또는
        None(라이브러리/피처 없음). 자기 자신(파일명 일치)은 제외해 LOO 의미 유지."""
        lib = (self.dl_model.get("_knn_library")
               if isinstance(self.dl_model, dict) else None)
        z = getattr(r, "conc_feature_z", None)
        if not lib or z is None:
            return None
        z = np.asarray(z, float)
        base = os.path.basename(self.test or "")
        rows = [e for e in lib if not e.get("imb100")]
        if len(rows) < 3:
            return None
        Z = np.array([e["z"] for e in rows], float)
        Y = np.array([e["y"] for e in rows], float)
        d = np.linalg.norm(Z - z[None, :], axis=1)
        loo = getattr(self, "chk_loo", None) is None or self.chk_loo.isChecked()
        for j, e in enumerate(rows):
            if loo and e["name"] == base:
                d[j] = np.inf
        idx = np.argsort(d)[:3]
        w = 1.0 / np.maximum(d[idx], 1e-9); w = w / w.sum()
        return {"dmin": float(d[idx][0]),
                "uM": (Y[idx] * w[:, None]).sum(0),
                "names": [rows[j]["name"] for j in idx]}

    def _pxknn_lookup(self, r):
        """픽셀 하나 = 액적 하나: hit 픽셀 각각을 학습 맵 픽셀 라이브러리에서 조회.
        반환: (P (n_hit,3) µM, d_med) 또는 None. 자기 파일명 조건 픽셀은 제외."""
        lib = (self.dl_model.get("_pxknn")
               if isinstance(self.dl_model, dict) else None)
        if not lib or getattr(r, "ratio_nb", None) is None:
            return None
        from dl_model import _band_signal
        u = self.dl_model.get("uM") or {}
        bands = np.asarray(u.get("bands_cm", ()), float)
        if bands.size != 3:
            return None
        hitm = r.hit if r.hit.any() else np.ones(r.n_pixels, bool)
        wn = np.asarray(r.wn, float)
        X = np.clip(np.asarray(r.spectra, float), 0, None)[hitm]
        sig = np.log1p(np.clip(_band_signal(X, wn, bands), 0, None))
        tot = np.log1p(X.sum(1))[:, None]
        Rq = np.clip(np.asarray(r.ratio_nb, float), 0, None)[hitm]
        F = (np.hstack([sig, tot, Rq]) - lib["mu"]) / lib["sd"]
        Fz = np.asarray(lib["Fz"], float); Y = np.asarray(lib["Y"], float)
        cond = np.asarray(lib["cond"]).astype(str)
        excl = ((cond == os.path.basename(self.test or ""))
                if getattr(self, "chk_loo", None) is None or self.chk_loo.isChecked()
                else np.zeros(len(cond), bool))
        Zl = Fz[~excl]; Yl = Y[~excl]
        # k=15 + 로그공간(기하) 가중평균: 농도는 로그 성질이라 기하평균이 맞고,
        # 라벨 격자(3·6·…·500)에 스냅되던 계단 밴딩과 고농도 꼬리 인공물을 없앤다
        # (검증: LOO 정확도 동등~개선, 글씨맵 DQ 최대 403→42 µM).
        P = np.zeros((len(F), 3)); dmin = np.zeros(len(F))
        _EPS = 0.25
        for i, q in enumerate(F):
            d = np.linalg.norm(Zl - q[None, :], axis=1)
            idx = np.argpartition(d, 15)[:15]
            w = 1.0 / np.maximum(d[idx], 1e-9); w = w / w.sum()
            P[i] = np.exp((w[:, None] * np.log(Yl[idx] + _EPS)).sum(0)) - _EPS
            dmin[i] = d[idx].min()
        return {"P": np.clip(P, 0, None), "hit": hitm,
                "d_med": float(np.median(dmin))}

    def _apparent_medians(self, r):
        """Median RAW apparent µM per non-background substance, mirroring the
        distribution panel's filtering (hit px, finite, positive, OOD/above-range
        dropped). Returns (names, medians, at_ceiling) — ceiling = median ≥ 80 % of
        the validated hi bound, where a saturating response pins the inversion."""
        nb = [r.comps[i] for i in r.nonbg]
        um_all = r.conc * 1e6
        hit = self._hit(r)
        ood = getattr(r, "conc_ood", None)
        rngs = getattr(r, "conc_ranges", None)
        hi_um = None
        if rngs is not None:
            _h = np.asarray(rngs, float)[:, 1] * 1e6
            hi_um = np.where(np.isfinite(_h), _h, np.inf)
        med = np.full(len(nb), np.nan)
        at_ceiling = np.zeros(len(nb), bool)
        for i in range(len(nb)):
            sel = hit if hit.any() else np.ones(r.n_pixels, bool)
            v = um_all[sel, i]
            fin = np.isfinite(v) & (v > 0)
            bad = np.zeros(len(v), bool)
            if ood is not None:
                bad |= np.asarray(ood, bool)[sel, i]
            if hi_um is not None and np.isfinite(hi_um[i]):
                bad |= fin & (v > hi_um[i])
            vv = v[fin & ~bad]
            if vv.size:
                med[i] = float(np.median(vv))
                if hi_um is not None and np.isfinite(hi_um[i]) \
                        and med[i] >= 0.8 * hi_um[i]:
                    at_ceiling[i] = True
        return nb, med, at_ceiling

    def _set_anchor(self):
        """Adopt the CURRENT map as the session's one-point batch recalibration."""
        r = self._res
        if r is None or not getattr(r, "calibrated", False) or r.conc is None:
            self.status.setText("unmix a map with a µM model first, then set the anchor")
            self.status.setStyleSheet(f"color:{RED};"); return
        nb, med, ceil = self._apparent_medians(r)
        txt = self.true_edit.text().strip()
        try:
            parts = [float(t) for t in txt.replace(" ", "").split(",")] if txt else []
        except ValueError:
            parts = []
        if len(parts) != len(nb) or any(p <= 0 for p in parts):
            self.status.setText("type this map's true µM (a,b,c) in the box first — "
                                "the grey text is only an example")
            self.status.setStyleSheet(f"color:{RED};")
            self.true_edit.setFocus(); return
        factor = np.ones(len(nb)); tags = []
        for i, nm in enumerate(nb):
            if np.isfinite(med[i]) and med[i] > 0 and not ceil[i]:
                factor[i] = parts[i] / med[i]
                tags.append(f"{nm} ×{factor[i]:.2f}")
            else:                       # saturated/empty — no honest factor exists
                tags.append(f"{nm} skipped")
        self._anchor = {"file": os.path.basename(self.test or "?"),
                        "subs": list(nb), "factor": factor}
        self.anchor_lbl.setText("anchor: " + " · ".join(tags))
        self.anchor_x.setVisible(True)
        self._plot_conc(r)

    def _clear_anchor(self):
        self._anchor = None
        self.anchor_lbl.setText(""); self.anchor_x.setVisible(False)
        if self._res is not None:
            self._plot_conc(self._res)

    def _draw_conc_spiral(self, ax, r, nb, nbcols, um_all, hit, med,
                          out_of_lib):
        """Circular heatmap inspired by radial expression plots.

        Angle is the NNLS THI-share rank. Each coloured ring is one analyte's
        concentration relative to truth (when supplied) or its map median:
        purple < reference, ivory = reference, gold > reference. The centre curve
        is the local mean absolute MLP-vs-NNLS composition change, so this panel
        supports the dashboard's main comparison instead of being decorative.
        """
        truth = None
        try:
            tv = [float(x) for x in
                  self.true_edit.text().replace(" ", "").split(",") if x]
            if len(tv) == len(nb) and any(v > 0 for v in tv):
                truth = tv
        except Exception:
            truth = None
        ref = [(truth[i] if truth and truth[i] > 0 else
                (med[i] if np.isfinite(med[i]) and med[i] > 0 else None))
               for i in range(len(nb))]
        shown = False
        rnb = getattr(r, "ratio_nb", None)
        if not out_of_lib and hit.any() and rnb is not None:
            thi = nb.index("THI") if "THI" in nb else 0
            share = np.clip(np.asarray(rnb, float), 0, None)[:, thi]
            sel = np.where(hit)[0]
            order = sel[np.argsort(share[sel])]
            n = len(order)
            bins = min(n, self.sl_spiral.value() if hasattr(self, "sl_spiral") else 36)
            bins = max(1, bins)
            edges_i = np.linspace(0, n, bins + 1).astype(int)
            heat = np.full((len(nb), bins), np.nan)
            for i in range(len(nb)):
                if ref[i] is None:
                    continue
                v = um_all[order, i]
                for b in range(bins):
                    w = v[edges_i[b]:edges_i[b + 1]]
                    w = w[np.isfinite(w) & (w > 0)]
                    if w.size:
                        heat[i, b] = np.log2(np.median(w) / ref[i])
                        shown = True
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                "radial_ratio", ["#7775a5", "#f7f5ed", "#e6a63a"])
            # Full, contiguous clockwise rings, matching the reference visual grammar.
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            theta_e = np.linspace(0, 2 * np.pi, bins + 1)
            theta_c = (theta_e[:-1] + theta_e[1:]) / 2.0
            ax.grid(False)
            # Mean in log2 space = geometric mean fold-change across DQ/TBZ/THI.
            integrated = np.nanmean(heat, axis=0)
            heat_rows = np.vstack([heat, integrated])
            ring_names = list(nb) + ["Integrated"]
            ring_h, pitch, inner = 0.76, 0.79, 1.35
            for i, nm in enumerate(ring_names):
                bottom = inner + i * pitch
                colors = [cmap(np.clip((v + 1.0) / 2.0, 0.0, 1.0))
                          if np.isfinite(v) else (0.92, 0.92, 0.92, 1.0)
                          for v in heat_rows[i]]
                ax.bar(theta_c, np.full(bins, ring_h), bottom=bottom,
                       width=np.diff(theta_e) * 0.98, align="center",
                       color=colors, edgecolor="white", linewidth=0.35, zorder=1)
                # Like the reference figure, ring names sit together at 12 o'clock.
                ax.text(0.0, bottom + ring_h / 2,
                        nm, ha="center", va="center", fontsize=6.5,
                        fontweight="bold", color="#30343a",
                        bbox=dict(facecolor="white", alpha=0.72,
                                  edgecolor="none", pad=0.35),
                        zorder=6)
            # Actual THI fractions around the perimeter, analogous to the reference's
            # outer feature labels. Numeric values remove the old rank ambiguity.
            outer = inner + len(ring_names) * pitch
            label_bins = np.unique(np.linspace(0, bins - 1, min(8, bins)).astype(int))
            for b in label_bins:
                sw = share[order[edges_i[b]:edges_i[b + 1]]]
                if not len(sw):
                    continue
                angle = theta_c[b]
                deg = float(np.degrees(angle))
                rotation = -deg
                ha = "left"
                if 90 < deg < 270:
                    rotation += 180
                    ha = "right"
                ax.text(angle, outer + 0.10, f"THI {np.median(sw) * 100:.0f}%",
                         rotation=rotation, rotation_mode="anchor",
                         ha=ha, va="center", fontsize=5.5, color="#30343a")
            # The reference graphic uses the centre for a separate summary.  Keep the
            # fold colours exclusive to the rings and report model disagreement as a
            # number, so %p can never be mistaken for a reference fold.
            before = np.asarray(self._spectral_ratio_nb(r), float)
            after = np.asarray(self._ratio_nb(r), float)
            delta = np.mean(np.abs(after - before), axis=1) * 100.0
            dmed = float(np.nanmedian(delta[hit])) if hit.any() else np.nan
            ax.text(0.0, inner * 0.46,
                    "MLP−NNLS\nmedian Δ\n" + (f"{dmed:.1f} %p" if np.isfinite(dmed) else "—"),
                    ha="center", va="center", fontsize=6.3, fontweight="bold",
                    color="#30343a", zorder=8)
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            ax.set_xticks([]); ax.set_yticks([])
        if shown:
            # Match the reference: a compact vertical expression scale at upper-right.
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            cax = ax.inset_axes([1.01, 0.64, 0.035, 0.27])
            sm = ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap=cmap)
            cb = ax.figure.colorbar(sm, cax=cax, orientation="vertical")
            cb.set_ticks([-1.0, 0.0, np.log2(1.5), 1.0])
            cb.set_ticklabels(["0.5×", "1×", "1.5×", "2×"])
            cb.ax.tick_params(labelsize=5, length=2, pad=1)
            cb.outline.set_linewidth(0.45)
            cax.set_title("Fold", fontsize=6, pad=2)
        ax.set_ylim(0.0, inner + (len(nb) + 1) * pitch + 0.55
                    if shown else len(nb) + 1.0)
        ax.spines["polar"].set_visible(False)
        ax.set_title("" if shown else "radial readout · no answer",
                     fontsize=8, color="#c0392b", pad=8)

    def _ensure_conc_scale_controls(self, names, auto_max):
        """One editable colour maximum per apparent-concentration map."""
        names = tuple(names)
        if names == getattr(self, "_conc_scale_names", ()):
            return
        while self.conc_scale_lay.count():
            item = self.conc_scale_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._conc_scale_names = names
        self._conc_scale_spins = {}
        self.conc_scale_lay.addStretch(1)
        self.conc_scale_lay.addWidget(self._mk_lbl("map max (µM):"))
        for nm in names:
            self.conc_scale_lay.addWidget(self._mk_lbl(nm))
            sp = QDoubleSpinBox(); sp.setDecimals(1); sp.setRange(0.1, 1e6)
            sp.setValue(max(float(auto_max), 0.1)); sp.setFixedWidth(68)
            sp.setToolTip(f"{nm} concentration-map colour maximum")
            sp.editingFinished.connect(
                lambda: self._res is not None and self._plot_conc(self._res))
            self.conc_scale_lay.addWidget(sp)
            self._conc_scale_spins[nm] = sp
        self.conc_scale_lay.addStretch(1)

    def _plot_conc(self, r):
        """Per-substance apparent SERS-equivalent concentration (µM) heat-maps — only when a
        dilution-series calibration has been applied."""
        if not getattr(r, "calibrated", False) or r.conc is None:
            self.card_conc.setVisible(False)
            self.result_grid.setRowStretch(2, 0)
            return
        self.card_conc.setVisible(True)
        self.result_grid.setRowStretch(2, 4)
        self.c_conc.fig.clear()
        self._exp_conc = []
        nb = [r.comps[i] for i in r.nonbg]; nbcols = self._nb_colors(r)
        rows, cc, ny, nx, ux, uy = self._grid_rc(r)
        origin, extent = self._extent_origin(ux, uy)
        # This row does not need to match the other rows' map size — use the width:
        # merged + 3 map cells + spacer + distribution(3) + spacer + radial(3).
        map_slots = 12
        hit = self._hit(r)                                 # exclude saturated/low-R² px
        # SHARED µM colour axis across substances, so the maps are directly comparable.
        # The maps and dots stay RAW (what signal + learning alone report); the batch
        # anchor is drawn as an OVERLAY so the before/after is visible in one panel.
        um_all = r.conc * 1e6
        # 판독 레시피(채택): 각 픽셀의 모델 조성 비율 × 그 픽셀의 µM 총량.
        # 픽셀 단위에서 농도 비율이 조성과 정확히 정합한다. (외부 정보 없음 —
        # 두 값 모두 모델 자신의 출력이다.)
        _Rnb = getattr(r, "ratio_nb", None)
        est_total = None
        if _Rnb is not None:
            _Rpx = np.clip(np.asarray(_Rnb, float), 0, None)
            if _Rpx.shape == um_all.shape:
                _fin = np.isfinite(um_all) & (um_all > 0)
                _T = np.where(_fin, um_all, 0.0).sum(axis=1)
                um_all = _Rpx * _T[:, None]
                _hs = hit & (_T > 0) if hit.any() else (_T > 0)
                if _hs.any():
                    est_total = float(np.median(_T[_hs]))
        # ── 판독 경로: model head vs library k-NN — 무응답 규칙은 공통 ──
        route_sel = (self.cmb_umroute.currentData()
                     if hasattr(self, "cmb_umroute") else "auto")
        knn = self._knn_lookup(r)
        out_of_lib = (knn["dmin"] > self.KNN_MAX_DIST) if knn is not None \
            else bool(getattr(r, "conc_batch_mismatch", False))
        route = route_sel
        if route_sel == "auto":
            # 손 스위칭 없음: head가 영역 안이면 head, 밖이면 픽셀 장부가 커버할
            # 때 자동으로 그 값을 (라벨 "pixel-library"로 출처 명시). 둘 다 밖이면
            # 무응답.
            route = "model"
            if out_of_lib:
                _pxa = self._pxknn_lookup(r)
                if _pxa is not None and _pxa["d_med"] <= self.KNN_MAX_DIST:
                    route = "pxknn"
        knn_used = False
        px_used = False
        if route == "pxknn":
            _px = self._pxknn_lookup(r)
            if _px is not None:
                out_of_lib = _px["d_med"] > self.KNN_MAX_DIST
                if not out_of_lib:
                    # 픽셀 값 자체가 조회 결과: hit 픽셀만 채우고 나머지는 0
                    um_all = np.zeros_like(um_all)
                    um_all[_px["hit"]] = _px["P"]
                    px_used = True
                knn = knn or {"dmin": _px["d_med"]}
                knn["dmin"] = _px["d_med"]
        if route == "knn" and knn is not None and not out_of_lib:
            # k-NN 모드의 값 = 이웃 학습 맵 실측 농도의 내분점. 픽셀 패턴은 그대로
            # 두고 성분별 중앙값만 조회값에 맞춘다.
            for i in range(len(nb)):
                v = um_all[hit, i] if hit.any() else um_all[:, i]
                fin = np.isfinite(v) & (v > 0)
                mmed = float(np.median(v[fin])) if fin.any() else 0.0
                if mmed > 0 and np.isfinite(knn["uM"][i]):
                    um_all[:, i] = um_all[:, i] * (float(knn["uM"][i]) / mmed)
            knn_used = True
        raw_sig = False
        if route_sel in ("raw", "mlpsig"):
            # 판독 대신 실측: 성분별 VIP 밴드(현재 밴드 스핀 값)의 픽셀 세기.
            # 무응답 규칙·검량창·OOD는 해당 없음(측정값 그 자체).
            um_all = np.stack([np.clip(self._band_image(r, self._band_of(r, nm)),
                                       0, None) for nm in nb], axis=1)
            if route_sel == "mlpsig" and getattr(r, "ratio_nb", None) is not None:
                # 총 밴드 신호(counts)를 MLP 조성으로 배분 — 경쟁 보정된 몫.
                # 게이트 밖 픽셀(배경·잉크 지배)은 0.
                _R = np.clip(np.asarray(r.ratio_nb, float), 0, None)
                _R = _R / (_R.sum(axis=1, keepdims=True) + 1e-12)
                um_all = _R * um_all.sum(axis=1, keepdims=True)
                um_all[~hit] = 0.0
            out_of_lib = False; px_used = False; knn_used = False
            route = route_sel; raw_sig = True
        # 픽셀 파이 크기용 스냅샷 — 현재 판독 경로의 픽셀 값 (2026-09-04)
        self._um_display = None if out_of_lib else um_all.copy()
        self._um_display_route = route
        self._um_display_units = "counts" if raw_sig else "uM"
        anch = getattr(self, "_anchor", None)
        anchored = anch is not None and anch.get("subs") == nb
        afac = (np.asarray(anch["factor"], float) if anchored else None)
        # the model's own out-of-range judgment: per-pixel OOD flags plus the stored
        # reportable range. A weak binder (DQ) has a nearly flat response, so its
        # inversion EXPLODES on spurious signal — thousands of µM on a 3–500 µM
        # training range. Those pixels may not silently drive the medians.
        ood = getattr(r, "conc_ood", None)
        rngs = getattr(r, "conc_ranges", None)
        if raw_sig:
            ood = None; rngs = None            # counts에는 검량창/OOD 개념이 없다
        hi_um = None
        lo_um = None
        if rngs is not None:
            _h = np.asarray(rngs, float)[:, 1] * 1e6
            hi_um = np.where(np.isfinite(_h), _h, np.inf)
        if rngs is not None:                    # validated lo bound counts too
            _l = np.asarray(rngs, float)[:, 0] * 1e6
            lo_um = np.where(np.isfinite(_l), _l, 0.0)
        vmask = hit[:, None] & np.isfinite(um_all) & (um_all > 0)
        vals = um_all[vmask]
        vmax = float(np.quantile(vals, 0.90)) if vals.size else 1.0
        if hi_um is not None and np.isfinite(hi_um).any():
            # never stretch the shared ramp past the calibrated range — one exploding
            # substance was blanking every other panel
            vmax = min(vmax, float(np.nanmax(hi_um[np.isfinite(hi_um)])))
        vmax = vmax or 1.0
        vmax *= (self.sl_map_scale.value() / 100.0
                 if hasattr(self, "sl_map_scale") else 1.0)
        self._ensure_conc_scale_controls(nb, vmax)
        vmaxes = [float(self._conc_scale_spins[nm].value()) for nm in nb]
        if raw_sig:
            vmaxes = [float(np.quantile(um_all[hit, i], 0.99)) if hit.any() and
                      np.isfinite(um_all[hit, i]).any() else 1.0
                      for i in range(len(nb))]
        # Snapshot the exact displayed values for a clean export-only grouped figure.
        # Reusing screen axes can capture the neighbouring distribution's labels.
        self._conc_export = dict(nb=list(nb), colors=list(nbcols),
                                 um=np.asarray(um_all, float).copy(),
                                 hit=np.asarray(hit, bool).copy(), rows=rows.copy(),
                                 cols=cc.copy(), ny=ny, nx=nx, extent=list(extent),
                                 origin=origin, vmax=list(vmaxes))
        from matplotlib.colors import LinearSegmentedColormap
        # Full-width spec of its own: _row_gs centres this row to the OTHER cards'
        # map width, which left both gutters empty. This row uses everything.
        _old_cid = getattr(self.c_conc, "_rowgs_cid", None)
        if _old_cid is not None:
            self.c_conc.mpl_disconnect(_old_cid); self.c_conc._rowgs_cid = None
        gs = self.c_conc.fig.add_gridspec(
            2, map_slots, height_ratios=[1.0, 0.06],
            width_ratios=[1, 1, 1, 1, 0.22, 1, 1, 1, 0.22, 1, 1, 1],
            hspace=0.05, wspace=0.04,
            left=0.012, right=0.988, bottom=0.13, top=0.84)
        # merged (R/G/B): 성분별 µM ÷ 그 성분의 map max 를 채널로, raw/composition
        # 카드와 같은 색상 규칙(색조 유지·합>1 정규화) + 회색 총신호 밑그림.
        _tot = np.clip(np.asarray(r.spectra, float), 0, None).sum(axis=1)
        _g_lo, _g_hi = np.percentile(_tot, [2, 98])
        _gray = 0.06 + 0.24 * np.clip((_tot - _g_lo) / max(_g_hi - _g_lo, 1e-9), 0, 1)
        _img = np.zeros((ny, nx, 3)); _img[rows, cc] = _gray[:, None]
        _chan = np.stack([np.clip(np.nan_to_num(um_all[:, i], nan=0.0)
                                  / max(vmaxes[i], 1e-9), 0.0, 1.0)
                          for i in range(len(nb))], axis=1)
        _cols = np.array([to_rgb(c) for c in nbcols])
        _wsum = np.maximum(_chan.sum(axis=1, keepdims=True), 1.0)
        _rgb = np.clip((_chan / _wsum) @ _cols
                       * np.minimum(_chan.sum(axis=1, keepdims=True), 1.0), 0, 1)
        _lit = hit & (_chan.sum(axis=1) > 0)
        _img[rows[_lit], cc[_lit]] = _rgb[_lit]
        axm = self.c_conc.style(self.c_conc.fig.add_subplot(gs[0, 0]))
        axm.imshow(_img, extent=extent, origin=origin, aspect="equal",
                   interpolation="nearest")
        self._pixel_grid(axm, r, extent, origin)
        self._leaf_outline(axm, r, extent, origin)
        axm.set_anchor("S")
        axm.set_title("merged (R/G/B)", fontsize=10)
        axm.set_xticks([]); axm.set_yticks([])
        self._exp_conc.append(("map_merged", axm, None))
        self._click_axes.append(axm)
        for i, nm in enumerate(nb):
            ax = self.c_conc.style(self.c_conc.fig.add_subplot(gs[0, i + 1]))
            cax = self.c_conc.fig.add_subplot(gs[1, i + 1])
            um = np.where(hit & np.isfinite(um_all[:, i]) & (um_all[:, i] > 0),
                          um_all[:, i], np.nan)
            grid = np.full((ny, nx), np.nan); grid[rows, cc] = um
            cmap = LinearSegmentedColormap.from_list("m", ["#0b0d10", nbcols[i]])
            cmap.set_bad("#0b0d10")
            ax.set_facecolor("#0b0d10")
            im = ax.imshow(grid, extent=extent, origin=origin, aspect="equal",
                           interpolation="nearest", cmap=cmap, vmin=0.0,
                           vmax=vmaxes[i])
            self._pixel_grid(ax, r, extent, origin)
            self._leaf_outline(ax, r, extent, origin)
            # 스케일 바 — 세 맵이 같은 0..vmax 램프를 쓴다는 것까지 같이 보인다.
            # aspect=equal 로 축 상자가 줄어들 때 맵은 아래(S), 바는 위(N)로
            # 붙여 둘 사이가 벌어지지 않게 한다 (2026-09-02).
            ax.set_anchor("S")
            cax.set_anchor("N")
            cb = self.c_conc.fig.colorbar(im, cax=cax,
                                          orientation="horizontal")
            cb.set_ticks([0.0, vmaxes[i]])
            cb.set_ticklabels(["0", f"{vmaxes[i]:.0f}" + ("" if raw_sig else " µM")])
            cax.tick_params(labelsize=8, length=2, pad=1)
            cb.outline.set_linewidth(0.4)
            self._exp_conc.append((f"map_{nm}", ax, cax))
            ax.set_title(nm, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            self._click_axes.append(ax)
        # ---- pixel distribution: show the measurements, not only one median bar ----
        # Three balanced zones: concentration maps | distribution | radial heatmap.
        dist_start = len(nb) + 2
        dist_end = min(dist_start + 3, map_slots)
        axb = self.c_conc.style(
            self.c_conc.fig.add_subplot(gs[:, dist_start:dist_end]))
        med = np.full(len(nb), np.nan); q1 = med.copy(); q3 = med.copy()
        bad_frac = np.zeros(len(nb))
        total_n = np.zeros(len(nb), dtype=int)
        valid_n = np.zeros(len(nb), dtype=int)
        distributions = []
        for i in range(len(nb)):
            sel = hit if hit.any() else np.ones(r.n_pixels, bool)
            v = um_all[sel, i]
            fin = np.isfinite(v) & (v > 0)
            bad = np.zeros(len(v), bool)
            if ood is not None:
                bad |= np.asarray(ood, bool)[sel, i]
            if hi_um is not None and np.isfinite(hi_um[i]):
                bad |= fin & (v > hi_um[i])
            # The DISPLAY shows every pixel — small and large — and the median of
            # all of them: censoring over-range pixels out of the plot left whole
            # substances looking empty. Over-range is a caveat on the number, not
            # a reason to hide the measurements.
            bad_frac[i] = float(bad[fin].mean()) if fin.any() else 0.0
            vv = v[fin]
            total_n[i] = int(fin.sum())
            valid_n[i] = int((fin & ~bad).sum())
            distributions.append(vv)
            if vv.size:
                med[i], q1[i], q3[i] = (float(np.median(vv)),
                                        float(np.quantile(vv, 0.25)),
                                        float(np.quantile(vv, 0.75)))
        # ── 판독 스파이럴 (그림 44k/60 문법의 배선판) — 각도 = 픽셀 표면 THI
        #    분율 순위, 반지름 = 픽셀 판독 ÷ 기준(truth 입력 시 참값, 없으면 맵
        #    중앙값 = 픽셀 산포). 무응답이면 점 없이 과녁만.
        ax_sp = self.c_conc.fig.add_subplot(
            gs[:, dist_end + 1:map_slots], projection="polar")
        self._draw_conc_spiral(ax_sp, r, nb, nbcols, um_all, hit, med,
                               out_of_lib)
        self._exp_conc.append(("radial_readout", ax_sp, None))
        xs = np.arange(len(nb))
        ok = np.isfinite(med)
        rng = np.random.default_rng(260819)       # stable jitter across redraws
        # A light, genuinely jittered strip plot + black IQR/median marks.  The old
        # violin body made dense samples read as one fat solid object and hid the
        # individual pixels, which are the useful evidence here.
        # 라이브러리 밖이면 µM 축의 점을 아예 그리지 않는다 — "무응답"이라면서
        # 외삽 구름을 계속 보여주는 모순을 없앤다. 공간 지도는 상대 패턴이라 유지.
        for i, vv in enumerate(distributions if not out_of_lib else []):
            if vv.size:
                # Wider horizontal jitter, smaller translucent dots and no outlines:
                # repeated/quantised y values remain separable instead of fusing.
                jitter = rng.uniform(-0.34, 0.34, size=vv.size)
                axb.scatter(np.full(vv.size, i) + jitter, vv, s=6,
                            color=nbcols[i], alpha=0.30, edgecolors="none",
                            linewidths=0.0, zorder=2, rasterized=True)
                # black IQR and median marks are a compact summary on top of all dots
                axb.vlines(i, q1[i], q3[i], color=INK, lw=1.2, zorder=3)
                axb.hlines([q1[i], q3[i]], i - 0.08, i + 0.08,
                           color=INK, lw=1.0, zorder=3)
                axb.plot(i, med[i], marker="_", ms=14, mew=2.0,
                         color=INK, zorder=4)
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
                     zorder=5)
        # known-total reconstruction: Ci = pi × Ctotal from the SAME pooled
        # composition the pie reports. A CONSTRAINED number (the total is declared
        # sample-prep metadata) — held-out it lifts within-2× from 78% to 91% on
        # ≤100 µM maps, but the lift comes from the added information, not the model.
        kt = self._known_total_vec(r)
        total_uM = self._known_total_uM()
        if kt is not None:                                 # blue tick = known-total
            axb.plot(xs, kt, ls="none", marker="_", ms=16, mew=1.8, color=BLUE,
                     zorder=5)
        amed = med * afac if anchored else None
        if amed is not None:                               # orange tick = batch-anchored
            _ai = [i for i in range(len(nb))
                   if np.isfinite(amed[i]) and afac[i] != 1.0]
            if _ai:
                axb.plot(np.asarray(xs)[_ai], amed[_ai], ls="none", marker="_",
                         ms=16, mew=1.8, color="#e08214", zorder=5)
        # 축은 데이터가 사는 구간(99퍼센타일)에 맞춘다 — 소수 이상치가 축을 늘려
        # 대부분(1–100 µM)의 분포를 뭉개던 문제. 위쪽 1.45배는 라벨 밴드 여유.
        _allv = (np.concatenate([v for v in distributions if v.size])
                 if any(v.size for v in distributions) else np.array([]))
        _base = float(np.percentile(_allv, 99)) * 1.15 if _allv.size else 1.0
        for _extra in (kt, tv):
            if _extra is not None:
                _base = max(_base, float(np.nanmax(np.asarray(_extra, float))) * 1.15)
        _yfactor = (self.sl_dist_y.value() / 100.0
                    if hasattr(self, "sl_dist_y") else 1.0)
        axb.set_ylim(0, max(_base, 1e-6) * 1.45 * _yfactor)
        self._um_display_ymax = float(max(_base, 1e-6) * 1.45 * _yfactor)
        if out_of_lib and kt is None and tv is None:
            axb.set_ylim(0, 1); axb.set_yticks([])
            axb.set_ylabel("")
        for i in range(len(nb)):
            # One reported number per substance. With a declared total, the
            # composition-based reconstruction IS the report — that is the whole
            # point of the trained composition — and it never depends on whether
            # the signal-only inversion stayed in range. The signal reading is a
            # diagnostic line, and beyond the validated window (grid64, ≤24 µM —
            # competition breaks above ~50) it is never shown as a bare number.
            has_kt = kt is not None and np.isfinite(kt[i])
            if out_of_lib and not has_kt:
                continue          # 성분별 반복 대신 중앙의 빨간 무응답 메시지 하나로
            if px_used and not has_kt:
                # 픽셀 조회 모드 + KT 없음: 회색 중앙값 한 줄만 — 숫자가 아예
                # 없으면 무응답으로 오독된다 (2026-09-02).
                if ok[i]:
                    _iqr = (f" ({q1[i]:.0f}–{q3[i]:.0f})"
                            if np.isfinite(q1[i]) and np.isfinite(q3[i])
                            else "")
                    axb.annotate(f"pixel-library {med[i]:.1f}{_iqr} µM",
                                 (float(xs[i]), 0.995),
                                 xycoords=("data", "axes fraction"),
                                 ha="center", va="top", fontsize=9,
                                 color=MUTE)
                continue
            if not ok[i] and not has_kt:
                if bad_frac[i] > 0:
                    axb.annotate("no readable concentration —\n"
                                 + f"all {total_n[i]} px over the calibrated range",
                                 (xs[i], 0), xytext=(0, 8),
                                 textcoords="offset points",
                                 ha="center", fontsize=9, color=RED)
                continue
            n_above = total_n[i] - valid_n[i]
            over_major = bad_frac[i] > 0.5
            sat = (hi_um is not None and np.isfinite(hi_um[i])
                   and np.isfinite(med[i]) and med[i] >= 0.8 * hi_um[i])
            # The signal number is ALWAYS printed; range/batch problems are caveats
            # appended to it, never a reason to withhold the value. When the batch
            # detector fired, the signal line says so in as many words.
            # 최소 텍스트: 숫자 하나 + 상태 기호. ⚠ = 검증창/배치 밖 (하단 빨간
            # 줄과 제목이 뜻을 설명한다).
            mm = bool(getattr(r, "conc_batch_mismatch", False))
            _flag = " ⚠" if (mm or over_major or sat) else ""
            # 숫자 하나로 단정하지 않는다: 산점에 보이는 퍼짐(IQR)을 괄호로
            # 병기 — 넓으면 무른 판독이라는 게 리포트 줄에서 바로 보인다.
            _iqr = (f" ({q1[i]:.0f}–{q3[i]:.0f})"
                    if np.isfinite(q1[i]) and np.isfinite(q3[i]) else "")
            if raw_sig:
                raw_line = (("corrected" if route == "mlpsig" else "band")
                            + f" signal {med[i]:.0f}{_iqr} counts")
            elif out_of_lib:
                # 라이브러리에 닮은 맵이 없다 — 농도는 무응답이다. 숫자 없음.
                raw_line = "no answer — outside library"
            elif px_used:
                raw_line = f"pixel-library {med[i]:.1f}{_iqr} µM"
            elif knn_used:
                raw_line = f"library {knn['uM'][i]:.1f} µM"
            elif not ok[i]:
                raw_line = "no signal"
            elif hi_um is not None and np.isfinite(hi_um[i]) and med[i] > hi_um[i]:
                # 검증 천장 캡: 상한 위는 어떤 값도 주장하지 않는다 — 보고는
                # "최소 천장"까지다.
                raw_line = f"signal ≥{hi_um[i]:g} µM{_flag}"
            else:
                raw_line = f"signal {med[i]:.1f}{_iqr} µM{_flag}"
            sub_lines = []
            if has_kt:
                main = f"reported {kt[i]:.1f} µM"
                sub_lines.append(raw_line)
            else:
                main = raw_line
            if amed is not None and np.isfinite(amed[i]) \
                    and not (over_major or sat) and afac[i] != 1.0:
                sub_lines.append(f"anchored {amed[i]:.1f}")
            if vol > 0:                                    # µM × µL = pmol
                _v = kt[i] if has_kt else med[i]
                sub_lines.append(f"≈{_v * vol:.0f} pmol")
            # A reading outside the validated window / batch is a WARNING —
            # coloured so nobody quotes the number. The reported (declared-total)
            # line stays ink-black above it.
            _bad = out_of_lib or over_major or sat or mm
            _warn = "#b3421a" if _bad else MUTE
            _mcol = INK if has_kt else ("#b3421a" if _bad else INK)
            axb.annotate(main, (float(xs[i]), 0.995),
                         xycoords=("data", "axes fraction"),
                         ha="center", va="top", fontsize=9, color=_mcol)
            if sub_lines:
                axb.annotate("\n".join(sub_lines), (float(xs[i]), 0.995),
                             xycoords=("data", "axes fraction"),
                             xytext=(0, -11), textcoords="offset points",
                             ha="center", va="top", fontsize=8, color=_warn)
        axb.set_xticks(xs)
        xt = ([f"{nm}\ntruth {tv[i]:g}" for i, nm in enumerate(nb)]
              if tv is not None else nb)
        axb.set_xticklabels(xt, fontsize=9)
        if out_of_lib:
            _dtxt = (f"nearest training map at distance {knn['dmin']:.1f} "
                     f"(limit {self.KNN_MAX_DIST:g})" if knn is not None
                     else "intensity scale does not match the calibration batch")
            _empty = kt is None and tv is None            # 점도 눈금도 없는 상태
            axb.text(0.5, 0.5 if _empty else 0.015,
                     "⚠ concentration not answered\n"
                     f"outside the training library — {_dtxt}\n"
                     "use a batch anchor or a declared total",
                     transform=axb.transAxes, ha="center",
                     va="center" if _empty else "bottom",
                     fontsize=10 if _empty else 8.5, color=RED, zorder=6)
        # 두 경로의 차이는 총량 스칼라 하나다 — 그걸 제목이 직접 보여준다.
        _tparts = []
        _dtot = self._known_total_uM()
        if _dtot:
            _tparts.append(f"declared total {_dtot:g} µM")
        _rt = ({"knn": "library k-NN", "pxknn": "pixel k-NN",
                "raw": "raw VIP band signal",
                "mlpsig": "MLP-corrected signal"}.get(route, "model head"))
        if route_sel == "auto":
            _rt = "auto → " + _rt
        _rt += f" · nearest d={knn['dmin']:.1f}" if knn is not None else ""
        axb.set_title((" · ".join(_tparts) + "  —  " if _tparts else "")
                      + f"readout: {_rt} · " + ("per-pixel counts" if raw_sig else "per-pixel µM") + " · black – median"
                      + (" · orange – anchored" if anchored else "")
                      + (" · red – truth" if tv is not None else "")
                      + (" · blue – declared-total" if kt is not None else ""),
                      fontsize=9)
        axb.set_ylabel("band signal (counts) per pixel" if raw_sig else "µM per pixel", fontsize=9)
        axb.tick_params(labelsize=9)
        self._exp_conc.append(("pixel_distribution", axb, None))
        self.c_conc.draw_idle()

    # Final pie-map style (settled with the 260812 trio map): pure black ground,
    # the full measurement grid in white so the map reads as MAP DATA, cell-filling
    # pies, and a heavier white outline tracing the hit region. Low-R² ✕ marks are
    # gone for good — on a badly-fit map they covered every pixel and said nothing;
    # the R² still reaches the user via the pixel click-out and the CSV export.
    PIE_BG = "#000000"
    PIE_GRID = "#ffffff"

    def _plot_pies(self, r):
        """Draw NNLS and final per-pixel composition at a stable, fixed layout."""
        from matplotlib.collections import LineCollection
        old = getattr(self.c_pie, "_rowgs_cid", None)
        if old is not None:
            self.c_pie.mpl_disconnect(old)
            self.c_pie._rowgs_cid = None
        self.c_pie.fig.clear()
        # 4패널: NNLS 조성 | MLP 조성 | MLP 농도(size ~ µM) | Δ  (2026-09-04)
        pgs = self.c_pie.fig.add_gridspec(
            1, 4, wspace=0.16, left=0.02, right=0.98, bottom=0.08, top=0.88)
        cols = self._nb_colors(r)
        x, y = r.coords[:, 0], r.coords[:, 1]
        ux, uy = np.unique(x), np.unique(y)
        sx = float(np.median(np.diff(ux))) if len(ux) > 1 else 1.0
        sy = float(np.median(np.diff(uy))) if len(uy) > 1 else 1.0
        rad = min(sx, sy) * 0.5
        hit = self._hit(r)
        before = self._spectral_ratio_nb(r)
        after = self._ratio_nb(r)
        # 파이 크기 = 픽셀 µM 총량 (면적 비례 → 반지름 ∝ sqrt). 기준은 hit 픽셀
        # 총량의 p95(그 이상은 셀을 꽉 채움), 하한 0.22·rad로 존재만 표시.
        # 판독 경로가 무응답이면 크기 정보 없음 → 기존처럼 균일.
        um_d = getattr(self, "_um_display", None)
        prad = np.full(r.n_pixels, rad)
        size_by_um = False
        if um_d is not None and len(um_d) == r.n_pixels:
            tot = np.where(np.isfinite(um_d), np.clip(um_d, 0, None), 0.0).sum(axis=1)
            ref = float(np.quantile(tot[hit], 0.95)) if hit.any() and (tot[hit] > 0).any() else 0.0
            if ref > 0:
                prad = rad * np.clip(np.sqrt(tot / ref), 0.22, 1.0)
                size_by_um = True
        after_name = "MLP" if r.method == "dlpx" else f"After · {r.method.upper()}"
        # (조성행렬, 제목, µM 크기 여부): 조성 파이 둘은 균일 크기, 셋째가 농도 파이
        panels = [(before, "NNLS (raw spectral)", False),
                  (after, after_name, False),
                  (after, after_name + " · size ~ µM" if size_by_um
                   else after_name + " · (no µM)", True)]
        axes = []
        for panel, (ratios, title, sized) in enumerate(panels, 1):
            ax = self.c_pie.style(self.c_pie.fig.add_subplot(pgs[0, panel - 1]))
            axes.append(ax); self._click_axes.append(ax)
            ax.set_facecolor(self.PIE_BG)
            gsegs = [[(ux[0] - sx/2 + j*sx, uy[0] - sy/2),
                      (ux[0] - sx/2 + j*sx, uy[-1] + sy/2)]
                     for j in range(len(ux) + 1)]
            gsegs += [[(ux[0] - sx/2, uy[0] - sy/2 + i*sy),
                       (ux[-1] + sx/2, uy[0] - sy/2 + i*sy)]
                      for i in range(len(uy) + 1)]
            ax.add_collection(LineCollection(gsegs, colors=self.PIE_GRID,
                                             linewidths=0.35, zorder=1))
            wedges, wcols = [], []
            for i in np.where(hit)[0]:
                a0 = 90.0
                for k, frac in enumerate(ratios[i]):
                    if frac <= 0.002:
                        continue
                    a1 = a0 - frac * 360.0
                    wedges.append(Wedge((x[i], y[i]), prad[i] if sized else rad,
                                        a1, a0))
                    wcols.append(cols[k]); a0 = a1
            if wedges:
                ax.add_collection(PatchCollection(wedges, facecolors=wcols,
                                                  edgecolors="none"))
            ax.set_xlim(x.min()-sx, x.max()+sx)
            ax.set_ylim(*((y.max()+sy, y.min()-sy) if self._flip()
                          else (y.min()-sy, y.max()+sy)))
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9, fontweight="bold", pad=2)
        # A direct difference panel removes the need to mentally subtract thousands
        # of tiny pies. Value = mean absolute component change in percentage points.
        axd = self.c_pie.style(self.c_pie.fig.add_subplot(pgs[0, 3]))
        axes.append(axd); self._click_axes.append(axd)
        delta = np.mean(np.abs(after - before), axis=1) * 100.0
        dgrid = np.full((len(uy), len(ux)), np.nan)
        xi = {v: i for i, v in enumerate(ux)}; yi = {v: i for i, v in enumerate(uy)}
        for i in np.where(hit)[0]:
            dgrid[yi[y[i]], xi[x[i]]] = delta[i]
        vmax = float(np.nanquantile(delta[hit], 0.99)) if hit.any() else 1.0
        vmax = max(vmax, 1.0)
        delta_manual = self._parse_scale("delta")
        dmin = delta_manual[0] if delta_manual is not None else 0.0
        if delta_manual is not None:
            vmax = delta_manual[1]
        dim = axd.imshow(dgrid, extent=[ux.min()-sx/2, ux.max()+sx/2,
                                      uy.min()-sy/2, uy.max()+sy/2],
                         origin="lower", aspect="equal", interpolation="nearest",
                         cmap="magma", vmin=dmin, vmax=vmax)
        if self._flip():
            axd.invert_yaxis()
        axd.set_xticks([]); axd.set_yticks([])
        axd.set_title("Mean |MLP − NNLS| (%p)", fontsize=9,
                      fontweight="bold", pad=2)
        # Use the same dedicated bottom row as the two blank peers above. This keeps
        # all three map axes exactly aligned instead of shrinking only the delta map.
        cb = self._side_colorbar(self.c_pie.fig, axd, dim,
                                 ticks=[dmin, vmax],
                                 labels=[f"{dmin:.0f}", f"{vmax:.0f} %p"])
        self._sel_arts = []
        self._pie_ax = axes[1]
        self._exp_pie = [("composition_before", axes[0], None),
                         ("composition_after", axes[1], None),
                         ("concentration_pies", axes[2], None),
                         ("composition_difference", axes[3], cb.ax)]
        # The persistent colour chips above the dashboard already identify each
        # substance; repeating them here only covers the bottom of both maps.
        self.c_pie.draw_idle()
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
        self.c_comp.fig.clear()
        self.c_comp.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.17,
                                        top=0.80, wspace=0.18)
        cols = self._nb_colors(r); nb = [r.comps[i] for i in r.nonbg]
        hit = self._hit(r)
        weights = np.clip(np.asarray(r.spectra, float), 0.0, None).sum(axis=1)

        def aggregate(ratios):
            use = hit if hit.any() else np.ones(len(ratios), bool)
            w = weights[use]
            return ((ratios[use] * w[:, None]).sum(axis=0) / w.sum()
                    if w.sum() > 0 else ratios[use].mean(axis=0))

        pairs = ((aggregate(self._spectral_ratio_nb(r)), "Before\nspectral/NNLS"),
                  (aggregate(self._ratio_nb(r)),
                   "After\nMLP" if r.method == "dlpx" else f"After\n{r.method.upper()}"))
        changes = (pairs[1][0] - pairs[0][0]) * 100.0
        change_text = "   ".join(f"{nm} {d:+.0f}%p" for nm, d in zip(nb, changes))
        self.c_comp.fig.suptitle("Change:  " + change_text, fontsize=9,
                                 fontweight="bold", y=0.98)
        for panel, (mr, title) in enumerate(pairs, 1):
            ax = self.c_comp.style(self.c_comp.fig.add_subplot(1, 2, panel))
            keep = [i for i in range(len(nb)) if mr[i] >= 0.01] or [int(mr.argmax())]
            ax.pie([mr[i] for i in keep], labels=None,
                   colors=[cols[i] for i in keep], autopct="%1.0f%%",
                   pctdistance=0.62,
                   textprops={"fontsize": 11, "fontweight": "bold", "color": INK},
                   radius=0.9)          # 1.05는 이웃 파이와 겹쳤다 (2026-09-04)
            ax.set_title(title, fontsize=9, fontweight="bold", pad=1)
            ax.set_aspect("equal")
        handles = [Patch(facecolor=cols[i], label=nm) for i, nm in enumerate(nb)]
        self.c_comp.fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                               fontsize=8, frameon=False, handlelength=1.0,
                               columnspacing=1.1, bbox_to_anchor=(0.5, 0.01))
        self.c_comp.draw_idle()
    def _plot_spec(self, r, i):
        ax = self.c_spec.new_ax()
        axis = r.wn if r.wn is not None else np.arange(r.spectra.shape[1])
        meas = np.asarray(r.spectra[i], float)
        mm = meas.max() or 1.0
        ax.plot(axis, meas / mm, lw=1.3, color=INK, label="measured")
        ratio_nb = self._ratio_nb(r)                      # corrected when toggle on
        evidence = np.asarray(getattr(r, "A_evidence", r.A), float)
        ev_nb = evidence[:, r.nonbg]
        ev_sum = ev_nb.sum(axis=1, keepdims=True)
        ev_ratio = np.divide(ev_nb, ev_sum, out=np.zeros_like(ev_nb), where=ev_sum > 0)

        if r.templates is not None:
            # ONE curve per substance — A_k x template_k in the substance's colour —
            # instead of a single summed "reference mix" that matched nothing visibly
            # ("green이 뭘 의미하는지 모르겠음"). Which peak belongs to whom, and how
            # much, is what this panel is for. The thin grey dashed curve is their sum:
            # the actual fit under NNLS/MCR; under the composition model A holds
            # probabilities, so it only shows the references the model leaned on,
            # not a goodness-of-fit.
            recon = evidence[i] @ r.templates
            rmax = float(recon.max()) or 1.0
            cols = self._nb_colors(r)
            for k, j in enumerate(r.nonbg):
                contrib = evidence[i, j] * r.templates[j]
                if float(contrib.max()) / rmax < 0.02:
                    continue                       # absent substance — no clutter
                ax.plot(axis, contrib / rmax, lw=1.0, color=cols[k], alpha=0.9,
                        label=(f"{r.comps[j]} model {ratio_nb[i, k] * 100:.0f}% · "
                               f"spectral {ev_ratio[i, k] * 100:.0f}%"))
            lab = "full-spectrum spectral fit (sum)"
            ax.plot(axis, recon / rmax, lw=0.9, color=FAINT, ls="--", label=lab)
        xp, yp = r.coords[i]
        rat = "  ·  ".join(f"{r.comps[j]} {ratio_nb[i, k] * 100:.0f}%"
                           for k, j in enumerate(r.nonbg) if ratio_nb[i, k] > 0.02)
        tag = rat if r.hit[i] else "background"
        weak = [r.comps[j] for k, j in enumerate(r.nonbg)
                if ratio_nb[i, k] > 0.02
                and ev_ratio[i, k] < 0.5 * ratio_nb[i, k]]
        if r.hit[i] and weak:
            tag += "  |  weak full-spectrum support: " + ", ".join(weak)
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
        readout = f"({xp:g}, {yp:g})  ·  {tag}"
        title = "\n".join(textwrap.wrap(
            readout, width=72, break_long_words=False, break_on_hyphens=False))
        n_title_lines = max(1, title.count("\n") + 1)
        ax.set_title(title, fontsize=8, pad=3, loc="left")
        ax.set_xlabel("Raman shift (cm$^{-1}$)", labelpad=1); ax.set_yticks([])
        # legend inside the axes — a short card cannot spare a strip under the plot
        ax.legend(fontsize=7, framealpha=0.0, labelcolor="black",
                  loc="upper right", ncol=2, frameon=False,
                  handlelength=1.4, columnspacing=0.9)
        self.c_spec.fig.subplots_adjust(left=0.055, right=0.99,
                                        top=max(0.58, 0.88 - 0.07 * n_title_lines), bottom=0.20)
        self.c_spec.draw_idle()

    # ---- interaction ----
    def _on_click(self, event):
        r = self._res
        if (r is None or event.xdata is None or event.inaxes not in self._click_axes
                or event.button != 1):
            return
        self._roi_drag = (float(event.xdata), float(event.ydata), event.inaxes)

    def _pitch(self, r):
        _ri, _ci, _ny, _nx, ux, uy = self._grid_rc(r)
        px = float(np.min(np.diff(ux))) if len(ux) > 1 else 1.0
        py = float(np.min(np.diff(uy))) if len(uy) > 1 else 1.0
        return px, py

    def _on_drag(self, event):
        st = self._roi_drag
        if st is None or event.xdata is None or event.inaxes is not st[2]:
            return
        from matplotlib.patches import Rectangle
        x0, y0, ax = st
        if self._roi_rubber is None:
            self._roi_rubber = Rectangle((x0, y0), 0, 0, fill=True, alpha=0.18,
                                         facecolor="white", edgecolor="white",
                                         linestyle="--", linewidth=1.0, zorder=8)
            ax.add_patch(self._roi_rubber)
        self._roi_rubber.set_bounds(min(x0, event.xdata), min(y0, event.ydata),
                                    abs(event.xdata - x0), abs(event.ydata - y0))
        ax.figure.canvas.draw_idle()

    def _on_release(self, event):
        r = self._res; st = self._roi_drag; self._roi_drag = None
        if self._roi_rubber is not None:
            try:
                self._roi_rubber.remove()
            except Exception:
                pass
            self._roi_rubber = None
        if r is None or st is None:
            return
        x0, y0, ax = st
        x1 = float(event.xdata) if event.xdata is not None else x0
        y1 = float(event.ydata) if event.ydata is not None else y0
        px, py = self._pitch(r)
        if abs(x1 - x0) >= px and abs(y1 - y0) >= py:        # a real drag → ROI
            hx, hy = px / 2, py / 2
            self._roi = (min(x0, x1) - hx, max(x0, x1) + hx,
                         min(y0, y1) - hy, max(y0, y1) + hy)
            if not self._roi_mask(r).any():
                self._roi = None
                return
            self.btn_roi.setEnabled(True)
            self._apply(r)
            return
        d = ((r.coords[:, 0] - x0) ** 2 + (r.coords[:, 1] - y0) ** 2)   # a click
        self._sel = int(d.argmin())
        self._plot_spec(r, self._sel)
        self._update_sel_rings(r)                # rings on every map, never a rebuild

    # ---- export ----
    def _save_concentration_group(self, folder):
        """Export only the three real µM maps on a fresh, aligned canvas."""
        data = getattr(self, "_conc_export", None)
        if not data:
            return False
        from matplotlib.figure import Figure
        from matplotlib.colors import LinearSegmentedColormap
        fig = Figure(figsize=(12.0, 4.3))
        gs = fig.add_gridspec(2, len(data["nb"]), height_ratios=[1.0, 0.065],
                              hspace=0.08, wspace=0.06,
                              left=0.015, right=0.985, bottom=0.08, top=0.91)
        for i, (nm, color) in enumerate(zip(data["nb"], data["colors"])):
            ax = fig.add_subplot(gs[0, i]); cax = fig.add_subplot(gs[1, i])
            values = data["um"][:, i]
            values = np.where(data["hit"] & np.isfinite(values) & (values > 0),
                              values, np.nan)
            grid = np.full((data["ny"], data["nx"]), np.nan)
            grid[data["rows"], data["cols"]] = values
            cmap = LinearSegmentedColormap.from_list("m", ["#0b0d10", color])
            cmap.set_bad("#0b0d10")
            ax.set_facecolor("#0b0d10")
            im = ax.imshow(grid, extent=data["extent"], origin=data["origin"],
                           aspect="equal", interpolation="nearest", cmap=cmap,
                           vmin=0.0, vmax=data["vmax"][i])
            ax.set_title(nm, fontsize=11, fontweight="bold", pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_ticks([0.0, data["vmax"][i]])
            cb.set_ticklabels(["0", f'{data["vmax"][i]:.0f} µM'])
            cax.tick_params(labelsize=8, length=2, pad=1)
            cb.outline.set_linewidth(0.5)
        fig.savefig(os.path.join(folder, "real_concentration_maps.png"),
                    dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
        return True

    def _export(self):
        if self._res is None:
            self.status.setText("run first, then export")
            self.status.setStyleSheet(f"color:{RED};"); return
        sel = self._pick_export_items()
        if sel is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        want = lambda k: sel.get(k, True)
        _written = []
        from io_utils import write_csv as _wc

        def write_csv(path, head, rows):          # shadows the module function here
            rel = os.path.relpath(path, d).replace("\\", "/")
            key = "matrix" if rel.startswith("matrix/") else rel
            if want(key):
                _wc(path, head, rows); _written.append(path)
        r = self._res; nb = [r.comps[i] for i in r.nonbg]
        # 내보내는 hit = 화면의 유효 hit (게이트 ∧ 신뢰도 ∧ 포화 ∧ signal floor ∧
        # 잎 마스크). 슬라이더로 걸러낸 픽셀은 파일에서도 hit=0 (사용자 2026-09-04).
        _eff = self._hit(r)
        evidence = np.asarray(getattr(r, "A_evidence", r.A), float)
        ev_nb = evidence[:, r.nonbg]
        ev_sum = ev_nb.sum(axis=1, keepdims=True)
        ev_ratio = np.divide(ev_nb, ev_sum, out=np.zeros_like(ev_nb), where=ev_sum > 0)
        _eh = self._hit(r)
        ev_mean = ev_ratio[_eh].mean(axis=0) if _eh.any() else ev_ratio.mean(axis=0)
        # both aggregates, so the pie can be redrawn either way: the plain average
        # and the signal-weighted one the app now displays (see _mean_ratio)
        mr_un = (r.ratio_nb[self._hit(r)].mean(axis=0) if self._hit(r).any()
                 else r.ratio_nb.mean(axis=0))
        mr_w = self._mean_ratio(r)
        write_csv(os.path.join(d, "composition.csv"),
                  ["substance", "mean_ratio_unweighted", "mean_ratio_signal_weighted",
                   "mean_spectral_evidence_share"],
                  [[nm, f"{mr_un[i]:.4f}", f"{mr_w[i]:.4f}", f"{ev_mean[i]:.4f}"]
                   for i, nm in enumerate(nb)])
        inten = r.spectra.sum(axis=1)                      # total baseline-removed signal
        cal = getattr(r, "conc", None) is not None
        has_bg = getattr(r, "bg_score", None) is not None
        has_sat = getattr(r, "sat_frac", None) is not None
        head = (["x", "y", "hit", "total_intensity"]
                + [f"ratio_{nm}" for nm in nb] + [f"A_{c}" for c in r.comps]
                + [f"spectral_A_{c}" for c in r.comps]
                + ([f"conc_uM_{nm}" for nm in nb] if cal else []) + ["reliability_r2"]
                + (["clipped_frac"] if has_sat else [])
                + (["bg_match"] if has_bg else []))
        rows = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}", int(_eff[i]),
                 f"{inten[i]:.4f}"]
                + [f"{r.ratio_nb[i, k]:.4f}" for k in range(len(nb))]
                + [f"{r.A[i, k]:.5f}" for k in range(len(r.comps))]
                + [f"{evidence[i, k]:.5f}" for k in range(len(r.comps))]
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
        # Standalone map tables make the exported figures reproducible without
        # opening the app: each row is one pixel and each column is one panel.
        _ah = ["x", "y"] + [f"abundance_{c}" for c in r.comps]
        _ar = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}"]
               + [f"{evidence[i, k]:.6g}" for k in range(len(r.comps))]
               for i in range(r.n_pixels)]
        write_csv(os.path.join(d, "abundance_maps.csv"), _ah, _ar)
        ncsv += 1
        if cal:
            _ch = ["x", "y"] + [f"conc_uM_{nm}" for nm in nb]
            _cr = [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}"]
                   + [f"{r.conc[i, k] * 1e6:.6g}" if np.isfinite(r.conc[i, k]) else ""
                      for k in range(len(nb))] for i in range(r.n_pixels)]
            write_csv(os.path.join(d, "concentration_maps.csv"), _ch, _cr)
            ncsv += 1
        # 맵별 매트릭스(ny×nx) — long-form과 별도로, Origin 등에서 채널 하나를
        # 그대로 히트맵으로 다시 그릴 수 있는 형태. 첫 행 = x 좌표, 첫 열 = y.
        mdir = os.path.join(d, "matrix")
        os.makedirs(mdir, exist_ok=True)
        _ri, _ci, _ny, _nx, _ux, _uy = self._grid_rc(r)

        def _mat(name, vec):
            g = np.full((_ny, _nx), np.nan)
            g[_ri, _ci] = np.asarray(vec, float)
            mh = ["y\\x"] + [f"{v:g}" for v in _ux]
            mb = [[f"{_uy[j]:g}"]
                  + ["" if not np.isfinite(g[j, k]) else f"{g[j, k]:.6g}"
                     for k in range(_nx)] for j in range(_ny)]
            write_csv(os.path.join(mdir, name), mh, mb)

        nmat = 0
        for nm, wl in _bands:
            _mat(f"band_{nm}_{wl:.0f}.csv", self._band_image(r, wl)); nmat += 1
        for wl in _extras:
            _mat(f"band_extra_{wl:.0f}.csv", self._band_image(r, wl)); nmat += 1
        for k, c in enumerate(r.comps):
            _mat(f"abundance_{c}.csv", evidence[:, k]); nmat += 1
        for k, nm in enumerate(nb):
            _mat(f"ratio_{nm}.csv", r.ratio_nb[:, k]); nmat += 1
        if cal:
            for k, nm in enumerate(nb):
                _mat(f"conc_uM_{nm}.csv", r.conc[:, k] * 1e6); nmat += 1
        _mat("hit.csv", np.asarray(_eff, float)); nmat += 1
        ncsv += nmat
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
            # known-total columns are CONSTRAINED numbers (declared total × pooled
            # composition; cap at the total) — flagged so no reader mistakes them
            # for the spectrum-only prediction.
            _kt = self._known_total_vec(r)
            _tot = self._known_total_uM()
            _sr = []
            for i, nm in enumerate(nb):
                v = _um[_hit, i] if _hit.any() else _um[:, i]
                v = v[np.isfinite(v) & (v > 0)]
                if not v.size:
                    _sr.append([nm, "0", "", "", "", "", "", "", "", "", ""])
                    continue
                med = float(np.median(v))
                _sr.append([nm, str(int(v.size)), f"{med:.4f}",
                            f"{float(np.quantile(v, 0.25)):.4f}",
                            f"{float(np.quantile(v, 0.75)):.4f}",
                            f"{_tv[i]:g}" if _tv else "",
                            f"{100 * med / _tv[i]:.1f}" if _tv else "",
                            f"{med * _vol:.2f}" if _vol > 0 else "",
                            f"{_kt[i]:.4f}" if _kt is not None else "",
                            f"{min(med, _tot):.4f}" if _tot is not None else "",
                            ("known-total capped" if _tot is not None and med > _tot
                             else "known-total" if _tot is not None else "")])
            write_csv(os.path.join(d, "um_summary.csv"),
                      ["substance", "n_hit_px", "median_uM", "q1_uM", "q3_uM",
                       "true_uM", "recovery_pct", "apparent_amount_pmol",
                       "known_total_uM", "median_capped_uM", "constraint_flag"], _sr)
            ncsv += 1
        # 화면의 픽셀 분포(활성 판독 경로 값 그대로) — Origin에서 다시 그리기용.
        um_d = getattr(self, "_um_display", None)
        if um_d is not None:
            _hd = self._hit(r)
            _units = getattr(self, "_um_display_units", "uM")
            _route = {"knn": "library k-NN", "pxknn": "pixel k-NN",
                      "raw": "raw VIP band signal", "mlpsig": "MLP-corrected signal"
                      }.get(getattr(self, "_um_display_route", ""), "model head")
            _idx = np.where(_hd)[0]
            _umz = np.where(_hd[:, None] & np.isfinite(um_d), um_d, 0.0)
            # strip-plot data: hit 픽셀만(0 없음). 전체 격자(비-hit = 0)는 matrix/display_*.
            write_csv(os.path.join(d, "pixel_distribution.csv"),
                      ["x", "y"] + [f"{nm}_{_units}" for nm in nb],
                      [[f"{r.coords[i, 0]:g}", f"{r.coords[i, 1]:g}"]
                       + [f"{um_d[i, k]:.6g}" for k in range(len(nb))] for i in _idx])
            # 같은 값을 ny×nx 매트릭스로도 (Origin 히트맵): 비-hit 픽셀 = 0
            for k, nm in enumerate(nb):
                _mat(f"display_{_units}_{nm}.csv", _umz[:, k])
            _srows = []
            _ymax = getattr(self, "_um_display_ymax", None)
            for k, nm in enumerate(nb):
                v = um_d[_idx, k]; v = v[np.isfinite(v)]
                if v.size:
                    _srows.append([nm, _route, _units, str(v.size),
                                   f"{np.median(v):.4f}", f"{np.quantile(v, .25):.4f}",
                                   f"{np.quantile(v, .75):.4f}",
                                   f"{np.percentile(v, 90):.4f}",
                                   f"{np.percentile(v, 99):.4f}", f"{v.max():.4f}",
                                   f"{_ymax:.4f}" if _ymax is not None else "",
                                   str(self.sl_floor.value()
                                       if getattr(self, "sl_floor", None) is not None else 0)])
            # app_y_max = 앱 strip plot의 y축 상한(99퍼센타일 기반 × 슬라이더) —
            # Origin에서 같은 축 범위를 쓰면 화면과 같은 그림이 된다.
            write_csv(os.path.join(d, "pixel_distribution_summary.csv"),
                      ["substance", "readout_route", "units", "n_hit_px",
                       "median", "q1", "q3", "p90", "p99", "max", "app_y_max",
                       "signal_floor_pct"], _srows)
        # figures export WITHOUT the selection ring — the clicked-pixel highlight
        # is a working aid, not figure content. Redraw clean, save, then restore.
        _sel = self._sel
        if _sel is not None:
            self._sel = None; self._plot_pies(r)
        figs = [("real_band_maps", self.c_maps),
                ("real_composition_maps", self.c_abund),
                ("real_composition_pies", self.c_pie),
                ("real_composition", self.c_comp),
                ("real_pixel_spectrum", self.c_spec)]
        # Concentration is intentionally omitted from the composite figure list.
        # Its maps, distribution and radial readout are exported separately below.
        figs = [f for f in figs if want("fig:" + f[0])]
        n = _save_figs(figs, d) if figs else 0
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
            for label, ax, cbax in entries:
                if not want("panel:" + label):
                    continue
                original_size = cv.fig.get_size_inches().copy()
                # 패널 파일은 "이미지만": 제목·축라벨·눈금글자·주석·범례를 잠시
                # 숨기고 크롭한다 (설명 글자는 슬라이드에서 따로 단다 — 2026-09-04).
                hidden = []
                for a_ in ([ax] + ([cbax] if cbax is not None else [])):
                    arts = ([a_.title, a_.xaxis.label, a_.yaxis.label]
                            + list(a_.texts) + a_.get_xticklabels()
                            + a_.get_yticklabels())
                    lg = a_.get_legend()
                    if lg is not None:
                        arts.append(lg)
                    for t_ in arts:
                        hidden.append((t_, t_.get_visible())); t_.set_visible(False)
                    a_.tick_params(length=0)
                try:
                    cv.draw()
                    ren = cv.get_renderer() if hasattr(cv, "get_renderer") else None
                    bb = ax.get_tightbbox(ren)
                    if cbax is not None:
                        bb = _mt.Bbox.union([bb, cbax.get_tightbbox(ren)])
                    bb = bb.transformed(cv.fig.dpi_scale_trans.inverted()).padded(0.05)
                    # A cropped on-screen axes can be physically tiny even at 300 DPI.
                    # Temporarily enlarge the complete figure so the crop itself has
                    # publication-size inches, then render at a genuine 300 DPI.
                    if label == "radial_readout":
                        min_w, min_h = 5.0, 5.0       # >= 1500 × 1500 px
                    elif label == "pixel_distribution":
                        min_w, min_h = 7.0, 4.0       # >= 2100 × 1200 px
                    elif label.startswith("map_"):
                        min_w, min_h = 7.0, 3.5       # >= 2100 × 1050 px
                    else:
                        min_w, min_h = 5.0, 3.5
                    scale = max(1.0, min_w / max(bb.width, 1e-6),
                                min_h / max(bb.height, 1e-6))
                    if scale > 1.0:
                        cv.fig.set_size_inches(original_size * scale, forward=False)
                        cv.draw()
                        ren = cv.get_renderer() if hasattr(cv, "get_renderer") else None
                        bb = ax.get_tightbbox(ren)
                        if cbax is not None:
                            bb = _mt.Bbox.union([bb, cbax.get_tightbbox(ren)])
                        bb = bb.transformed(
                            cv.fig.dpi_scale_trans.inverted()).padded(0.05)
                    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_"
                                   for ch in label)
                    if cv is self.c_conc:
                        out_path = os.path.join(d, f"real_concentration_{safe}.png")
                    else:
                        out_path = os.path.join(pdir, f"{safe}.png")
                    cv.fig.savefig(out_path, dpi=300, bbox_inches=bb,
                                   transparent=True)
                    np_ += 1
                except Exception:
                    pass                                 # one bad crop must not kill export
                finally:
                    for t_, vis in hidden:
                        t_.set_visible(vis)
                    for a_ in ([ax] + ([cbax] if cbax is not None else [])):
                        a_.tick_params(length=2)
                    cv.fig.set_size_inches(original_size, forward=False)
                    cv.draw_idle()
        if (getattr(r, "calibrated", False) and r.conc is not None
                and want("fig:real_concentration_maps")):
            if self._save_concentration_group(d):
                n += 1
        if _sel is not None:
            self._sel = _sel; self._plot_pies(r)         # put the highlight back
        if want("README"):
            self._export_readme(d, r, nb, [f[0] for f in figs])
        ncsv = len(_written)
        self.status.setText(f"exported {ncsv} CSV + {n} PNG + {np_} panel PNG "
                            f"→ {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _pick_export_items(self):
        """Export 선택 대화상자. None = 취소. 항목 키: CSV 파일명 / "matrix" /
        "README" / "fig:<name>" / "panel:<label>"."""
        r = self._res
        cal = getattr(r, "conc", None) is not None
        tables = [("composition.csv", "composition.csv — mean ratios"),
                  ("per_pixel.csv", "per_pixel.csv — every pixel, all channels"),
                  ("band_maps.csv", "band_maps.csv — displayed band values"),
                  ("abundance_maps.csv", "abundance_maps.csv — NNLS spectral evidence (not composition)")]
        if cal:
            tables += [("concentration_maps.csv", "concentration_maps.csv — model-head µM per pixel"),
                       ("um_summary.csv", "um_summary.csv — µM medians / recovery")]
        if getattr(self, "_um_display", None) is not None:
            tables += [("pixel_distribution.csv",
                        "pixel_distribution.csv — the strip plot's per-pixel values (Origin)"),
                       ("pixel_distribution_summary.csv",
                        "pixel_distribution_summary.csv — median / q1 / q3 per substance")]
        tables += [("matrix", "matrix/ — one ny×nx table per map (Origin heatmap)"),
                   ("README", "README.txt")]
        figs = [("fig:real_band_maps", "raw band maps"),
                ("fig:real_composition_maps", "MLP reconstruction (composition)"),
                ("fig:real_composition_pies", "primary comparison"),
                ("fig:real_composition", "overall fractions"),
                ("fig:real_pixel_spectrum", "selected pixel spectrum")]
        if cal:
            figs.append(("fig:real_concentration_maps", "apparent concentration maps"))
        panels = []
        for title, attr in (("Raw band maps", "_exp_maps"),
                            ("MLP reconstruction", "_exp_abund"),
                            ("Primary comparison", "_exp_pie"),
                            ("Apparent concentration", "_exp_conc")):
            ents = [("panel:" + lab, lab) for lab, _a, _c in getattr(self, attr, [])]
            if ents:
                panels.append((f"Panels (image only) — {title}", ents))
        groups = [("Tables (CSV)", tables), ("Figures (composite PNG)", figs)] + panels
        dlg = _ExportPicker(self, groups, getattr(self, "_export_sel", {}))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        self._export_sel = dlg.selection()
        return self._export_sel

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
            "real_composition_maps": ("per-pixel composition (share of each analyte, "
                                      "composition model) on gated pixels, shaded by the "
                                      "pixel's relative VIP-band signal; grey underlay = "
                                      "total signal for context; plus the background gate."),
            "real_composition_pies": ("per-pixel composition map from the loaded "
                                      "composition model."),
            "real_composition": ("signal-weighted mean composition (pie) over the "
                                 "hit pixels."),
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
                f"- Calibration: {os.path.basename(self.calib_path) if self.calib_path else 'none (ratio only)'}",
                "- Pixel filters as displayed (the 'hit' column in per_pixel.csv and "
                "matrix/hit.csv, and every hit-only table, use pixels AFTER these): "
                f"signal floor {getattr(self, 'sl_floor', None).value() if getattr(self, 'sl_floor', None) is not None else 0} % of hit p99"
                f"{' (' + self.lbl_floor.text() + ')' if getattr(self, 'lbl_floor', None) is not None and self.lbl_floor.text() != 'off' else ''}; "
                f"low-R² drop {'on' if self.chk_rel.isChecked() else 'off'}; "
                f"saturation quarantine {'on' if self.chk_sat.isChecked() else 'off'}; "
                f"ink-area-only {'on' if getattr(self, 'chk_leaf', None) is not None and self.chk_leaf.isChecked() else 'off'}"
                + (f"; ROI x {self._roi[0]:.0f}–{self._roi[1]:.0f}, y {self._roi[2]:.0f}–{self._roi[3]:.0f} "
                   f"({int(self._hit(r).sum())} px)" if getattr(self, '_roi', None) is not None else "")],
            "Results": [
                f"- Dominant substance: {r.dominant}",
                f"- Substance pixels (hit fraction): {r.hit_frac:.0%} by the gate; "
                f"{int(self._hit(r).sum())} px ({self._hit(r).mean():.0%}) after the filters above",
                f"- Mean composition over hit pixels: {ratio_str}",
                f"- Mean reconstruction R²: {r.mean_r2:.2f}",
                f"- Median concentration across hit pixels: {conc_str}"],
        }
        figures = [(fn, fig_docs[fn]) for fn in fig_names if fn in fig_docs]
        write_readme(d, "UNMIXR — Real-data export", sections, figures)
