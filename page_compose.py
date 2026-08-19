"""page_compose.py — the composition-model half of the Model tab.

Pick a composition method (MLP / PLS / RF / 1D-CNN — the best one is data-dependent),
train ONCE on the known-ratio mixtures you prepared in Samples (Step 1), and save a
portable .dlm that Recovery and Real-data just APPLY (no retraining downstream).
The mixture list is shared from Samples via MIXTURE_BUS — not re-entered here.
"""
from __future__ import annotations

import os
import traceback

import matplotlib
import numpy as np
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QProgressBar,
)

from ui_common import *
from real_data import PEST_DEFAULT
from dataset import load_mixture_list, load_mixture_roles
from validate import simplify_ratio


def _train_subprocess(params, q):
    """Run the heavy training in a SEPARATE process (own GIL) so the GUI never freezes.
    Progress + result travel back over a multiprocessing Queue. Module-level so it is
    importable by the spawned interpreter on Windows."""
    try:
        import sys as _sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in _sys.path:                        # spawn may not inherit sys.path
            _sys.path.insert(0, _here)
        params = dict(params); items = params.pop("items")

        def _accepted(fn, kw):
            """Keep only the keywords ``fn`` actually declares. One options dict feeds
            train_model, benchmark_loo and kfold_stability, and their signatures differ —
            benchmark_loo takes `methods`, not `method`, and knows nothing about
            `include_blank`. Passing the dict wholesale raised TypeError inside the
            worker process, which surfaced as the run dying rather than as a message."""
            import inspect
            sig = inspect.signature(fn).parameters
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values()):
                return dict(kw)                       # **kw absorbs anything
            return {k: v for k, v in kw.items() if k in sig}

        if params.pop("_kfold", False):
            from dl_model import kfold_stability
            folds = params.pop("folds", 5); params.pop("loo", None); params.pop("test_items", None)
            q.put(("kfold", kfold_stability(items=items, folds=folds,
                                            progress=lambda s: q.put(("progress", s)),
                                            **_accepted(kfold_stability, params))))
            return
        if params.pop("_benchmark", False):
            from dl_model import benchmark_loo
            params.pop("loo", None)
            q.put(("bench", benchmark_loo(items=items,
                                          progress=lambda s: q.put(("progress", s)),
                                          **_accepted(benchmark_loo, params))))
            return
        from dl_model import train_model
        model = train_model(items=items, progress=lambda s: q.put(("progress", s)),
                            **_accepted(train_model, params))
        q.put(("progress", "packaging results — this can take a minute on low RAM"))
        q.put(("done", model))
    except Exception:
        import traceback
        q.put(("error", traceback.format_exc()))


class TrainComposeWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            proc = ctx.Process(target=_train_subprocess, args=(self.params, q), daemon=True)
            self.proc = proc                              # so Cancel can terminate it
            proc.start()
            import queue as _queue
            model = None
            while True:                                   # q.get releases the GIL → GUI free
                try:
                    tag, payload = q.get(timeout=0.5)
                except _queue.Empty:
                    if not proc.is_alive():               # killed (Cancel) or crashed child
                        # Drain first: a child that finished can exit while its result is
                        # still in flight, and reporting failure then would throw away a
                        # completed run. Only after the queue is dry is the run really dead.
                        drained = []
                        while True:
                            try:
                                drained.append(q.get_nowait())
                            except Exception:
                                break
                        for tg, pl in drained:
                            if tg in ("bench", "kfold"):
                                self.done.emit((tg, pl)); return
                            if tg == "error":
                                self.fail.emit(pl); return
                            if tg == "done":
                                model = pl; break
                        if model is None:
                            # No message at all: the process was killed rather than raising,
                            # so there is no traceback to show. The exit code is the only
                            # evidence, and the usual cause is the OS reclaiming memory.
                            code = proc.exitcode
                            self.fail.emit(
                                f"the worker process was killed (exit code {code}) without "
                                "reporting an error.\nThis is almost always the operating "
                                "system reclaiming memory, or Cancel.\nIf you did not press "
                                "Cancel: close other applications and re-run, and check the "
                                "status line says 'one mean spectrum per map' — if it does "
                                "not, the app is still running the older code and needs a "
                                "restart.")
                            return
                        break
                    continue
                if tag == "progress":
                    self.progress.emit(payload)
                elif tag == "error":
                    proc.join(timeout=2); self.fail.emit(payload); return
                elif tag == "kfold":                      # repeated held-out check
                    proc.join(timeout=5); self.done.emit(("kfold", payload)); return
                elif tag == "bench":                      # LOO method comparison
                    proc.join(timeout=5); self.done.emit(("bench", payload)); return
                else:                                     # "done"
                    model = payload; break
            proc.join(timeout=5)
            # train-set recovery is computed IN-MEMORY inside train_model (no map reload,
            # which was holding the GIL and freezing the GUI). Reorder to SUBSTANCES.
            from dataset import is_blank
            all_subs = list(model["subs"])
            subs = [s for s in all_subs if not is_blank(s)]
            sidx = {s: j for j, s in enumerate(all_subs)}
            # prefer the leave-one-out predictions when they were computed — the honest number
            te = (model.get("loo_eval") or model.get("test_eval")
                  or model.get("train_eval", {}))
            tvs = te.get("true", []); pvs = te.get("pred", [])
            errs = []; rows = []
            for i in range(len(tvs)):
                tv = [float(tvs[i][sidx[s]]) for s in subs]
                pv = [float(pvs[i][sidx[s]]) for s in subs]
                errs.append(0.5 * sum(abs(pv[j] - tv[j]) for j in range(len(subs))))
                rows.append((f"mix {i + 1}", tv, pv))
            self.done.emit((model, float(np.mean(errs)) if errs else float("nan"), rows))
        except Exception:
            self.fail.emit(traceback.format_exc())


class PixelMapWorker(QObject):
    """Re-unmix every scored map with the trained model so the per-pixel composition
    maps can be tiled. Runs off the GUI thread — 65 conditions is minutes, not seconds."""
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from unmix import unmix_map
            p = self.params
            out = []
            paths = list(dict.fromkeys(p["paths"]))       # repeats of one map draw once
            for i, path in enumerate(paths):
                self.progress.emit(f"per-pixel map {i + 1}/{len(paths)} — "
                                   f"{os.path.basename(path)}")
                try:
                    r = unmix_map(p["data_dir"], path, method="dlpx",
                                  baseline=p["baseline"], trim=p["trim"],
                                  dl_model=p["model"])
                except Exception:
                    continue                              # one bad map must not kill the set
                out.append((os.path.basename(path).replace("_corrected.csv", ""),
                            np.asarray(r.coords, float), np.asarray(r.ratio_nb, float),
                            np.asarray(r.hit, bool)))
            self.done.emit(out)
        except Exception:
            self.fail.emit(traceback.format_exc())


class ComposePanel(QWidget):
    METHODS = [("MLP", "mlp"), ("PLS", "pls"),
               ("Random Forest", "rf"), ("1D-CNN", "cnn")]

    def __init__(self):
        super().__init__()
        self.data_dir = PEST_DEFAULT
        self.calib_path = None
        self._calib_manual = False     # a CSV browsed here beats the Quantify bus
        self._items_cache = []
        self._test_items = []
        self._model = None
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(12)

        sub = QLabel("Pick a method (the best one is data-dependent) and train once on the "
                     "mixtures prepared in Samples. Save the model so Recovery and Real-data "
                     "just apply it — no retraining.")
        sub.setObjectName("sub"); sub.setWordWrap(True); root.addWidget(sub)

        # ---- files: pure refs + (mixtures come from Samples) ----
        frow = QHBoxLayout(); frow.setSpacing(8)
        pure_b = QPushButton("Pure refs…"); pure_b.setObjectName("ghost")
        pure_b.setToolTip("folder of pure reference maps (one class per substance)")
        pure_b.clicked.connect(self._browse_pure)
        self.ref_lbl = QLabel(self._short(self.data_dir)); self.ref_lbl.setObjectName("field")
        self.mix_lbl = QLabel("mixtures: from Samples"); self.mix_lbl.setObjectName("field")
        reload_b = QPushButton("Reload"); reload_b.setObjectName("ghost")
        reload_b.setToolTip("re-read the known-ratio mixtures prepared in the Samples tab")
        reload_b.clicked.connect(self._load_from_samples)
        calib_b = QPushButton("Calibration…"); calib_b.setObjectName("ghost")
        calib_b.setToolTip("dilution-series CSV. Without one the physics pre-training cannot "
                           "run — train_model needs it to build the simulated mixtures.")
        calib_b.clicked.connect(self._browse_calib)
        self.calib_lbl = QLabel("no calibration"); self.calib_lbl.setObjectName("field")
        frow.addWidget(pure_b); frow.addWidget(self.ref_lbl, 1)
        frow.addWidget(calib_b); frow.addWidget(self.calib_lbl)
        frow.addWidget(self.mix_lbl); frow.addWidget(reload_b)
        root.addLayout(frow)

        # ---- step 1: which method? (leave-one-out comparison, trains nothing) ----
        brow = QHBoxLayout(); brow.setSpacing(8)
        blbl = QLabel("1 · compare methods:"); blbl.setObjectName("field")
        self.bench_b = QPushButton("Benchmark (LOO)"); self.bench_b.setObjectName("ghost")
        self.bench_b.setToolTip("score VIP band / NNLS / PLS / RF / 1D-CNN / MLP on the "
                                "same mixtures with leave-one-out — mean deviation (%p), "
                                "recovery %±SE per substance, detection ROC, and accuracy "
                                "at the cut declared in advanced (if any). VIP band = NNLS "
                                "restricted to each compound's marker windows, the simplest "
                                "rung of the ladder. Tells you which method to train; "
                                "saves no model.")
        self.bench_b.clicked.connect(self._benchmark)
        self.kfold_b = QPushButton("5-fold check"); self.kfold_b.setObjectName("ghost")
        self.kfold_b.setToolTip("repeat the held-out test over all 5 one-in-five splits for "
                                "the method chosen below — reports mean ± spread, so a single "
                                "lucky split cannot set the headline number")
        self.kfold_b.clicked.connect(self._kfold)
        self.bench_lbl = QLabel("run this first to see which method fits your data")
        self.bench_lbl.setObjectName("field")
        brow.addWidget(blbl); brow.addWidget(self.bench_b); brow.addWidget(self.kfold_b)
        brow.addWidget(self.bench_lbl, 1)
        root.addLayout(brow)

        # ---- step 2: train the model that gets deployed ----
        mrow = QHBoxLayout(); mrow.setSpacing(8)
        mlbl = QLabel("2 · train:"); mlbl.setObjectName("field")
        self.cmb = QComboBox(); self.cmb.setObjectName("field")
        for text, data in self.METHODS:
            self.cmb.addItem(text, data)
        self.cmb.currentIndexChanged.connect(self._update_params)
        self.sp_ep = QSpinBox(); self.sp_ep.setRange(20, 3000); self.sp_ep.setSingleStep(50)
        self.sp_ep.setValue(350); self.sp_ep.setPrefix("epochs "); self.sp_ep.setObjectName("field")
        self.sp_ep.setToolTip("training iterations (MLP / CNN). Higher = better but slower; "
                              "the GUI is busy while it trains — raise for a final model.")
        self.sp_seed = QSpinBox(); self.sp_seed.setRange(0, 999); self.sp_seed.setPrefix("seed ")
        self.sp_seed.setObjectName("field")
        self.sp_nc = QSpinBox(); self.sp_nc.setRange(1, 20); self.sp_nc.setValue(8)
        self.sp_nc.setPrefix("components "); self.sp_nc.setObjectName("field")
        self.chk_blank = QCheckBox("learn background (BLK)"); self.chk_blank.setObjectName("field")
        # ON by default (2026-08-15): the blank references are already sitting in
        # Samples, and a model shipped without the class reads ink/paper as its
        # most-similar analyte — the lettering film put 63% into DQ that way.
        # Untick only to reproduce an old no-blank run.
        self.chk_blank.setChecked(True)
        self.chk_blank.setToolTip("add pixels from the blank reference map as a 4th class, so "
                                  "the model can answer 'nothing here' instead of spreading "
                                  "every spectrum across the substances — OFF reproduces "
                                  "old no-blank runs only")
        self.sp_px = QSpinBox(); self.sp_px.setRange(0, 400); self.sp_px.setSingleStep(20)
        self.sp_px.setValue(400); self.sp_px.setPrefix("pixels/map "); self.sp_px.setObjectName("field")
        self.sp_px.setToolTip("individual hit-pixel training rows per map: 0 = one mean spectrum per map "
                              "(one row per map), higher = use that many individual "
                              "individual pixels, all labelled with the map's ratio. More rows "
                              "fight overfitting; splits stay grouped by map either way. "
                              "Affects TRAINING only — Benchmark always scores one mean "
                              "spectrum per map, since refitting six methods per condition "
                              "on 400 px/map does not fit in memory.")
        # OFF by default (2026-08-19): defaulting ON cost two accidental all-day
        # runs — LOO refits once per condition (~100×, hours). Tick it exactly once,
        # for the final model whose held-out numbers go in the paper.
        self.chk_loo = QCheckBox("held-out scoring (LOO)")
        self.chk_loo.setChecked(False); self.chk_loo.setObjectName("field")
        self.chk_loo.setToolTip(
            "score every condition held-out (leave-one-condition-out) after training. "
            "HONEST but SLOW — it refits the model once per condition (~100×, hours). "
            "Tick for the FINAL model only; everyday retrains leave it off "
            "(the saved .dlm then carries train-set numbers only).")
        self.chk_screen = QCheckBox("NNLS-screen ink first")
        self.chk_screen.setChecked(True); self.chk_screen.setObjectName("field")
        self.chk_screen.setToolTip("Use the same NNLS hit/background gate as Real data, then train "
                                   "the model only on accepted SERS-ink pixels. The model estimates "
                                   "composition/concentration but cannot change the spatial hit mask.")
        self.sp_hit = QDoubleSpinBox(); self.sp_hit.setRange(0.0, 1.0)
        self.sp_hit.setDecimals(3); self.sp_hit.setSingleStep(0.025); self.sp_hit.setValue(0.15)
        self.sp_hit.setPrefix("hit fraction "); self.sp_hit.setObjectName("field")
        self.sp_hit.setToolTip("Minimum NNLS substance fraction for the hit/background gate.")
        # OFF by default (2026-08-19): the current label set (260814_mixture_final)
        # carries FINAL µM in the filenames. Ticked, every µM target is divided by
        # the number of components (÷2 binary, ÷3 ternary) — the 260815 model was
        # trained that way by accident and its apparent µM came out low and
        # order-dependent. Tick ONLY for labels that really are pre-mix stocks.
        self.chk_equal_volume = QCheckBox("µM labels are pre-mix stocks (÷N)")
        self.chk_equal_volume.setChecked(False); self.chk_equal_volume.setObjectName("field")
        self.chk_equal_volume.setToolTip(
            "tick only when Samples µM values are SOURCE solutions mixed at equal "
            "volumes — targets are then divided by the number of components. "
            "260814_mixture_final filenames are FINAL concentrations: leave this off.")
        self.sp_nt = QSpinBox(); self.sp_nt.setRange(20, 1000); self.sp_nt.setSingleStep(20)
        self.sp_nt.setValue(300); self.sp_nt.setPrefix("trees "); self.sp_nt.setObjectName("field")

        # ---- two knobs that only matter for Benchmark, where every method is refit once
        # per condition. Left at the library defaults a five-method run over 68 conditions
        # takes most of a day: RandomForestRegressor reads all ~2000 spectral channels at
        # every split, and the 1-D CNN convolves the full spectrum full-batch.
        self.cmb_rfmf = QComboBox(); self.cmb_rfmf.setObjectName("field")
        self.cmb_rfmf.addItem("RF features: all", None)
        self.cmb_rfmf.addItem("RF features: sqrt", "sqrt")
        self.cmb_rfmf.addItem("RF features: log2", "log2")
        self.cmb_rfmf.setToolTip("how many spectral channels each forest split may consider. "
                                 "sklearn's regressor default is ALL of them, which on ~2000 "
                                 "channels costs minutes per fit; 'sqrt' is the usual forest "
                                 "recipe and runs in seconds. Affects Benchmark only.")
        self.sp_cnnep = QSpinBox(); self.sp_cnnep.setRange(0, 3000)
        self.sp_cnnep.setSingleStep(20); self.sp_cnnep.setValue(0)
        self.sp_cnnep.setPrefix("CNN epochs "); self.sp_cnnep.setSpecialValueText("CNN epochs: same")
        self.sp_cnnep.setObjectName("field")
        self.sp_cnnep.setToolTip("separate epoch budget for the 1-D CNN in Benchmark. "
                                 "'same' gives it the epochs above, like every other method — "
                                 "the honest comparison. Lower it only to get a run finished, "
                                 "and say so wherever the numbers are used: the CNN is then "
                                 "scored undertrained against the rest.")
        # the accuracy cut is DECLARED here, before the run — picking it after seeing the
        # results is cherry-picking (HANDOFF 2026-08-19 §3). Mean deviation (%p) and
        # per-substance recovery are always reported regardless; the cut only adds an
        # accuracy column. 15 %p ≈ reproducibility floor (11.6) + 3; set 0 for no cut.
        self.sp_cut = QDoubleSpinBox(); self.sp_cut.setObjectName("field")
        self.sp_cut.setRange(0.0, 50.0); self.sp_cut.setDecimals(1)
        self.sp_cut.setSingleStep(0.5); self.sp_cut.setValue(15.0)
        self.sp_cut.setPrefix("declared cut "); self.sp_cut.setSuffix(" %p")
        self.sp_cut.setSpecialValueText("no accuracy cut")
        self.sp_cut.setToolTip("Benchmark only: a condition counts as 'correct' when its "
                               "composition deviation is within this many percentage "
                               "points. DECLARE it before running — choosing a cut after "
                               "seeing the results is cherry-picking. 15 %p ≈ the measured "
                               "repeat-preparation floor (11.6 %p) + 3; set to 0 to report "
                               "mean deviation and recovery only, with no cut at all.")

        # ---- physics pre-training. These were hardcoded off (use_pretrain=False, and
        # calib_path never set), so a model trained here could not be the one the
        # 64-condition work adopted no matter what else you ticked. The knobs exist in
        # dl_model.train_model; they just had no way in.
        self.chk_pretrain = QCheckBox("physics pre-training")
        self.chk_pretrain.setObjectName("field")
        self.chk_pretrain.setToolTip("warm the composition head up on mixtures simulated from "
                                     "the calibration's response coefficients before fitting the "
                                     "real maps. Needs a calibration CSV — without one it is "
                                     "silently skipped inside train_model.")
        self.chk_pretrain.toggled.connect(self._update_params)
        self.cmb_iso = QComboBox(); self.cmb_iso.setObjectName("field")
        self.cmb_iso.addItem("isotherm: Langmuir", None)
        self.cmb_iso.addItem("isotherm: Sips (DQ)", "sips_dq")
        self.cmb_iso.setToolTip("how the simulated mixtures respond to concentration. "
                                "'Sips (DQ)' gives DQ its measured exponent (m 0.66) instead of "
                                "an ideal Langmuir; TBZ and THI are left at m≈1 either way.")
        self.chk_nuisance = QCheckBox("simulate ink + substrate")
        self.chk_nuisance.setObjectName("field")
        self.chk_nuisance.setToolTip("inject the ink and blank references into the simulated "
                                     "spectra. Tried on the 64-condition grid and it pushed TBZ "
                                     "and THI the wrong way — off is the adopted setting.")

        mrow.addWidget(mlbl); mrow.addWidget(self.cmb)
        # every knob lives in a FOLDED advanced box — the defaults are the adopted
        # settings, so the normal run is: calibration arrives from Quantify, Train.
        self.adv_tgl = QPushButton("▸ advanced"); self.adv_tgl.setObjectName("ghost")
        self.adv_tgl.setCheckable(True)
        self.adv_tgl.setToolTip("epochs/seed, NNLS screen, blank class, physics "
                                "pre-training, benchmark knobs — defaults are the "
                                "adopted settings; open only to deviate")
        mrow.addWidget(self.adv_tgl)
        mrow.addStretch(1)
        self.advw = QWidget(); adv = QVBoxLayout(self.advw)
        adv.setContentsMargins(0, 0, 0, 0); adv.setSpacing(6)
        _adv1 = QHBoxLayout(); _adv1.setSpacing(8)
        for w in (self.sp_ep, self.sp_seed, self.sp_nc, self.sp_nt, self.sp_px,
                  self.chk_screen, self.sp_hit, self.chk_loo):
            _adv1.addWidget(w)
        _adv1.addStretch(1)
        _adv2 = QHBoxLayout(); _adv2.setSpacing(8)
        for w in (self.chk_equal_volume, self.chk_blank,
                  self.chk_pretrain, self.cmb_iso, self.chk_nuisance,
                  self.cmb_rfmf, self.sp_cnnep, self.sp_cut):
            _adv2.addWidget(w)
        _adv2.addStretch(1)
        adv.addLayout(_adv1); adv.addLayout(_adv2)
        self.advw.setVisible(False)

        def _adv_tgl(on):
            self.advw.setVisible(on)
            self.adv_tgl.setText(("▾ " if on else "▸ ") + "advanced")

        self.adv_tgl.toggled.connect(_adv_tgl)
        self.train_b = QPushButton("Train"); self.train_b.setObjectName("primary")
        self.train_b.clicked.connect(self._train)
        self.cancel_b = QPushButton("Cancel"); self.cancel_b.setObjectName("ghost")
        self.cancel_b.setVisible(False); self.cancel_b.clicked.connect(self._cancel)
        self.load_b = QPushButton("Load model…"); self.load_b.setObjectName("ghost")
        self.load_b.setToolTip("re-open a .dlm trained earlier and publish it to Recovery "
                               "and Real-data — train once, keep using it across restarts")
        self.load_b.clicked.connect(self._load)
        self.save_b = QPushButton("Save model…"); self.save_b.setObjectName("ghost")
        self.save_b.setEnabled(False); self.save_b.clicked.connect(self._save)
        self.export_b = QPushButton("Export…"); self.export_b.setObjectName("ghost")
        self.export_b.setEnabled(False); self.export_b.clicked.connect(self._export)
        self.export_b.setToolTip("write the plots (PNG), the per-mixture predictions and the "
                                 "per-substance metrics (CSV) + a README to a folder. "
                                 "Includes the whole-experiment figures: a pie per "
                                 "condition, the design grid true vs predicted, and the "
                                 "grid as an error heat-map.")
        self.pix_b = QPushButton("Pixel maps…"); self.pix_b.setObjectName("ghost")
        self.pix_b.setEnabled(False); self.pix_b.clicked.connect(self._export_pixel_maps)
        self.pix_b.setToolTip("after an Export, re-unmix every scored map and tile the "
                              "per-pixel composition maps into one figure. Slow — it runs "
                              "the Real-data path once per map.")
        mrow.addWidget(self.load_b); mrow.addWidget(self.export_b)
        mrow.addWidget(self.pix_b); mrow.addWidget(self.save_b)
        mrow.addWidget(self.cancel_b); mrow.addWidget(self.train_b)
        root.addLayout(mrow)
        root.addWidget(self.advw)                          # folded advanced knobs
        self._update_params()

        self.pbar = QProgressBar(); self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(6); self.pbar.setVisible(False); root.addWidget(self.pbar)

        # read-only view of the shared mixtures (managed in Samples) — collapsible
        self.mix_tgl = QPushButton(); self.mix_tgl.setObjectName("ghost")
        self.mix_tgl.setCheckable(True); self.mix_tgl.setChecked(True)
        self.mix_tgl.setStyleSheet("text-align:left; padding:4px 8px;")
        self.mix_tgl.toggled.connect(self._toggle_table)
        root.addWidget(self.mix_tgl)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["mixture file (from Samples)", "true ratio"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(140)
        root.addWidget(self.table)
        self._toggle_table(True)

        # the Benchmark table — one row per method: mean deviation (%p), accuracy at the
        # declared cut (only when one was declared), recovery %±SE per substance,
        # detection AUC. Hidden until a benchmark has run. Recovery is direction-aware
        # but over/under CANCEL in its mean, so it is shown next to the deviation, never
        # alone (HANDOFF 2026-08-19 §3).
        self.bench_lbl2 = QLabel(
            "Benchmark table — leave-one-out, same folds for every method. "
            "Mean dev and RMSE are percentage points of composition; log-ratio distance "
            "(Aitchison) compares the ratios themselves, so a 2-fold miss costs the same "
            "on a minor component as on the major one (~0.57 per 2-fold on a ternary). "
            "Recovery = mean(pred/true)·100 where the substance is present (true=0 excluded).")
        self.bench_lbl2.setObjectName("sub"); self.bench_lbl2.setWordWrap(True)
        self.bench_lbl2.setVisible(False)
        root.addWidget(self.bench_lbl2)
        self.bench_table = QTableWidget(0, 0)
        self.bench_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.bench_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.bench_table.setMaximumHeight(220)
        self.bench_table.setVisible(False)
        root.addWidget(self.bench_table)

        plots = QHBoxLayout(); plots.setSpacing(12)
        lcard, llay = _card("Learning curve — training loss vs epoch (MLP / CNN)")
        self.c_loss = Canvas(); self.c_loss.setMinimumHeight(300)
        self.c_loss.placeholder("Train an MLP or CNN to see the loss curve")
        llay.addWidget(self.c_loss); plots.addWidget(lcard, 1)
        tcard, tlay = _card("Held-out recovery (leave-one-out) — true (○) vs predicted (●, colour = accuracy)")
        self.c_tri = Canvas(); self.c_tri.setMinimumHeight(300)
        self.c_tri.placeholder("Train to see held-out composition recovery")
        tlay.addWidget(self.c_tri); plots.addWidget(tcard, 1)
        self._tri_title = tlay.itemAt(0).widget()
        root.addLayout(plots, 1)

        plots2 = QHBoxLayout(); plots2.setSpacing(12)
        pcard, play = _card("Parity per substance — predicted vs true fraction "
                            "(on the line = exact)")
        self.c_parity = Canvas(); self.c_parity.setMinimumHeight(300)
        self.c_parity.placeholder("Train to see the per-substance parity")
        play.addWidget(self.c_parity); plots2.addWidget(pcard, 1)
        self._parity_title = play.itemAt(0).widget()
        ecard, elay = _card("Per-substance error — mean |predicted − true| fraction "
                            "(lower = better)")
        self.c_err = Canvas(); self.c_err.setMinimumHeight(300)
        self.c_err.placeholder("Train to see the per-substance error")
        elay.addWidget(self.c_err); plots2.addWidget(ecard, 1)
        self._err_title = elay.itemAt(0).widget()
        rcard, rlay = _card("Detection ROC — is each substance present? "
                            "(threshold the predicted fraction)")
        self.c_roc = Canvas(); self.c_roc.setMinimumHeight(300)
        self.c_roc.placeholder("Train to see the detection ROC / AUC")
        rlay.addWidget(self.c_roc); plots2.addWidget(rcard, 1)
        self._roc_title = rlay.itemAt(0).widget()
        root.addLayout(plots2, 1)

        self.status = QLabel(""); self.status.setObjectName("sub"); root.addWidget(self.status)

        MIXTURE_BUS.changed.connect(self._load_from_samples)   # Samples edits → refresh here
        self._load_from_samples()
        CALIB_BUS.changed.connect(self._adopt_calib_bus)       # Quantify → calibration
        self._adopt_calib_bus()                # Quantify may have published already

    # ---- data dir shared with the Model tab / Samples ----
    def set_data_dir(self, path):
        self.data_dir = path; self.ref_lbl.setText(self._short(path))
        self._load_from_samples()

    def _short(self, p):
        tail = "…" + p[-38:] if len(p) > 38 else p
        return f"pure refs: {tail}"

    def _browse_pure(self):
        d = QFileDialog.getExistingDirectory(self, "Pure references folder", self.data_dir)
        if d:
            self.set_data_dir(d)

    def _browse_calib(self):
        p, _ = QFileDialog.getOpenFileName(self, "Calibration (dilution series) CSV",
                                           self.data_dir, "CSV (*.csv);;all files (*)")
        if not p:
            return
        self._calib_manual = True
        self.calib_path = p
        self.calib_lbl.setText("calib: " + os.path.basename(p))
        # a calibration is here to be used — physics pre-training defaults ON
        self.chk_pretrain.setChecked(True)
        self._update_params()

    def _adopt_calib_bus(self):
        """Quantify published its dilution series: adopt it unless one was browsed
        here by hand. Physics pre-training follows — that is the point of having it."""
        if getattr(self, "_calib_manual", False) or not CALIB_BUS.path:
            return
        self.calib_path = CALIB_BUS.path
        self.calib_lbl.setText("calib: from Quantify"
                               + (f" ({CALIB_BUS.origin})" if CALIB_BUS.origin else ""))
        self.chk_pretrain.setChecked(True)
        self._update_params()

    def _load_from_samples(self):
        """Pull the known-ratio mixtures prepared in Samples (Step 1)."""
        roles = load_mixture_roles(self.data_dir)
        allm = load_mixture_list(self.data_dir)
        self._items_cache = [it for it in allm if roles.get(it[0], "train") != "test"]
        self._test_items = [it for it in allm if roles.get(it[0], "train") == "test"]
        self.table.setRowCount(0)
        for it in self._items_cache + self._test_items:
            row = self.table.rowCount(); self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(os.path.basename(it[0])))
            self.table.setItem(row, 1, QTableWidgetItem(
                ", ".join(f"{k}:{v:.3g}" for k, v in simplify_ratio(it[1]).items())))
        n = len(self._items_cache); nt = len(self._test_items)
        has_uM = any(len(it) > 2 and it[2] for it in self._items_cache)
        self.mix_lbl.setText(f"mixtures: {n} train" + (f" · {nt} test" if nt else "")
                             + " (Samples)" + (" · µM" if has_uM else ""))
        if hasattr(self, "mix_tgl"):
            self._toggle_table(self.mix_tgl.isChecked())
        if not self.status.text().startswith("●"):
            self.status.setText(f"{n} mixtures from Samples" if n
                                else "no mixtures yet — add them in the Samples tab")
            self.status.setStyleSheet(f"color:{MUTE};")

    def _toggle_table(self, on):
        self.table.setVisible(on)
        n = self.table.rowCount()
        self.mix_tgl.setText(("▾  " if on else "▸  ") + f"Mixtures from Samples ({n})")

    def _update_params(self):
        m = self.cmb.currentData()
        self.sp_ep.setVisible(m in ("mlp", "cnn"))
        self.sp_seed.setVisible(m in ("mlp", "cnn", "rf"))
        self.sp_nc.setVisible(m == "pls")
        self.sp_nt.setVisible(m == "rf")
        # 벤치마크 전용 노브 — 학습(Train)에는 안 쓰이지만 항상 보여야 누를 수 있다
        self.cmb_rfmf.setVisible(True); self.sp_cnnep.setVisible(True)
        self.sp_px.setVisible(True)          # applies to every method
        self.chk_blank.setVisible(True)
        self.chk_screen.setVisible(True)
        self.sp_hit.setVisible(self.chk_screen.isChecked())
        # pre-training only feeds the torch heads, and only means anything with a calibration
        self.chk_pretrain.setVisible(m in ("mlp", "cnn"))
        self.chk_pretrain.setEnabled(self.calib_path is not None)
        if self.calib_path is None:
            self.chk_pretrain.setChecked(False)
        on = self.chk_pretrain.isVisible() and self.chk_pretrain.isChecked()
        self.cmb_iso.setVisible(on); self.chk_nuisance.setVisible(on)

    def _opts(self):
        from dataset import load_preprocess
        cfg = load_preprocess(self.data_dir)
        return dict(data_dir=self.data_dir, calib_path=self.calib_path,
                    baseline=cfg["baseline"], trim=cfg["trim"],
                    method=self.cmb.currentData(), epochs=self.sp_ep.value(),
                    seed=self.sp_seed.value(),
                    use_pretrain=bool(self.chk_pretrain.isChecked() and self.calib_path),
                    sim_iso=self.cmb_iso.currentData(),
                    sim_nuisance=bool(self.chk_nuisance.isChecked()),
                    n_components=self.sp_nc.value(), n_trees=self.sp_nt.value(),
                    rf_max_features=self.cmb_rfmf.currentData(),
                    cnn_epochs=(self.sp_cnnep.value() or None),
                    px_per_map=self.sp_px.value(),
                    include_blank=self.chk_blank.isChecked(),
                    nnls_screen=self.chk_screen.isChecked(),
                    screen_min_frac=self.sp_hit.value(),
                    equal_volume_mix=self.chk_equal_volume.isChecked())

    def _train(self):
        items = self._items_cache
        if len(items) < 3:
            self.status.setText("prepare ≥3 known-ratio mixtures in the Samples tab first")
            self.status.setStyleSheet(f"color:{RED};"); return
        params = self._opts(); params["items"] = items
        if self._test_items:
            params["test_items"] = self._test_items
        params["loo"] = self.chk_loo.isChecked()   # the honest-but-slow switch, in the open
        self.train_b.setEnabled(False); self.train_b.setText("Training…")
        self.save_b.setEnabled(False); self._cancelled = False; self.cancel_b.setVisible(True)
        self.pbar.setRange(0, 0); self.pbar.setVisible(True)
        self.status.setText("● training…"); self.status.setStyleSheet(f"color:{MUTE};")
        start_worker(self, TrainComposeWorker(params), done=self._done,
                     fail=self._fail,
                     progress=lambda m: self.status.setText("● " + m))

    def _kfold(self):
        """Repeat the held-out check over every 1-in-5 split for the chosen method."""
        items = self._items_cache + getattr(self, "_test_items", [])
        if len(items) < 5:
            self.status.setText("need ≥5 mixtures for a 5-fold check")
            self.status.setStyleSheet(f"color:{RED};"); return
        params = self._opts(); params["items"] = items
        params["_kfold"] = True; params["folds"] = 5
        self.train_b.setEnabled(False); self.bench_b.setEnabled(False)
        self.kfold_b.setEnabled(False); self.kfold_b.setText("Checking…")
        self._cancelled = False; self.cancel_b.setVisible(True)
        self.pbar.setRange(0, 0); self.pbar.setVisible(True)
        self.status.setText("● 5-fold held-out check…"); self.status.setStyleSheet(f"color:{MUTE};")
        start_worker(self, TrainComposeWorker(params), done=self._done,
                     fail=self._fail,
                     progress=lambda m: self.status.setText("● " + m))

    def _done_kfold(self, res):
        import numpy as _np
        errs = res.get("errors", [])
        self._err_title.setText("5-fold held-out check — composition error per split")
        ax = self.c_err.new_ax()
        if errs:
            x = _np.arange(len(errs))
            ax.bar(x, [e * 100 for e in errs], color=TEAL, edgecolor="white", alpha=0.9)
            ax.axhline(res["mean"] * 100, color=MUTE, ls="--", lw=1.2)
            ax.set_xticks(x); ax.set_xticklabels([f"fold {i + 1}" for i in x], fontsize=8.5)
            ax.set_ylabel("composition error (%)  — held out")
        self.c_err.fig.tight_layout(); self.c_err.draw_idle()
        self._kfold_res = res; self.export_b.setEnabled(True)
        meth = self.cmb.currentData().upper()
        self.status.setText(f"5-fold ({meth}) — " + " · ".join(f"{e:.0%}" for e in errs)
                            + f"   → mean {res['mean']:.1%} ± {res['sd']:.1%}")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _benchmark(self):
        """Leave-one-out comparison of VIP band / NNLS / PLS / RF / CNN / MLP on the same
        mixtures."""
        items = self._items_cache
        if len(items) < 3:
            self.status.setText("prepare ≥3 known-ratio mixtures in the Samples tab first")
            self.status.setStyleSheet(f"color:{RED};"); return
        params = self._opts(); params.pop("method", None); params.pop("loo", None)
        params["items"] = items; params["_benchmark"] = True
        # ONE mean spectrum per map, whatever "pixels/map" says. That spinbox sizes the
        # TRAINING set, where 400 px/map is affordable because the model is fit once; the
        # benchmark refits six methods per condition, so the same setting means a
        # 102x400 = 40,800-row matrix in every one of ~100 folds. Measured on this code:
        # the 1-D CNN runs full-batch, so its activations alone come to ~18 GB (the
        # machine has 15.7) and one method would need ~830 h. At map level the whole
        # six-method run is an overnight job. Said out loud in the status line below,
        # because it makes the benchmark's MLP column a map-level number that will not
        # equal the deployed model's pixel-level one.
        params["px_per_map"] = 0
        # the accuracy cut is frozen NOW, before any result exists — changing the spinbox
        # afterwards must not re-score a finished run
        v = float(self.sp_cut.value())
        self._bench_cut_pending = v if v > 0 else None
        self.train_b.setEnabled(False); self.bench_b.setEnabled(False)
        self.bench_b.setText("Benchmarking…")
        self._cancelled = False; self.cancel_b.setVisible(True)
        self.pbar.setRange(0, 0); self.pbar.setVisible(True)
        # maps, not folds: repeats of one ratio are held out together (that is the point
        # of grouping), so the fold count is the number of distinct ratios and is only
        # known once the maps are loaded — the progress line reports it.
        self.status.setText(f"● leave-one-out benchmark — {len(items)} maps × 6 methods, "
                            "one mean spectrum per map (pixel-level would need ~18 GB per "
                            "CNN fold)")
        self.status.setStyleSheet(f"color:{MUTE};")
        start_worker(self, TrainComposeWorker(params), done=self._done,
                     fail=self._fail,
                     progress=lambda m: self.status.setText("● " + m))

    def _reset_buttons(self):
        self.train_b.setEnabled(True); self.train_b.setText("Train")
        self.bench_b.setEnabled(True); self.bench_b.setText("Benchmark (LOO)")
        self.kfold_b.setEnabled(True); self.kfold_b.setText("5-fold check")
        self.pbar.setVisible(False); self.cancel_b.setVisible(False)

    def _done_bench(self, bench):
        """Plot the LOO comparison: composition error per method + overlaid detection ROC,
        and fill the benchmark table (mean deviation %p · accuracy at the declared cut ·
        recovery %±SE per substance · detection AUC)."""
        import numpy as _np
        subs = list(bench.get("subs", []))
        methods = [m for m in ("band", "nnls", "pls", "rf", "cnn", "mlp") if m in bench]
        label = {"band": "VIP band", "nnls": "NNLS", "pls": "PLS", "rf": "RF",
                 "cnn": "1D-CNN", "mlp": "MLP"}
        self._bench = {}
        self._bench_raw = bench
        self._bench_subs = subs
        self._canvas_shows = "bench"        # c_err / c_roc now hold the method comparison
        self._err_title.setText("Method comparison — composition error, leave-one-out "
                                "(lower = better)")
        self._roc_title.setText("Method comparison — detection ROC, leave-one-out")
        errs, aucs = [], {}
        for m in methods:
            T = _np.asarray(bench[m]["true"], float); P = _np.asarray(bench[m]["pred"], float)
            errs.append(float((0.5 * _np.abs(P - T).sum(1)).mean()))
            rows = [(f"mix {i+1}",
                     [float(v) for v in T[i]],
                     [float(v) for v in P[i]])
                    for i in range(len(T))]
            self._bench[m] = rows
            aucs[m] = self._roc(rows)

        # summary FIRST — the error chart and the table below both read from it
        from dl_model import benchmark_summary
        cut = getattr(self, "_bench_cut_pending", None)
        self._bench_cut = cut
        summ = benchmark_summary(bench, cut)
        self._bench_summary = summ

        # the benchmark figure: how well each method recovers the mixture composition —
        # ONE bar per method, mean deviation over every condition with its standard
        # error. (The by-component-count split still rides along in the CSV for anyone
        # who wants it, but the headline is the single recovery number.)
        ax = self.c_err.new_ax()
        vals = [summ.get(m, {}).get("dev_by_k", {}).get(
                    "mean", (float("nan"), float("nan"), 0)) for m in methods]
        x = _np.arange(len(methods))
        ax.bar(x, [v[0] for v in vals], 0.62,
               yerr=[v[1] if v[1] == v[1] else 0.0 for v in vals], capsize=4,
               color=[TEAL if m == "mlp" else MUTE for m in methods],
               edgecolor="white", alpha=0.9, error_kw=dict(lw=0.9, ecolor=INK))
        ax.set_xticks(x); ax.set_xticklabels([label[m] for m in methods], fontsize=8.5)
        ax.set_ylabel("composition error (%p)  — leave-one-out")
        self.c_err.fig.tight_layout(); self.c_err.draw_idle()

        # The parity and ternary canvases belong to a TRAINED model and sit idle during a
        # benchmark, so the two comparisons the table leads with get a figure each rather
        # than living only in the CSV.
        self._parity_title.setText(
            f"Accuracy — conditions recovered within the declared cut (≤{cut:g} %p)"
            if cut is not None else "Accuracy — no cut was declared for this run")
        if cut is None:
            self.c_parity.placeholder("No accuracy cut declared — set one in advanced "
                                      "BEFORE running, then re-run the benchmark")
        else:
            axa = self.c_parity.new_ax()
            acc = [100 * summ.get(m, {}).get("acc_at_cut", float("nan")) for m in methods]
            axa.bar(x, acc, 0.62, color=[TEAL if m == "mlp" else MUTE for m in methods],
                    edgecolor="white", alpha=0.9)
            for xi, a in zip(x, acc):
                if a == a:
                    axa.text(xi, a + 1.5, f"{a:.0f}%", ha="center", fontsize=8, color=INK)
            axa.set_xticks(x); axa.set_xticklabels([label[m] for m in methods], fontsize=8.5)
            axa.set_ylim(0, 105); axa.set_ylabel(f"conditions within {cut:g} %p (%)")
            self.c_parity.fig.tight_layout(); self.c_parity.draw_idle()

        # recovery: mean(pred/true)·100 per substance, ±SE, with 100% marked. Read it
        # WITH the deviation chart — over- and under-recovery cancel in this mean.
        self._tri_title.setText("Recovery per substance — mean(predicted / prepared) × 100 "
                                "± SE (100% = exact; read next to the error chart)")
        axv = self.c_tri.new_ax()
        bw = 0.8 / max(len(subs), 1)
        for si, s in enumerate(subs):
            vals = [summ.get(m, {}).get("recovery", {}).get(
                        s, (float("nan"), float("nan"), 0)) for m in methods]
            axv.bar(x + (si - (len(subs) - 1) / 2) * bw, [v[0] for v in vals], bw * 0.9,
                    yerr=[v[1] if v[1] == v[1] else 0.0 for v in vals], capsize=3,
                    color=substance_color(s, si), edgecolor="white", alpha=0.9,
                    label=s, error_kw=dict(lw=0.9, ecolor=INK))
        axv.axhline(100, color=INK, ls="--", lw=1.0)
        axv.set_xticks(x); axv.set_xticklabels([label[m] for m in methods], fontsize=8.5)
        axv.set_ylabel("recovery (%)")
        axv.legend(fontsize=8, framealpha=0, ncol=len(subs))
        self.c_tri.fig.tight_layout(); self.c_tri.draw_idle()

        axr = self.c_roc.new_ax()                           # ROC overlay
        axr.plot([0, 1], [0, 1], ls="--", color=MUTE, lw=1.0)
        for j, m in enumerate(methods):
            fpr, tpr, a = aucs[m]
            if fpr is None:
                continue
            axr.plot(fpr, tpr, lw=2.4 if m == "mlp" else 1.4,
                     color=TEAL if m == "mlp" else SERIES[j % len(SERIES)],
                     label=f"{label[m]} (AUC {a:.3f}; n={len(rows)})")
        axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate")
        axr.set_xlim(-0.02, 1.02); axr.set_ylim(-0.02, 1.02); axr.set_aspect("equal")
        axr.legend(fontsize=8, framealpha=0, loc="lower right")
        self.c_roc.fig.tight_layout(); self.c_roc.draw_idle()
        self._bench_err = dict(zip(methods, errs))
        self._bench_auc = {m: aucs[m][2] for m in methods}

        # the benchmark table — same benchmark_summary the chart above and the CSV
        # export read, so screen and file cannot disagree
        cols = (["mean dev (%p)", "RMSE (%p)", "log-ratio dist", "vs NNLS (Δ%p, p)"]
                + ([f"accuracy ≤{cut:g} %p"] if cut is not None else [])
                + [f"{s} recovery %±SE" for s in subs] + ["detection AUC"])
        self.bench_table.setColumnCount(len(cols))
        self.bench_table.setHorizontalHeaderLabels(cols)
        self.bench_table.setRowCount(len(methods))
        for i, m in enumerate(methods):
            e = summ.get(m, {})
            self.bench_table.setVerticalHeaderItem(i, QTableWidgetItem(label[m]))
            def _ms(key, dec=1):
                mu, se, _n = e.get(key, (float("nan"), float("nan"), 0))
                return f"{mu:.{dec}f} ± {se:.{dec}f}" if se == se else f"{mu:.{dec}f}"

            vr = e.get("vs_ref")
            vals = [f"{e.get('mean_dev_pp', float('nan')):.1f}", _ms("rmse_pp"),
                    _ms("logratio", 2),
                    "reference" if m == "nnls" else
                    ("—" if not vr else f"{vr[0]:+.1f}, p={vr[1]:.3f}")]
            if cut is not None:
                vals.append(f"{e.get('acc_at_cut', float('nan')) * 100:.0f}%")
            for s in subs:
                mu, se, n = e.get("recovery", {}).get(s, (float("nan"), float("nan"), 0))
                vals.append("—" if not n else
                            (f"{mu:.0f}±{se:.0f}% (n={n})" if se == se
                             else f"{mu:.0f}% (n={n})"))
            vals.append(f"{self._bench_auc.get(m, float('nan')):.3f}")
            for c, v in enumerate(vals):
                self.bench_table.setItem(i, c, QTableWidgetItem(v))
        self.bench_lbl2.setVisible(True)
        self.bench_table.setVisible(True)

        self.export_b.setEnabled(True)
        best = methods[int(_np.argmin(errs))]
        self.status.setText("leave-one-out benchmark — " + " · ".join(
            f"{label[m]} {e:.0%}" for m, e in zip(methods, errs)) + f"   → best: {label[best]}"
            + (f"   (accuracy cut declared: ≤{cut:g} %p)" if cut is not None
               else "   (no accuracy cut — deviation only)"))
        self.status.setStyleSheet(f"color:{MUTE};")

    def _cancel(self):
        """Non-blocking cancel: kill the child, DON'T wait on anything in the GUI thread.
        (The old join/wait here blocked the main thread — Cancel itself froze the app.)"""
        self._cancelled = True
        proc = getattr(getattr(self, "_worker", None), "proc", None)
        if proc is not None and proc.is_alive():
            proc.terminate()                              # no join — the queue reader unblocks
        th = getattr(self, "_thread", None)
        if th is not None:
            th.quit()                                     # no wait — it dies on its own
        self._reset_buttons()
        self.status.setText("cancelled"); self.status.setStyleSheet(f"color:{MUTE};")

    def _done(self, res):
        if getattr(self, "_cancelled", False):
            return
        if isinstance(res, tuple) and len(res) == 2 and res[0] == "kfold":
            self._reset_buttons(); self._done_kfold(res[1]); return
        if isinstance(res, tuple) and len(res) == 2 and res[0] == "bench":
            self._reset_buttons(); self._done_bench(res[1]); return
        model, err, rows = res
        self._model = model
        self._reset_buttons()
        self.save_b.setEnabled(True)
        MODEL_BUS.set(model, origin=f"Model tab · {model.get('method', 'mlp').upper()}")
        self._rows = rows
        self._restore_model_titles()      # the canvases go back to the model's plots
        try:
            self._plot_triangle(rows)
        except Exception:
            import traceback as _tb
            print(_tb.format_exc(), file=sys.stderr)
        self._plot_loss(model.get("train_eval", {}).get("loss", []))
        try:
            self._plot_parity(rows); self._plot_error(rows); self._plot_roc(rows)
        except Exception:
            import traceback as _tb
            print(_tb.format_exc(), file=sys.stderr)
            self.status.setText("trained OK — a result plot failed to draw (see console); "
                                "the model itself is live and can be Saved")
        self.export_b.setEnabled(True)
        m = model.get("method", "mlp").upper() + ("  +µM" if model.get("has_uM") else "")
        errtxt = f"{err:.0%}" if err == err else "—"
        kind = ("leave-one-map-out" if model.get("loo_eval")
                else "independent test batch" if model.get("test_eval") else "train-set")
        independent = ""
        if model.get("test_eval"):
            _te = model["test_eval"]
            _tt = np.asarray(_te.get("true", []), float)
            _tp = np.asarray(_te.get("pred", []), float)
            if len(_tt):
                _ie = (0.5 * np.abs(_tp - _tt).sum(1)).mean()
                independent = f"; independent test (n={len(_tt)}) error {_ie:.0%}"
        selected = model.get("selected_epochs")
        conc_selected = (model.get("uM") or {}).get("selected_epochs")
        epoch_info = (f" · selected epochs: composition {selected}"
                      + (f", concentration {conc_selected}" if conc_selected else "")
                      if selected else "")
        self.status.setText(f"done — {m} · {model.get('n_train', 0)} training spectra · {kind} error {errtxt}"
                            f"{epoch_info}. Recovery & Real-data now use this model{independent}; "
                            "Save to keep it.")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _fail(self, tb):
        import sys
        if getattr(self, "_cancelled", False):            # cancel path already reset the UI
            return
        self._reset_buttons()
        # A Python traceback says the most in its LAST line; anything else (a killed
        # worker, a plain message) says it in the FIRST. Guessing wrong buries the
        # reason — the old code always took the last line, so a multi-line explanation
        # showed up as its closing clause. The whole text goes to the tooltip and to a
        # log file, since the app is usually started without a console to print to.
        lines = [l for l in tb.strip().splitlines() if l.strip()]
        msg = (lines[-1] if any(l.startswith("Traceback") for l in lines) else lines[0]) \
            if lines else "no message"
        log = os.path.join(self.data_dir, "unmixr_last_error.txt")
        try:
            with open(log, "w", encoding="utf-8") as f:
                f.write(tb)
        except Exception:
            log = None
        self.status.setText("failed — " + msg[:120]
                            + (f"   (full text: {os.path.basename(log)} in the pure-refs "
                               "folder, and hover this line)" if log else ""))
        self.status.setToolTip(tb)
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _result_subs(self):
        if self._model and self._model.get("subs"):
            from dataset import is_blank
            return [s for s in self._model["subs"] if not is_blank(s)]
        return list(getattr(self, "_bench_subs", []))

    def _plot_parity(self, rows):
        """Predicted vs true fraction, one series per substance (on the diagonal = exact)."""
        subs = self._result_subs()
        ax = self.c_parity.new_ax()                        # app's cnsplots style
        ax.plot([0, 1], [0, 1], ls="--", color=MUTE, lw=1.0, zorder=1)
        for j, s in enumerate(subs):
            tv = [r[1][j] for r in rows]; pv = [r[2][j] for r in rows]
            ax.scatter(tv, pv, s=34, alpha=0.8, edgecolors="white", linewidths=0.5,
                       color=substance_color(s, j), label=s, zorder=3)
        ax.set_xlabel("true fraction"); ax.set_ylabel("predicted fraction")
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect("equal")
        ax.legend(fontsize=8, framealpha=0, loc="upper left")
        self.c_parity.fig.tight_layout(); self.c_parity.draw_idle()

    def _plot_error(self, rows):
        """Mean absolute fraction error per substance (with the spread as an error bar)."""
        import numpy as _np
        subs = self._result_subs()
        ax = self.c_err.new_ax()                           # app's cnsplots style
        means, sds, cols = [], [], []
        for j, s in enumerate(subs):
            e = _np.array([abs(r[2][j] - r[1][j]) for r in rows]) if rows else _np.zeros(1)
            means.append(float(e.mean())); sds.append(float(e.std()))
            cols.append(substance_color(s, j))
        x = _np.arange(len(subs))
        ax.bar(x, means, yerr=sds, capsize=4, color=cols, edgecolor="white", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(subs)
        ax.set_ylabel("mean |predicted − true| fraction")
        self.c_err.fig.tight_layout(); self.c_err.draw_idle()

    def _roc(self, rows, thr=0.05):
        """Detection framing of the regression: 'substance present' = true fraction > thr,
        score = predicted fraction. Returns (fpr, tpr, auc) micro-averaged over substances."""
        import numpy as _np
        subs = self._result_subs()
        y = _np.concatenate([[1 if r[1][j] > thr else 0 for r in rows]
                             for j in range(len(subs))]) if rows else _np.zeros(0)
        s = _np.concatenate([[r[2][j] for r in rows] for j in range(len(subs))]) \
            if rows else _np.zeros(0)
        if len(y) == 0 or y.min() == y.max():
            return None, None, float("nan")
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _t = roc_curve(y, s)
            return fpr, tpr, float(auc(fpr, tpr))
        except Exception:
            return None, None, float("nan")

    def _plot_roc(self, rows):
        ax = self.c_roc.new_ax()                           # app's cnsplots style
        fpr, tpr, a = self._roc(rows)
        if fpr is None:
            ax.text(0.5, 0.5, "needs both present and absent components", ha="center",
                    va="center", color=MUTE, transform=ax.transAxes); ax.axis("off")
        else:
            ax.plot([0, 1], [0, 1], ls="--", color=MUTE, lw=1.0)
            ax.plot(fpr, tpr, color=TEAL, lw=2.0, label=f"AUC {a:.3f} (n={len(rows)} maps; presence only)")
            ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
            ax.legend(fontsize=9, framealpha=0, loc="lower right")
        self.c_roc.fig.tight_layout(); self.c_roc.draw_idle()

    def _plot_loss(self, loss):
        ax = self.c_loss.new_ax()                          # app's cnsplots style
        if loss:
            ax.plot(range(1, len(loss) + 1), loss, color=TEAL, lw=1.5)
            ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
        else:
            ax.text(0.5, 0.5, "no epochs for this method (PLS / RF)", ha="center", va="center",
                    color=MUTE, transform=ax.transAxes); ax.axis("off")
        self.c_loss.fig.tight_layout(); self.c_loss.draw_idle()

    def _plot_triangle(self, rows):
        """Ternary simplex: true (open) → predicted (filled, green=accurate) per mixture."""
        from composition import bary
        subs = self._result_subs()
        if len(subs) != 3:
            self.c_tri.placeholder("Ternary plot is available only for exactly 3 substances")
            return
        ax = self.c_tri.new_ax()                           # app's cnsplots style
        fig = self.c_tri.fig
        A, B, C = bary([1, 0, 0]), bary([0, 0, 1]), bary([0, 1, 0])   # DQ · THI · TBZ corners
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color=INK, lw=1.0, zorder=1)
        # corner labels OFFSET away from the vertex so they never sit on the lines/points
        for f, s, col, dx, dy, ha, va in [
                ([1, 0, 0], subs[0], substance_color(subs[0], 0), 0, 0.05, "center", "bottom"),
                ([0, 0, 1], subs[2], substance_color(subs[2], 2), -0.03, -0.05, "right", "top"),
                ([0, 1, 0], subs[1], substance_color(subs[1], 1), 0.03, -0.05, "left", "top")]:
            p = bary(f)
            ax.text(p[0] + dx, p[1] + dy, s, ha=ha, va=va, fontsize=11,
                    fontweight="bold", color=col)
        # matplotlib.cm.get_cmap was removed in 3.9 — the registry is the API now
        cmap = matplotlib.colormaps["RdYlGn"]
        for _name, tv, pv in rows:
            p0 = bary(tv); p1 = bary(pv); e = 0.5 * sum(abs(pv[j] - tv[j]) for j in range(len(tv)))
            if (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 > 1e-5:
                ax.annotate("", xy=p1, xytext=p0,
                            arrowprops=dict(arrowstyle="-|>", color="#b6bcc4", lw=0.8))
            ax.scatter(*p0, s=26, facecolors="none", edgecolors=MUTE, linewidths=1.0, zorder=3)
            ax.scatter(*p1, s=42, color=cmap(1 - min(e / 0.6, 1.0)), edgecolors="white",
                       linewidths=0.5, zorder=4)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.18, 1.02)
        fig.tight_layout(); self.c_tri.draw_idle()

    def _export(self):
        """Write the plots + the numbers behind them (so every figure is redrawable)."""
        if self._model is None or not getattr(self, "_rows", None):
            return self._export_eval_only()
        import numpy as _np
        from collections import OrderedDict
        subs = self._result_subs()
        from io_utils import write_csv, write_readme
        d = QFileDialog.getExistingDirectory(self, "Export composition-model results",
                                             self.data_dir)
        if not d:
            return
        rows = self._rows; m = self._model
        # per-mixture true vs predicted fraction
        write_csv(os.path.join(d, "composition_loo_predictions.csv"),
                  ["mixture"] + [f"true_{s}" for s in subs]
                  + [f"pred_{s}" for s in subs] + ["composition_error"],
                  [[r[0]] + [f"{v:.4f}" for v in r[1]] + [f"{v:.4f}" for v in r[2]]
                   + [f"{0.5 * sum(abs(r[2][j] - r[1][j]) for j in range(len(subs))):.4f}"]
                   for r in rows])
        # per-substance metrics
        mrows = []
        for j, s in enumerate(subs):
            e = _np.array([abs(r[2][j] - r[1][j]) for r in rows])
            tv = _np.array([r[1][j] for r in rows]); pv = _np.array([r[2][j] for r in rows])
            ss = float(((tv - tv.mean()) ** 2).sum())
            r2 = 1 - float(((tv - pv) ** 2).sum()) / ss if ss > 1e-12 else float("nan")
            mrows.append([s, f"{e.mean():.4f}", f"{e.std():.4f}",
                          f"{float(_np.sqrt(((tv - pv) ** 2).mean())):.4f}", f"{r2:+.3f}"])
        write_csv(os.path.join(d, "composition_metrics.csv"),
                  ["substance", "mean_abs_error", "sd", "rmse", "r2"], mrows)
        loss = m.get("train_eval", {}).get("loss", [])
        if loss:
            write_csv(os.path.join(d, "composition_learning_curve.csv"), ["epoch", "loss"],
                      [[i + 1, f"{v:.5f}"] for i, v in enumerate(loss)])
        fpr, tpr, auc_v = self._roc(rows)                  # detection ROC (present/absent)
        if fpr is not None:
            write_csv(os.path.join(d, "composition_roc.csv"), ["fpr", "tpr"],
                      [[f"{a:.5f}", f"{b:.5f}"] for a, b in zip(fpr, tpr)])
        map_errors = [0.5 * sum(abs(r[2][j] - r[1][j]) for j in range(len(subs)))
                      for r in rows]
        write_csv(os.path.join(d, "evaluation_summary.csv"),
                  ["metric", "value", "unit_or_note"], [
                  ["evaluation_unit", "map", "pixels remain grouped by map"],
                  ["experimental_batches", "1", "batch IDs not yet supplied"],
                  ["evaluation_maps", str(len(rows)), "leave-one-map-out"],
                  ["training_maps", str(m.get("n_maps", len(self._items_cache))), "maps"],
                  ["training_spectra", str(m.get("n_train", 0)), "pixel spectra"],
                  ["pixels_per_map", str(m.get("px_per_map", 0)), "configured maximum"],
                  ["mean_map_composition_error", f"{_np.mean(map_errors):.6f}", "0.5*L1"],
                  ["presence_threshold", "0.05", "true fraction > threshold"],
                  ["roc_auc_micro", f"{auc_v:.6f}" if auc_v == auc_v else "",
                   "pooled component presence; not composition accuracy"]])
        # c_err / c_roc are SHARED with the benchmark: whichever ran last is on screen.
        # Name them for what they actually show, or a benchmark chart would leave here
        # as "composition_error.png" and be read as the model's per-substance error.
        # A benchmark takes over four of the five canvases (error, ROC, parity, ternary),
        # so a session that both trained and benchmarked can only export the panels the
        # LAST run drew. Name them for what they actually show, or the benchmark's charts
        # would leave as composition_*.png and be read as the model's own results.
        showing_bench = getattr(self, "_canvas_shows", "model") == "bench"
        figs = [("composition_learning_curve", self.c_loss)]
        figs += (self._bench_figs() if showing_bench else
                 [("composition_triangle", self.c_tri),
                  ("composition_parity", self.c_parity),
                  ("composition_error", self.c_err),
                  ("composition_roc", self.c_roc)])
        n = _save_figs(figs, d)
        n += self._export_grid_figs(d, rows, subs)
        # a benchmark run in this session leaves with the model, not silently dropped
        if getattr(self, "_bench", None):
            self._write_bench_csvs(d, self._result_subs())
        # ONE ternary only — the on-screen `composition_triangle` above. The extra
        # ternaries this used to write (vs-NNLS side-by-side, accuracy-shaded, RGB)
        # said the same thing three more times and buried the rest of the export.
        err = float(_np.mean([0.5 * sum(abs(r[2][j] - r[1][j]) for j in range(len(subs)))
                              for r in rows]))
        write_readme(d, "UNMIXR — Composition model (Step 2)", OrderedDict([
            ("What this is", [
                "The composition model trained once on the known-ratio mixtures prepared in "
                "Samples. Recovery and Real-data APPLY this model — they do not retrain. "
                "Numbers here are LEAVE-ONE-OUT: every mixture was scored by a model that "
                "never saw it. The saved model itself is fit on all of them."]),
            ("How it was produced", [
                f"- Pure references: {self.data_dir}",
                f"- Mixture maps: {m.get('n_maps') or len(self._items_cache)} "
                f"(training rows after aggregation/pixel sampling: {m.get('n_train', 0)})",
                f"- Method: {m.get('method', 'mlp').upper()}"
                + (f" · epochs {self.sp_ep.value()} · seed {self.sp_seed.value()}"
                   if m.get('method') in ('mlp', 'cnn') else ""),
                f"- Concentration (µM) head: {'trained' if m.get('has_uM') else 'not trained'}",
                f"- Spectral window: {m.get('lo')}–{m.get('hi')} cm⁻¹"]),
            ("Results", [
                f"- Mean composition error: {err:.1%} (½·Σ|pred − true|, 0 = perfect)",
                f"- Detection ROC-AUC: {auc_v:.3f} (is each substance present? score = "
                "predicted fraction, present = true fraction > 5%)"
                if auc_v == auc_v else "- Detection ROC-AUC: n/a",
                "- Per-substance error / RMSE / R²: `composition_metrics.csv`",
                "- Every figure is redrawable from the CSVs next to it."]),
        ]), [("composition_learning_curve", "training loss vs epoch (MLP / CNN only)."),
             ("composition_triangle", "leave-one-out true (○) → predicted (●) on the simplex; "
                                      "colour = accuracy."),
             ("composition_parity", "predicted vs true fraction per substance; on the "
                                    "diagonal = exact."),
             ("composition_error", "mean |predicted − true| fraction per substance (± SD)."),
             ("composition_roc", "detection ROC — present/absent per substance, scored by "
                                 "the predicted fraction.")])
        self.status.setText(f"exported {n} PNG + CSVs + README → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")

    # ------------------------------------------------------------------ whole-experiment figures
    def _row_context(self, subs):
        """(concentrations, paths) aligned to ``self._rows``, or (None, None).

        Held-out scoring reports one row per map and keeps the map path alongside, so
        the concentrations come back from the Samples items by path. Without that the
        rows are just 'mix 1..n' and there is no grid to lay out."""
        m = self._model or {}
        ev = m.get("test_eval") or m.get("loo_eval") or {}
        paths = list(ev.get("paths") or [])
        rows = getattr(self, "_rows", [])
        if not paths or len(paths) != len(rows):
            return None, None
        by_path = {it[0]: it[1] for it in (self._items_cache + self._test_items)}
        concs = []
        for p in paths:
            c = by_path.get(p)
            if c is None:
                return None, None
            concs.append(tuple(float(c.get(s, 0.0)) for s in subs))
        return concs, paths

    def _export_grid_figs(self, d, rows, subs):
        """Write the whole experiment at once: a pie per condition, the design grid with
        true vs predicted bars, and the grid as an error heat-map. Returns the count."""
        import grid_figs as GF
        from ui_common import substance_color, EXPORT_DPI
        cols = [substance_color(s, i) for i, s in enumerate(subs)]
        pct = [(r[0], [100.0 * v for v in r[1]], [100.0 * v for v in r[2]]) for r in rows]
        concs, _paths = self._row_context(subs)
        if concs:
            pct, concs = GF.group_by_condition(pct, concs, fmt="{} µM")

        def _w(fig, name):
            if fig is None:
                return 0
            fig.savefig(os.path.join(d, name + ".png"), dpi=EXPORT_DPI,
                        transparent=True, bbox_inches="tight", pad_inches=0.01)
            return 1

        n = _w(GF.composition_pies(pct, subs, cols,
                                   title="Predicted composition per condition — held out"),
               "grid_pies")
        if concs:
            n += _w(GF.grid_truepred(pct, concs, subs, cols), "grid_true_vs_predicted")
            n += _w(GF.grid_error(pct, concs, subs), "grid_error")
            # the numbers, not just the picture — these get redrawn in Origin
            from io_utils import write_csv
            h, b = GF.condition_table(pct, concs, subs)
            write_csv(os.path.join(d, "grid_conditions.csv"), h, b)
            for name, mh, mb in GF.error_matrix_tables(pct, concs, subs):
                write_csv(os.path.join(d, f"grid_error_matrix_{name}.csv"), mh, mb)
        self._grid_ctx = (d, pct, concs, subs, cols)
        self.pix_b.setEnabled(True)
        return n

    def _export_pixel_maps(self):
        """Run the per-pixel composition map for every scored map and tile them.

        This re-unmixes each map, so it runs in the worker rather than freezing the tab
        while a 65-condition experiment goes through."""
        ctx = getattr(self, "_grid_ctx", None)
        if not ctx or self._model is None:
            self.status.setText("export once first — the maps come from that run")
            self.status.setStyleSheet(f"color:{RED};"); return
        subs = ctx[3]
        _concs, paths = self._row_context(subs)
        if not paths:
            self.status.setText("no per-map paths in this run — cannot draw pixel maps")
            self.status.setStyleSheet(f"color:{RED};"); return
        from dataset import load_preprocess
        cfg = load_preprocess(self.data_dir)
        params = dict(data_dir=self.data_dir, paths=paths, model=self._model,
                      baseline=bool(self._model.get("baseline", cfg["baseline"])),
                      trim=cfg["trim"])
        self.pbar.setRange(0, 0); self.pbar.setVisible(True)
        self.status.setText("● per-pixel maps…"); self.status.setStyleSheet(f"color:{MUTE};")
        start_worker(self, PixelMapWorker(params), done=self._done_pixel_maps,
                     fail=self._fail, progress=lambda s: self.status.setText("● " + s))

    def _done_pixel_maps(self, entries):
        import grid_figs as GF
        from ui_common import EXPORT_DPI
        self.pbar.setVisible(False)
        d, _pct, _concs, subs, cols = self._grid_ctx
        if not entries:
            self.status.setText("per-pixel maps: nothing came back")
            self.status.setStyleSheet(f"color:{RED};"); return
        fig = GF.pixel_maps(entries, subs, cols,
                            title="Per-pixel composition map — NNLS decides hit, "
                                  "the model composes")
        fig.savefig(os.path.join(d, "grid_pixel_maps.png"), dpi=EXPORT_DPI,
                    transparent=True, bbox_inches="tight", pad_inches=0.01)
        from io_utils import write_csv
        h, b = GF.pixel_table(entries, subs)
        write_csv(os.path.join(d, "grid_pixel_maps.csv"), h, b)
        self.status.setText(f"per-pixel maps → grid_pixel_maps.png + .csv "
                            f"({len(entries)} maps)")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _restore_model_titles(self):
        """Hand the four shared panels back to the trained model. A benchmark relabels
        them (accuracy / recovery / method comparison), so training or loading a model
        afterwards has to put the captions back or the plots would describe the wrong run."""
        self._canvas_shows = "model"
        self._tri_title.setText("Held-out recovery (leave-one-out) — true (○) vs "
                                "predicted (●, colour = accuracy)")
        self._parity_title.setText("Parity per substance — predicted vs true fraction "
                                   "(on the line = exact)")
        self._err_title.setText("Per-substance error — mean |predicted − true| fraction "
                                "(lower = better)")
        self._roc_title.setText("Detection ROC — is each substance present? "
                                "(threshold the predicted fraction)")

    def _bench_figs(self):
        """The benchmark's four panels, as (filename, canvas). The accuracy panel is
        skipped when no cut was declared — it holds a placeholder, not a chart."""
        figs = [("benchmark_error", self.c_err), ("benchmark_recovery", self.c_tri),
                ("benchmark_roc", self.c_roc)]
        if getattr(self, "_bench_cut", None) is not None:
            figs.insert(1, ("benchmark_accuracy", self.c_parity))
        return figs

    def _write_bench_csvs(self, d, subs):
        """The benchmark table + its raw predictions and ROC points, into folder ``d``.
        Shared by both export paths so a benchmark leaves with its numbers whether or
        not a model was also trained this session. Metrics come from
        dl_model.benchmark_summary — the same call the on-screen table reads."""
        from io_utils import write_csv
        bench = getattr(self, "_bench", None)
        if not bench:
            return
        # mean deviation (%p) + per-substance recovery %±SE always; accuracy only when a
        # cut was DECLARED before the run.
        summ = getattr(self, "_bench_summary", None)
        if summ is None and getattr(self, "_bench_raw", None):
            from dl_model import benchmark_summary
            summ = benchmark_summary(self._bench_raw, getattr(self, "_bench_cut", None))
        summ = summ or {}
        cut = getattr(self, "_bench_cut", None)
        # the by-component-count split is off the chart now, but stays in the file
        kgroups = [g for g in ("pure", "binary", "ternary")
                   if any(g in e.get("dev_by_k", {}) for e in summ.values())]
        head = ["method", "mean_deviation_pp", "rmse_pp", "rmse_se",
                "logratio_distance", "logratio_se", "median_delta_vs_nnls_pp",
                "wilcoxon_p_vs_nnls", "composition_error", "detection_auc"]
        for g in kgroups:
            head += [f"deviation_pp_{g}", f"deviation_se_{g}", f"n_{g}"]
        for s in subs:
            head += [f"recovery_pct_{s}", f"recovery_se_{s}", f"recovery_n_{s}"]
        if cut is not None:
            head.append(f"accuracy_within_{cut:g}pp")
        mrows = []
        for m in bench:
            e = summ.get(m, {})
            rm, rs, _ = e.get("rmse_pp", (float("nan"),) * 3)
            lm, ls, _ = e.get("logratio", (float("nan"),) * 3)
            vr = e.get("vs_ref") or (float("nan"), float("nan"), 0)
            row = [m, f"{e.get('mean_dev_pp', float('nan')):.2f}",
                   f"{rm:.2f}", f"{rs:.2f}", f"{lm:.4f}", f"{ls:.4f}",
                   f"{vr[0]:.2f}", f"{vr[1]:.5f}",
                   f"{self._bench_err[m]:.4f}",
                   f"{self._bench_auc.get(m, float('nan')):.4f}"]
            for g in kgroups:
                mu, se, nn = e.get("dev_by_k", {}).get(g, (float("nan"), float("nan"), 0))
                row += [f"{mu:.2f}", f"{se:.2f}", str(nn)]
            for s in subs:
                mu, se, nn = e.get("recovery", {}).get(s, (float("nan"), float("nan"), 0))
                row += [f"{mu:.2f}", f"{se:.2f}", str(nn)]
            if cut is not None:
                row.append(f"{e.get('acc_at_cut', float('nan')):.4f}")
            mrows.append(row)
        write_csv(os.path.join(d, "benchmark_metrics.csv"), head, mrows)
        rows_out = []
        for m, rws in bench.items():
            for nm, tv, pv in rws:
                rows_out.append([m, nm] + [f"{v:.4f}" for v in tv]
                                + [f"{v:.4f}" for v in pv])
        write_csv(os.path.join(d, "benchmark_predictions.csv"),
                  ["method", "mixture"] + [f"true_{s}" for s in subs]
                  + [f"pred_{s}" for s in subs], rows_out)
        # the ROC panel was the one figure leaving without its numbers — only the pooled
        # AUC was written, and a curve cannot be redrawn from a single scalar.
        curve = []
        for m, rws in bench.items():
            fpr, tpr, _a = self._roc(rws)
            if fpr is None:
                continue
            curve += [[m, f"{a:.5f}", f"{b:.5f}"] for a, b in zip(fpr, tpr)]
        if curve:
            write_csv(os.path.join(d, "benchmark_roc_curves.csv"),
                      ["method", "fpr", "tpr"], curve)

    def _export_eval_only(self):
        """Save a Benchmark / 5-fold result when no model has been trained in this session."""
        import numpy as _np
        subs = self._result_subs()
        from io_utils import write_csv
        bench = getattr(self, "_bench", None); kf = getattr(self, "_kfold_res", None)
        if not bench and not kf:
            self.status.setText("run Benchmark, 5-fold or Train first")
            self.status.setStyleSheet(f"color:{RED};"); return
        d = QFileDialog.getExistingDirectory(self, "Export evaluation", self.data_dir)
        if not d:
            return
        n = 0
        if bench:
            self._write_bench_csvs(d, subs)
            n += _save_figs(self._bench_figs(), d)
        if kf:
            write_csv(os.path.join(d, "kfold_errors.csv"), ["fold", "composition_error"],
                      [[i + 1, f"{e:.4f}"] for i, e in enumerate(kf.get("errors", []))]
                      + [["mean", f"{kf.get('mean', float('nan')):.4f}"],
                         ["sd", f"{kf.get('sd', float('nan')):.4f}"]])
            n += _save_figs([("kfold_errors", self.c_err)], d)
        self.status.setText(f"exported evaluation ({n} PNG + CSV) → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _load(self):
        """Re-open a saved .dlm and publish it, so a model trained in an earlier session
        is usable everywhere without retraining. Its stored held-out evaluation is
        redrawn when present, so the plots describe the model you just loaded."""
        p, _ = QFileDialog.getOpenFileName(self, "Composition model (.dlm)", "",
                                           "DL model (*.dlm);;all (*)")
        if not p:
            return
        try:
            from dl_model import load_model
            model = load_model(p)
            if not isinstance(model, dict) or "subs" not in model:
                raise ValueError("not a composition model")
        except Exception as e:
            self.status.setText(f"could not load {os.path.basename(p)} — {e}")
            self.status.setStyleSheet(f"color:{RED};"); return
        self._model = model
        self.save_b.setEnabled(True)
        from dataset import is_blank
        all_subs = list(model["subs"])
        subs = [s for s in all_subs if not is_blank(s)]
        sidx = {s: j for j, s in enumerate(all_subs)}
        te = model.get("test_eval") or model.get("loo_eval") or model.get("train_eval") or {}
        tvs = te.get("true", []); pvs = te.get("pred", [])
        rows = []
        for i in range(min(len(tvs), len(pvs))):
            tv = [float(tvs[i][sidx[s]]) if s in sidx else 0.0 for s in subs]
            pv = [float(pvs[i][sidx[s]]) if s in sidx else 0.0 for s in subs]
            rows.append((f"mix {i + 1}", tv, pv))
        if rows:
            self._rows = rows
            self._restore_model_titles()
            self._plot_triangle(rows); self._plot_parity(rows)
            self._plot_error(rows); self._plot_roc(rows)
            self._plot_loss(model.get("train_eval", {}).get("loss", []))
            self.export_b.setEnabled(True)
        blank = model.get("blank")
        MODEL_BUS.set(model, origin=f"loaded · {os.path.basename(p)}")
        self.status.setText(
            f"loaded {os.path.basename(p)} — {model.get('method', 'mlp').upper()} · "
            f"{len(subs)} classes ({', '.join(subs)})"
            + (f" · blank class {blank}" if blank else " · no blank class")
            + f" · {model.get('n_train', 0)} training rows. Recovery and Real-data now use it.")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _save(self):
        if self._model is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "Save composition model", "composition.dlm",
                                           "DL model (*.dlm)")
        if not p:
            return
        from dl_model import save_model
        save_model(self._model, p)
        self.status.setText("saved → " + os.path.basename(p))
        self.status.setStyleSheet(f"color:{MUTE};")
