"""page_validate.py — Validate tab: check the pure-reference unmixing against
KNOWN-ratio mixtures, recover per-substance response factors, and export a
correction the Real-data tab can apply (surface ratio → solution ratio)."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFileDialog, QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QLineEdit, QCheckBox, QProgressBar, QSpinBox,
    QComboBox,
)

from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from ui_common import *
from real_data import PEST_DEFAULT
from dataset import load_preprocess
from io_utils import write_csv, write_readme
from validate import validate_mixtures, parse_mixture_label, parse_amount
from composition import compute_composition, SUBSTANCES, bary, composition_distance


class ValidateWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            vres = validate_mixtures(progress=self.progress.emit, **self.params["validate"])
            cp = self.params.get("composition")
            cres = compute_composition(progress=self.progress.emit, **cp) if cp else None
            dres = None
            if self.params.get("shared_model") is not None:      # inherit the Model-tab model
                from dl_model import apply_recovery
                dres = apply_recovery(self.params["shared_model"],
                                      self.params["shared_items"], progress=self.progress.emit)
            elif self.params.get("dl"):
                from dl_recovery import dl_recovery
                dres = dl_recovery(progress=self.progress.emit, **self.params["dl"])
            self.done.emit((vres, cres, dres))
        except Exception:
            self.fail.emit(traceback.format_exc())


class ExplainWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from dl_explain import dl_explain
            self.done.emit(dl_explain(progress=self.progress.emit, **self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


class SaveModelWorker(QObject):
    done = pyqtSignal(str)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from dl_model import train_model, save_model
            out = self.params.pop("out")
            model = train_model(progress=self.progress.emit, **self.params)
            save_model(model, out)
            self.done.emit(f"{out}  (n={model['n_train']}, µM {'yes' if model['has_uM'] else 'no'})")
        except Exception:
            self.fail.emit(traceback.format_exc())


class ValidatePage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self._cres = None                # composition (per-pixel) result
        self._shared_model = MODEL_BUS.model         # model trained in the Model tab (Step 2)
        COLOR_BUS.changed.connect(self._recolor)     # top-bar picker → recolour
        MODEL_BUS.changed.connect(self._on_model_bus)
        MIXTURE_BUS.changed.connect(lambda: self._load_from_samples(force=True))
        self._files = []                 # full paths, aligned with table rows
        self.data_dir = PEST_DEFAULT
        self.calib_path = None           # dilution-series CSV → recovery (measured µM)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(12)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Recovery — known-ratio mixtures → response factors"); h1.setObjectName("h1")
        sub = QLabel("Load mixtures whose true ratio you know (e.g. DQ_TBZ_1to3). Each "
                     "is unmixed against your pure references; the observed surface "
                     "ratio is compared to the true ratio to recover each substance's "
                     "response factor — why one substance (e.g. THI) can dominate every "
                     "map. Export the factors and Real data can report the corrected "
                     "solution ratio.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        self.ref_lbl = QLabel(self._short(self.data_dir)); self.ref_lbl.setObjectName("field")
        add_b = QPushButton("Add mixtures…"); add_b.setObjectName("ghost")
        add_b.setToolTip("load one or more known-ratio mixture maps")
        add_b.clicked.connect(self._add)
        cal_b = QPushButton("Load calibration…"); cal_b.setObjectName("ghost")
        cal_b.setToolTip("dilution-series CSV (calibration_spectra.csv). Enables RECOVERY "
                         "when you enter true concentrations (e.g. DQ:100uM) in the table")
        cal_b.clicked.connect(self._browse_calib)
        self.cal_lbl = QLabel("no calib (ratio only)"); self.cal_lbl.setObjectName("field")
        self.dl_chk = QCheckBox("DL predict"); self.dl_chk.setObjectName("field")
        self.dl_chk.setToolTip("physics-informed DL composition (leave-one-map-out over the "
                               "loaded mixtures). The composition view (triangle · recovery · "
                               "drift) then shows the DL prediction instead of NNLS. Slower.")
        self.explain_btn = QPushButton("DL explain"); self.explain_btn.setObjectName("ghost")
        self.explain_btn.setToolTip("train the DL on the loaded mixtures and show which "
                                    "spectral bands it uses (Integrated Gradients + band "
                                    "permutation importance + ligand ablation)")
        self.explain_btn.clicked.connect(self._run_explain)
        # (the composition model is trained + saved ONCE in the Model tab; Recovery applies it)
        self.savemodel_btn = QPushButton("Save DL model"); self.savemodel_btn.setObjectName("ghost")
        self.savemodel_btn.setToolTip("deprecated — train and save the composition model in the "
                                      "Model tab instead; Recovery applies it")
        self.savemodel_btn.clicked.connect(self._save_dl_model)
        self.savemodel_btn.setVisible(False)
        clr_b = QPushButton("Clear"); clr_b.setObjectName("ghost"); clr_b.clicked.connect(self._clear)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost"); exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Validate"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setObjectName("ghost")
        self.cancel_btn.setToolTip("stop the running Validate / DL job")
        self.cancel_btn.clicked.connect(self._cancel); self.cancel_btn.setVisible(False)
        ctl.addWidget(self.ref_lbl); ctl.addStretch(1)
        ctl.addWidget(cal_b); ctl.addWidget(self.cal_lbl); ctl.addWidget(self.dl_chk)
        ctl.addWidget(add_b); ctl.addWidget(self.explain_btn); ctl.addWidget(self.savemodel_btn)
        ctl.addWidget(clr_b); ctl.addWidget(exp_b); ctl.addWidget(self.cancel_btn); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        # collapsible input detail (fixed components · VIP · mixture table) — collapse it
        # (auto-collapses after a run) to give the result plots the full window height
        self.tgl = QPushButton(); self.tgl.setObjectName("ghost")
        self.tgl.setCheckable(True); self.tgl.setChecked(True)
        self.tgl.setStyleSheet("text-align:left; padding:4px 8px;")
        self.tgl.toggled.connect(self._toggle_inputs)
        root.addWidget(self.tgl)

        self.inbox = QWidget(); ibl = QVBoxLayout(self.inbox)
        ibl.setContentsMargins(0, 0, 0, 0); ibl.setSpacing(12)

        # fixed-component base ratio — auto-added to files that don't name them
        brow = QHBoxLayout(); brow.setSpacing(8)
        bl = QLabel("fixed components:"); bl.setObjectName("field")
        self.base_txt = QLineEdit()
        self.base_txt.setPlaceholderText("e.g. TBZ:1, DQ:1  — added to files that only "
                                         "name the varied substance (THI001 → +TBZ:1, DQ:1)")
        self.base_txt.setToolTip("the components held constant across your mixtures; "
                                 "filenames only need to encode what changes")
        repar_b = QPushButton("Re-parse names"); repar_b.setObjectName("ghost")
        repar_b.setToolTip("re-read every filename's true ratio using the fixed-components base")
        repar_b.clicked.connect(self._reparse)
        brow.addWidget(bl); brow.addWidget(self.base_txt, 1); brow.addWidget(repar_b)
        ibl.addLayout(brow)

        # VIP-band NNLS — decompose the composition on each compound's discriminative
        # marker band(s) only, instead of the whole spectrum (less mixture cross-talk)
        vrow = QHBoxLayout(); vrow.setSpacing(8)
        self.vip_chk = QCheckBox("VIP-band NNLS"); self.vip_chk.setObjectName("field")
        self.vip_chk.setToolTip("fit each mixture's composition only on the compounds' "
                                "VIP marker bands (least cross-talk) rather than the full "
                                "spectrum — matches how Quantify picks bands")
        self.vip_txt = QLineEdit()
        self.vip_txt.setPlaceholderText("e.g. DQ:1177, TBZ:1001, THI:1369  — bands the "
                                        "composition is decomposed on (multi-band: DQ:1177+1566)")
        self.vip_txt.setToolTip("one or more bands per compound as name:wavenumber "
                                "(join a compound's bands with '+'); 'auto VIP' fills the pick")
        vip_b = QPushButton("auto VIP"); vip_b.setObjectName("ghost")
        vip_b.setToolTip("recommend each compound's least-cross-talk discriminative band "
                         "from the references")
        vip_b.clicked.connect(self._auto_vip)
        vrow.addWidget(self.vip_chk); vrow.addWidget(self.vip_txt, 1); vrow.addWidget(vip_b)
        ibl.addLayout(vrow)

        # DL training settings — used by BOTH 'DL predict' (leave-one-out) and 'Save DL
        # model'. Method dropdown; each method's own sub-parameters appear automatically.
        # Downstream (Real-data apply) needs no knobs; tuning lives here.
        drow = QHBoxLayout(); drow.setSpacing(8)
        dlbl = QLabel("DL method:"); dlbl.setObjectName("field")
        self.dl_method = QComboBox(); self.dl_method.setObjectName("field")
        for text, data in [("MLP (deep)", "mlp"), ("PLS", "pls"),
                           ("Random Forest", "rf"), ("1D-CNN", "cnn")]:
            self.dl_method.addItem(text, data)
        self.dl_method.setToolTip("composition model — the best one is data-dependent, so pick "
                                  "per experiment. Sub-parameters below change with the method.")
        self.dl_method.currentIndexChanged.connect(self._update_dl_params)
        # per-method sub-parameter widgets (shown/hidden by _update_dl_params)
        self.dl_ep = QSpinBox(); self.dl_ep.setRange(20, 3000); self.dl_ep.setSingleStep(50)
        self.dl_ep.setValue(350); self.dl_ep.setPrefix("epochs "); self.dl_ep.setObjectName("field")
        self.dl_ep.setToolTip("training iterations (MLP / CNN)")
        self.dl_seed = QSpinBox(); self.dl_seed.setRange(0, 999); self.dl_seed.setValue(0)
        self.dl_seed.setPrefix("seed "); self.dl_seed.setObjectName("field")
        self.dl_seed.setToolTip("RNG seed (MLP / CNN / RF)")
        self.dl_pre = QCheckBox("physics pretrain"); self.dl_pre.setChecked(True)
        self.dl_pre.setObjectName("field")
        self.dl_pre.setToolTip("MLP only — warm up on physics-simulated mixtures from the "
                               "calibration before fitting the real ones (needs a calibration)")
        self.dl_nc = QSpinBox(); self.dl_nc.setRange(1, 20); self.dl_nc.setValue(8)
        self.dl_nc.setPrefix("components "); self.dl_nc.setObjectName("field")
        self.dl_nc.setToolTip("PLS latent components")
        self.dl_nt = QSpinBox(); self.dl_nt.setRange(20, 1000); self.dl_nt.setSingleStep(20)
        self.dl_nt.setValue(300); self.dl_nt.setPrefix("trees "); self.dl_nt.setObjectName("field")
        self.dl_nt.setToolTip("Random-Forest trees")
        drow.addWidget(dlbl); drow.addWidget(self.dl_method)
        for w in (self.dl_ep, self.dl_seed, self.dl_pre, self.dl_nc, self.dl_nt):
            drow.addWidget(w)
        drow.addStretch(1)
        ibl.addLayout(drow)
        self._update_dl_params()

        # editable table: file  |  true ratio (name:parts, comma-separated)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["mixture file", "true ratio  (e.g. DQ:1, TBZ:3)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        self.table.setMaximumHeight(170)
        ibl.addWidget(self.table)
        root.addWidget(self.inbox)
        self._toggle_inputs(True)                          # set the toggle label

        # progress: a busy bar (shown only while working) + the live status text
        prow = QHBoxLayout(); prow.setSpacing(8)
        self.status = QLabel(""); self.status.setObjectName("sub")
        self.progbar = QProgressBar(); self.progbar.setTextVisible(False)
        self.progbar.setFixedHeight(6); self.progbar.setMaximumWidth(180)
        self.progbar.setVisible(False)
        prow.addWidget(self.progbar); prow.addWidget(self.status, 1)
        root.addLayout(prow)

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_mix = Kpi("mixtures"); self.k_sub = Kpi("substances")
        self.k_max = Kpi("max response ×"); self.k_err = Kpi("mean error → corrected")
        for k in (self.k_mix, self.k_sub, self.k_max, self.k_err):
            kpis.addWidget(k)
        root.addLayout(kpis)

        body = QVBoxLayout(); body.setSpacing(12)
        self.c_parity = Canvas(); self.c_resp = Canvas(); self.c_corr = Canvas()
        prow = QHBoxLayout(); prow.setSpacing(12)
        for cv, title in [
            (self.c_parity, "Observed (surface) vs true ratio — points above the line "
                            "are over-reported"),
            (self.c_corr, "Corrected (solution) vs true ratio — should sit on the line"),
        ]:
            card, lay = _card(title); lay.addWidget(cv); cv.setMinimumHeight(320)
            prow.addWidget(card, 1)
        prow_w = QWidget(); prow_w.setLayout(prow); body.addWidget(prow_w)
        rcard, rlay = _card("Response factor per substance (×, relative — higher = "
                            "dominates the surface signal)")
        rlay.addWidget(self.c_resp); self.c_resp.setMinimumHeight(300)
        body.addWidget(rcard)

        # ---- composition view (drift · recovery) ----
        self.c_tri = Canvas()
        self.c_rel = Canvas(); self.c_rec = Canvas(); self.c_explain = Canvas()
        crow = QHBoxLayout(); crow.setSpacing(12)
        for cv, title in [
            (self.c_tri, "Composition recovery (drift triangle)"),
            (self.c_rec, "Apparent recovery (predicted / real, mean ± SE over mixtures)"),
        ]:
            card, lay = _card(title); lay.addWidget(cv); cv.setMinimumHeight(300)
            crow.addWidget(card, 1)
        crow_w = QWidget(); crow_w.setLayout(crow); body.addWidget(crow_w)
        dcard, dlay = _card("Relative drift — predicted vs real, grouped by dominant substance")
        dlay.addWidget(self.c_rel); self.c_rel.setMinimumHeight(300)
        body.addWidget(dcard)
        ecard, elay = _card("DL interpretability — IG attribution · band importance · ligand "
                            "ablation  (run 'DL explain')")
        elay.addWidget(self.c_explain); self.c_explain.setMinimumHeight(430)
        body.addWidget(ecard)

        bodyw = QWidget(); bodyw.setLayout(body)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidget(bodyw)
        scroll.setStyleSheet("QScrollArea{background:transparent;}")
        root.addWidget(scroll, 1)
        for cv, m in [(self.c_parity, "Add mixtures, then Validate"),
                      (self.c_corr, "Corrected ratio appears here"),
                      (self.c_resp, "Response factors appear here"),
                      (self.c_tri, "Drift triangle appears here"),
                      (self.c_rec, "Recovery appears here"),
                      (self.c_rel, "Relative drift appears here"),
                      (self.c_explain, "Run 'DL explain' → which bands the model uses (IG · "
                                       "permutation · ablation)")]:
            cv.placeholder(m)

        self.readout = QLabel(""); self.readout.setObjectName("sub")
        self.readout.setWordWrap(True); self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setStyleSheet(f"font-size:15px; color:{INK};")
        root.addWidget(self.readout)

    # ---- helpers ----
    def _short(self, p):
        """Name the reference SUBSTANCES, not just the folder — the folder is only where
        the pure maps live, which reads confusingly as 'refs = Pure'."""
        names = []
        try:
            from dataset import discover_dataset, is_blank
            names = [c for c, _m in discover_dataset(p) if not is_blank(c)]
        except Exception:
            pass
        if names:
            return "refs (from Samples): " + " · ".join(names)
        return "refs (from Samples): " + ("…" + p[-34:] if len(p) > 34 else p)

    def set_data_dir(self, path):
        self.data_dir = path; self.ref_lbl.setText(self._short(path))
        self._load_from_samples()

    def _load_from_samples(self, force=False):
        """Pull the known-ratio mixtures prepared in Samples (Step 1). Non-destructive:
        fills only when the table is empty, unless ``force`` (a Samples edit)."""
        from dataset import load_mixture_list
        if self._files and not force:
            return
        items = load_mixture_list(self.data_dir)
        if not items:
            return
        self._files = []; self.table.setRowCount(0)
        for it in items:
            self._files.append(it[0])
            row = self.table.rowCount(); self.table.insertRow(row)
            f_item = QTableWidgetItem(os.path.basename(it[0]))
            f_item.setFlags(f_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, f_item)
            self.table.setItem(row, 1, QTableWidgetItem(
                ", ".join(f"{k}:{v:.3g}" for k, v in it[1].items())))
        self.status.setText(f"{len(items)} mixtures from Samples — Validate, or add more")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _toggle_inputs(self, on):
        """Show/hide the input detail (fixed components · VIP · mixture table)."""
        self.inbox.setVisible(on)
        self.tgl.setText(("▾  " if on else "▸  ")
                         + "inputs  (fixed components · VIP bands · mixture table)")

    def _ref_names(self):
        try:
            from dataset import discover_dataset, is_blank
            return [c for c, _m in discover_dataset(self.data_dir) if not is_blank(c)]
        except Exception:
            return []

    def _auto_vip(self):
        """Fill the VIP field with each compound's least-cross-talk marker band(s),
        computed from the references — the same pick Quantify uses."""
        from unmix import compute_vip_bands
        try:
            cfg = load_preprocess(self.data_dir)
            _nb, bands = compute_vip_bands(self.data_dir, baseline=cfg["baseline"],
                                           trim=cfg["trim"])
        except Exception as e:
            self.status.setText(f"VIP failed — {e}")
            self.status.setStyleSheet(f"color:{RED};"); return
        self.vip_txt.setText(", ".join(
            f"{n}:" + "+".join(f"{b:.0f}" for b in bs) for n, bs in bands.items() if bs))
        self.vip_chk.setChecked(True)
        self.status.setText("VIP bands filled — composition will be fit on these bands")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _vip_peak_map(self):
        """Parse the VIP field into {name: [wavenumbers]}, or None when unchecked/empty
        (→ full-spectrum NNLS as before)."""
        if not self.vip_chk.isChecked():
            return None
        out = {}
        for tok in self.vip_txt.text().replace(";", ",").split(","):
            if ":" not in tok:
                continue
            name, rest = tok.split(":", 1); name = name.strip()
            bands = []
            for part in rest.replace("+", " ").split():
                try:
                    bands.append(float(part))
                except ValueError:
                    pass
            if name and bands:
                out[name] = bands
        return out or None

    def _add(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Known-ratio mixture maps", "",
                                                "maps (*.csv *.txt);;all files (*)")
        if not paths:
            return
        refs = self._ref_names(); base = self._base()
        for p in paths:
            if p in self._files:
                continue
            self._files.append(p)
            row = self.table.rowCount(); self.table.insertRow(row)
            f_item = QTableWidgetItem(os.path.basename(p))
            f_item.setFlags(f_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, f_item)
            self.table.setItem(row, 1, QTableWidgetItem(self._guess(p, refs, base)))
        self.status.setText(f"{len(self._files)} mixtures — edit any true ratio, then Validate")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _browse_calib(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Calibration spectra CSV (compound, concentration_M, wavenumbers…)",
            "", "CSV (*.csv)")
        if not p:
            return
        self.calib_path = p
        self.cal_lbl.setText("calib: " + os.path.basename(p)); self.cal_lbl.setStyleSheet("")

    def _base(self):
        return self._parse_true(self.base_txt.text()) or None

    def _guess(self, path, refs, base):
        g = parse_mixture_label(os.path.splitext(os.path.basename(path))[0], refs, base)
        return ", ".join(f"{k}:{v:.3g}" for k, v in g.items()) if g else ""

    def _reparse(self):
        """Re-fill every row's true ratio from its filename + the fixed-components base."""
        refs = self._ref_names(); base = self._base()
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 1, QTableWidgetItem(self._guess(self._files[row], refs, base)))
        self.status.setText("re-parsed filenames with the fixed-components base")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _clear(self):
        self._files = []; self.table.setRowCount(0)
        self.status.setText("cleared"); self.status.setStyleSheet(f"color:{MUTE};")

    @staticmethod
    def _parse_true(text):
        out = {}
        for tok in text.replace(";", ",").split(","):
            if ":" not in tok:
                continue
            k, v = tok.split(":", 1)
            val, _ = parse_amount(v)
            if val is not None:
                out[k.strip()] = val
        return out

    def _items(self):
        """Build (path, ratio_dict, true_conc_or_None). Values with units (100uM, 1e-4M)
        give absolute concentrations → recovery; bare numbers are ratios only."""
        items = []
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 1)
            if not cell:
                continue
            ratio, conc, all_abs = {}, {}, True
            for tok in cell.text().replace(";", ",").split(","):
                if ":" not in tok:
                    continue
                k, v = tok.split(":", 1); k = k.strip()
                val, is_abs = parse_amount(v)
                if val is None:
                    continue
                ratio[k] = val
                if is_abs:
                    conc[k] = val
                else:
                    all_abs = False
            if len(ratio) >= 2:
                tc = conc if (all_abs and len(conc) == len(ratio)) else None
                items.append((self._files[row], ratio, tc))
        return items

    def _update_dl_params(self):
        """Show only the sub-parameters that the chosen composition method uses."""
        m = self.dl_method.currentData()
        self.dl_ep.setVisible(m in ("mlp", "cnn"))
        self.dl_seed.setVisible(m in ("mlp", "cnn", "rf"))
        self.dl_pre.setVisible(m == "mlp")
        self.dl_nc.setVisible(m == "pls")
        self.dl_nt.setVisible(m == "rf")

    def _on_model_bus(self):
        """A model was trained in the Model tab → Recovery applies it (no retraining)."""
        self._shared_model = MODEL_BUS.model
        active = self._shared_model is not None
        self.dl_chk.setText("DL predict (Model-tab model)" if active else "DL predict")
        for w in (self.dl_method, self.dl_ep, self.dl_seed, self.dl_pre, self.dl_nc, self.dl_nt):
            w.setEnabled(not active)                 # applying, not training → knobs inert
        if active:
            self.dl_chk.setChecked(True)

    def _dl_opts(self):
        """DL training knobs shared by 'DL predict' (leave-one-out) and 'Save DL model'."""
        return dict(method=self.dl_method.currentData(),
                    epochs=self.dl_ep.value(), seed=self.dl_seed.value(),
                    use_pretrain=self.dl_pre.isChecked(),
                    n_components=self.dl_nc.value(), n_trees=self.dl_nt.value())

    # ---- run ----
    def _run(self):
        items = self._items()
        if len(items) < 1:
            self.status.setText("add ≥1 mixture with a true ratio (e.g. DQ:1, TBZ:3)")
            self.status.setStyleSheet(f"color:{RED};"); return
        cfg = load_preprocess(self.data_dir)
        params = dict(
            validate=dict(data_dir=self.data_dir, items=items, method="nnls",
                          baseline=cfg["baseline"], trim=cfg["trim"],
                          calib_path=self.calib_path, peak_map=self._vip_peak_map()),
            composition=dict(data_dir=self.data_dir, baseline=cfg["baseline"],
                             files=[it[0] for it in items],
                             nominals=[it[1] for it in items]))
        if self.dl_chk.isChecked():                        # physics-informed DL composition
            if self._shared_model is not None:             # inherit the Model-tab model (no retrain)
                params["shared_model"] = self._shared_model
                params["shared_items"] = items
            else:
                params["dl"] = dict(data_dir=self.data_dir, items=items,
                                    calib_path=self.calib_path, baseline=cfg["baseline"],
                                    trim=cfg["trim"], **self._dl_opts())
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self._cancelled = False; self.cancel_btn.setVisible(True)
        self.status.setText("● starting…"); self.status.setStyleSheet(f"color:{MUTE};")
        self.progbar.setRange(0, 0); self.progbar.setVisible(True)   # busy indicator
        self._thread = QThread(); self._worker = ValidateWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda m: self.status.setText("● " + m))
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _cancel(self):
        """Stop whichever background job is running and free the UI."""
        self._cancelled = True
        for name in ("_thread", "_ethread", "_sthread"):
            th = getattr(self, name, None)
            if th is not None and th.isRunning():
                th.quit()
                if not th.wait(300):
                    th.terminate(); th.wait()
        self.cancel_btn.setVisible(False); self.progbar.setVisible(False)
        self.btn.setEnabled(True); self.btn.setText("Validate")
        self.explain_btn.setEnabled(True); self.explain_btn.setText("DL explain")
        self.savemodel_btn.setEnabled(True); self.savemodel_btn.setText("Save DL model")
        self.status.setText("cancelled"); self.status.setStyleSheet(f"color:{MUTE};")

    # ---- DL explain (interpretability) ----
    def _run_explain(self):
        items = self._items()
        if len(items) < 3:
            self.status.setText("add ≥3 mixtures for DL explain")
            self.status.setStyleSheet(f"color:{RED};"); return
        cfg = load_preprocess(self.data_dir)
        params = dict(data_dir=self.data_dir, items=items, calib_path=self.calib_path,
                      baseline=cfg["baseline"], trim=cfg["trim"])
        self.explain_btn.setEnabled(False); self.explain_btn.setText("Explaining…")
        self._cancelled = False; self.cancel_btn.setVisible(True)
        self.progbar.setRange(0, 0); self.progbar.setVisible(True)
        self.status.setText("● DL explain — training…"); self.status.setStyleSheet(f"color:{MUTE};")
        self._ethread = QThread(); self._eworker = ExplainWorker(params)
        self._eworker.moveToThread(self._ethread)
        self._ethread.started.connect(self._eworker.run)
        self._eworker.progress.connect(lambda m: self.status.setText("● " + m))
        self._eworker.done.connect(self._apply_explain)
        self._eworker.fail.connect(self._error_explain)
        self._eworker.done.connect(self._ethread.quit)
        self._eworker.fail.connect(self._ethread.quit)
        self._ethread.start()

    def _apply_explain(self, r):
        if getattr(self, "_cancelled", False):
            return
        self.explain_btn.setEnabled(True); self.explain_btn.setText("DL explain")
        self.progbar.setVisible(False); self.cancel_btn.setVisible(False)
        self.status.setText("done (DL explain)"); self.status.setStyleSheet(f"color:{MUTE};")
        self._plot_explain(r)

    def _error_explain(self, tb):
        self.explain_btn.setEnabled(True); self.explain_btn.setText("DL explain")
        self.progbar.setVisible(False); self.cancel_btn.setVisible(False)
        self.status.setText("DL explain failed — " + tb.strip().splitlines()[-1][:80])
        self.status.setStyleSheet(f"color:{RED};")

    # ---- train + save a DL model (for the Real-data tab) ----
    def _save_dl_model(self):
        items = self._items()
        if len(items) < 3:
            self.status.setText("add ≥3 mixtures to train a DL model")
            self.status.setStyleSheet(f"color:{RED};"); return
        p, _ = QFileDialog.getSaveFileName(self, "Save DL model", "dl_model.dlm",
                                           "DL model (*.dlm)")
        if not p:
            return
        cfg = load_preprocess(self.data_dir)
        params = dict(data_dir=self.data_dir, items=items, calib_path=self.calib_path,
                      baseline=cfg["baseline"], trim=cfg["trim"], out=p, **self._dl_opts())
        self.savemodel_btn.setEnabled(False); self.savemodel_btn.setText("Saving…")
        self._cancelled = False; self.cancel_btn.setVisible(True)
        self.progbar.setRange(0, 0); self.progbar.setVisible(True)
        self.status.setText("● training DL model…"); self.status.setStyleSheet(f"color:{MUTE};")
        self._sthread = QThread(); self._sworker = SaveModelWorker(params)
        self._sworker.moveToThread(self._sthread)
        self._sthread.started.connect(self._sworker.run)
        self._sworker.progress.connect(lambda m: self.status.setText("● " + m))
        self._sworker.done.connect(self._saved_model)
        self._sworker.fail.connect(self._error_save)
        self._sworker.done.connect(self._sthread.quit)
        self._sworker.fail.connect(self._sthread.quit)
        self._sthread.start()

    def _saved_model(self, info):
        if getattr(self, "_cancelled", False):
            return
        self.savemodel_btn.setEnabled(True); self.savemodel_btn.setText("Save DL model")
        self.progbar.setVisible(False); self.cancel_btn.setVisible(False)
        self.status.setText("DL model saved → " + os.path.basename(info))
        self.status.setStyleSheet(f"color:{MUTE};")

    def _error_save(self, tb):
        self.savemodel_btn.setEnabled(True); self.savemodel_btn.setText("Save DL model")
        self.progbar.setVisible(False); self.cancel_btn.setVisible(False)
        self.status.setText("DL model save failed — " + tb.strip().splitlines()[-1][:80])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _plot_explain(self, r):
        """3-panel interpretability: IG attribution per compound · band permutation
        importance · ligand ablation (drawn into the c_explain canvas)."""
        wnv = r["wn"]; subs = r["subs"]; attr = r["attr"]; vip = r["vip"]
        perm = r["perm"]; abl = r["abl"]
        col = {s: substance_color(s, SUBSTANCES.index(s)) for s in subs}
        fig = self.c_explain.fig; fig.clear()
        n = len(subs)
        gs = fig.add_gridspec(n, 2, width_ratios=[1.1, 1], hspace=0.55, wspace=0.28)
        for c, name in enumerate(subs):
            ax = fig.add_subplot(gs[c, 0])
            ax.plot(wnv, attr[name], color=col[name], lw=0.8)
            ax.fill_between(wnv, attr[name], color=col[name], alpha=0.25)
            for wv in vip.get(name, []):
                ax.axvline(wv, color="#555", ls=":", lw=0.9)
            ax.set_ylabel(name, color=col[name], fontweight="bold", fontsize=9)
            ax.set_xlim(wnv.min(), wnv.max()); ax.set_yticks([])
            if c < n - 1:
                ax.set_xticklabels([])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.axes[0].set_title("IG attribution  (dotted = VIP bands)", fontsize=9, fontweight="bold")
        fig.axes[n - 1].set_xlabel("wavenumber (cm-1)")
        axp = fig.add_subplot(gs[0:max(1, n - 1), 1])
        if len(perm):
            axp.plot(perm[:, 0], np.clip(perm[:, 1], 0, None) * 100, color=INK, lw=1.1)
            axp.fill_between(perm[:, 0], np.clip(perm[:, 1], 0, None) * 100, color="#8b95a1", alpha=0.3)
        for name in subs:
            for wv in vip.get(name, []):
                axp.axvline(wv, color=col[name], ls=":", lw=0.9)
        axp.set_title("band permutation importance", fontsize=9, fontweight="bold")
        axp.set_ylabel("Δ error %"); axp.set_xlim(wnv.min(), wnv.max())
        for s in ("top", "right"):
            axp.spines[s].set_visible(False)
        axa = fig.add_subplot(gs[n - 1, 1])
        x = np.arange(len(subs)); w = 0.38
        axa.bar(x - w / 2, [abl[s][0] for s in subs], w, color=[col[s] for s in subs],
                alpha=0.9, label="full")
        axa.bar(x + w / 2, [abl[s][1] for s in subs], w, color=[col[s] for s in subs],
                alpha=0.35, hatch="//", label="VIP ablated")
        axa.set_xticks(x); axa.set_xticklabels(subs)
        axa.set_title("ligand ablation", fontsize=9, fontweight="bold")
        axa.legend(fontsize=7, framealpha=0.0)
        for s in ("top", "right"):
            axa.spines[s].set_visible(False)
        fig.tight_layout(); self.c_explain.draw_idle()

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Validate")
        self.progbar.setVisible(False); self.cancel_btn.setVisible(False)
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, pair):
        if getattr(self, "_cancelled", False):
            return
        if isinstance(pair, tuple):
            res, cres = pair[0], pair[1]
            dres = pair[2] if len(pair) > 2 else None
        else:
            res, cres, dres = pair, None, None
        # composition view uses the DL prediction when it was requested, else NNLS
        comp = dres if dres else cres
        self._res = res; self._cres = comp; self._is_dl = bool(dres)
        self.btn.setEnabled(True); self.btn.setText("Validate")
        self.progbar.setVisible(False); self.cancel_btn.setVisible(False)
        self.tgl.setChecked(False)                          # auto-collapse inputs → show results
        self.status.setText("done (DL composition)" if dres else "done")
        self.status.setStyleSheet(f"color:{MUTE};")
        names = res.names
        self.k_mix.set(str(len(res.rows)), TEAL)
        self.k_sub.set(str(len(names)), AMBER)
        se = res.response_se or {}
        dom = max(res.response, key=res.response.get)
        self.k_max.set((f"{res.response[dom]:.0f}±{se.get(dom,0):.0f}×" if se.get(dom)
                        else f"{res.response[dom]:.1f}×"), CORAL)
        e0 = self._mean_err([r["obs"] for r in res.rows], res.rows, names)
        e1 = self._mean_err(res.corrected, res.rows, names)
        self.k_err.set(f"{e0:.0%} → {e1:.0%}", BLUE)
        self._plot_parity(res); self._plot_corr(res); self._plot_resp(res)
        if comp:                                          # composition view (NNLS or DL)
            self._plot_triangle(comp)
            self._plot_reldrift(comp); self._plot_recovery(comp)
        rf = "  ·  ".join(f"{n} {res.response[n]:.1f}±{se.get(n,0):.1f}×" for n in names)
        txt = (f"<b>response factors</b> (anchor {res.ref}, mean±SE): {rf}<br>"
               f"<b>{dom}</b> is over-reported on the surface by "
               f"{res.response[dom]:.0f}±{se.get(dom,0):.0f}× — that is why it tends to dominate every map. "
               f"Mean ratio error drops {e0:.0%} → {e1:.0%} after correction.")
        if getattr(res, "calibrated", False) and res.mean_recovery:
            rec = "  ·  ".join(f"{n} {res.mean_recovery[n]:.0f}%" for n in names
                               if np.isfinite(res.mean_recovery.get(n, float('nan'))))
            if rec:
                good = all(80 <= res.mean_recovery[n] <= 120 for n in names
                           if np.isfinite(res.mean_recovery.get(n, float('nan'))))
                col = TEAL if good else CORAL
                txt += (f"<br><b style='color:{col}'>recovery</b> (measured / true "
                        f"concentration): {rec}"
                        + ("  ✓ within 80–120%" if good
                           else "  ⚠ off 80–120% — calibration/competition needs work"))
        else:
            txt += ("<br><span style='color:%s'>load a calibration + enter true "
                    "concentrations (e.g. DQ:100uM) to get recovery %%.</span>" % FAINT)
        if dres and any("uM_pred" in r for r in dres):        # DL order-of-magnitude µM
            lr = []
            for r in dres:
                tp, pp = r.get("uM_true", {}), r.get("uM_pred", {})
                for s in tp:
                    if tp[s] > 0 and s in pp:
                        lr.append(abs(np.log10(max(pp[s], 1e-4) / tp[s])))
            if lr:
                fac = 10 ** float(np.median(lr)); wo = float(np.mean(np.array(lr) < 1))
                txt += ("<br><b style='color:%s'>approx. concentration (DL, semi-quantitative)</b>: "
                        "median within ~%.1f× of true, %.0f%% within order-of-magnitude "
                        "— screening level (not precise µM)." % (BLUE, fac, 100 * wo))
        self.readout.setText(txt)

    @staticmethod
    def _mean_err(fracs, rows, names):
        errs = []
        for f, r in zip(fracs, rows):
            for n in names:
                if r["true"].get(n, 0) > 0 or (f[n] if isinstance(f, dict) else 0) > 0:
                    errs.append(abs((f.get(n, 0.0) if isinstance(f, dict) else 0.0)
                                    - r["true"].get(n, 0.0)))
        return float(np.mean(errs)) if errs else 0.0

    # ---- plots ----
    def _recolor(self):
        """Re-draw all plots when the shared substance colours change."""
        if self._res is not None:
            self._apply((self._res, self._cres))

    def _plot_parity(self, res, corrected=False, canvas=None, title_obs="observed"):
        cv = canvas or self.c_parity
        ax = cv.new_ax()
        names = res.names
        fracs = res.corrected if corrected else [r["obs"] for r in res.rows]
        for i, n in enumerate(names):
            col = substance_color(n, i)
            xs = [r["true"].get(n, 0.0) for r in res.rows]
            ys = [f[n] if isinstance(f, dict) else f[i] for f in fracs]
            ax.scatter(xs, ys, color=col, s=42, edgecolors="white", linewidths=0.6,
                       label=n, zorder=3)
        ax.plot([0, 1], [0, 1], color=MUTE, ls="--", lw=1.0, zorder=1)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")                    # fraction-vs-fraction → square, like the
        ax.set_xlabel("true fraction")            # other parity plots in the app
        ax.set_ylabel(f"{title_obs} fraction")
        ax.legend(fontsize=10, framealpha=0.0, labelcolor="black")
        cv.fig.tight_layout(); cv.draw_idle()

    def _plot_corr(self, res):
        self._plot_parity(res, corrected=True, canvas=self.c_corr,
                          title_obs="corrected (solution)")

    def _plot_resp(self, res):
        ax = self.c_resp.new_ax()
        names = res.names
        vals = [res.response[n] for n in names]
        se = res.response_se or {}
        errs = [se.get(n, 0.0) for n in names]
        cols = [substance_color(names[i], i) for i in range(len(names))]
        x = np.arange(len(names))
        ax.bar(x, vals, color=cols, yerr=errs, capsize=4,
               error_kw=dict(ecolor=INK, elinewidth=0.9))
        ax.axhline(1.0, color=MUTE, ls="--", lw=1.0)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + 0.03 * max(vals), f"{v:.1f}±{e:.1f}×", ha="center",
                    fontsize=8.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("response factor (×)")
        ax.set_ylim(0, max(vals) * 1.2 + 0.2)
        self.c_resp.fig.tight_layout(); self.c_resp.draw_idle()

    # ---- composition view (drift · recovery) ----
    def _recovery(self, res, min_frac=0.03):
        """Per-substance apparent recovery (%) over true mixtures (≥2 nominal
        components). Components below ``min_frac`` (trace, e.g. a 1% ternary spike) are
        EXCLUDED: recovery = pred/true blows up for a tiny denominator (0.01 → 0.04 =
        400%) and makes the metric meaningless. {substance: [pct, ...]}."""
        per = {s: [] for s in SUBSTANCES}
        for r in res:
            nom = r["nominal"]
            if nom is None or int(np.count_nonzero(nom)) < 2:
                continue
            for i, s in enumerate(SUBSTANCES):
                if nom[i] > min_frac:
                    per[s].append(r["mean"][i] / nom[i] * 100)
        return per

    def _tri_frame(self, ax, rec=None):
        A, B, C = bary([1, 0, 0]), bary([0, 0, 1]), bary([0, 1, 0])   # DQ, THI, TBZ
        # faint interior grid parallel to each edge (25/50/75%) → gives the plot a scale
        for t in (0.25, 0.5, 0.75):
            for P, Q, R in [(A, B, C), (B, A, C), (C, A, B)]:
                ax.plot([P[0] + t * (Q[0] - P[0]), P[0] + t * (R[0] - P[0])],
                        [P[1] + t * (Q[1] - P[1]), P[1] + t * (R[1] - P[1])],
                        color=FAINT, lw=0.5, alpha=0.45, zorder=0)
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color=INK, lw=1.0, zorder=1)
        for f, s, ha, va, dx, dy in [([1, 0, 0], "DQ", "center", "bottom", 0, 0.05),
                                     ([0, 0, 1], "THI", "right", "top", -0.03, -0.02),
                                     ([0, 1, 0], "TBZ", "left", "top", 0.03, -0.02)]:
            p = bary(f)
            lab = s
            col = substance_color(s, SUBSTANCES.index(s))
            if rec and rec.get(s):
                v = rec[s]
                mu = float(np.mean(v))
                se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                # over/under tag by arrow (ASCII-safe); in-range gets no tag
                tag = " ↑over" if mu > 120 else (" ↓under" if mu < 80 else "")
                lab = f"{s}\n{mu:.0f}±{se:.0f}%{tag}"
            ax.text(p[0] + dx, p[1] + dy, lab, ha=ha, va=va, fontsize=10, linespacing=1.25,
                    fontweight="bold", color=col)
        ax.set_aspect("equal"); ax.axis("off")

    def _plot_triangle(self, res, annotated=False):
        """Draw the recovery drift triangle. ``annotated`` adds the title + description
        + reading guide — baked in only on EXPORT, so the standalone PNG is readable
        with zero context; in-app the card header already labels it, kept uncluttered."""
        import matplotlib.pyplot as _plt
        cmap = _plt.get_cmap("RdYlGn")                    # green = accurate, red = wrong
        EMAX = 0.6
        ax = self.c_tri.new_ax()
        self._tri_frame(ax, self._recovery(res))
        for rec in res:
            if rec["nominal"] is None:
                continue
            nom = np.asarray(rec["nominal"], float); mn = np.asarray(rec["mean"], float)
            p0 = bary(nom); p1 = bary(mn)
            e = 0.5 * float(np.abs(mn - nom).sum())       # composition error → accuracy colour
            ax.scatter(*p0, s=44, facecolors="none", edgecolors=FAINT,   # real ratio
                       linewidths=1.2, zorder=3)
            if np.linalg.norm(p1 - p0) > 1e-3:
                ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                             color="#8b95a1", lw=1.1, zorder=2,
                                             shrinkA=3, shrinkB=3))
            ax.scatter(*p1, s=52, color=cmap(1 - min(e / EMAX, 1)),      # measured, by accuracy
                       edgecolors="white", linewidths=0.7, zorder=4)
        # legend decodes the marks — SHORT labels in the empty top-left corner so the box
        # never overflows into the DQ apex / recovery labels (a long label overflowed the
        # narrow in-app panel). Colour meaning is a small caption below.
        handles = [
            Line2D([], [], marker="o", mfc="none", mec=FAINT, mew=1.2, ls="", ms=7, label="real"),
            Line2D([], [], marker="o", mfc=cmap(0.85), mec="white", ls="", ms=7, label="measured"),
            Line2D([], [], marker=r"$\rightarrow$", color="#8b95a1", ls="", ms=10, label="drift")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.04, 1.02),
                  fontsize=7.5, framealpha=0.0, handletextpad=0.3, labelspacing=0.25)
        ax.text(0.5, -0.11, "dot colour = accuracy  (green good → red off)",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=FAINT)
        if annotated:                    # export only: full self-documenting caption
            ax.set_title("Composition recovery — real vs measured mixture ratio",
                         fontsize=11.5, fontweight="bold", color=INK, pad=24)
            ax.text(0.5, 1.04, "arrow: real → measured   ·   corner %: recovery "
                    "(100% = perfect, green 80–120%)   ·   closer to a corner = more of "
                    "that substance", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7.5, color=MUTE)
        ax.set_xlim(-0.16, 1.16); ax.set_ylim(-0.12, 1.10)
        self.c_tri.fig.tight_layout(); self.c_tri.draw_idle()

    def _plot_reldrift(self, res):
        recs = [r for r in res if r["nominal"] is not None]
        if not recs:
            self.c_rel.new_ax(); self.c_rel.placeholder("no nominal ratios"); return
        # give each mixture ~19 px of vertical room so the labels never overlap (the panel
        # lives in a scroll area, so a tall figure just scrolls)
        self.c_rel.setMinimumHeight(max(300, len(recs) * 19 + 110))
        ax = self.c_rel.new_ax()
        dom = [int(np.argmax(r["nominal"])) for r in recs]
        gap = np.array([composition_distance(r["nominal"], r["mean"]) for r in recs])
        gap_n = gap / (gap.max() or 1.0) * 100
        bars, ypos, ynames, ycols, spans = [], [], [], [], []
        cursor, first = 0.0, True
        for si in (2, 1, 0):                                   # THI-, TBZ-, DQ-dominant
            members = [k for k in range(len(recs)) if dom[k] == si]
            if not members:
                continue
            if not first:
                cursor -= 0.8
            first = False
            members.sort(key=lambda k: -gap_n[k])
            start = cursor
            for k in members:
                bars.append((cursor, k, si)); ypos.append(cursor)
                ynames.append(recs[k]["name"].replace("_corrected", ""))
                ycols.append(substance_color(SUBSTANCES[si], si))
                cursor -= 1.0
            spans.append((si, (start + cursor + 1.0) / 2))
        # 2- vs 3-component mixtures are different regimes (3-way competition is harder);
        # hatch the ternaries so both read in one chart instead of needing separate runs
        ncomp = [int(sum(1 for v in r["nominal"] if v > 0.02)) for r in recs]
        for y, k, si in bars:
            ax.barh(y, gap_n[k], height=0.66, alpha=0.85,
                    color=substance_color(SUBSTANCES[si], si),
                    hatch="//" if ncomp[k] >= 3 else None,
                    edgecolor="white" if ncomp[k] >= 3 else "none", linewidth=0.0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ynames, fontsize=7 if len(recs) > 24 else 8)
        ax.tick_params(axis="y", length=0)
        for t, c in zip(ax.get_yticklabels(), ycols):
            t.set_color(c)
        for si, yc in spans:
            ax.text(103, yc, f"{SUBSTANCES[si]}-dom", va="center", ha="left", fontsize=9,
                    fontweight="bold", color=substance_color(SUBSTANCES[si], si))
        ax.set_xlim(0, 120)
        ax.set_xlabel("predicted vs real  —  drift (% of worst)")
        if any(n >= 3 for n in ncomp):                     # explain the hatch
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor=MUTE, alpha=0.85, label="2 components"),
                               Patch(facecolor=MUTE, alpha=0.85, hatch="//",
                                     edgecolor="white", label="3 components")],
                      fontsize=8, framealpha=0, loc="lower right")
        self.c_rel.fig.tight_layout(); self.c_rel.draw_idle()

    def _plot_recovery(self, res):
        ax = self.c_rec.new_ax()
        per = self._recovery(res)
        x = np.arange(len(SUBSTANCES))
        for i, s in enumerate(SUBSTANCES):
            col = substance_color(s, i)
            v = per[s]
            mu = float(np.mean(v)) if v else 0.0
            se = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
            ax.bar(i, mu, width=0.6, color=col, alpha=0.45, yerr=se, capsize=4,
                   error_kw=dict(ecolor=INK, elinewidth=0.9))
            if v:
                ax.scatter(np.full(len(v), i), v, s=18, color=col,
                           edgecolors="white", linewidths=0.5, zorder=3)
                ax.text(i, max(v) + 4, f"{mu:.0f}±{se:.0f}%", ha="center", va="bottom",
                        fontsize=9, color=INK)
        ax.axhline(100, color=MUTE, ls="--", lw=1.0)
        ax.set_xticks(x); ax.set_xticklabels(SUBSTANCES)
        ax.set_ylabel("apparent recovery (%)")
        allv = [w for s in SUBSTANCES for w in per[s]] + [100]
        ax.set_ylim(0, max(allv) * 1.15 + 8)
        self.c_rec.fig.tight_layout(); self.c_rec.draw_idle()

    # ---- export ----
    def _export(self):
        if self._res is None:
            self.status.setText("validate first, then export"); self.status.setStyleSheet(f"color:{RED};"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        res = self._res
        rse = res.response_se or {}
        write_csv(os.path.join(d, "response_factors.csv"),
                  ["substance", "response_factor", "SE", "anchor"],
                  [[n, f"{res.response[n]:.5f}", f"{rse.get(n, 0.0):.5f}", res.ref]
                   for n in res.names])
        cal = getattr(res, "calibrated", False)
        head = ["mixture"] + [f"true_{n}" for n in res.names] \
            + [f"obs_{n}" for n in res.names] + [f"corr_{n}" for n in res.names] \
            + ([f"trueM_{n}" for n in res.names] + [f"measM_{n}" for n in res.names]
               + [f"recovery%_{n}" for n in res.names] if cal else [])
        rows = []
        rec_list = res.recovery or [{}] * len(res.rows)
        for r, corr, rec in zip(res.rows, res.corrected, rec_list):
            row = [os.path.basename(r["path"])] \
                + [f"{r['true'].get(n, 0):.4f}" for n in res.names] \
                + [f"{r['obs'].get(n, 0):.4f}" for n in res.names] \
                + [f"{corr[n]:.4f}" for n in res.names]
            if cal:
                tc = r.get("true_conc") or {}; me = r.get("meas") or {}
                row += [f"{tc.get(n, ''):.4e}" if tc.get(n) else "" for n in res.names] \
                    + [f"{me.get(n, ''):.4e}" if me.get(n) is not None else "" for n in res.names] \
                    + [f"{rec.get(n, ''):.1f}" if rec.get(n) is not None else "" for n in res.names]
            rows.append(row)
        write_csv(os.path.join(d, "validation_table.csv"), head, rows)
        if self._cres:                                   # composition data (redraws triangle/recovery/drift)
            crecs = [r for r in self._cres if r.get("nominal") is not None]
            chead = ["mixture"] + [f"true_{s}" for s in SUBSTANCES] + \
                    [f"measured_{s}" for s in SUBSTANCES] + \
                    [f"recovery_{s}_pct" for s in SUBSTANCES] + ["drift"]
            crows = []
            for r in crecs:
                nom = np.asarray(r["nominal"], float); mn = np.asarray(r["mean"], float)
                rec = [f"{mn[i] / nom[i] * 100:.1f}" if nom[i] > 0 else "" for i in range(len(SUBSTANCES))]
                crows.append([r["name"]] + [f"{v:.4f}" for v in nom] + [f"{v:.4f}" for v in mn] +
                             rec + [f"{composition_distance(nom, mn):.4f}"])
            write_csv(os.path.join(d, "composition_view.csv"), chead, crows)
        figs = [("validate_parity", self.c_parity),
                ("validate_corrected", self.c_corr),
                ("validate_response", self.c_resp)]
        if self._cres:                                   # composition view
            figs += [("drift_triangle", self.c_tri),
                     ("relative_drift", self.c_rel), ("recovery", self.c_rec)]
            self._plot_triangle(self._cres, annotated=True)   # bake caption into the PNG
        n = _save_figs(figs, d)
        if self._cres:
            self._plot_triangle(self._cres)                   # restore clean in-app view
            n += self._export_pub_triangles(d)                # publication-style variants
        self._export_readme(d, res, [f[0] for f in figs], is_dl=getattr(self, "_is_dl", False))
        self.status.setText(f"exported README + response_factors.csv + table + {n} PNG "
                            f"→ {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _export_pub_triangles(self, d):
        """Also write the publication-style ternaries (accuracy field + RGB composition)
        from the same composition data the in-app drift triangle uses."""
        import os as _os
        try:
            from triangle_figs import accuracy_triangle, rgb_triangle
        except Exception as e:
            print(e, file=sys.stderr); return 0
        recs = [r for r in self._cres if r.get("nominal") is not None]
        if not recs:
            return 0
        rows = [(r["name"], list(np.asarray(r["nominal"], float)),
                 list(np.asarray(r["mean"], float))) for r in recs]
        cols = [substance_color(s, i) for i, s in enumerate(SUBSTANCES)]
        n = 0
        for fn, name in ((accuracy_triangle, "drift_triangle_accuracy"),
                         (rgb_triangle, "drift_triangle_rgb")):
            try:
                fig = fn(rows, list(SUBSTANCES), cols)
                fig.savefig(_os.path.join(d, name + ".png"), dpi=130, facecolor="white",
                            bbox_inches="tight")
                n += 1
            except Exception as e:
                print(e, file=sys.stderr)
        return n

    def _export_readme(self, d, res, fig_names, is_dl=False):
        """Write README.md describing WHAT the export is, HOW it was produced (settings),
        and the RESULT numbers — so a PNG/CSV read out of context is still understandable."""
        names = res.names
        cfg = load_preprocess(self.data_dir)
        pm = self._vip_peak_map()
        vip = ("VIP marker bands only — "
               + " · ".join(f"{k} {'+'.join(f'{b:.0f}' for b in v)}" for k, v in pm.items())
               if pm else "the full spectrum")
        trim = cfg.get("trim")
        window = f"{trim[0]:.0f}–{trim[1]:.0f} cm⁻¹" if trim else "full range"
        rse = res.response_se or {}
        rf = sorted(res.response.items(), key=lambda kv: kv[1], reverse=True)
        rf_str = " · ".join(f"{k} {v:.1f}±{rse.get(k, 0):.1f}×" for k, v in rf)
        e0 = self._mean_err([r["obs"] for r in res.rows], res.rows, names)
        e1 = self._mean_err(res.corrected, res.rows, names)
        rec_str = "not computed (load a calibration + enter true concentrations)"
        if res.mean_recovery:
            rec_str = " · ".join(
                f"{n} {res.mean_recovery[n]:.0f}%" for n in names
                if np.isfinite(res.mean_recovery.get(n, float("nan")))) + \
                "   (100% = perfect; 80–120% acceptable)"
        comp_src = ("physics-informed DL (leave-one-map-out prediction)" if is_dl
                    else "NNLS surface unmixing")
        fig_docs = {
            "drift_triangle": f"real (○) vs measured (●) composition per mixture [{comp_src}]; "
                              "arrow = drift real→measured; corner % = recovery.",
            "validate_response": "response factor per substance (× relative; higher = "
                                 "over-reported on the surface).",
            "validate_parity": "observed (surface) ratio vs true ratio — above the line = over-reported.",
            "validate_corrected": "corrected (solution) ratio vs true ratio — should sit on the line.",
            "relative_drift": f"predicted vs real drift, grouped by dominant substance [{comp_src}].",
            "recovery": f"apparent recovery % per substance (mean ± SE, trace <3% excluded) [{comp_src}].",
        }
        sections = {
            "What this is": [
                "Known-ratio mixtures unmixed against the pure references to check whether "
                "the MEASURED composition matches the TRUE (solution) ratio, and to recover "
                "each substance's response factor (relative surface sensitivity). A "
                "high-response substance dominates the surface signal even in a balanced "
                "mixture; the response factors convert an observed surface ratio back to a "
                "solution ratio."],
            "How it was produced": [
                f"- References: {self.data_dir}",
                f"- Mixtures: {len(res.rows)} known-ratio maps",
                f"- Response-factor unmixing: NNLS against pure reference templates, fit on {vip}",
                f"- Composition view (triangle · recovery · drift): "
                + ("physics-informed DL, leave-one-map-out over the loaded mixtures "
                   "(spectrum→composition MLP, physics-pretrained)" if is_dl
                   else "NNLS surface composition"),
                f"- Baseline removal: {'on' if cfg.get('baseline') else 'off'}; "
                f"spectral window: {window}",
                f"- Response factors anchored to {res.ref} (its factor ≈ 1)",
                f"- Calibration: {os.path.basename(self.calib_path) if self.calib_path else 'none (ratio only)'}"],
            "Results": [
                f"- Response factors (×, higher = over-reported): {rf_str}",
                f"- Recovery (measured / true): {rec_str}",
                f"- Mean composition error: {e0:.0%} → {e1:.0%} after response-factor correction."],
            "Data files (every figure is redrawable from these)": [
                "- `response_factors.csv` — response factor ± SE per substance (→ validate_response).",
                "- `validation_table.csv` — true / observed / corrected ratio + recovery per mixture "
                "(→ validate_parity, validate_corrected).",
                "- `composition_view.csv` — true & measured composition, per-substance recovery %, and "
                "drift per mixture (→ drift_triangle, relative_drift, recovery).",
                "Each PNG is a rendering of one of these tables — re-plot from the CSV in any tool."],
        }
        figures = [(fn, fig_docs[fn]) for fn in fig_names if fn in fig_docs]
        write_readme(d, "UNMIXR — Recovery export", sections, figures)
