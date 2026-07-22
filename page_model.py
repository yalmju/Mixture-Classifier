"""page_model.py — Model tab: train a classifier on the reference maps."""
from __future__ import annotations

import os
import re
import sys
import traceback

import numpy as np
from matplotlib.patches import Patch

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSpinBox, QComboBox, QCheckBox, QFileDialog, QProgressBar,
)

from ui_common import *
from model_training import train_model, TrainResult
from real_data import PEST_DEFAULT
from dataset import load_preprocess
from io_utils import write_csv


class TrainWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.done.emit(train_model(progress=self.progress.emit, **self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


# --------------------------------------------------------------------------
# Model page
# --------------------------------------------------------------------------
class ModelPage(QWidget):
    ALGOS = [("RandomForest", "rf"), ("ResNet1D (torch)", "resnet"),
             ("SVM (RBF)", "svm"), ("k-NN", "knn"),
             ("Logistic Reg.", "logreg"), ("Gradient Boosting", "gbm"),
             ("PLS-DA (VIP)", "pls")]

    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self._train_params = None
        self.pest_dir = PEST_DEFAULT
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(14)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Model training"); h1.setObjectName("h1")
        sub = QLabel("Step 2 — train. Uses the train/test division and "
                     "preprocessing you already set in Samples (split = manual). "
                     "Pick an algorithm; read the learning curve, confusion matrix, "
                     "per-class F1, PCA and discriminative bands. The other splits "
                     "(spatial / random / batch-CV) are there for comparison only.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        # ---- controls: row 1 = model + data + actions ----
        ctl = QHBoxLayout(); ctl.setSpacing(10)
        ctl.addLayout(self._combo_col("algorithm", "cmb", self.ALGOS))
        self.sp_ep = self._spin(QSpinBox(), 2, 100, 25, "epochs (ResNet)")
        self.sp_tr = self._spin(QSpinBox(), 60, 600, 300, "trees (RF)", step=20)
        self.sp_seed = self._spin(QSpinBox(), 0, 999, 0, "seed")
        for w in (self.sp_ep, self.sp_tr, self.sp_seed):
            ctl.addLayout(w)
        self.src = QLabel(self._short(self.pest_dir)); self.src.setObjectName("field")
        ctl.addWidget(self.src, 1)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Train + evaluate"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._train)
        ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        # ---- controls: row 2 = split only (preprocessing comes from Samples) ----
        ctl2 = QHBoxLayout(); ctl2.setSpacing(10)
        ctl2.addLayout(self._combo_col("split", "cmb_split",
                                       [("manual (Samples train/test)", "manual"),
                                        ("spatial (honest)", "spatial"),
                                        ("random (leaky)", "random"),
                                        ("batch (leave-1-out)", "batch"),
                                        ("batch-CV (mean±SD)", "batch-cv")]))
        self.sp_test = self._spin(QSpinBox(), 5, 90, 50, "test %", step=5)
        ctl2.addLayout(self.sp_test)
        self.prep_lbl = QLabel(""); self.prep_lbl.setObjectName("field")
        ctl2.addSpacing(8); ctl2.addWidget(self.prep_lbl)
        ctl2.addStretch(1)
        root.addLayout(ctl2)
        self._refresh_prep()

        # progress bar — visible (busy) only while training, so it never reads as frozen
        self.pbar = QProgressBar(); self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(6); self.pbar.setVisible(False)
        root.addWidget(self.pbar)

        # KPI row
        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_acc = Kpi("test accuracy"); self.k_f1 = Kpi("macro F1")
        self.k_tr = Kpi("train pixels"); self.k_te = Kpi("test pixels")
        for k in (self.k_acc, self.k_f1, self.k_tr, self.k_te):
            kpis.addWidget(k)
        root.addLayout(kpis)

        # plot grid: 2x2 + a full-width discriminative-band row
        grid = QGridLayout(); grid.setSpacing(12)
        self.c_curve = Canvas(); self.c_cm = Canvas()
        self.c_pca = Canvas(); self.c_bar = Canvas(); self.c_bands = Canvas()
        self.c_box = Canvas()
        for (cv, title, r, c) in [
            (self.c_curve, "Learning curve", 0, 0),
            (self.c_cm, "Confusion matrix (held-out test)", 0, 1),
            (self.c_pca, "PCA — real per-pixel spectra by class", 1, 0),
            (self.c_bar, "Per-class precision / recall / F1", 1, 1),
        ]:
            card, lay = _card(title)
            lay.addWidget(cv)
            grid.addWidget(card, r, c)
        bcard, blay = _card("Discriminative bands — ANOVA F per wavenumber "
                            "(or PLS-DA VIP when that backend is used)")
        blay.addWidget(self.c_bands); grid.addWidget(bcard, 2, 0, 1, 2)
        xcard, xlay = _card("Top bands — intensity by class (box plot: which "
                            "substance is high at each discriminative peak)")
        xlay.addWidget(self.c_box); grid.addWidget(xcard, 3, 0, 1, 2)
        for rr in (0, 1, 2, 3):
            grid.setRowStretch(rr, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        root.addLayout(grid, 1)

        for cv, msg in [(self.c_curve, "Train to watch the learning curve"),
                        (self.c_cm, "Train to compute confusion matrix"),
                        (self.c_pca, "Train to compute PCA"),
                        (self.c_bar, "Train to compute per-class F1"),
                        (self.c_bands, "Train to rank discriminative bands (ANOVA F)"),
                        (self.c_box, "Train to see intensity-by-class at the top bands")]:
            cv.placeholder(msg)

    def _short(self, p):
        tail = "…" + p[-40:] if len(p) > 40 else p
        return f"dataset (from Samples): {tail}"

    def set_data_dir(self, path):
        """Adopt the dataset folder chosen in Samples (single source of truth)."""
        self.pest_dir = path; self.src.setText(self._short(path)); self._refresh_prep()

    def _spin(self, spin, lo, hi, val, label, step=1):
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        spin.setSingleStep(step)
        spin.setRange(lo, hi); spin.setValue(val)
        col.addWidget(lb); col.addWidget(spin)
        return col

    def _combo_col(self, label, attr, items):
        """A labelled combo box; stores the widget on self.<attr>."""
        col = QVBoxLayout(); col.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("field")
        cb = QComboBox()
        for text, data in items:
            cb.addItem(text, data)
        setattr(self, attr, cb)
        col.addWidget(lb); col.addWidget(cb)
        return col

    def _refresh_prep(self):
        """Show the preprocessing that Samples saved for this folder (read-only)."""
        cfg = load_preprocess(self.pest_dir)
        deriv = {0: "none", 1: "1st", 2: "2nd"}.get(cfg["deriv"], "none")
        trim = f"{cfg['trim'][0]}–{cfg['trim'][1]}" if cfg["trim"] else "full"
        base = "ALS" if cfg["baseline"] else "no-baseline"
        self.prep_lbl.setText(
            f"preprocessing (from Samples): {base} · deriv {deriv} · "
            f"{cfg['norm'].upper()} · trim {trim}")

    def _export(self):
        if self._res is None:
            self.src.setText("run first, then export"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res
        # --- per-graph CSVs (each plot gets its underlying numbers) ---
        rows = [[nm, f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.4f}", v[3]]
                for nm, v in r.per_component.items()]
        rows.append(["accuracy", f"{r.acc:.4f}", "", f"{r.macro_f1:.4f}", ""])
        write_csv(os.path.join(d, "model_metrics.csv"),
                  ["class", "precision", "recall", "f1", "support"], rows)
        write_csv(os.path.join(d, "model_confusion.csv"),
                  [""] + list(r.classes),
                  [[r.classes[i]] + list(map(int, r.confusion[i]))
                   for i in range(len(r.classes))])
        write_csv(os.path.join(d, "model_learning_curve.csv"),
                  [r.curve_xlabel, r.curve_label],
                  [[f"{x:g}", f"{y:.6f}"] for x, y in zip(r.curve_x, r.curve_y)])
        write_csv(os.path.join(d, "model_pca.csv"),
                  ["class", "PC1", "PC2"],
                  [[r.classes[int(lab)], f"{e[0]:.6f}", f"{e[1]:.6f}"]
                   for e, lab in zip(r.pca_emb, r.pca_lab)])
        if r.wn is not None and r.band_f is not None:
            has_vip = r.vip is not None and len(r.vip) == len(r.wn)
            head = ["wavenumber", "anova_f"] + (["vip"] if has_vip else [])
            brows = [[f"{r.wn[i]:.2f}", f"{r.band_f[i]:.6f}"]
                     + ([f"{r.vip[i]:.6f}"] if has_vip else [])
                     for i in range(len(r.wn))]
            write_csv(os.path.join(d, "model_bands.csv"), head, brows)
        # top-band intensity-by-class (box-plot data): per band, per class stats
        if r.box_wn is not None and r.box_vals is not None:
            brows = []
            for j, w in enumerate(r.box_wn):
                for c, nm in enumerate(r.classes):
                    v = r.box_vals[r.box_lab == c, j]
                    if len(v):
                        brows.append([f"{w:.2f}", nm, f"{v.mean():.6f}",
                                      f"{np.median(v):.6f}", f"{v.std():.6f}"])
            write_csv(os.path.join(d, "model_top_bands_by_class.csv"),
                      ["wavenumber", "class", "mean", "median", "std"], brows)
        # --- PNGs (one per plot) ---
        n = _save_figs([("model_learning_curve", self.c_curve),
                        ("model_confusion", self.c_cm),
                        ("model_pca", self.c_pca), ("model_prf", self.c_bar),
                        ("model_bands", self.c_bands), ("model_top_bands_box", self.c_box)], d)
        # --- the fitted model itself, so it can be reused later ---
        saved = self._save_model(d, r)
        tail = f" + {saved}" if saved else ""
        self.src.setText(f"exported CSV + {n} PNG{tail} → {os.path.basename(d)}")

    def _save_model(self, d, r):
        """Persist the fitted estimator (+ classes and preprocessing) so it can be
        loaded and applied to new maps later. Returns the saved filename or ""."""
        model = getattr(r, "model", None)
        if model is None:                                 # e.g. batch-CV has no single model
            return ""
        bundle = dict(model=model, backend=r.backend, classes=list(r.classes),
                      comps=list(r.comps), wn=r.wn,
                      preprocessing=self._train_params)
        try:
            import joblib
            joblib.dump(bundle, os.path.join(d, "unmixr_model.joblib"))
            return "model(.joblib)"
        except Exception:
            try:                                          # torch fallback for ResNet
                import torch
                torch.save({"state_dict": model.state_dict(),
                            "classes": list(r.classes)},
                           os.path.join(d, "unmixr_model.pt"))
                return "model(.pt)"
            except Exception as exc:
                print("model save failed:", exc, file=sys.stderr)
                return ""

    # ---- training ----
    def _train(self):
        cfg = load_preprocess(self.pest_dir)              # preprocessing set in Samples
        self._refresh_prep()
        params = dict(pest_dir=self.pest_dir, backend=self.cmb.currentData(),
                      epochs=self.sp_ep.itemAt(1).widget().value(),
                      n_estimators=self.sp_tr.itemAt(1).widget().value(),
                      seed=self.sp_seed.itemAt(1).widget().value(),
                      baseline=cfg["baseline"], trim=cfg["trim"],
                      deriv=cfg["deriv"], norm=cfg["norm"],
                      split=self.cmb_split.currentData(),
                      test_frac=self.sp_test.itemAt(1).widget().value() / 100.0)
        self._train_params = dict(params)                 # remember for model export
        self.btn.setEnabled(False); self.btn.setText("Training…")
        self.c_curve.placeholder("Training…")
        self.pbar.setVisible(True); self.pbar.setRange(0, 0)   # busy until first step
        self._thread = QThread()
        self._worker = TrainWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _progress(self, msg):
        # live status so a long run reads as working, not frozen
        self.btn.setText("Training…  " + msg.split("  ")[0])
        self.c_curve.placeholder("● " + msg)
        m = re.search(r"(\d+)\s*/\s*(\d+)", msg)          # "epoch 12/25" -> determinate
        if m:
            self.pbar.setRange(0, int(m.group(2))); self.pbar.setValue(int(m.group(1)))
        else:
            self.pbar.setRange(0, 0)                       # loading / finalising -> busy

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Train + evaluate")
        self.pbar.setVisible(False)
        first = tb.strip().splitlines()[-1][:90]
        self.c_curve.placeholder("Training failed — " + first)
        print(tb, file=sys.stderr)

    def _apply(self, res: TrainResult):
        self._res = res
        self.btn.setEnabled(True); self.btn.setText("Train + evaluate")
        self.pbar.setVisible(False)
        # random split is leaky; batch-CV reports cross-fold mean ± SD
        leaky = getattr(res, "split", "spatial") == "random"
        if getattr(res, "split", "") == "batch-cv" and getattr(res, "acc_std", 0):
            self.k_acc.set(f"{res.acc:.0%} ±{res.acc_std:.0%}", TEAL)
        else:
            self.k_acc.set(f"{res.acc:.0%}" + ("  ⚠leaky" if leaky else ""),
                           CORAL if leaky else TEAL)
        self.k_f1.set(f"{res.macro_f1:.3f}", BLUE)
        self.k_tr.set(f"{res.n_train:,}", AMBER)
        self.k_te.set(f"{res.n_test:,}", PURPLE)
        self._plot_curve(res); self._plot_cm(res)
        self._plot_pca(res); self._plot_bar(res); self._plot_bands(res)
        self._plot_box(res)

    # ---- plots ----
    def _plot_curve(self, res):
        ax = self.c_curve.new_ax()
        col = TEAL if res.backend == "resnet" else BLUE
        ax.plot(res.curve_x, res.curve_y, marker="o", ms=4, lw=1.4, color=col)
        ax.set_xlabel(res.curve_xlabel); ax.set_ylabel(res.curve_label)
        ax.set_ylim(bottom=0)
        self.c_curve.fig.tight_layout(); self.c_curve.draw_idle()

    def _plot_cm(self, res):
        ax = self.c_cm.new_ax()
        cm = res.confusion; names = res.classes
        # row-normalise: colour + label by % of each true class, so unequal class
        # sizes (e.g. a 1-batch class with far fewer test pixels) don't make the raw
        # counts look arbitrary. The count is kept in parentheses below the %.
        row = cm.sum(axis=1, keepdims=True)
        frac = np.divide(cm, row, out=np.zeros(cm.shape, float), where=row > 0)
        ax.imshow(frac, cmap=CM_CMAP, aspect="equal", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
        ax.grid(which="minor", color=PANEL, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j]:
                    ax.text(j, i, f"{frac[i, j] * 100:.0f}%\n({cm[i, j]})",
                            ha="center", va="center", fontsize=7,
                            color="#ffffff" if frac[i, j] > 0.5 else INK)
        self.c_cm.fig.tight_layout(); self.c_cm.draw_idle()

    def _plot_pca(self, res):
        ax = self.c_pca.new_ax()
        for i, nm in enumerate(res.classes):
            m = res.pca_lab == i
            if m.any():
                ax.scatter(res.pca_emb[m, 0], res.pca_emb[m, 1], s=12,
                           color=SERIES[i % len(SERIES)], alpha=0.6,
                           edgecolors="none", label=nm)
        var = getattr(res, "pca_var", None)
        if var is not None and len(var) >= 2:
            ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
            ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
        else:
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="best", framealpha=0.0, labelcolor=MUTE)
        self.c_pca.fig.tight_layout(); self.c_pca.draw_idle()

    def _plot_bar(self, res):
        ax = self.c_bar.new_ax()
        names = res.classes
        x = np.arange(len(names)); w = 0.26
        P = [res.per_component[n][0] for n in names]
        R = [res.per_component[n][1] for n in names]
        F = [res.per_component[n][2] for n in names]
        ax.bar(x - w, P, w, color=BLUE, label="precision")
        ax.bar(x, R, w, color=AMBER, label="recall")
        ax.bar(x + w, F, w, color=TEAL, label="F1")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
        ax.set_ylim(0, 1.05); ax.set_ylabel("score")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=3)
        self.c_bar.fig.tight_layout(); self.c_bar.draw_idle()

    def _plot_bands(self, res):
        ax = self.c_bands.new_ax()
        wn = res.wn
        # prefer PLS-DA VIP when present, else the always-computed ANOVA F
        vip = getattr(res, "vip", None)
        use_vip = vip is not None and wn is not None and len(vip) == len(wn)
        f = vip if use_vip else getattr(res, "band_f", None)
        if f is None or wn is None or len(f) != len(wn):
            self.c_bands.placeholder("no band statistics"); return
        ax.plot(wn, f, lw=1.0, color=PURPLE)
        ax.fill_between(wn, f, color=PURPLE, alpha=0.15)
        if use_vip:                                       # VIP > 1 = important band
            ax.axhline(1.0, color=MUTE, lw=0.8, ls="--")
            ax.text(wn[-1], 1.0, " VIP=1", fontsize=6, color=MUTE, va="bottom", ha="right")
        # label the strongest discriminative bands with their cm⁻¹
        k = min(6, len(f))
        top = np.argsort(f)[-k:]
        for idx in top:
            ax.annotate(f"{wn[idx]:.0f}", (wn[idx], f[idx]), fontsize=7, color=INK,
                        ha="center", va="bottom",
                        xytext=(0, 2), textcoords="offset points")
            ax.plot([wn[idx]], [f[idx]], "o", ms=3, color=CORAL)
        ax.set_xlabel("wavenumber (cm⁻¹)")
        ax.set_ylabel("PLS-DA VIP" if use_vip else "ANOVA F")
        ax.margins(y=0.15)
        self.c_bands.fig.tight_layout(); self.c_bands.draw_idle()

    def _plot_box(self, res):
        """Grouped box plot: for each top discriminative band, the intensity
        distribution of every class — so you can read off which substance is high
        at each good peak (and thus why the band discriminates)."""
        ax = self.c_box.new_ax()
        wn = getattr(res, "box_wn", None)
        vals = getattr(res, "box_vals", None)
        lab = getattr(res, "box_lab", None)
        if wn is None or vals is None or lab is None or len(wn) == 0:
            self.c_box.placeholder("no band statistics"); return
        classes = res.classes; nC = len(classes)
        k = len(wn); width = 0.8 / nC
        for c in range(nC):
            data = [vals[lab == c, j] for j in range(k)]
            data = [d if len(d) else [0.0] for d in data]
            pos = np.arange(k) + (c - (nC - 1) / 2) * width
            col = SERIES[c % len(SERIES)]
            bp = ax.boxplot(data, positions=pos, widths=width * 0.9,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK, linewidth=0.8))
            for box in bp["boxes"]:
                box.set(facecolor=col, edgecolor=col, alpha=0.65)
            for w in ("whiskers", "caps"):
                for ln in bp[w]:
                    ln.set(color=col, linewidth=0.8)
        ax.set_xticks(np.arange(k))
        ax.set_xticklabels([f"{w:.0f}" for w in wn], fontsize=8)
        ax.set_xlabel("top discriminative band (cm⁻¹)")
        ax.set_ylabel("intensity")
        ax.legend(handles=[Patch(facecolor=SERIES[c % len(SERIES)], label=classes[c])
                           for c in range(nC)],
                  fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=min(nC, 4))
        self.c_box.fig.tight_layout(); self.c_box.draw_idle()
