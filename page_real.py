"""page_real.py — Real data tab: single-component / mixture / composition / calib."""
from __future__ import annotations

import os
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFileDialog,
)
from matplotlib.patches import Patch, Rectangle

from ui_common import *
from real_data import compute_real, PEST_DEFAULT
from io_utils import write_csv


class RealWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)

    def __init__(self, pest_dir):
        super().__init__()
        self.pest_dir = pest_dir

    def run(self):
        try:
            self.done.emit(compute_real(self.pest_dir))
        except Exception:
            self.fail.emit(traceback.format_exc())


# --------------------------------------------------------------------------
# Real-data page
# --------------------------------------------------------------------------
class RealDataPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self.pest_dir = PEST_DEFAULT
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(12)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Real-data analysis"); h1.setObjectName("h1")
        sub = QLabel("Single-component classification, mixture detection (per-pixel), "
                     "composition confusion, and response-factor correction on your "
                     "loaded maps.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        self.folder_lbl = QLabel(self._short(self.pest_dir)); self.folder_lbl.setObjectName("field")
        browse = QPushButton("Data folder…"); browse.setObjectName("ghost")
        browse.clicked.connect(self._browse)
        self.status = QLabel(""); self.status.setObjectName("sub")
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost")
        exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Run analysis"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(browse); ctl.addWidget(self.folder_lbl, 1)
        ctl.addWidget(self.status); ctl.addStretch(1)
        ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self.k_pure = Kpi("single-component acc")
        self.k_f1 = Kpi("mixture detection F1")
        self.k_combo = Kpi("exact composition")
        self.k_r = Kpi("dominant response  R")
        for k in (self.k_pure, self.k_f1, self.k_combo, self.k_r):
            kpis.addWidget(k)
        root.addLayout(kpis)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_pca = Canvas(); self.c_pure = Canvas(); self.c_combo = Canvas()
        self.c_det = Canvas(); self.c_cal = Canvas(); self.c_strat = Canvas()
        for (cv, title, r, c) in [
            (self.c_pca, "PCA — real per-pixel spectra", 0, 0),
            (self.c_pure, "Single-component confusion  (spatial split)", 0, 1),
            (self.c_strat, "Detection strategy — RF vs per-pixel vs matched", 0, 2),
            (self.c_det, "Detection outcome  (per-pixel voting)", 1, 0),
            (self.c_combo, "Composition confusion  (per-pixel detector)", 1, 1),
            (self.c_cal, "Ratio: raw vs response-calibrated", 1, 2),
        ]:
            card, lay = _card(title); lay.addWidget(cv)
            grid.addWidget(card, r, c)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        root.addLayout(grid, 1)
        for cv, m in [(self.c_pca, "Run for PCA"),
                      (self.c_pure, "Run to classify pure substances"),
                      (self.c_strat, "Run to compare strategies"),
                      (self.c_det, "Run to show detection outcomes"),
                      (self.c_combo, "Run to build composition confusion"),
                      (self.c_cal, "Run to correct the ratios")]:
            cv.placeholder(m)

    def _short(self, p):
        return "…" + p[-42:] if len(p) > 42 else p

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Data folder (Reference/ + Ratio/ maps)",
                                             self.pest_dir)
        if d:
            self.pest_dir = d; self.folder_lbl.setText(self._short(d))

    def _export(self):
        if self._res is None:
            self.status.setText("run first, then export")
            self.status.setStyleSheet(f"color:{RED};"); return
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        r = self._res
        # per-mixture detection (true vs per-pixel predicted)
        det = []
        for k, nm in enumerate(r.mix_names):
            t = "+".join(c for i, c in enumerate(r.comps) if r.yt[k, i])
            p = "+".join(c for i, c in enumerate(r.comps) if r.yp[k, i])
            det.append([nm, t, p, "hit" if t == p else "miss"])
        write_csv(os.path.join(d, "detection.csv"),
                  ["mixture", "true", "predicted", "exact"], det)
        write_csv(os.path.join(d, "strategies.csv"),
                  ["strategy", "recall", "precision", "f1", "exact"],
                  [[s[0], f"{s[1]:.3f}", f"{s[2]:.3f}", f"{s[3]:.3f}", f"{s[4]:.3f}"]
                   for s in r.strategies])
        write_csv(os.path.join(d, "response_factors.csv"),
                  ["compound", "R"], [[c, f"{r.R[i]:.3f}"]
                                      for i, c in enumerate(r.comps)])
        n = _save_figs([("real_pca", self.c_pca), ("real_pure_confusion", self.c_pure),
                        ("real_strategy", self.c_strat), ("real_detection", self.c_det),
                        ("real_composition", self.c_combo),
                        ("real_calibration", self.c_cal)], d)
        self.status.setText(f"exported 3 CSV + {n} PNG → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _run(self):
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self.status.setText("")
        self._thread = QThread(); self._worker = RealWorker(self.pest_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Run analysis")
        first = tb.strip().splitlines()[-1][:80]
        self.status.setText("failed — " + first)
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, r):
        self._res = r
        self.btn.setEnabled(True); self.btn.setText("Run analysis")
        self.status.setText("done"); self.status.setStyleSheet(f"color:{MUTE};")
        self.k_pure.set(f"{r.acc4:.0%}", TEAL)
        self.k_f1.set(f"{r.micro['micro_f1']:.2f}", BLUE)
        self.k_combo.set(f"{r.combo_exact:.0%}", AMBER)
        di = int(np.argmax(r.R))
        self.k_r.set(f"{r.comps[di]} {r.R[di]:.1f}×", CORAL)
        self._plot_pca(r); self._plot_pure(r); self._plot_strat(r)
        self._plot_det(r); self._plot_combo(r); self._plot_cal(r)

    def _plot_pca(self, r):
        ax = self.c_pca.new_ax()
        for i, nm in enumerate(r.classes4):
            m = r.pca_lab == i
            if m.any():
                ax.scatter(r.pca_emb[m, 0], r.pca_emb[m, 1], s=10,
                           color=SERIES[i % len(SERIES)], alpha=0.6,
                           edgecolors="none", label=nm)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE)
        self.c_pca.fig.tight_layout(); self.c_pca.draw_idle()

    def _plot_strat(self, r):
        ax = self.c_strat.new_ax()
        short = ["RF\n(mean)", "per-pixel\nvote", "matched\n(mean)"]
        rec = [s[1] for s in r.strategies]
        f1 = [s[3] for s in r.strategies]
        ex = [s[4] for s in r.strategies]
        x = np.arange(len(r.strategies)); w = 0.26
        ax.bar(x - w, rec, w, color=BLUE, label="recall")
        ax.bar(x, f1, w, color=TEAL, label="F1")
        ax.bar(x + w, ex, w, color=AMBER, label="exact")
        for xi, v in zip(x, f1):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7)
        ax.set_ylim(0, 1.08); ax.set_ylabel("score")
        ax.legend(fontsize=7, framealpha=0.0, labelcolor=MUTE, ncol=3)
        self.c_strat.fig.tight_layout(); self.c_strat.draw_idle()

    # ---- plots ----
    def _confusion(self, ax, M, xlabels, ylabels, box_diag=False):
        ax.imshow(M, cmap=CM_CMAP, aspect="auto", vmin=0)
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ylabels)), minor=True)
        ax.grid(which="minor", color=PANEL, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        thr = M.max() / 2 if M.max() else 0.5
        for (rr, cc), v in np.ndenumerate(M):
            if v:
                ax.text(cc, rr, str(v), ha="center", va="center", fontsize=9,
                        color="#ffffff" if v > thr else INK, fontweight="bold")
        if box_diag:
            xi = {c: i for i, c in enumerate(xlabels)}
            for rr, yl in enumerate(ylabels):
                if yl in xi:
                    ax.add_patch(Rectangle((xi[yl] - 0.5, rr - 0.5), 1, 1,
                                 fill=False, edgecolor=INK, lw=1.6))

    def _plot_pure(self, r):
        ax = self.c_pure.new_ax()
        self._confusion(ax, r.cm4, r.classes4, r.classes4)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"spatial split {r.acc4:.0%}   ·   random {r.acc4_random:.0%} "
                     "(leaky)", fontsize=9, color=INK)
        self.c_pure.fig.tight_layout(); self.c_pure.draw_idle()

    def _plot_combo(self, r):
        ax = self.c_combo.new_ax()
        self._confusion(ax, r.combo_M, r.combo_cols, r.combo_rows, box_diag=True)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
        self.c_combo.fig.tight_layout(); self.c_combo.draw_idle()

    def _plot_det(self, r):
        ax = self.c_det.new_ax()
        nS, nC = r.yt.shape
        for row in range(nS):
            for col in range(nC):
                t, p = r.yt[row, col], r.yp[row, col]
                if t and p:
                    fc, mark = TEAL, "O"
                elif t and not p:
                    fc, mark = RED, "X"
                elif p and not t:
                    fc, mark = AMBER, "!"
                else:
                    fc, mark = TNGRAY, ""
                ax.add_patch(Rectangle((col, nS - 1 - row), 1, 1, facecolor=fc,
                             edgecolor="white", linewidth=1.5))
                if mark:
                    ax.text(col + 0.5, nS - 1 - row + 0.5, mark, ha="center",
                            va="center", color="white", fontsize=9, fontweight="bold")
        ax.set_xlim(0, nC); ax.set_ylim(0, nS)
        ax.set_xticks(np.arange(nC) + 0.5); ax.set_xticklabels(r.comps, fontsize=9)
        ax.set_yticks(np.arange(nS) + 0.5); ax.set_yticklabels(r.mix_names[::-1], fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(handles=[Patch(facecolor=TEAL, label="hit"),
                           Patch(facecolor=RED, label="miss"),
                           Patch(facecolor=AMBER, label="false +")],
                  loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3,
                  frameon=False, fontsize=8, labelcolor=MUTE)
        self.c_det.fig.tight_layout(); self.c_det.draw_idle()

    def _plot_cal(self, r):
        self.c_cal.fig.clear()
        for idx, (key, title) in enumerate([("raw", "raw signal"),
                                            ("cal", "calibrated")]):
            ax = self.c_cal.style(self.c_cal.fig.add_subplot(1, 2, idx + 1))
            for name, present, nom, raw, cal in r.calib_rows:
                val = raw if key == "raw" else cal
                for i in present:
                    ax.scatter(nom[i], val[i], s=30, color=SERIES[i % len(SERIES)],
                               edgecolors="white", linewidths=0.5, zorder=3)
            ax.plot([0, 1], [0, 1], ls="--", color=FAINT, lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("nominal", fontsize=8); ax.set_title(title, fontsize=9)
            if idx == 0:
                ax.set_ylabel("recovered", fontsize=8)
        self.c_cal.fig.suptitle(f"mean error  {r.err_raw:.0%}  →  {r.err_cal:.0%}",
                                fontsize=9, color=INK)
        self.c_cal.fig.tight_layout(); self.c_cal.draw_idle()
