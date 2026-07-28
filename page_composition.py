"""page_composition.py — Composition tab: reuse the NNLS unmixing to show the
mixtures' composition as a colour blend, the nominal→measured drift on a ternary
triangle, a relative-drift ranking, and the apparent (intensity) recovery.
Colours come from the shared substance palette (top-bar picker)."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Patch

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QProgressBar,
)

from ui_common import *                                    # noqa: F401,F403
from dataset import load_preprocess
from real_data import PEST_DEFAULT
from composition import (compute_composition, SUBSTANCES, bary,
                         composition_distance, aitchison_distance)


class CompWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.done.emit(compute_composition(progress=self.progress.emit, **self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


class CompositionPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        COLOR_BUS.changed.connect(self._recolor)           # top-bar picker → recolour
        self.data_dir = PEST_DEFAULT
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(12)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Composition — colour blend · drift · recovery"); h1.setObjectName("h1")
        sub = QLabel("Unmix every known-ratio mixture (NNLS, background separated) and "
                     "read composition as a colour blend, the nominal→measured drift on a "
                     "ternary triangle, a relative-drift ranking, and the apparent "
                     "(intensity) recovery. References + preprocessing come from Samples.")
        sub.setObjectName("sub"); sub.setWordWrap(True)
        head.addWidget(h1); head.addWidget(sub)
        root.addLayout(head)

        ctl = QHBoxLayout(); ctl.setSpacing(8)
        self.btn = QPushButton("Run"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        self.exp = QPushButton("Export…"); self.exp.setObjectName("ghost")
        self.exp.clicked.connect(self._export)
        self.status = QLabel(""); self.status.setObjectName("field")
        ctl.addWidget(self.btn); ctl.addWidget(self.exp)
        ctl.addSpacing(8); ctl.addWidget(self.status); ctl.addStretch(1)
        root.addLayout(ctl)

        self.pbar = QProgressBar(); self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(6); self.pbar.setVisible(False)
        root.addWidget(self.pbar)

        grid = QGridLayout(); grid.setSpacing(12)
        self.c_maps = Canvas(); self.c_tri = Canvas()
        self.c_rel = Canvas(); self.c_rec = Canvas()
        for cv, title, r, c in [
            (self.c_maps, "Composition maps (colour blend)", 0, 0),
            (self.c_tri, "Drift — nominal → measured", 0, 1),
            (self.c_rel, "Relative drift (ranked)", 1, 0),
            (self.c_rec, "Apparent recovery (measured / nominal)", 1, 1),
        ]:
            card, lay = _card(title)
            lay.addWidget(cv); grid.addWidget(card, r, c)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 1)
        root.addLayout(grid, 1)

        self.readout = QLabel(""); self.readout.setObjectName("sub")
        self.readout.setWordWrap(True)
        root.addWidget(self.readout)

        for cv in (self.c_maps, self.c_tri, self.c_rel, self.c_rec):
            cv.placeholder("Run to compute composition from the reference mixtures")

    def set_data_dir(self, path):
        self.data_dir = path

    # ---- run ----
    def _run(self):
        cfg = load_preprocess(self.data_dir)
        params = dict(data_dir=self.data_dir, baseline=cfg["baseline"])
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self.status.setText(""); self.pbar.setVisible(True); self.pbar.setRange(0, 0)
        self.c_maps.placeholder("Unmixing…")
        self._thread = QThread(); self._worker = CompWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _progress(self, msg):
        self.btn.setText("Working…  " + msg.split("  ")[0]); self.status.setText("● " + msg)
        m = __import__("re").search(r"(\d+)\s*/\s*(\d+)", msg)
        if m:
            self.pbar.setRange(0, int(m.group(2))); self.pbar.setValue(int(m.group(1)))

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Run"); self.pbar.setVisible(False)
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, res):
        self._res = res
        self.btn.setEnabled(True); self.btn.setText("Run"); self.pbar.setVisible(False)
        self.status.setText(f"done — {len(res)} maps"); self.status.setStyleSheet("")
        self._draw()

    def _recolor(self):
        if self._res is not None:
            self._draw()

    def _draw(self):
        self._plot_maps(self._res); self._plot_triangle(self._res)
        self._plot_reldrift(self._res); self._plot_recovery(self._res)
        self._readout(self._res)

    # ---- colour helper ----
    def _rgb(self):
        """Current substance colours as RGB rows aligned with SUBSTANCES."""
        return np.array([to_rgb(substance_color(s, i)) for i, s in enumerate(SUBSTANCES)])

    # ---- plots ----
    def _plot_maps(self, res):
        C = self._rgb()
        n = len(res)
        nc = int(np.ceil(np.sqrt(n))); nr = int(np.ceil(n / nc))
        self.c_maps.fig.clear()
        for k, rec in enumerate(res):
            ax = self.c_maps.fig.add_subplot(nr, nc, k + 1)
            x, y = rec["coords"][:, 0], rec["coords"][:, 1]
            ux, uy = np.unique(x), np.unique(y)
            xi = {v: i for i, v in enumerate(ux)}; yi = {v: i for i, v in enumerate(uy)}
            rows = np.array([yi[v] for v in y]); cols = np.array([xi[v] for v in x])
            img = np.ones((len(uy), len(ux), 3))          # white background
            blend = np.clip(rec["frac"] @ C, 0.0, 1.0)
            hit = rec["hit"]
            img[rows[hit], cols[hit]] = blend[hit]
            ax.imshow(img, origin="lower", interpolation="nearest", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            m = rec["mean"] * 100
            ax.set_xlabel(f"{rec['name']}\nDQ{m[0]:.0f} TBZ{m[1]:.0f} THI{m[2]:.0f}",
                          fontsize=8, color=INK, labelpad=2)
        self.c_maps.fig.tight_layout(); self.c_maps.draw_idle()

    def _recovery(self, res):
        """Per-substance apparent recovery values (%) over the true MIXTURES (≥2
        nominal components) only — pure / dilution maps are excluded because their
        low-signal misattribution inflates the spread. {substance: [pct, ...]}."""
        per = {s: [] for s in SUBSTANCES}
        for r in res:
            nom = r["nominal"]
            if nom is None or int(np.count_nonzero(nom)) < 2:   # mixtures only
                continue
            for i, s in enumerate(SUBSTANCES):
                if nom[i] > 0:
                    per[s].append(r["mean"][i] / nom[i] * 100)
        return per

    def _tri_frame(self, ax, rec=None):
        A, B, C = bary([1, 0, 0]), bary([0, 0, 1]), bary([0, 1, 0])   # DQ, THI, TBZ
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color=INK, lw=0.8)
        cen = (A + B + C) / 3
        for P, Q in [(A, B), (B, C), (C, A)]:              # quarter ticks (0·¼·½·¾·1) per edge
            u = Q - P
            n = np.array([-u[1], u[0]]); n = n / (np.hypot(*n) + 1e-9)
            if np.dot(n, (P + Q) / 2 - cen) < 0:
                n = -n
            for t in (0.25, 0.5, 0.75):
                pt = P + t * u
                ax.plot([pt[0], pt[0] + 0.022 * n[0]], [pt[1], pt[1] + 0.022 * n[1]],
                        color=MUTE, lw=0.7, zorder=1)
        for f, s, ha, va, dx, dy in [([1, 0, 0], "DQ", "center", "bottom", 0, 0.04),
                                     ([0, 0, 1], "THI", "right", "top", -0.03, -0.03),
                                     ([0, 1, 0], "TBZ", "left", "top", 0.03, -0.03)]:
            p = bary(f)
            lab = s
            if rec and rec.get(s):
                v = rec[s]
                se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                lab = f"{s}\n{np.mean(v):.0f}±{se:.0f}%"
            ax.text(p[0] + dx, p[1] + dy, lab, ha=ha, va=va, fontsize=10, linespacing=1.1,
                    fontweight="bold", color=substance_color(s, SUBSTANCES.index(s)))
        ax.set_aspect("equal"); ax.axis("off")

    def _plot_triangle(self, res):
        ax = self.c_tri.new_ax()
        self._tri_frame(ax, self._recovery(res))               # corners: name + recovery %
        for rec in res:                                        # arrows only, no clutter
            if rec["nominal"] is None:
                continue
            p0 = bary(rec["nominal"]); p1 = bary(rec["mean"])
            dom = int(np.argmax(rec["nominal"]))               # colour by TRUE (nominal) mix
            col = substance_color(SUBSTANCES[dom], dom)
            if np.linalg.norm(p1 - p0) > 1e-3:                 # drift arrow from the tick
                ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                             color="#98a1ac", lw=1.1, zorder=2))
            ax.scatter(*p1, s=38, color=col, edgecolors="white", linewidths=0.6, zorder=4)
        ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.10, 1.06)
        self.c_tri.fig.tight_layout(); self.c_tri.draw_idle()

    def _plot_reldrift(self, res):
        ax = self.c_rel.new_ax()
        recs = [r for r in res if r["nominal"] is not None]
        if not recs:
            self.c_rel.placeholder("no nominal ratios parsed"); return
        dom = [int(np.argmax(r["nominal"])) for r in recs]     # group by TRUE (nominal) mix
        comp = np.array([composition_distance(r["nominal"], r["mean"]) for r in recs])
        ait = np.array([aitchison_distance(r["nominal"], r["mean"]) for r in recs])
        comp_n = comp / (comp.max() or 1.0) * 100
        ait_n = ait / (ait.max() or 1.0) * 100
        DARK, LIGHT, h = "#3a4453", "#b7bfc9", 0.38            # neutral metric colours

        bars, ypos, ynames, ycols, spans = [], [], [], [], []
        cursor, first = 0.0, True
        for si in (2, 1, 0):                                   # THI-, TBZ-, DQ-dominant
            members = [k for k in range(len(recs)) if dom[k] == si]
            if not members:
                continue
            if not first:
                cursor -= 1.0                                 # gap between groups
            first = False
            members.sort(key=lambda k: -comp_n[k])            # worst on top
            start = cursor
            for k in members:
                bars.append((cursor, k)); ypos.append(cursor)
                ynames.append(recs[k]["name"].replace("_corrected", ""))
                ycols.append(substance_color(SUBSTANCES[si], si))
                cursor -= 1.0
            spans.append((si, (start + cursor + 1.0) / 2))
        for y, k in bars:
            ax.barh(y + h / 2, comp_n[k], height=h, color=DARK)
            ax.barh(y - h / 2, ait_n[k], height=h, color=LIGHT)
        ax.set_yticks(ypos); ax.set_yticklabels(ynames, fontsize=8)
        for t, c in zip(ax.get_yticklabels(), ycols):
            t.set_color(c)
        for si, yc in spans:                                  # dominance group tag
            ax.text(104, yc, f"{SUBSTANCES[si]}-dom", va="center", ha="left", fontsize=9,
                    fontweight="bold", color=substance_color(SUBSTANCES[si], si))
        ax.set_xlim(0, 122)
        ax.set_xlabel("relative drift (% of worst)")
        ax.legend(handles=[Patch(color=DARK, label="composition"),
                           Patch(color=LIGHT, label="Aitchison")],
                  fontsize=8, framealpha=0.0, labelcolor="black", loc="lower right")
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

    def _readout(self, res):
        per = self._recovery(res)
        def _pm(s, i):
            v = per[s]
            se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            return (f"<b style='color:{substance_color(s, i)}'>{s}</b> "
                    f"{np.mean(v):.0f}±{se:.0f}%")
        rec = "  ·  ".join(_pm(s, i) for i, s in enumerate(SUBSTANCES) if per[s])
        self.readout.setText(
            f"<b>apparent recovery</b> (intensity fraction / nominal): {rec} &nbsp;·&nbsp; "
            "&gt;100% = over-reported on the surface (larger SERS cross-section), "
            "&lt;100% = under-reported. This is an <i>intensity</i> recovery, not a "
            "concentration recovery.")

    # ---- export ----
    def _export(self):
        if self._res is None:
            self.status.setText("run first, then export"); return
        from PyQt6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, "Export folder")
        if not d:
            return
        n = _save_figs([("composition_maps", self.c_maps),
                        ("drift_triangle", self.c_tri),
                        ("relative_drift", self.c_rel),
                        ("recovery", self.c_rec)], d)
        self.status.setText(f"exported {n} PNG → {os.path.basename(d)}")
