"""page_validate.py — Validate tab: check the pure-reference unmixing against
KNOWN-ratio mixtures, recover per-substance response factors, and export a
correction the Real-data tab can apply (surface ratio → solution ratio)."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout,
    QFileDialog, QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QLineEdit,
)

from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch

from ui_common import *
from real_data import PEST_DEFAULT
from dataset import load_preprocess
from io_utils import write_csv
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
            self.done.emit((vres, cres))
        except Exception:
            self.fail.emit(traceback.format_exc())


class ValidatePage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self._cres = None                # composition (per-pixel) result
        COLOR_BUS.changed.connect(self._recolor)     # top-bar picker → recolour
        self._files = []                 # full paths, aligned with table rows
        self.data_dir = PEST_DEFAULT
        self.calib_path = None           # dilution-series CSV → recovery (measured µM)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20); root.setSpacing(12)

        head = QVBoxLayout(); head.setSpacing(2)
        h1 = QLabel("Validate — known-ratio mixtures → response factors"); h1.setObjectName("h1")
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
        clr_b = QPushButton("Clear"); clr_b.setObjectName("ghost"); clr_b.clicked.connect(self._clear)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost"); exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Validate"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(self.ref_lbl); ctl.addStretch(1)
        ctl.addWidget(cal_b); ctl.addWidget(self.cal_lbl)
        ctl.addWidget(add_b); ctl.addWidget(clr_b); ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

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
        root.addLayout(brow)

        self.status = QLabel(""); self.status.setObjectName("sub")
        root.addWidget(self.status)

        # editable table: file  |  true ratio (name:parts, comma-separated)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["mixture file", "true ratio  (e.g. DQ:1, TBZ:3)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        self.table.setMaximumHeight(170)
        root.addWidget(self.table)

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

        # ---- composition view (colour blend · drift · recovery) ----
        self.c_maps = Canvas(); self.c_tri = Canvas()
        self.c_rel = Canvas(); self.c_rec = Canvas()
        mcard, mlay = _card("Composition maps — per-pixel colour blend of each mixture")
        mlay.addWidget(self.c_maps); self.c_maps.setMinimumHeight(320)
        body.addWidget(mcard)
        crow = QHBoxLayout(); crow.setSpacing(12)
        for cv, title in [
            (self.c_tri, "Drift — real → predicted composition (arrows), corners = recovery"),
            (self.c_rec, "Apparent recovery (predicted / real, mean ± SE over mixtures)"),
        ]:
            card, lay = _card(title); lay.addWidget(cv); cv.setMinimumHeight(300)
            crow.addWidget(card, 1)
        crow_w = QWidget(); crow_w.setLayout(crow); body.addWidget(crow_w)
        dcard, dlay = _card("Relative drift — predicted vs real, grouped by dominant substance")
        dlay.addWidget(self.c_rel); self.c_rel.setMinimumHeight(300)
        body.addWidget(dcard)

        bodyw = QWidget(); bodyw.setLayout(body)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidget(bodyw)
        scroll.setStyleSheet("QScrollArea{background:transparent;}")
        root.addWidget(scroll, 1)
        for cv, m in [(self.c_parity, "Add mixtures, then Validate"),
                      (self.c_corr, "Corrected ratio appears here"),
                      (self.c_resp, "Response factors appear here"),
                      (self.c_maps, "Composition colour maps appear here"),
                      (self.c_tri, "Drift triangle appears here"),
                      (self.c_rec, "Recovery appears here"),
                      (self.c_rel, "Relative drift appears here")]:
            cv.placeholder(m)

        self.readout = QLabel(""); self.readout.setObjectName("sub")
        self.readout.setWordWrap(True); self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setStyleSheet(f"font-size:15px; color:{INK};")
        root.addWidget(self.readout)

    # ---- helpers ----
    def _short(self, p):
        return "refs (from Samples): " + ("…" + p[-34:] if len(p) > 34 else p)

    def set_data_dir(self, path):
        self.data_dir = path; self.ref_lbl.setText(self._short(path))

    def _ref_names(self):
        try:
            from dataset import discover_dataset, is_blank
            return [c for c, _m in discover_dataset(self.data_dir) if not is_blank(c)]
        except Exception:
            return []

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

    # ---- run ----
    def _run(self):
        if worker_busy(self):                             # already running — ignore
            return
        items = self._items()
        if len(items) < 1:
            self.status.setText("add ≥1 mixture with a true ratio (e.g. DQ:1, TBZ:3)")
            self.status.setStyleSheet(f"color:{RED};"); return
        cfg = load_preprocess(self.data_dir)
        params = dict(
            validate=dict(data_dir=self.data_dir, items=items, method="nnls",
                          baseline=cfg["baseline"], trim=cfg["trim"],
                          calib_path=self.calib_path),
            composition=dict(data_dir=self.data_dir, baseline=cfg["baseline"],
                             files=[it[0] for it in items],
                             nominals=[it[1] for it in items]))
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self.status.setText(""); self.status.setStyleSheet(f"color:{MUTE};")
        start_worker(self, ValidateWorker(params), done=self._apply, fail=self._error,
                     progress=lambda m: self.status.setText("● " + m))

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Validate")
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, pair):
        res, cres = pair if isinstance(pair, tuple) else (pair, None)
        self._res = res; self._cres = cres
        self.btn.setEnabled(True); self.btn.setText("Validate")
        self.status.setText("done"); self.status.setStyleSheet(f"color:{MUTE};")
        names = res.names
        self.k_mix.set(str(len(res.rows)), TEAL)
        self.k_sub.set(str(len(names)), AMBER)
        self.k_max.set(f"{max(res.response.values()):.1f}×", CORAL)
        e0 = self._mean_err([r["obs"] for r in res.rows], res.rows, names)
        e1 = self._mean_err(res.corrected, res.rows, names)
        self.k_err.set(f"{e0:.0%} → {e1:.0%}", BLUE)
        self._plot_parity(res); self._plot_corr(res); self._plot_resp(res)
        if cres:                                          # composition view
            self._plot_maps(cres); self._plot_triangle(cres)
            self._plot_reldrift(cres); self._plot_recovery(cres)
        rf = "  ·  ".join(f"{n} {res.response[n]:.2f}×" for n in names)
        dom = max(res.response, key=res.response.get)
        txt = (f"<b>response factors</b> (anchor {res.ref}): {rf}<br>"
               f"<b>{dom}</b> is over-reported on the surface by "
               f"{res.response[dom]:.1f}× — that is why it tends to dominate every map. "
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
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)   # fill panel (uniform export size)
        ax.set_xlabel("true fraction"); ax.set_ylabel(f"{title_obs} fraction")
        ax.legend(fontsize=10, framealpha=0.0, labelcolor="black")
        cv.fig.tight_layout(); cv.draw_idle()

    def _plot_corr(self, res):
        self._plot_parity(res, corrected=True, canvas=self.c_corr,
                          title_obs="corrected (solution)")

    def _plot_resp(self, res):
        ax = self.c_resp.new_ax()
        names = res.names
        vals = [res.response[n] for n in names]
        cols = [substance_color(names[i], i) for i in range(len(names))]
        x = np.arange(len(names))
        ax.bar(x, vals, color=cols)
        ax.axhline(1.0, color=MUTE, ls="--", lw=1.0)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}×", ha="center", fontsize=9, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("response factor (×)")
        ax.set_ylim(0, max(vals) * 1.2 + 0.2)
        self.c_resp.fig.tight_layout(); self.c_resp.draw_idle()

    # ---- composition view (colour blend · drift · recovery) ----
    def _rgb(self):
        return np.array([to_rgb(substance_color(s, i)) for i, s in enumerate(SUBSTANCES)])

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
            img = np.ones((len(uy), len(ux), 3))
            blend = np.clip(rec["frac"] @ C, 0.0, 1.0); hit = rec["hit"]
            img[rows[hit], cols[hit]] = blend[hit]
            ax.imshow(img, origin="lower", interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            m = rec["mean"] * 100
            ax.set_xlabel(f"{rec['name']}\nDQ{m[0]:.0f} TBZ{m[1]:.0f} THI{m[2]:.0f}",
                          fontsize=8, color=INK, labelpad=2)
        self.c_maps.fig.tight_layout(); self.c_maps.draw_idle()

    def _recovery(self, res):
        """Per-substance apparent recovery (%) over true mixtures (≥2 nominal
        components); pure/dilution maps excluded. {substance: [pct, ...]}."""
        per = {s: [] for s in SUBSTANCES}
        for r in res:
            nom = r["nominal"]
            if nom is None or int(np.count_nonzero(nom)) < 2:
                continue
            for i, s in enumerate(SUBSTANCES):
                if nom[i] > 0:
                    per[s].append(r["mean"][i] / nom[i] * 100)
        return per

    def _tri_frame(self, ax, rec=None):
        A, B, C = bary([1, 0, 0]), bary([0, 0, 1]), bary([0, 1, 0])   # DQ, THI, TBZ
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color=INK, lw=0.8)
        cen = (A + B + C) / 3
        for P, Q in [(A, B), (B, C), (C, A)]:              # quarter ticks per edge
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
        self._tri_frame(ax, self._recovery(res))
        for rec in res:
            if rec["nominal"] is None:
                continue
            p0 = bary(rec["nominal"]); p1 = bary(rec["mean"])
            dom = int(np.argmax(rec["nominal"]))
            col = substance_color(SUBSTANCES[dom], dom)
            if np.linalg.norm(p1 - p0) > 1e-3:
                ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                             color="#98a1ac", lw=1.1, zorder=2))
            ax.scatter(*p1, s=38, color=col, edgecolors="white", linewidths=0.6, zorder=4)
        ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.10, 1.06)
        self.c_tri.fig.tight_layout(); self.c_tri.draw_idle()

    def _plot_reldrift(self, res):
        ax = self.c_rel.new_ax()
        recs = [r for r in res if r["nominal"] is not None]
        if not recs:
            self.c_rel.placeholder("no nominal ratios"); return
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
        for y, k, si in bars:
            ax.barh(y, gap_n[k], height=0.66, alpha=0.85,
                    color=substance_color(SUBSTANCES[si], si))
        ax.set_yticks(ypos); ax.set_yticklabels(ynames, fontsize=8)
        for t, c in zip(ax.get_yticklabels(), ycols):
            t.set_color(c)
        for si, yc in spans:
            ax.text(103, yc, f"{SUBSTANCES[si]}-dom", va="center", ha="left", fontsize=9,
                    fontweight="bold", color=substance_color(SUBSTANCES[si], si))
        ax.set_xlim(0, 120)
        ax.set_xlabel("predicted vs real  —  drift (% of worst)")
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
        write_csv(os.path.join(d, "response_factors.csv"),
                  ["substance", "response_factor", "anchor"],
                  [[n, f"{res.response[n]:.5f}", res.ref] for n in res.names])
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
        figs = [("validate_parity", self.c_parity),
                ("validate_corrected", self.c_corr),
                ("validate_response", self.c_resp)]
        if self._cres:                                   # composition view
            figs += [("composition_maps", self.c_maps), ("drift_triangle", self.c_tri),
                     ("relative_drift", self.c_rel), ("recovery", self.c_rec)]
        n = _save_figs(figs, d)
        self.status.setText(f"exported response_factors.csv + table + {n} PNG → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")
