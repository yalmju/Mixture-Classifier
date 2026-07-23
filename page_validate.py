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
    QHeaderView, QAbstractItemView,
)

from ui_common import *
from real_data import PEST_DEFAULT
from dataset import load_preprocess
from io_utils import write_csv
from validate import validate_mixtures, parse_mixture_label


class ValidateWorker(QObject):
    done = pyqtSignal(object)
    fail = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.done.emit(validate_mixtures(progress=self.progress.emit, **self.params))
        except Exception:
            self.fail.emit(traceback.format_exc())


class ValidatePage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._res = None
        self._files = []                 # full paths, aligned with table rows
        self.data_dir = PEST_DEFAULT
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
        clr_b = QPushButton("Clear"); clr_b.setObjectName("ghost"); clr_b.clicked.connect(self._clear)
        exp_b = QPushButton("Export…"); exp_b.setObjectName("ghost"); exp_b.clicked.connect(self._export)
        self.btn = QPushButton("Validate"); self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._run)
        ctl.addWidget(self.ref_lbl); ctl.addStretch(1)
        ctl.addWidget(add_b); ctl.addWidget(clr_b); ctl.addWidget(exp_b); ctl.addWidget(self.btn)
        root.addLayout(ctl)

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

        bodyw = QWidget(); bodyw.setLayout(body)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame); scroll.setWidget(bodyw)
        scroll.setStyleSheet("QScrollArea{background:transparent;}")
        root.addWidget(scroll, 1)
        for cv, m in [(self.c_parity, "Add mixtures, then Validate"),
                      (self.c_corr, "Corrected ratio appears here"),
                      (self.c_resp, "Response factors appear here")]:
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
        refs = self._ref_names()
        for p in paths:
            if p in self._files:
                continue
            self._files.append(p)
            row = self.table.rowCount(); self.table.insertRow(row)
            f_item = QTableWidgetItem(os.path.basename(p))
            f_item.setFlags(f_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, f_item)
            guess = parse_mixture_label(os.path.splitext(os.path.basename(p))[0], refs)
            txt = ", ".join(f"{k}:{v:.2g}" for k, v in guess.items()) if guess else ""
            self.table.setItem(row, 1, QTableWidgetItem(txt))
        self.status.setText(f"{len(self._files)} mixtures — edit any true ratio, then Validate")
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
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
        return out

    def _items(self):
        items = []
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 1)
            true = self._parse_true(cell.text()) if cell else {}
            if len(true) >= 2:
                items.append((self._files[row], true))
        return items

    # ---- run ----
    def _run(self):
        items = self._items()
        if len(items) < 1:
            self.status.setText("add ≥1 mixture with a true ratio (e.g. DQ:1, TBZ:3)")
            self.status.setStyleSheet(f"color:{RED};"); return
        cfg = load_preprocess(self.data_dir)
        params = dict(data_dir=self.data_dir, items=items, method="nnls",
                      baseline=cfg["baseline"], trim=cfg["trim"])
        self.btn.setEnabled(False); self.btn.setText("Working…")
        self.status.setText(""); self.status.setStyleSheet(f"color:{MUTE};")
        self._thread = QThread(); self._worker = ValidateWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda m: self.status.setText("● " + m))
        self._worker.done.connect(self._apply)
        self._worker.fail.connect(self._error)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _error(self, tb):
        self.btn.setEnabled(True); self.btn.setText("Validate")
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _apply(self, res):
        self._res = res
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
        rf = "  ·  ".join(f"{n} {res.response[n]:.2f}×" for n in names)
        dom = max(res.response, key=res.response.get)
        txt = (f"<b>response factors</b> (anchor {res.ref}): {rf}<br>"
               f"<b>{dom}</b> is over-reported on the surface by "
               f"{res.response[dom]:.1f}× — that is why it tends to dominate every map. "
               f"Mean ratio error drops {e0:.0%} → {e1:.0%} after correction. "
               "Export to apply the correction in Real data.")
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
    def _plot_parity(self, res, corrected=False, canvas=None, title_obs="observed"):
        cv = canvas or self.c_parity
        ax = cv.new_ax()
        names = res.names
        fracs = res.corrected if corrected else [r["obs"] for r in res.rows]
        for i, n in enumerate(names):
            col = SERIES[i % len(SERIES)]
            xs = [r["true"].get(n, 0.0) for r in res.rows]
            ys = [f[n] if isinstance(f, dict) else f[i] for f in fracs]
            ax.scatter(xs, ys, color=col, s=42, edgecolors="white", linewidths=0.6,
                       label=n, zorder=3)
        ax.plot([0, 1], [0, 1], color=MUTE, ls="--", lw=1.0, zorder=1)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
        ax.set_xlabel("true fraction"); ax.set_ylabel(f"{title_obs} fraction")
        ax.legend(fontsize=8, framealpha=0.0, labelcolor=MUTE)
        cv.fig.tight_layout(); cv.draw_idle()

    def _plot_corr(self, res):
        self._plot_parity(res, corrected=True, canvas=self.c_corr,
                          title_obs="corrected (solution)")

    def _plot_resp(self, res):
        ax = self.c_resp.new_ax()
        names = res.names
        vals = [res.response[n] for n in names]
        cols = [SERIES[i % len(SERIES)] for i in range(len(names))]
        x = np.arange(len(names))
        ax.bar(x, vals, color=cols)
        ax.axhline(1.0, color=MUTE, ls="--", lw=1.0)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}×", ha="center", fontsize=9, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("response factor (×)")
        ax.set_ylim(0, max(vals) * 1.2 + 0.2)
        self.c_resp.fig.tight_layout(); self.c_resp.draw_idle()

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
        head = ["mixture"] + [f"true_{n}" for n in res.names] \
            + [f"obs_{n}" for n in res.names] + [f"corr_{n}" for n in res.names]
        rows = []
        for r, corr in zip(res.rows, res.corrected):
            rows.append([os.path.basename(r["path"])]
                        + [f"{r['true'].get(n, 0):.4f}" for n in res.names]
                        + [f"{r['obs'].get(n, 0):.4f}" for n in res.names]
                        + [f"{corr[n]:.4f}" for n in res.names])
        write_csv(os.path.join(d, "validation_table.csv"), head, rows)
        n = _save_figs([("validate_parity", self.c_parity),
                        ("validate_corrected", self.c_corr),
                        ("validate_response", self.c_resp)], d)
        self.status.setText(f"exported response_factors.csv + table + {n} PNG → {os.path.basename(d)}")
        self.status.setStyleSheet(f"color:{MUTE};")
