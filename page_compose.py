"""page_compose.py — the composition-model half of the Model tab.

Pick a composition method (MLP / PLS / RF / 1D-CNN — the best one is data-dependent),
train ONCE on the known-ratio mixtures you prepared in Samples (Step 1), and save a
portable .dlm that Recovery and Real-data just APPLY (no retraining downstream).
The mixture list is shared from Samples via MIXTURE_BUS — not re-entered here.
"""
from __future__ import annotations

import os
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QComboBox, QSpinBox,
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
        if params.pop("_benchmark", False):
            from dl_model import benchmark_loo
            params.pop("loo", None)
            q.put(("bench", benchmark_loo(items=items,
                                          progress=lambda s: q.put(("progress", s)), **params)))
            return
        from dl_model import train_model
        model = train_model(items=items, progress=lambda s: q.put(("progress", s)), **params)
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
            from composition import SUBSTANCES
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
                        self.fail.emit("training stopped"); return
                    continue
                if tag == "progress":
                    self.progress.emit(payload)
                elif tag == "error":
                    proc.join(timeout=2); self.fail.emit(payload); return
                elif tag == "bench":                      # LOO method comparison
                    proc.join(timeout=5); self.done.emit(("bench", payload)); return
                else:                                     # "done"
                    model = payload; break
            proc.join(timeout=5)
            # train-set recovery is computed IN-MEMORY inside train_model (no map reload,
            # which was holding the GIL and freezing the GUI). Reorder to SUBSTANCES.
            subs = model["subs"]; sidx = {s: j for j, s in enumerate(subs)}
            # prefer the leave-one-out predictions when they were computed — the honest number
            te = (model.get("test_eval") or model.get("loo_eval")
                  or model.get("train_eval", {}))
            tvs = te.get("true", []); pvs = te.get("pred", [])
            errs = []; rows = []
            for i in range(len(tvs)):
                tv = [float(tvs[i][sidx[s]]) if s in sidx else 0.0 for s in SUBSTANCES]
                pv = [float(pvs[i][sidx[s]]) if s in sidx else 0.0 for s in SUBSTANCES]
                errs.append(0.5 * sum(abs(pv[j] - tv[j]) for j in range(len(SUBSTANCES))))
                rows.append((f"mix {i + 1}", tv, pv))
            self.done.emit((model, float(np.mean(errs)) if errs else float("nan"), rows))
        except Exception:
            self.fail.emit(traceback.format_exc())


class ComposePanel(QWidget):
    METHODS = [("MLP (deep)", "mlp"), ("PLS", "pls"),
               ("Random Forest", "rf"), ("1D-CNN", "cnn")]

    def __init__(self):
        super().__init__()
        self.data_dir = PEST_DEFAULT
        self.calib_path = None
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
        frow.addWidget(pure_b); frow.addWidget(self.ref_lbl, 1)
        frow.addWidget(self.mix_lbl); frow.addWidget(reload_b)
        root.addLayout(frow)

        # ---- step 1: which method? (leave-one-out comparison, trains nothing) ----
        brow = QHBoxLayout(); brow.setSpacing(8)
        blbl = QLabel("1 · compare methods:"); blbl.setObjectName("field")
        self.bench_b = QPushButton("Benchmark (LOO)"); self.bench_b.setObjectName("ghost")
        self.bench_b.setToolTip("score NNLS / PLS / RF / 1D-CNN / MLP on the same mixtures "
                                "with leave-one-out — composition error and detection ROC. "
                                "Tells you which method to train; saves no model.")
        self.bench_b.clicked.connect(self._benchmark)
        self.bench_lbl = QLabel("run this first to see which method fits your data")
        self.bench_lbl.setObjectName("field")
        brow.addWidget(blbl); brow.addWidget(self.bench_b)
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
        self.sp_nt = QSpinBox(); self.sp_nt.setRange(20, 1000); self.sp_nt.setSingleStep(20)
        self.sp_nt.setValue(300); self.sp_nt.setPrefix("trees "); self.sp_nt.setObjectName("field")
        mrow.addWidget(mlbl); mrow.addWidget(self.cmb)
        for w in (self.sp_ep, self.sp_seed, self.sp_nc, self.sp_nt):
            mrow.addWidget(w)
        mrow.addStretch(1)
        self.train_b = QPushButton("Train"); self.train_b.setObjectName("primary")
        self.train_b.clicked.connect(self._train)
        self.cancel_b = QPushButton("Cancel"); self.cancel_b.setObjectName("ghost")
        self.cancel_b.setVisible(False); self.cancel_b.clicked.connect(self._cancel)
        self.save_b = QPushButton("Save model…"); self.save_b.setObjectName("ghost")
        self.save_b.setEnabled(False); self.save_b.clicked.connect(self._save)
        self.export_b = QPushButton("Export…"); self.export_b.setObjectName("ghost")
        self.export_b.setEnabled(False); self.export_b.clicked.connect(self._export)
        self.export_b.setToolTip("write the plots (PNG), the per-mixture predictions and the "
                                 "per-substance metrics (CSV) + a README to a folder")
        mrow.addWidget(self.export_b); mrow.addWidget(self.save_b)
        mrow.addWidget(self.cancel_b); mrow.addWidget(self.train_b)
        root.addLayout(mrow)
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

        plots = QHBoxLayout(); plots.setSpacing(12)
        lcard, llay = _card("Learning curve — training loss vs epoch (MLP / CNN)")
        self.c_loss = Canvas(); self.c_loss.setMinimumHeight(300)
        self.c_loss.placeholder("Train an MLP or CNN to see the loss curve")
        llay.addWidget(self.c_loss); plots.addWidget(lcard, 1)
        tcard, tlay = _card("Held-out recovery (leave-one-out) — true (○) vs predicted (●, colour = accuracy)")
        self.c_tri = Canvas(); self.c_tri.setMinimumHeight(300)
        self.c_tri.placeholder("Train to see held-out composition recovery")
        tlay.addWidget(self.c_tri); plots.addWidget(tcard, 1)
        root.addLayout(plots, 1)

        plots2 = QHBoxLayout(); plots2.setSpacing(12)
        pcard, play = _card("Parity per substance — predicted vs true fraction "
                            "(on the line = exact)")
        self.c_parity = Canvas(); self.c_parity.setMinimumHeight(300)
        self.c_parity.placeholder("Train to see the per-substance parity")
        play.addWidget(self.c_parity); plots2.addWidget(pcard, 1)
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

    def _opts(self):
        from dataset import load_preprocess
        cfg = load_preprocess(self.data_dir)
        return dict(data_dir=self.data_dir, calib_path=self.calib_path,
                    baseline=cfg["baseline"], trim=cfg["trim"],
                    method=self.cmb.currentData(), epochs=self.sp_ep.value(),
                    seed=self.sp_seed.value(), use_pretrain=False,
                    n_components=self.sp_nc.value(), n_trees=self.sp_nt.value())

    def _train(self):
        items = self._items_cache
        if len(items) < 3:
            self.status.setText("prepare ≥3 known-ratio mixtures in the Samples tab first")
            self.status.setStyleSheet(f"color:{RED};"); return
        params = self._opts(); params["items"] = items
        if self._test_items:
            params["test_items"] = self._test_items
        params["loo"] = True   # always score held-out: train-set numbers are meaningless here
        self.train_b.setEnabled(False); self.train_b.setText("Training…")
        self.save_b.setEnabled(False); self._cancelled = False; self.cancel_b.setVisible(True)
        self.pbar.setRange(0, 0); self.pbar.setVisible(True)
        self.status.setText("● training…"); self.status.setStyleSheet(f"color:{MUTE};")
        self._thread = QThread(); self._worker = TrainComposeWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda m: self.status.setText("● " + m))
        self._worker.done.connect(self._done)
        self._worker.fail.connect(self._fail)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _benchmark(self):
        """Leave-one-out comparison of NNLS / PLS / RF / CNN / MLP on the same mixtures."""
        items = self._items_cache
        if len(items) < 3:
            self.status.setText("prepare ≥3 known-ratio mixtures in the Samples tab first")
            self.status.setStyleSheet(f"color:{RED};"); return
        params = self._opts(); params.pop("method", None); params.pop("loo", None)
        params["items"] = items; params["_benchmark"] = True
        self.train_b.setEnabled(False); self.bench_b.setEnabled(False)
        self.bench_b.setText("Benchmarking…")
        self._cancelled = False; self.cancel_b.setVisible(True)
        self.pbar.setRange(0, 0); self.pbar.setVisible(True)
        self.status.setText("● leave-one-out benchmark…"); self.status.setStyleSheet(f"color:{MUTE};")
        self._thread = QThread(); self._worker = TrainComposeWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda m: self.status.setText("● " + m))
        self._worker.done.connect(self._done)
        self._worker.fail.connect(self._fail)
        self._worker.done.connect(self._thread.quit)
        self._worker.fail.connect(self._thread.quit)
        self._thread.start()

    def _reset_buttons(self):
        self.train_b.setEnabled(True); self.train_b.setText("Train")
        self.bench_b.setEnabled(True); self.bench_b.setText("Benchmark (LOO)")
        self.pbar.setVisible(False); self.cancel_b.setVisible(False)

    def _done_bench(self, bench):
        """Plot the LOO comparison: composition error per method + overlaid detection ROC."""
        import numpy as _np
        from composition import SUBSTANCES
        subs = bench.get("subs", list(SUBSTANCES))
        methods = [m for m in ("nnls", "pls", "rf", "cnn", "mlp") if m in bench]
        label = {"nnls": "NNLS", "pls": "PLS", "rf": "RF", "cnn": "1D-CNN", "mlp": "MLP"}
        self._bench = {}
        self._err_title.setText("Method comparison — composition error, leave-one-out "
                                "(lower = better)")
        self._roc_title.setText("Method comparison — detection ROC, leave-one-out")
        ax = self.c_err.new_ax()                            # error bar chart per method
        errs, aucs = [], {}
        for m in methods:
            T = _np.asarray(bench[m]["true"], float); P = _np.asarray(bench[m]["pred"], float)
            errs.append(float((0.5 * _np.abs(P - T).sum(1)).mean()))
            rows = [(f"mix {i+1}",
                     [T[i][subs.index(s)] if s in subs else 0.0 for s in SUBSTANCES],
                     [P[i][subs.index(s)] if s in subs else 0.0 for s in SUBSTANCES])
                    for i in range(len(T))]
            self._bench[m] = rows
            aucs[m] = self._roc(rows)
        x = _np.arange(len(methods))
        ax.bar(x, [e * 100 for e in errs], color=[TEAL if m == "mlp" else MUTE for m in methods],
               edgecolor="white", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels([label[m] for m in methods], fontsize=8.5)
        ax.set_ylabel("composition error (%)  — leave-one-out")
        self.c_err.fig.tight_layout(); self.c_err.draw_idle()
        axr = self.c_roc.new_ax()                           # ROC overlay
        axr.plot([0, 1], [0, 1], ls="--", color=MUTE, lw=1.0)
        for j, m in enumerate(methods):
            fpr, tpr, a = aucs[m]
            if fpr is None:
                continue
            axr.plot(fpr, tpr, lw=2.4 if m == "mlp" else 1.4,
                     color=TEAL if m == "mlp" else SERIES[j % len(SERIES)],
                     label=f"{label[m]}  (AUC {a:.3f})")
        axr.set_xlabel("false positive rate"); axr.set_ylabel("true positive rate")
        axr.set_xlim(-0.02, 1.02); axr.set_ylim(-0.02, 1.02); axr.set_aspect("equal")
        axr.legend(fontsize=8, framealpha=0, loc="lower right")
        self.c_roc.fig.tight_layout(); self.c_roc.draw_idle()
        best = methods[int(_np.argmin(errs))]
        self.status.setText("leave-one-out benchmark — " + " · ".join(
            f"{label[m]} {e:.0%}" for m, e in zip(methods, errs)) + f"   → best: {label[best]}")
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
        if isinstance(res, tuple) and len(res) == 2 and res[0] == "bench":
            self._reset_buttons(); self._done_bench(res[1]); return
        model, err, rows = res
        self._model = model
        self._reset_buttons()
        self.save_b.setEnabled(True)
        self._rows = rows
        self._plot_triangle(rows)
        self._plot_loss(model.get("train_eval", {}).get("loss", []))
        self._err_title.setText("Per-substance error — mean |predicted − true| fraction "
                                "(lower = better)")
        self._roc_title.setText("Detection ROC — is each substance present? "
                                "(threshold the predicted fraction)")
        self._plot_parity(rows); self._plot_error(rows); self._plot_roc(rows)
        self.export_b.setEnabled(True)
        MODEL_BUS.set(model, origin=f"Model tab · {model.get('method', 'mlp').upper()}")
        m = model.get("method", "mlp").upper() + ("  +µM" if model.get("has_uM") else "")
        errtxt = f"{err:.0%}" if err == err else "—"
        kind = ("independent test batch" if model.get("test_eval")
                else "leave-one-out" if model.get("loo_eval") else "train-set")
        self.status.setText(f"done — {m} · {model.get('n_train', 0)} train mixtures · {kind} error {errtxt}. "
                            "Recovery & Real-data now use this model; Save to keep it.")
        self.status.setStyleSheet(f"color:{MUTE};")

    def _fail(self, tb):
        import sys
        if getattr(self, "_cancelled", False):            # cancel path already reset the UI
            return
        self._reset_buttons()
        self.status.setText("failed — " + tb.strip().splitlines()[-1][:90])
        self.status.setStyleSheet(f"color:{RED};")
        print(tb, file=sys.stderr)

    def _plot_parity(self, rows):
        """Predicted vs true fraction, one series per substance (on the diagonal = exact)."""
        from composition import SUBSTANCES
        ax = self.c_parity.new_ax()                        # app's cnsplots style
        ax.plot([0, 1], [0, 1], ls="--", color=MUTE, lw=1.0, zorder=1)
        for j, s in enumerate(SUBSTANCES):
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
        from composition import SUBSTANCES
        ax = self.c_err.new_ax()                           # app's cnsplots style
        means, sds, cols = [], [], []
        for j, s in enumerate(SUBSTANCES):
            e = _np.array([abs(r[2][j] - r[1][j]) for r in rows]) if rows else _np.zeros(1)
            means.append(float(e.mean())); sds.append(float(e.std()))
            cols.append(substance_color(s, j))
        x = _np.arange(len(SUBSTANCES))
        ax.bar(x, means, yerr=sds, capsize=4, color=cols, edgecolor="white", alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(SUBSTANCES)
        ax.set_ylabel("mean |predicted − true| fraction")
        self.c_err.fig.tight_layout(); self.c_err.draw_idle()

    def _roc(self, rows, thr=0.05):
        """Detection framing of the regression: 'substance present' = true fraction > thr,
        score = predicted fraction. Returns (fpr, tpr, auc) micro-averaged over substances."""
        import numpy as _np
        from composition import SUBSTANCES
        y = _np.concatenate([[1 if r[1][j] > thr else 0 for r in rows]
                             for j in range(len(SUBSTANCES))]) if rows else _np.zeros(0)
        s = _np.concatenate([[r[2][j] for r in rows] for j in range(len(SUBSTANCES))]) \
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
            ax.plot(fpr, tpr, color=TEAL, lw=2.0, label=f"AUC {a:.3f}")
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
        from matplotlib import cm
        ax = self.c_tri.new_ax()                           # app's cnsplots style
        fig = self.c_tri.fig
        A, B, C = bary([1, 0, 0]), bary([0, 0, 1]), bary([0, 1, 0])   # DQ · THI · TBZ corners
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color=INK, lw=1.0, zorder=1)
        # corner labels OFFSET away from the vertex so they never sit on the lines/points
        for f, s, col, dx, dy, ha, va in [
                ([1, 0, 0], "DQ", substance_color("DQ", 0), 0, 0.05, "center", "bottom"),
                ([0, 0, 1], "THI", substance_color("THI", 2), -0.03, -0.05, "right", "top"),
                ([0, 1, 0], "TBZ", substance_color("TBZ", 1), 0.03, -0.05, "left", "top")]:
            p = bary(f)
            ax.text(p[0] + dx, p[1] + dy, s, ha=ha, va=va, fontsize=11,
                    fontweight="bold", color=col)
        cmap = cm.get_cmap("RdYlGn")
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
            return
        import numpy as _np
        from collections import OrderedDict
        from composition import SUBSTANCES
        from io_utils import write_csv, write_readme
        d = QFileDialog.getExistingDirectory(self, "Export composition-model results",
                                             self.data_dir)
        if not d:
            return
        rows = self._rows; m = self._model
        # per-mixture true vs predicted fraction
        write_csv(os.path.join(d, "composition_loo_predictions.csv"),
                  ["mixture"] + [f"true_{s}" for s in SUBSTANCES]
                  + [f"pred_{s}" for s in SUBSTANCES] + ["composition_error"],
                  [[r[0]] + [f"{v:.4f}" for v in r[1]] + [f"{v:.4f}" for v in r[2]]
                   + [f"{0.5 * sum(abs(r[2][j] - r[1][j]) for j in range(len(SUBSTANCES))):.4f}"]
                   for r in rows])
        # per-substance metrics
        mrows = []
        for j, s in enumerate(SUBSTANCES):
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
        n = _save_figs([("composition_learning_curve", self.c_loss),
                        ("composition_triangle", self.c_tri),
                        ("composition_parity", self.c_parity),
                        ("composition_error", self.c_err),
                        ("composition_roc", self.c_roc)], d)
        # the same publication-style ternaries the Recovery export writes
        try:
            from triangle_figs import accuracy_triangle, rgb_triangle
            cols = [substance_color(s, j) for j, s in enumerate(SUBSTANCES)]
            for fn, name in ((accuracy_triangle, "composition_triangle_accuracy"),
                             (rgb_triangle, "composition_triangle_rgb")):
                fn(rows, list(SUBSTANCES), cols).savefig(
                    os.path.join(d, name + ".png"), dpi=300, transparent=True,
                    bbox_inches="tight")
                n += 1
        except Exception as e:
            import sys as _s; print(e, file=_s.stderr)
        err = float(_np.mean([0.5 * sum(abs(r[2][j] - r[1][j]) for j in range(len(SUBSTANCES)))
                              for r in rows]))
        write_readme(d, "UNMIXR — Composition model (Step 2)", OrderedDict([
            ("What this is", [
                "The composition model trained once on the known-ratio mixtures prepared in "
                "Samples. Recovery and Real-data APPLY this model — they do not retrain. "
                "Numbers here are LEAVE-ONE-OUT: every mixture was scored by a model that "
                "never saw it. The saved model itself is fit on all of them."]),
            ("How it was produced", [
                f"- Pure references: {self.data_dir}",
                f"- Mixtures: {m.get('n_train', 0)} known-ratio maps (from Samples)",
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
                                 "the predicted fraction."),
             ("composition_triangle_accuracy", "ternary with the interior shaded by "
                                               "accuracy and per-corner recovery ± SE."),
             ("composition_triangle_rgb", "ternary with the interior coloured by "
                                          "composition itself (RGB blend).")])
        self.status.setText(f"exported {n} PNG + CSVs + README → {os.path.basename(d)}")
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
