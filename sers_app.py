"""
sers_app.py
===========
Desktop GUI for the SERS mixture kit — same layout idiom as the MP Report Node
(customtkinter sidebar + dashboard cards + embedded matplotlib + export).

Wraps the analysis core:
  - preprocess + SERSMixtureClassifier  (component detection, trained on pures)
  - competitive.recover_ratios          (concentration-ratio read-out)
  - competitive_compare.additive_residual (non-additivity / fit quality)

Run:
    pip install customtkinter matplotlib scikit-learn scipy numpy
    python sers_app.py

Data format (CSV, shared wavenumber axis):
    pure.csv      wavenumber, DQ, THI, TBZ, ...     (one column per pure)
    unknown.csv   wavenumber, mix1, mix2, ...       (spectra to analyse)
(Adaptable to XY-map CSVs — swap load_csv for your map loader.)
"""
from __future__ import annotations
import csv, os
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.size"] = 10
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import brand
import family
from sers_mixture import preprocess, SERSMixtureClassifier, AugmentConfig
from competitive import recover_ratios, calibrate_response
from competitive_compare import additive_residual

family.apply()   # shared UNMIXR light look
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    names = [h.strip() for h in rows[0][1:] if h.strip() != ""]
    arr, axis = [], []
    for r in rows[1:]:
        vals = [v for v in r if v.strip() != ""]
        if len(vals) < 2:
            continue
        axis.append(float(vals[0]))
        arr.append([float(v) for v in vals[1:len(names) + 1]])
    axis = np.array(axis)
    spectra = np.array(arr).T                     # (n_spectra, n_feat)
    return axis, names, spectra


class SERSApp:
    def __init__(self, root, embedded=False):
        self.root = root
        if not embedded:
            root.title(brand.APP_NAME)
            root.geometry("1180x760")
        self.axis = None
        self.comp_names = None
        self.pures = None            # preprocessed pure templates
        self.unk_axis = None
        self.unk_names = None
        self.unk = None              # preprocessed unknown spectra
        self.clf = None
        self.R = None                # response factors (optional)
        self.last_results = None
        self._build_ui()

    # button color helpers (all colors come from brand.py)
    def _btn(self):
        return dict(fg_color=brand.BTN_FILL, hover_color=brand.BTN_HOVER)

    def _runbtn(self):
        return dict(fg_color=brand.RUN_FILL, hover_color=brand.RUN_HOVER)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = self.root
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=1)

        header = family.make_header(root, "Mixture", "detect components + ratio")
        header.grid(row=0, column=0, columnspan=2, sticky="ew")

        side = ctk.CTkScrollableFrame(root, width=340, corner_radius=0,
                                      fg_color=brand.SIDE_FILL)
        side.grid(row=1, column=0, sticky="nsew")

        ctk.CTkLabel(side, text=brand.APP_NAME,
                     font=ctk.CTkFont(size=15, weight="bold"), anchor="w"
                     ).pack(fill="x", padx=14, pady=(14, 0))
        ctk.CTkLabel(side, text=brand.APP_TAGLINE,
                     font=ctk.CTkFont(size=12), text_color=brand.SUBTLE, anchor="w"
                     ).pack(fill="x", padx=14, pady=(0, 10))

        ctk.CTkButton(side, text="Load pure references", height=40, **self._btn(),
                      command=self.load_references).pack(fill="x", padx=14, pady=6)
        self.pure_var = tk.StringVar(value="no references loaded")
        ctk.CTkLabel(side, textvariable=self.pure_var, anchor="w", justify="left",
                     wraplength=300, font=ctk.CTkFont(size=11), text_color=brand.SUBTLE
                     ).pack(fill="x", padx=14)

        ctk.CTkButton(side, text="Load unknown CSV", height=40, **self._btn(),
                      command=self.load_unknown).pack(fill="x", padx=14, pady=(10, 6))
        self.unk_var = tk.StringVar(value="no unknown loaded")
        ctk.CTkLabel(side, textvariable=self.unk_var, anchor="w", justify="left",
                     wraplength=300, font=ctk.CTkFont(size=11), text_color=brand.SUBTLE
                     ).pack(fill="x", padx=14)

        # options
        opt = ctk.CTkFrame(side, fg_color="transparent")
        opt.pack(fill="x", padx=14, pady=10)
        ctk.CTkLabel(opt, text="detection threshold", anchor="w",
                     font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        self.thr_var = tk.StringVar(value="0.30")
        ctk.CTkEntry(opt, textvariable=self.thr_var, width=80, height=26
                     ).grid(row=0, column=1, padx=6, pady=3)
        ctk.CTkLabel(opt, text="max components", anchor="w",
                     font=ctk.CTkFont(size=12)).grid(row=1, column=0, sticky="w")
        self.maxc_var = tk.StringVar(value="3")
        ctk.CTkEntry(opt, textvariable=self.maxc_var, width=80, height=26
                     ).grid(row=1, column=1, padx=6, pady=3)

        ctk.CTkButton(side, text="Run analysis", height=48, **self._runbtn(),
                      font=ctk.CTkFont(size=13, weight="bold"),
                      command=self.run).pack(fill="x", padx=14, pady=(4, 8))

        self.summary_var = tk.StringVar(value="load references + unknown, then Run.")
        ctk.CTkLabel(side, textvariable=self.summary_var, anchor="w", justify="left",
                     wraplength=300, font=ctk.CTkFont(size=13)
                     ).pack(fill="x", padx=14, pady=(2, 8))

        ctk.CTkButton(side, text="Export results CSV", height=32, **self._btn(),
                      command=self.export_csv).pack(fill="x", padx=14, pady=3)
        ctk.CTkButton(side, text="Export dashboard PNG", height=32, **self._btn(),
                      command=self.export_png).pack(fill="x", padx=14, pady=3)

        ctk.CTkLabel(side, text="appearance", anchor="w",
                     font=ctk.CTkFont(size=12)).pack(fill="x", padx=14, pady=(12, 0))
        ctk.CTkOptionMenu(side, values=["System", "Light", "Dark"], width=120,
                          command=ctk.set_appearance_mode).pack(padx=14, pady=(2, 14),
                                                                anchor="w")

        # ---------- main dashboard ----------
        main = ctk.CTkScrollableFrame(root, corner_radius=0, fg_color=brand.MAIN_FILL)
        main.grid(row=1, column=1, sticky="nsew")
        main.grid_columnconfigure((0, 1), weight=1)

        spec_card = ctk.CTkFrame(main, corner_radius=10, fg_color=brand.CARD_FILL)
        spec_card.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=12, pady=(12, 6))
        ctk.CTkLabel(spec_card, text="Spectrum + NNLS reconstruction",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(8, 0))
        self.fig_spec = Figure(figsize=(7.4, 2.8), dpi=100)
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=spec_card)
        self.canvas_spec.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=8)

        comp_card = ctk.CTkFrame(main, corner_radius=10, fg_color=brand.CARD_FILL)
        comp_card.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        ctk.CTkLabel(comp_card, text="Composition (ratio)",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(8, 0))
        self.fig_comp = Figure(figsize=(3.6, 3.0), dpi=100)
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=comp_card)
        self.canvas_comp.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=8)

        tbl_card = ctk.CTkFrame(main, corner_radius=10, fg_color=brand.CARD_FILL)
        tbl_card.grid(row=1, column=1, sticky="nsew", padx=12, pady=6)
        ctk.CTkLabel(tbl_card, text="Per-component read-out",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=12, pady=(8, 0))
        self.table = ctk.CTkTextbox(tbl_card, height=230, font=ctk.CTkFont(family="Courier", size=12))
        self.table.pack(fill="both", expand=True, padx=10, pady=8)

    # --------------------------------------------------------------- actions
    def load_references(self):
        path = filedialog.askopenfilename(
            title="pure.csv (wavenumber + one column per pure compound)",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.axis, self.comp_names, raw = load_csv(path)
            self.pures = preprocess(raw)
            self.clf = None
            self.pure_var.set(f"{os.path.basename(path)}\n{len(self.comp_names)} pures: "
                              + ", ".join(self.comp_names))
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def load_unknown(self):
        path = filedialog.askopenfilename(
            title="unknown.csv (wavenumber + spectra to analyse)",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            self.unk_axis, self.unk_names, raw = load_csv(path)
            self.unk = preprocess(raw)
            self.unk_var.set(f"{os.path.basename(path)}\n{len(self.unk_names)} spectra")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def run(self):
        if self.pures is None or self.unk is None:
            messagebox.showwarning("Missing data", "Load pure references and an unknown CSV first.")
            return
        try:
            thr = float(self.thr_var.get()); maxc = int(self.maxc_var.get())
        except ValueError:
            messagebox.showwarning("Bad option", "threshold/max components must be numbers.")
            return
        # train on pures (cache)
        if self.clf is None:
            self.summary_var.set("training on pure spectra…"); self.root.update()
            self.clf = SERSMixtureClassifier(
                self.comp_names, prob_threshold=thr, max_components=maxc,
                augment=AugmentConfig(n_per_pure=150))
            self.clf.fit(self.pures)
        else:
            self.clf.prob_threshold = thr; self.clf.max_components = maxc

        # analyse the FIRST unknown spectrum for the dashboard
        y = self.unk[0]
        det = self.clf.predict(y, return_details=True)[0]
        B, yhat, res = additive_residual(y, self.pures)
        # equal-response ratio proxy (no calibration): B normalized
        ratio = B / (B.sum() + 1e-12)

        self.last_results = {"names": self.comp_names, "B": B, "ratio": ratio,
                             "detected": det, "residual": res}
        self._draw_spectrum(y, yhat, res)
        self._draw_composition(ratio)
        self._fill_table(det, ratio, res)
        top = det["components"]
        self.summary_var.set(f"detected: {', '.join(top)}\nfit residual {res*100:.1f}%")

    # --------------------------------------------------------------- drawing
    def _draw_spectrum(self, y, yhat, res):
        f = self.fig_spec; f.clear(); ax = f.add_subplot(111)
        x = self.axis if self.axis is not None else np.arange(len(y))
        ax.plot(x, y, lw=1.2, label="measured", color=brand.SERIES[0])
        ax.plot(x, yhat, lw=1.0, ls="--", label=f"reconstruction (res {res*100:.1f}%)",
                color=brand.SERIES[3])
        ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_ylabel("intensity (norm.)")
        ax.legend(fontsize=8); f.tight_layout(); self.canvas_spec.draw()

    def _draw_composition(self, ratio):
        f = self.fig_comp; f.clear(); ax = f.add_subplot(111)
        names = self.comp_names
        keep = [(n, r) for n, r in zip(names, ratio) if r > 0.01]
        if keep:
            labels, vals = zip(*keep)
            ax.pie(vals, labels=[f"{n}\n{v*100:.0f}%" for n, v in keep],
                   colors=brand.SERIES[:len(vals)], startangle=90,
                   textprops={"fontsize": 9})
        ax.set_aspect("equal"); f.tight_layout(); self.canvas_comp.draw()

    def _fill_table(self, det, ratio, res):
        self.table.delete("1.0", "end")
        proba = det.get("proba", {})
        lines = [f"{'component':10s} {'present':>8} {'ratio':>7} {'prob':>6}", "-" * 34]
        for i, n in enumerate(self.comp_names):
            present = "yes" if n in det["components"] else "-"
            p = proba.get(n, float("nan"))
            lines.append(f"{n:10s} {present:>8} {ratio[i]*100:6.1f}% "
                         f"{('' if p!=p else f'{p:.2f}'):>6}")
        lines += ["-" * 34, f"reconstruction residual: {res*100:.1f}%",
                  "(ratio = signal-weighted; calibrate response",
                  " factors for molar ratio)"]
        self.table.insert("1.0", "\n".join(lines))

    # --------------------------------------------------------------- export
    def export_csv(self):
        if not self.last_results:
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        r = self.last_results
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["component", "detected", "signal_ratio", "B"])
            for i, n in enumerate(r["names"]):
                w.writerow([n, n in r["detected"]["components"],
                            f"{r['ratio'][i]:.4f}", f"{r['B'][i]:.4f}"])
        messagebox.showinfo("Exported", os.path.basename(path))

    def export_png(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            self.fig_spec.savefig(path, dpi=200, bbox_inches="tight")
            messagebox.showinfo("Exported", os.path.basename(path))


def main():
    root = ctk.CTk()
    SERSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
