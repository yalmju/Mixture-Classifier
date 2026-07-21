"""
SERS Discriminator — CustomTkinter UI shell.

Reuses all the computation, baseline correction, metric and saving logic
from sers_discriminator.py. This file only owns the GUI: a Plaspector-style
sidebar with step cards on the left and a tabbed analysis area on the right.

Run with:
    python sers_discriminator_ctk.py
or build the exe via the .spec file (see sers_discriminator.spec).
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

import customtkinter as ctk

# Pull all computation + saving code in unchanged
import sers_discriminator as sd
import family

BUILD_TAG = "build 2026-05-21 CTk v1"

family.apply()     # shared UNMIXR light look

# Matplotlib font setup: Arial 14pt for embedded charts (keep readable inside
# CTk frames; saved-PNG path still uses 18pt via sers_discriminator's rcParams)
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]


# =============================================================================
# CTk helpers (mirrored from Plaspector)
# =============================================================================

def section_frame(parent, title=None):
    """Rounded card with an optional small bold header label."""
    f = ctk.CTkFrame(parent, corner_radius=12, fg_color=family.CARD,
                     border_width=1, border_color=family.LINE)
    if title:
        ctk.CTkLabel(
            f, text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 2))
    return f


def plot_card(parent):
    """White card to host a matplotlib figure (so plots stay readable in
    both dark and light modes)."""
    return ctk.CTkFrame(parent, corner_radius=10,
                        fg_color=("#ffffff", "#ffffff"))


# Accent colors per analysis tile, mirroring Plaspector's convention.
# view-tab accents drawn from the UNMIXR family (teal primary, then blue / coral
# / purple for the other categories)
TILE_COLORS = {
    "maps":        {"fg": ("#0f9d6b", "#0f9d6b"), "hover": ("#0c855a", "#0c855a")},
    "reliability": {"fg": ("#1a73e8", "#1a73e8"), "hover": ("#155ec2", "#155ec2")},
    "validation":  {"fg": ("#d8542a", "#d8542a"), "hover": ("#b8441f", "#b8441f")},
    "stats":       {"fg": ("#6b5fd6", "#6b5fd6"), "hover": ("#574bc4", "#574bc4")},
}


# =============================================================================
# Spectrum popup (pixel-click target)
# =============================================================================

class SpectrumPopup(ctk.CTkToplevel):
    """Pop a window showing the spectrum at one pixel + reference overlays."""

    def __init__(self, master, wn, spectrum, title,
                 ref_specs=None, ref_names=None, ref_tints=None):
        super().__init__(master)
        self.title(title)
        self.geometry("960x560")
        self.minsize(640, 380)

        self._wn = np.asarray(wn)
        self._spec = np.asarray(spectrum)
        self._title = title

        hdr = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(14, 4))
        ctk.CTkLabel(
            hdr, text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).pack(side="left")

        ctk.CTkButton(
            hdr, text="Save CSV",
            command=self._save_csv,
            width=92, height=28, corner_radius=8,
            fg_color="transparent", border_width=1,
            border_color=("#d1d5db", "#374151"),
            text_color=("#374151", "#d1d5db"),
            hover_color=("#f3f4f6", "#1f2937"),
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            hdr, text="Save PNG",
            command=self._save_png,
            width=92, height=28, corner_radius=8,
            fg_color="transparent", border_width=1,
            border_color=("#d1d5db", "#374151"),
            text_color=("#374151", "#d1d5db"),
            hover_color=("#f3f4f6", "#1f2937"),
        ).pack(side="right", padx=4)

        card = plot_card(self)
        card.pack(fill="both", expand=True, padx=16, pady=8)

        self._fig = Figure(figsize=(8.4, 4.4), dpi=100, facecolor="white")
        ax = self._fig.add_subplot(111)
        ax.plot(wn, spectrum, color="#111", lw=1.0, label="Pixel spectrum")
        if ref_specs is not None and ref_names is not None:
            s_peak = max(np.max(np.abs(spectrum)), 1e-9)
            for j, (name, ref) in enumerate(zip(ref_names, ref_specs)):
                r_peak = max(np.max(np.abs(ref)), 1e-9)
                tint = (ref_tints[j] if ref_tints is not None
                        else "#888")
                ax.plot(wn, ref * (s_peak / r_peak) * 0.85,
                        color=tint, lw=0.9, alpha=0.7,
                        label=f"{name} (scaled)")
        ax.set_xlabel(r"Wavenumber ($\mathrm{cm^{-1}}$)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=9, loc="upper right",
                  frameon=True, facecolor="white", edgecolor="#e5e7eb")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._fig.tight_layout()

        canvas = FigureCanvasTkAgg(self._fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

        tb_frame = ctk.CTkFrame(self, corner_radius=0,
                                fg_color="transparent")
        tb_frame.pack(fill="x", padx=16)
        NavigationToolbar2Tk(canvas, tb_frame).update()

    def _save_png(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg")])
        if not path:
            return
        self._fig.savefig(path, dpi=200, bbox_inches="tight")

    def _save_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("Wavenumber_cm-1,Intensity\n")
            for w, v in zip(self._wn, self._spec):
                f.write(f"{w:.4f},{v:.6g}\n")


# =============================================================================
# Save-selection dialog (CTkToplevel replacement)
# =============================================================================

def ask_initial_peaks_ctk(master, refs, top_peaks_per_ref, tints):
    """Modal popup letting the user pick the initial wavenumber to display
    on each reference's intensity map.

    Args:
        master           : CTk root (used as parent for the Toplevel)
        refs             : list of dicts with 'name' key, one per reference
        top_peaks_per_ref: list of [(wn, intensity), ...] from
                           sd.detect_top_peaks, one list per reference
        tints            : per-ref tint colors for the colored ref-name label

    Returns:
        list[float] of chosen wavenumbers, one per ref (length == n_ref).
        Returns None if the user cancels — caller should fall back to the
        strongest peak in that case.
    """
    n_ref = len(refs)

    win = ctk.CTkToplevel(master)
    win.title("Pick initial wavenumber per reference")
    win.geometry("560x420")
    win.minsize(460, 320)
    win.transient(master)

    # Bottom button row (pinned)
    btn_row = ctk.CTkFrame(win, corner_radius=0, fg_color="transparent")
    btn_row.pack(side="bottom", fill="x", padx=14, pady=12)

    # Header
    hdr = ctk.CTkFrame(win, corner_radius=0, fg_color="transparent")
    hdr.pack(fill="x", padx=18, pady=(16, 4))
    ctk.CTkLabel(hdr,
                 text="Pick initial peak per reference",
                 font=ctk.CTkFont(size=14, weight="bold"),
                 anchor="w").pack(anchor="w")
    ctk.CTkLabel(hdr,
                 text=("Defaults to each reference's strongest peak. "
                       "Pick from detected top-5 peaks or type any "
                       "wavenumber (cm⁻¹)."),
                 font=ctk.CTkFont(size=10),
                 text_color=("#6b7280", "#9ca3af"),
                 anchor="w", wraplength=520, justify="left"
                 ).pack(anchor="w", pady=(2, 0))

    # Scrollable body
    body = ctk.CTkScrollableFrame(win, label_text="")
    body.pack(fill="both", expand=True, padx=14, pady=(8, 4))

    combos = []
    for i, r in enumerate(refs):
        row = ctk.CTkFrame(body, corner_radius=8)
        row.pack(fill="x", pady=4, padx=4)

        # Tint swatch + name
        tint_hex = mcolors.to_hex(tints[i] if i < len(tints) else (0.5, 0.5, 0.5))
        sw = ctk.CTkFrame(row, width=18, height=18, corner_radius=4,
                          fg_color=tint_hex,
                          border_width=1, border_color=("#999", "#666"))
        sw.pack(side="left", padx=(10, 6), pady=8)
        sw.pack_propagate(False)
        ctk.CTkLabel(row, text=r["name"],
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=tint_hex,
                     anchor="w", width=160
                     ).pack(side="left", padx=(0, 6), pady=8)

        # Top-5 peaks as combobox values (editable so user can type any wn)
        peaks = top_peaks_per_ref[i] if i < len(top_peaks_per_ref) else []
        values = [f"{w:.1f}" for w, _ in peaks] if peaks else ["1000.0"]
        cbo = ctk.CTkComboBox(
            row, values=values,
            width=140, height=28, corner_radius=6,
        )
        cbo.set(values[0])
        cbo.pack(side="left", padx=(0, 8), pady=8)
        ctk.CTkLabel(row, text="cm⁻¹",
                     font=ctk.CTkFont(size=11),
                     text_color=("#6b7280", "#9ca3af")
                     ).pack(side="left", padx=(0, 8), pady=8)
        combos.append(cbo)

    result = {"wn": None}

    def on_ok():
        out = []
        for cbo in combos:
            try:
                out.append(float(cbo.get().strip()))
            except ValueError:
                # Fall back to first value in that combobox's list
                vals = cbo.cget("values")
                out.append(float(vals[0]) if vals else 1000.0)
        result["wn"] = out
        win.destroy()

    def on_cancel():
        result["wn"] = None
        win.destroy()

    ctk.CTkButton(btn_row, text="Use these",
                  command=on_ok,
                  font=ctk.CTkFont(size=12, weight="bold"),
                  height=32, corner_radius=8,
                  ).pack(side="right", padx=4)
    ctk.CTkButton(btn_row, text="Use defaults",
                  command=on_cancel,
                  height=32, corner_radius=8,
                  fg_color="transparent", border_width=1,
                  border_color=("#d1d5db", "#374151"),
                  text_color=("#374151", "#d1d5db"),
                  hover_color=("#f3f4f6", "#1f2937"),
                  ).pack(side="right", padx=4)
    win.protocol("WM_DELETE_WINDOW", on_cancel)
    win.grab_set()
    win.focus_set()
    win.wait_window()
    return result["wn"]


def ask_save_selection_ctk(master, current_metric_name):
    """Modal popup with checkboxes for which output groups to save.
    Returns the selection dict (or None if cancelled)."""
    win = ctk.CTkToplevel(master)
    win.title("Save selection")
    win.geometry("620x720")
    win.minsize(520, 560)
    win.transient(master)

    flags = {
        "intensity":           tk.BooleanVar(value=True),
        "current_metric_only": tk.BooleanVar(value=True),
        "all_metrics":         tk.BooleanVar(value=False),
        "combined_rgb":        tk.BooleanVar(value=True),
        "mean_spectrum":       tk.BooleanVar(value=True),
        "mcr_spectra":         tk.BooleanVar(value=False),
        "histograms":          tk.BooleanVar(value=False),
        "confidence":          tk.BooleanVar(value=False),
        "agreement":           tk.BooleanVar(value=True),
        "validation":          tk.BooleanVar(value=True),
        "per_metric_csvs":     tk.BooleanVar(value=False),
        "summary_csvs":        tk.BooleanVar(value=True),
        "raw_versions":        tk.BooleanVar(value=False),
    }
    result = {"selection": None}

    def apply_preset(p):
        presets = {
            "essential": {
                "intensity": True, "current_metric_only": True,
                "all_metrics": False, "combined_rgb": True,
                "mean_spectrum": True, "mcr_spectra": False,
                "histograms": False, "confidence": False,
                "agreement": True, "validation": True,
                "per_metric_csvs": False, "summary_csvs": True,
                "raw_versions": False,
            },
            "current_only": {k: False for k in flags},
            "all": {k: True for k in flags},
        }
        v = presets[p]
        if p == "current_only":
            v["intensity"] = True
            v["current_metric_only"] = True
            v["combined_rgb"] = True
            v["summary_csvs"] = True
        if p == "all":
            v["current_metric_only"] = False
        for k, val in v.items():
            flags[k].set(val)

    # Bottom buttons (pinned)
    btn_row = ctk.CTkFrame(win, corner_radius=0, fg_color="transparent")
    btn_row.pack(side="bottom", fill="x", padx=14, pady=12)

    def on_ok():
        result["selection"] = {k: v.get() for k, v in flags.items()}
        result["selection"]["current_metric_name"] = current_metric_name
        win.destroy()

    def on_cancel():
        result["selection"] = None
        win.destroy()

    ctk.CTkButton(btn_row, text="Save",
                  command=on_ok,
                  height=34, corner_radius=8,
                  font=ctk.CTkFont(size=12, weight="bold"),
                  ).pack(side="right", padx=4)
    ctk.CTkButton(btn_row, text="Cancel",
                  command=on_cancel,
                  height=34, corner_radius=8,
                  fg_color="transparent", border_width=1,
                  border_color=("#d1d5db", "#374151"),
                  text_color=("#374151", "#d1d5db"),
                  hover_color=("#f3f4f6", "#1f2937"),
                  ).pack(side="right", padx=4)
    win.protocol("WM_DELETE_WINDOW", on_cancel)

    # Scrollable content
    body = ctk.CTkScrollableFrame(win, label_text="")
    body.pack(fill="both", expand=True, padx=14, pady=(14, 4))

    # Presets
    ps = section_frame(body, "Presets")
    ps.pack(fill="x", pady=(0, 8))
    ps_inner = ctk.CTkFrame(ps, fg_color="transparent")
    ps_inner.pack(fill="x", padx=12, pady=(0, 10))
    ctk.CTkButton(ps_inner, text="Essential",
                  command=lambda: apply_preset("essential"),
                  height=28, corner_radius=8,
                  ).pack(side="left", padx=4)
    ctk.CTkButton(ps_inner, text="Current only",
                  command=lambda: apply_preset("current_only"),
                  height=28, corner_radius=8,
                  ).pack(side="left", padx=4)
    ctk.CTkButton(ps_inner, text="All",
                  command=lambda: apply_preset("all"),
                  height=28, corner_radius=8,
                  ).pack(side="left", padx=4)

    # Maps
    m = section_frame(body, "Maps (per-ref PNGs)")
    m.pack(fill="x", pady=4)
    ctk.CTkCheckBox(m, text="Intensity @ selected WN",
                    variable=flags["intensity"]).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(m,
                    text=f"Only current metric ({current_metric_name})",
                    variable=flags["current_metric_only"]
                    ).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(m, text="All 9 metric maps",
                    variable=flags["all_metrics"]).pack(anchor="w", padx=14, pady=(2, 10))

    # RGB / Spectra
    r = section_frame(body, "Combined RGB / Spectra")
    r.pack(fill="x", pady=4)
    ctk.CTkCheckBox(r, text="Combined RGB (NNLS/MCR/CLS, 3 PNGs)",
                    variable=flags["combined_rgb"]).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(r, text="Mean spectrum overlay",
                    variable=flags["mean_spectrum"]).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(r, text="MCR-ALS resolved spectra",
                    variable=flags["mcr_spectra"]).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(r, text="Histograms vs each reference",
                    variable=flags["histograms"]).pack(anchor="w", padx=14, pady=(2, 10))

    # Reliability
    rel = section_frame(body, "Reliability")
    rel.pack(fill="x", pady=4)
    ctk.CTkCheckBox(rel, text="Confidence gap + entropy (NNLS+MCR)",
                    variable=flags["confidence"]).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(rel, text="Argmax + agreement + consensus",
                    variable=flags["agreement"]).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(rel, text="Classifier validation (CV+synth+sweep)",
                    variable=flags["validation"]).pack(anchor="w", padx=14, pady=(2, 10))

    # CSVs
    c = section_frame(body, "CSVs and raw variants")
    c.pack(fill="x", pady=4)
    ctk.CTkCheckBox(c, text="Per-metric pixel CSVs",
                    variable=flags["per_metric_csvs"]).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(c, text="Summary CSVs (statistics + reliability)",
                    variable=flags["summary_csvs"]).pack(anchor="w", padx=14, pady=2)
    ctk.CTkCheckBox(c, text="Raw (no-axis) PNG variants",
                    variable=flags["raw_versions"]).pack(anchor="w", padx=14, pady=(2, 10))

    # Modal
    win.grab_set()
    win.focus_set()
    win.wait_window()
    return result["selection"]


# =============================================================================
# Main app
# =============================================================================

class SERSDiscriminatorApp(ctk.CTkFrame):
    """Embeddable SERS Discriminator.

    Standalone:   SERSDiscriminatorApp()            -> owns its own CTk window
    Embedded:     SERSDiscriminatorApp(container,   -> fills the given container
                                       embedded=True)   frame (used by the suite
                                                         launcher)
    """

    def __init__(self, master=None, embedded=False):
        self._embedded = embedded
        self._own_root = None
        if master is None:                    # standalone: create our own window
            self._own_root = ctk.CTk()
            master = self._own_root
        super().__init__(master, fg_color="transparent")

        if self._own_root is not None:
            self._own_root.title(f"SERS Discriminator  —  {BUILD_TAG}")
            self._own_root.geometry("1540x940")
            self._own_root.minsize(1240, 800)
        # fill whatever container we were given (our own root, or the suite host)
        self.pack(fill="both", expand=True)

        # ---- State ----
        self.ref_files: list[str] = []
        self.test_file: str | None = None
        self.result: dict | None = None
        self.cmaps: list[str] = []
        self.tints: list[tuple] = []
        self.rgb_assignment: list[str] = []
        self.selected_wn: list[float] = []
        self.current_metric: str = "NNLS_norm"
        self.test_mean: np.ndarray | None = None

        # Embedded matplotlib resources (created when result arrives)
        self.maps_fig = None
        self.maps_canvas = None
        self.spec_ax = None
        self.test_line = None
        self.spec_overlay_lines: list = []
        self.sel_lines: list = []
        self.intensity_axes: list = []
        self.intensity_ims: list = []
        self.intensity_titles: list = []
        self.prob_axes: list = []
        self.prob_ims: list = []
        self.prob_titles: list = []
        self.rgb_ax = None
        self.rgb_im = None
        self.rgb_legend_box = None
        self.spec_state = {"mode": "mean", "pixel": None}

        # Per-ref control widgets (filled after load)
        self.ref_control_widgets: list[dict] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI scaffold
    # ------------------------------------------------------------------

    def _build_ui(self):
        # 3-column layout: [sidebar] [splitter] [content]
        # The splitter is a thin draggable bar that lets the user resize
        # the sidebar live with the mouse.
        self.SIDEBAR_W = 420
        self.SIDEBAR_MIN = 220
        self.SIDEBAR_MAX = 800
        self.grid_columnconfigure(0, weight=0, minsize=self.SIDEBAR_W)
        self.grid_columnconfigure(1, weight=0, minsize=6)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=0)     # header bar
        self.grid_rowconfigure(1, weight=1)     # content

        header = family.make_header(self, "SERS map", "per-pixel identify + unmix")
        header.grid(row=0, column=0, columnspan=3, sticky="ew")

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=self.SIDEBAR_W,
                                    corner_radius=0)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self._build_sidebar()
        self.update_idletasks()
        self.sidebar.configure(width=self.SIDEBAR_W)

        # --- Splitter (draggable) ---
        # Slightly lighter / darker than the sidebar so the user can see
        # where to grab; cursor switches to a horizontal-resize arrow.
        self.splitter = ctk.CTkFrame(
            self, width=6, corner_radius=0,
            fg_color=("#d4d4d8", "#3f3f46"),
        )
        self.splitter.grid(row=1, column=1, sticky="ns")
        self.splitter.configure(cursor="sb_h_double_arrow")
        self.splitter.bind("<B1-Motion>", self._on_splitter_drag)
        self.splitter.bind("<Double-Button-1>",
                            lambda _e: self._set_sidebar_width(420))

        # --- Content area ---
        self.content = ctk.CTkFrame(self, corner_radius=0,
                                    fg_color=family.PAGE)
        self.content.grid(row=1, column=2, sticky="nsew")

        # Panels created on demand
        self.panels: dict[str, ctk.CTkFrame] = {}
        self.current_panel: str | None = None
        self._build_welcome()

    # ---- Sidebar resize --------------------------------------------

    def _set_sidebar_width(self, new_w):
        new_w = max(self.SIDEBAR_MIN, min(int(new_w), self.SIDEBAR_MAX))
        self.SIDEBAR_W = new_w
        self.grid_columnconfigure(0, weight=0, minsize=new_w)
        self.sidebar.configure(width=new_w)
        # Inner scroll frame also needs to grow/shrink with the sidebar
        try:
            inner_w = max(new_w - 22, self.SIDEBAR_MIN - 22)
            self._scroll_frame.configure(width=inner_w)
            self._scroll_frame._parent_canvas.configure(width=inner_w)
        except Exception:
            pass
        self.update_idletasks()

    def _on_splitter_drag(self, event):
        # event.x_root is screen X; convert to width relative to window left.
        new_w = event.x_root - self.winfo_rootx()
        self._set_sidebar_width(new_w)

    # ---- Sidebar -----------------------------------------------------

    def _build_sidebar(self):
        # IMPORTANT: pack the bottom-anchored Appearance row FIRST so the
        # scrollable area below it gets the remaining space cleanly. (If we
        # pack it after a side=top expand=True widget, Tk can occasionally
        # squeeze it off-screen.)
        self.theme_row = ctk.CTkFrame(self.sidebar, corner_radius=0,
                                       fg_color="transparent")
        self.theme_row.pack(side="bottom", fill="x", padx=14, pady=12)

        # Scrollable inside the sidebar so all steps fit even on small
        # screens. We size the scroll frame to (sidebar - scrollbar) and
        # pin it to fill so the inner content gets the full visible width.
        SCROLL_W = self.SIDEBAR_W - 22
        scroll = ctk.CTkScrollableFrame(self.sidebar, corner_radius=0,
                                        fg_color="transparent",
                                        label_text="",
                                        width=SCROLL_W)
        scroll.pack(fill="both", expand=True)
        self._scroll_frame = scroll
        # The inner canvas needs to be told its width too, otherwise child
        # widgets may report wider than the visible area and clip.
        try:
            scroll._parent_canvas.configure(width=SCROLL_W)
            scroll.update_idletasks()
        except Exception:
            pass

        # Mouse wheel propagation: bind globally while the cursor is inside
        # the scroll frame, so wheel events from any child widget bubble up
        # to the canvas. CTkScrollableFrame doesn't do this by default.
        def _on_wheel(event):
            try:
                # Windows: event.delta is multiple of 120 per notch
                delta = -int(event.delta / 120) if event.delta else 0
                if delta == 0:
                    # Linux: button-4 / button-5 events have num set
                    delta = -1 if getattr(event, "num", 0) == 4 else 1
                scroll._parent_canvas.yview_scroll(delta, "units")
            except Exception:
                pass

        def _enter(_e):
            scroll._parent_canvas.bind_all("<MouseWheel>", _on_wheel)
            scroll._parent_canvas.bind_all("<Button-4>", _on_wheel)
            scroll._parent_canvas.bind_all("<Button-5>", _on_wheel)

        def _leave(_e):
            scroll._parent_canvas.unbind_all("<MouseWheel>")
            scroll._parent_canvas.unbind_all("<Button-4>")
            scroll._parent_canvas.unbind_all("<Button-5>")

        scroll.bind("<Enter>", _enter)
        scroll.bind("<Leave>", _leave)

        # Brand
        brand = ctk.CTkFrame(scroll, corner_radius=0,
                             fg_color="transparent")
        brand.pack(fill="x", padx=18, pady=(16, 6))
        ctk.CTkLabel(brand, text="SERS Discriminator",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(brand,
                     text="Reference-based SERS map classifier",
                     font=ctk.CTkFont(size=11),
                     text_color=("#6b7280", "#9ca3af"),
                     anchor="w").pack(anchor="w")

        # STEP 1 — References
        s1 = section_frame(scroll, "STEP 1 — REFERENCES")
        s1.pack(fill="x", padx=14, pady=(14, 6))
        self.ref_count_var = tk.StringVar(value="0 references")
        ctk.CTkLabel(s1, textvariable=self.ref_count_var,
                     font=ctk.CTkFont(size=11),
                     text_color=("#374151", "#d1d5db"),
                     anchor="w"
                     ).pack(fill="x", padx=14, pady=(0, 4))
        # Listbox (CTk has no listbox — use tk.Listbox styled to fit)
        self.ref_listbox = tk.Listbox(
            s1, height=4,
            font=("Segoe UI", 10),
            relief="flat", borderwidth=1,
            highlightthickness=0,
            background="#f9fafb",
            foreground="#111827",
            selectbackground="#3b82f6",
            selectforeground="#ffffff",
            selectmode=tk.EXTENDED,
        )
        self.ref_listbox.pack(fill="x", padx=14, pady=(0, 6))
        btn_row = ctk.CTkFrame(s1, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkButton(btn_row, text="Add",
                      command=self.add_refs,
                      width=70, height=28, corner_radius=8,
                      ).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="Remove",
                      command=self.remove_refs,
                      width=70, height=28, corner_radius=8,
                      fg_color="transparent", border_width=1,
                      border_color=("#d1d5db", "#374151"),
                      text_color=("#374151", "#d1d5db"),
                      hover_color=("#f3f4f6", "#1f2937"),
                      ).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="Clear",
                      command=self.clear_refs,
                      width=70, height=28, corner_radius=8,
                      fg_color="transparent", border_width=1,
                      border_color=("#d1d5db", "#374151"),
                      text_color=("#374151", "#d1d5db"),
                      hover_color=("#f3f4f6", "#1f2937"),
                      ).pack(side="left", padx=2)

        # STEP 2 — Test CSV
        s2 = section_frame(scroll, "STEP 2 — TEST CSV")
        s2.pack(fill="x", padx=14, pady=6)
        self.test_var = tk.StringVar(value="(no file selected)")
        ctk.CTkLabel(s2, textvariable=self.test_var,
                     font=ctk.CTkFont(size=11),
                     text_color=("#374151", "#d1d5db"),
                     anchor="w", wraplength=300, justify="left"
                     ).pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkButton(s2, text="Browse CSV…",
                      command=self.browse_test,
                      height=32, corner_radius=8,
                      ).pack(fill="x", padx=14, pady=(0, 12))

        # STEP 3 — Processing options
        s3 = section_frame(scroll, "STEP 3 — PROCESSING")
        s3.pack(fill="x", padx=14, pady=6)
        self.bl_refs_var = tk.BooleanVar(value=True)
        self.bl_test_var = tk.BooleanVar(value=True)
        self.run_mcr_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(s3, text="arPLS on references",
                        variable=self.bl_refs_var,
                        font=ctk.CTkFont(size=11)
                        ).pack(anchor="w", padx=14, pady=2)
        ctk.CTkCheckBox(s3, text="arPLS on test",
                        variable=self.bl_test_var,
                        font=ctk.CTkFont(size=11)
                        ).pack(anchor="w", padx=14, pady=2)
        ctk.CTkCheckBox(s3, text="Run MCR-ALS (slow on big maps)",
                        variable=self.run_mcr_var,
                        font=ctk.CTkFont(size=11)
                        ).pack(anchor="w", padx=14, pady=(2, 4))

        wn_row = ctk.CTkFrame(s3, fg_color="transparent")
        wn_row.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(wn_row, text="WN crop:",
                     font=ctk.CTkFont(size=11),
                     text_color=("#6b7280", "#9ca3af")
                     ).pack(side="left", padx=(0, 6))
        self.wn_min_var = tk.StringVar(value="400")
        self.wn_max_var = tk.StringVar(value="1800")
        ctk.CTkEntry(wn_row, textvariable=self.wn_min_var,
                     width=64, height=26).pack(side="left", padx=2)
        ctk.CTkLabel(wn_row, text="–",
                     text_color=("#6b7280", "#9ca3af")
                     ).pack(side="left", padx=4)
        ctk.CTkEntry(wn_row, textvariable=self.wn_max_var,
                     width=64, height=26).pack(side="left", padx=2)
        ctk.CTkLabel(wn_row, text="cm⁻¹",
                     font=ctk.CTkFont(size=10),
                     text_color=("#6b7280", "#9ca3af")
                     ).pack(side="left", padx=4)

        # Load button + status
        self.load_btn = ctk.CTkButton(
            scroll, text="Load & Process",
            command=self.load_and_process,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=42, corner_radius=10,
        )
        self.load_btn.pack(fill="x", padx=14, pady=(10, 6))

        self.status_var = tk.StringVar(value="No data loaded.")
        ctk.CTkLabel(scroll, textvariable=self.status_var,
                     text_color=("#059669", "#10b981"),
                     font=ctk.CTkFont(size=11),
                     anchor="w", wraplength=300, justify="left"
                     ).pack(fill="x", padx=14, pady=(0, 12))

        # STEP 4 — View (panel switcher)
        s4 = section_frame(scroll, "STEP 4 — VIEW")
        s4.pack(fill="x", padx=14, pady=6)
        self.btn_maps = ctk.CTkButton(
            s4, text="🗺  Maps",
            command=lambda: self.show_panel("maps"),
            font=ctk.CTkFont(size=12, weight="bold"),
            height=42, corner_radius=10,
            fg_color=TILE_COLORS["maps"]["fg"],
            hover_color=TILE_COLORS["maps"]["hover"],
            state="disabled",
        )
        self.btn_maps.pack(fill="x", padx=14, pady=(6, 4))
        self.btn_reliability = ctk.CTkButton(
            s4, text="📈  Reliability",
            command=lambda: self.show_panel("reliability"),
            font=ctk.CTkFont(size=12, weight="bold"),
            height=42, corner_radius=10,
            fg_color=TILE_COLORS["reliability"]["fg"],
            hover_color=TILE_COLORS["reliability"]["hover"],
            state="disabled",
        )
        self.btn_reliability.pack(fill="x", padx=14, pady=4)
        self.btn_validation = ctk.CTkButton(
            s4, text="🎯  Validation",
            command=lambda: self.show_panel("validation"),
            font=ctk.CTkFont(size=12, weight="bold"),
            height=42, corner_radius=10,
            fg_color=TILE_COLORS["validation"]["fg"],
            hover_color=TILE_COLORS["validation"]["hover"],
            state="disabled",
        )
        self.btn_validation.pack(fill="x", padx=14, pady=4)
        self.btn_stats = ctk.CTkButton(
            s4, text="📋  Stats",
            command=lambda: self.show_panel("stats"),
            font=ctk.CTkFont(size=12, weight="bold"),
            height=42, corner_radius=10,
            fg_color=TILE_COLORS["stats"]["fg"],
            hover_color=TILE_COLORS["stats"]["hover"],
            state="disabled",
        )
        self.btn_stats.pack(fill="x", padx=14, pady=(4, 10))

        # STEP 4b — Per-ref controls (populated after load)
        self.refctrl_frame = section_frame(scroll, "REF COLORS / CHANNELS")
        self.refctrl_frame.pack(fill="x", padx=14, pady=6)
        self.refctrl_inner = ctk.CTkFrame(self.refctrl_frame,
                                          fg_color="transparent")
        self.refctrl_inner.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(self.refctrl_inner,
                     text="(load data to enable)",
                     font=ctk.CTkFont(size=10),
                     text_color=("#6b7280", "#9ca3af")
                     ).pack(anchor="w")

        # Metric switcher
        m_frame = section_frame(scroll, "METRIC")
        m_frame.pack(fill="x", padx=14, pady=6)
        self.metric_menu = ctk.CTkOptionMenu(
            m_frame,
            values=["NNLS_norm", "MCR_contrib", "CLS_norm",
                    "Pearson_prob", "Cosine_sim", "Pearson_corr",
                    "NNLS_raw", "MCR_conc", "CLS_raw"],
            command=self.on_metric_change,
            height=32, corner_radius=8,
            state="disabled",
        )
        self.metric_menu.set("NNLS_norm")
        self.metric_menu.pack(fill="x", padx=14, pady=(0, 10))

        # STEP 5 — Save
        s5 = section_frame(scroll, "STEP 5 — SAVE")
        s5.pack(fill="x", padx=14, pady=6)

        # Output folder picker
        ctk.CTkLabel(s5, text="Output folder",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=("#374151", "#d1d5db"),
                     anchor="w"
                     ).pack(fill="x", padx=14, pady=(4, 2))
        self.out_dir_var = tk.StringVar(value="(auto: pick test CSV first)")
        ctk.CTkLabel(s5, textvariable=self.out_dir_var,
                     font=ctk.CTkFont(size=10),
                     text_color=("#6b7280", "#9ca3af"),
                     anchor="w", wraplength=300, justify="left"
                     ).pack(fill="x", padx=14, pady=(0, 4))
        ctk.CTkButton(s5, text="Browse folder…",
                      command=self.browse_out_dir,
                      height=28, corner_radius=8,
                      fg_color="transparent", border_width=1,
                      border_color=("#d1d5db", "#374151"),
                      text_color=("#374151", "#d1d5db"),
                      hover_color=("#f3f4f6", "#1f2937"),
                      ).pack(fill="x", padx=14, pady=(0, 10))

        self.btn_save_all = ctk.CTkButton(
            s5, text="💾  Save All",
            command=self.on_save_all,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=36, corner_radius=10,
            state="disabled",
        )
        self.btn_save_all.pack(fill="x", padx=14, pady=(6, 4))
        self.btn_save_sel = ctk.CTkButton(
            s5, text="⚙  Save…",
            command=self.on_save_selective,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=36, corner_radius=10,
            fg_color=("#f59e0b", "#b45309"),
            hover_color=("#d97706", "#92400e"),
            state="disabled",
        )
        self.btn_save_sel.pack(fill="x", padx=14, pady=4)
        self.btn_mean = ctk.CTkButton(
            s5, text="🔄  Mean spectrum",
            command=self.show_mean_spectrum,
            font=ctk.CTkFont(size=11),
            height=30, corner_radius=8,
            fg_color="transparent", border_width=1,
            border_color=("#d1d5db", "#374151"),
            text_color=("#374151", "#d1d5db"),
            hover_color=("#f3f4f6", "#1f2937"),
            state="disabled",
        )
        self.btn_mean.pack(fill="x", padx=14, pady=(4, 10))

        # Populate the bottom-pinned Appearance row that was created at the
        # very top of _build_sidebar (we packed it first to lock its anchor).
        ctk.CTkLabel(self.theme_row, text="Appearance:",
                     font=ctk.CTkFont(size=10),
                     text_color=("#6b7280", "#9ca3af")
                     ).pack(side="left", padx=(0, 6))
        self.theme_menu = ctk.CTkOptionMenu(
            self.theme_row, values=["System", "Light", "Dark"],
            command=lambda v: ctk.set_appearance_mode(v.lower()),
            width=110, height=26, font=ctk.CTkFont(size=10),
        )
        self.theme_menu.set("System")
        self.theme_menu.pack(side="left")

    # ---- Welcome placeholder -----------------------------------------

    def _build_welcome(self):
        self.placeholder = ctk.CTkFrame(self.content, corner_radius=0,
                                        fg_color="transparent")
        self.placeholder.pack(fill="both", expand=True)

        wrap = ctk.CTkFrame(self.placeholder, corner_radius=14,
                            fg_color=("#ffffff", "#1f2937"))
        wrap.place(relx=0.5, rely=0.5, anchor="center",
                   relwidth=0.6, relheight=0.6)
        ctk.CTkLabel(wrap, text="👋  Welcome",
                     font=ctk.CTkFont(size=26, weight="bold")
                     ).pack(pady=(34, 6))
        ctk.CTkLabel(
            wrap,
            text="Add reference CSVs, pick a test CSV, then click Load & Process.",
            font=ctk.CTkFont(size=13),
            text_color=("#6b7280", "#9ca3af"),
            wraplength=600,
        ).pack(pady=(0, 20))
        steps = (
            "1.   Add one or more reference SERS CSVs in STEP 1",
            "2.   Pick the testing SERS map CSV in STEP 2",
            "3.   Tune arPLS / MCR-ALS in STEP 3",
            "4.   Click Load & Process",
            "5.   Use the four view tabs to inspect results",
            "6.   Click any pixel in a map to pop its spectrum",
            "7.   Save all or pick categories with Save…",
        )
        for s in steps:
            ctk.CTkLabel(wrap, text=s,
                         font=ctk.CTkFont(size=12),
                         text_color=("#374151", "#d1d5db"),
                         anchor="w"
                         ).pack(anchor="w", padx=80, pady=2)

    # ---- File pickers ------------------------------------------------

    def _refresh_ref_listbox(self):
        self.ref_listbox.delete(0, tk.END)
        for p in self.ref_files:
            self.ref_listbox.insert(tk.END, os.path.basename(p))
        n = len(self.ref_files)
        self.ref_count_var.set(f"{n} reference{'s' if n != 1 else ''}")

    def add_refs(self):
        paths = filedialog.askopenfilenames(
            title="Select reference SERS CSV(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        for p in paths:
            if p not in self.ref_files:
                self.ref_files.append(p)
        self._refresh_ref_listbox()

    def remove_refs(self):
        for idx in reversed(list(self.ref_listbox.curselection())):
            del self.ref_files[idx]
        self._refresh_ref_listbox()

    def clear_refs(self):
        self.ref_files.clear()
        self._refresh_ref_listbox()

    def browse_test(self):
        p = filedialog.askopenfilename(
            title="Select testing SERS mapping CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p:
            self.test_file = p
            self.test_var.set(os.path.basename(p))
            # Auto-fill the output directory next to the test file (only if
            # the user hasn't already picked one explicitly).
            cur = self.out_dir_var.get()
            if cur.startswith("(") or not cur.strip():
                default_out = os.path.join(os.path.dirname(p),
                                            "discriminator_output")
                self.out_dir_var.set(default_out)

    def browse_out_dir(self):
        init = self.out_dir_var.get()
        if init.startswith("(") or not os.path.isdir(init):
            init = (os.path.dirname(self.test_file)
                    if self.test_file else os.path.expanduser("~"))
        path = filedialog.askdirectory(
            title="Select output folder", initialdir=init)
        if path:
            self.out_dir_var.set(path)

    # ---- Load + process ----------------------------------------------

    def load_and_process(self):
        if not self.ref_files:
            messagebox.showerror("Missing input",
                                 "Add at least one reference CSV in STEP 1.")
            return
        if not self.test_file:
            messagebox.showerror("Missing input",
                                 "Pick a testing CSV in STEP 2.")
            return
        try:
            wn_min = float(self.wn_min_var.get()) if self.wn_min_var.get().strip() else None
            wn_max = float(self.wn_max_var.get()) if self.wn_max_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Invalid input", "WN crop must be numbers.")
            return

        # Resolve output directory — user-picked overrides the default
        out_dir = self.out_dir_var.get().strip()
        if not out_dir or out_dir.startswith("("):
            out_dir = os.path.join(os.path.dirname(self.test_file),
                                    "discriminator_output")
            self.out_dir_var.set(out_dir)

        cfg = {
            "ref_files": list(self.ref_files),
            "test_file": self.test_file,
            "baseline_refs": self.bl_refs_var.get(),
            "baseline_test": self.bl_test_var.get(),
            "lam": 1e6, "ratio": 1e-6, "max_iter": 100,
            "wn_min": wn_min, "wn_max": wn_max,
            "out_dir": out_dir,
            "run_mcr": self.run_mcr_var.get(),
            "mcr_max_iter": 12,
            "mcr_tol": 1e-5,
            "cv_k": 5,
            "synth_n": 200,
            "synth_noise": 0.05,
            "sweep_n": 100,
        }

        self.load_btn.configure(state="disabled", text="Processing…")
        self.status_var.set("Working — this may take a minute.")
        self.update_idletasks()

        # Run in a background thread to keep UI responsive
        def worker():
            try:
                result = sd.process(cfg)
                self.after(0, lambda: self._on_processed(result))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_process_error(e, tb))
        threading.Thread(target=worker, daemon=True).start()

    def _on_process_error(self, e, tb):
        self.load_btn.configure(state="normal", text="Load & Process")
        self.status_var.set(f"Error: {e}")
        print(tb)
        messagebox.showerror("Processing failed", str(e))

    def _on_processed(self, result):
        # --- Cleanup from any previous Load (so panels don't stack) ---
        try:
            self.placeholder.pack_forget()
        except Exception:
            pass
        for old_name, old_panel in list(self.panels.items()):
            try:
                old_panel.pack_forget()
                old_panel.destroy()
            except Exception:
                pass
        self.panels = {}
        self.current_panel = None
        # Reset matplotlib state from the old maps panel
        self.maps_fig = None
        self.maps_canvas = None
        self.spec_ax = None
        self.test_line = None
        self.spec_overlay_lines = []
        self.sel_lines = []
        self.intensity_axes = []
        self.intensity_ims = []
        self.intensity_titles = []
        self.prob_axes = []
        self.prob_ims = []
        self.prob_titles = []
        self.rgb_ax = None
        self.rgb_im = None
        self.rgb_legend_box = None

        self.result = result
        n_ref = len(result["refs"])
        self.cmaps = list(sd.REF_CMAPS[:n_ref] + ["PastelGrey"] * max(0, n_ref - 6))[:n_ref]
        self.tints = list(sd.REF_TINTS[:n_ref] + [(0.5, 0.5, 0.5)] * max(0, n_ref - 6))[:n_ref]
        # Default: include every ref in the combined RGB (each ref gets its
        # own slot number 1..n_ref). 0 = exclude.
        self.rgb_assignment = [str(i + 1) for i in range(n_ref)]
        # Default selection: strongest peak per ref. Right after the popup,
        # let the user override.
        self.selected_wn = [p[0][0] for p in result["ref_top_peaks"]]
        self.test_mean = (result["test_cube"]
                          .reshape(-1, result["test_cube"].shape[2])
                          .mean(axis=0))

        # Initial peak picker — let the user override the auto-defaults
        # before we build the map panels.
        try:
            picked = ask_initial_peaks_ctk(
                self, result["refs"], result["ref_top_peaks"],
                self.tints,
            )
            if picked is not None and len(picked) == n_ref:
                self.selected_wn = picked
        except Exception as e:
            print(f"[initial peak picker failed: {e}; keeping defaults]")

        # Enable everything that depends on data
        self.load_btn.configure(state="normal", text="Reload")
        self.status_var.set(
            f"✓ Loaded {n_ref} ref(s) + {result['test_cube'].shape[1]}×"
            f"{result['test_cube'].shape[0]} test  "
            f"({result['agreement_summary']['full']}/"
            f"{result['agreement_summary']['n_pix']} 3/3-agree)"
        )
        for btn in (self.btn_maps, self.btn_reliability,
                    self.btn_validation, self.btn_stats,
                    self.btn_save_all, self.btn_save_sel,
                    self.btn_mean):
            btn.configure(state="normal")
        self.metric_menu.configure(state="normal")

        # Rebuild per-ref controls
        self._build_ref_controls()

        # Build all four analysis panels (creates the matplotlib figures)
        for name in ("maps", "reliability", "validation", "stats"):
            self.panels[name] = ctk.CTkFrame(self.content, corner_radius=0,
                                             fg_color="transparent")
            getattr(self, f"_build_{name}_panel")(self.panels[name])

        self.show_panel("maps")

    # ---- Per-ref control row ----------------------------------------

    def _build_ref_controls(self):
        # Wipe previous content
        for w in self.refctrl_inner.winfo_children():
            w.destroy()
        self.ref_control_widgets = []
        n_ref_total = len(self.result["refs"])
        for i, r in enumerate(self.result["refs"]):
            block = ctk.CTkFrame(self.refctrl_inner, corner_radius=8)
            block.pack(fill="x", pady=4, padx=0)

            # ---- Top row: swatch + name + ch dropdown ----
            top = ctk.CTkFrame(block, fg_color="transparent")
            top.pack(fill="x", padx=8, pady=(6, 2))

            tint_hex = mcolors.to_hex(self.tints[i])
            swatch = ctk.CTkButton(
                top, text="", width=22, height=22,
                corner_radius=6,
                fg_color=tint_hex, hover_color=tint_hex,
                border_width=1, border_color=("#999", "#666"),
                command=lambda idx=i: self._pick_color(idx),
            )
            swatch.pack(side="left", padx=(0, 6))

            name_lbl = ctk.CTkLabel(
                top, text=r["name"][:22],
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=tint_hex,
                anchor="w",
            )
            name_lbl.pack(side="left", expand=True, fill="x")

            ch_values = [str(j + 1) for j in range(n_ref_total)] + ["0"]
            ch_var = tk.StringVar(value=self.rgb_assignment[i])
            ch_menu = ctk.CTkOptionMenu(
                top, values=ch_values,
                variable=ch_var,
                command=lambda v, idx=i: self._set_channel(idx, v),
                width=58, height=24, corner_radius=6,
                font=ctk.CTkFont(size=11),
            )
            ch_menu.pack(side="left", padx=4)

            # ---- Bottom row: intensity vmin / vmax + auto ----
            bot = ctk.CTkFrame(block, fg_color="transparent")
            bot.pack(fill="x", padx=8, pady=(0, 6))

            ctk.CTkLabel(bot, text="min",
                         font=ctk.CTkFont(size=10),
                         text_color=("#6b7280", "#9ca3af")
                         ).pack(side="left", padx=(0, 2))
            vmin_var = tk.StringVar(value="—")
            vmin_entry = ctk.CTkEntry(
                bot, textvariable=vmin_var,
                width=78, height=24,
                font=ctk.CTkFont(size=11),
            )
            vmin_entry.pack(side="left", padx=(0, 6))
            vmin_entry.bind("<Return>",
                            lambda _e, idx=i: self._apply_vmin(idx))
            vmin_entry.bind("<FocusOut>",
                            lambda _e, idx=i: self._apply_vmin(idx))

            ctk.CTkLabel(bot, text="max",
                         font=ctk.CTkFont(size=10),
                         text_color=("#6b7280", "#9ca3af")
                         ).pack(side="left", padx=(0, 2))
            vmax_var = tk.StringVar(value="—")
            vmax_entry = ctk.CTkEntry(
                bot, textvariable=vmax_var,
                width=78, height=24,
                font=ctk.CTkFont(size=11),
            )
            vmax_entry.pack(side="left", padx=(0, 6))
            vmax_entry.bind("<Return>",
                            lambda _e, idx=i: self._apply_vmax(idx))
            vmax_entry.bind("<FocusOut>",
                            lambda _e, idx=i: self._apply_vmax(idx))

            auto_btn = ctk.CTkButton(
                bot, text="auto",
                command=lambda idx=i: self._reset_intensity_scale(idx),
                width=46, height=24, corner_radius=6,
                fg_color="transparent", border_width=1,
                border_color=("#d1d5db", "#374151"),
                text_color=("#374151", "#d1d5db"),
                hover_color=("#f3f4f6", "#1f2937"),
                font=ctk.CTkFont(size=10),
            )
            auto_btn.pack(side="left", padx=2)

            self.ref_control_widgets.append({
                "swatch": swatch, "name": name_lbl,
                "ch_var": ch_var, "ch_menu": ch_menu,
                "vmin_var": vmin_var, "vmax_var": vmax_var,
                "vmin_entry": vmin_entry, "vmax_entry": vmax_entry,
            })

    def _sync_intensity_scale_widgets(self):
        """Fill vmin/vmax textboxes from the current intensity_ims clims."""
        for i, w in enumerate(self.ref_control_widgets):
            if i >= len(self.intensity_ims):
                continue
            vmin, vmax = self.intensity_ims[i].get_clim()
            w["vmin_var"].set(f"{vmin:.1f}")
            w["vmax_var"].set(f"{vmax:.1f}")

    def _apply_vmin(self, idx):
        if idx >= len(self.intensity_ims):
            return
        w = self.ref_control_widgets[idx]
        try:
            v = float(w["vmin_var"].get())
        except ValueError:
            self._sync_intensity_scale_widgets()
            return
        cur_max = self.intensity_ims[idx].get_clim()[1]
        if v >= cur_max:
            v = cur_max - 1e-6
        self.intensity_ims[idx].set_clim(v, cur_max)
        w["vmin_var"].set(f"{v:.1f}")
        if self.maps_canvas is not None:
            self.maps_canvas.draw_idle()

    def _apply_vmax(self, idx):
        if idx >= len(self.intensity_ims):
            return
        w = self.ref_control_widgets[idx]
        try:
            v = float(w["vmax_var"].get())
        except ValueError:
            self._sync_intensity_scale_widgets()
            return
        cur_min = self.intensity_ims[idx].get_clim()[0]
        if v <= cur_min:
            v = cur_min + 1e-6
        self.intensity_ims[idx].set_clim(cur_min, v)
        w["vmax_var"].set(f"{v:.1f}")
        if self.maps_canvas is not None:
            self.maps_canvas.draw_idle()

    def _reset_intensity_scale(self, idx):
        """Reset to the underlying data's min/max for that ref's intensity map."""
        if idx >= len(self.intensity_ims) or self.result is None:
            return
        # Recompute from the cube slice at the currently displayed WN
        wn = self.result["wn"]
        cube = self.result["test_cube"]
        j = int(np.argmin(np.abs(wn - self.selected_wn[idx])))
        img = cube[:, :, j]
        vmin = float(img.min())
        vmax = float(img.max() + 1e-9)
        self.intensity_ims[idx].set_clim(vmin, vmax)
        w = self.ref_control_widgets[idx]
        w["vmin_var"].set(f"{vmin:.1f}")
        w["vmax_var"].set(f"{vmax:.1f}")
        if self.maps_canvas is not None:
            self.maps_canvas.draw_idle()

    def _pick_color(self, idx):
        cur_hex = mcolors.to_hex(self.tints[idx])
        picked = colorchooser.askcolor(
            color=cur_hex,
            title=f"Pick color for {self.result['refs'][idx]['name']}")
        if picked is None or picked[1] is None:
            return
        new_hex = picked[1]
        cmap_obj, tint = sd._resolve_user_cmap_and_tint(new_hex)
        if cmap_obj is None:
            return
        self.cmaps[idx] = new_hex
        self.tints[idx] = tint
        # Update swatch button color + name label
        w = self.ref_control_widgets[idx]
        new_hex_str = mcolors.to_hex(tint)
        w["swatch"].configure(fg_color=new_hex_str, hover_color=new_hex_str)
        w["name"].configure(text_color=new_hex_str)
        # Refresh maps
        self._refresh_maps_colors(idx, cmap_obj, tint)

    def _set_channel(self, idx, value):
        self.rgb_assignment[idx] = sd._ch_to_display(value)
        self._refresh_combined_rgb()

    def _refresh_maps_colors(self, idx, cmap_obj, tint):
        """Re-color one ref's intensity + metric maps after color picker."""
        if self.maps_fig is None:
            return
        if idx < len(self.intensity_ims):
            self.intensity_ims[idx].set_cmap(cmap_obj)
            self.intensity_titles[idx].set_color(tint)
        if idx < len(self.prob_ims):
            self.prob_ims[idx].set_cmap(cmap_obj)
            self.prob_titles[idx].set_color(tint)
        if idx < len(self.spec_overlay_lines):
            self.spec_overlay_lines[idx].set_color(tint)
            self.sel_lines[idx].set_color(tint)
            try:
                self.spec_ax.legend(loc="upper right", fontsize=8,
                                    ncol=min(len(self.tints) + 1, 4))
            except Exception:
                pass
        self._refresh_combined_rgb()
        self.maps_canvas.draw_idle()

    def _refresh_combined_rgb(self):
        if self.rgb_ax is None:
            return
        arr, (vmin, vmax), _ = self._metric_arrays(self.current_metric)
        span = max(vmax - vmin, 1e-12)
        norm = np.clip((arr - vmin) / span, 0.0, 1.0)
        rgb = sd._build_pastel_rgb(norm, self.rgb_assignment,
                                    tints=self.tints)
        self.rgb_im.set_data(rgb)
        # Refresh legend text in the RGB box
        self._update_rgb_legend()
        self.maps_canvas.draw_idle()

    def _update_rgb_legend(self):
        if self.rgb_legend_box is None:
            return
        # Channel numbers ≥4 are valid (each ref blends with its own tint),
        # so the R/G/B suffix would be misleading past 3. Just show slot
        # number + bullet in the ref's actual tint color.
        lines = []
        for i, r in enumerate(self.result["refs"]):
            disp = sd._ch_to_display(self.rgb_assignment[i])
            if disp == "0":
                continue
            lines.append(f"ch{disp}: ●  {r['name']}")
        self.rgb_legend_box.set_text("\n".join(lines))

    # ---- Panel switching ---------------------------------------------

    def show_panel(self, name):
        if not hasattr(self, "placeholder"):
            return
        try:
            self.placeholder.pack_forget()
        except Exception:
            pass
        if self.current_panel is not None and self.current_panel in self.panels:
            self.panels[self.current_panel].pack_forget()
        if name in self.panels:
            self.panels[name].pack(fill="both", expand=True)
            self.current_panel = name
        # Highlight active tile button
        for n, b in (("maps", self.btn_maps),
                     ("reliability", self.btn_reliability),
                     ("validation", self.btn_validation),
                     ("stats", self.btn_stats)):
            b.configure(border_width=2 if n == name else 0,
                        border_color=("#1f2937", "#ffffff"))

    # ---- Maps panel --------------------------------------------------

    def _metric_arrays(self, name):
        r = self.result
        opt = {
            "NNLS_norm":    (r["nnls_norm"], 0.0, 1.0, "contribution"),
            "MCR_contrib":  (r["mcr_contrib"], 0.0, 1.0, "contribution"),
            "CLS_norm":     (r["cls_norm"], 0.0, 1.0, "contribution"),
            "Pearson_prob": (r["prob_maps"], 0.0, 1.0, "probability"),
            "Cosine_sim":   (r["cos_maps"], 0.0, 1.0, "cosine"),
            "Pearson_corr": (r["corr_maps"], -1.0, 1.0, "Pearson r"),
            "NNLS_raw":     (r["nnls_raw"],
                             float(r["nnls_raw"].min()),
                             float(r["nnls_raw"].max() + 1e-9),
                             "weight"),
            "MCR_conc":     (r["mcr_conc"],
                             float(r["mcr_conc"].min()),
                             float(r["mcr_conc"].max() + 1e-9),
                             "concentration"),
            "CLS_raw":      (r["cls_raw"],
                             float(r["cls_raw"].min()),
                             float(r["cls_raw"].max() + 1e-9),
                             "coeff"),
        }
        arr, vmin, vmax, lbl = opt[name]
        return arr, (vmin, vmax), lbl

    def _build_maps_panel(self, parent):
        r = self.result
        n_ref = len(r["refs"])
        cube = r["test_cube"]
        wn = r["wn"]
        ny, nx, _ = cube.shape
        extent = [r["x_coords"].min(), r["x_coords"].max(),
                  r["y_coords"].min(), r["y_coords"].max()]

        # Header
        head = ctk.CTkFrame(parent, corner_radius=10, height=44)
        head.pack(fill="x", padx=14, pady=(14, 6))
        head.pack_propagate(False)
        ctk.CTkLabel(
            head, text="🗺  Maps",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left", padx=14)
        ctk.CTkLabel(
            head,
            text="Click any map pixel to pop its spectrum  •  use the METRIC "
                 "dropdown in the sidebar to switch what's shown",
            font=ctk.CTkFont(size=10),
            text_color=("#6b7280", "#9ca3af"),
        ).pack(side="left", padx=14)

        card = plot_card(parent)
        card.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # Build the matplotlib figure
        fig_w = max(4.6 * n_ref + 1.0, 12.0)
        fig = Figure(figsize=(fig_w, 11.6), dpi=100, facecolor="white")
        self.maps_fig = fig

        # Spectrum panel (top)
        spec_ax = fig.add_axes([0.06, 0.78, 0.90, 0.18])
        line, = spec_ax.plot(wn, self.test_mean, color="#222", lw=1.2,
                              label="Test mean")
        self.test_line = line
        self.spec_overlay_lines = []
        for i, (rf, spec) in enumerate(zip(r["refs"], r["ref_specs"])):
            scale = (self.test_mean.max() / (spec.max() + 1e-12)) * 0.9
            ln, = spec_ax.plot(wn, spec * scale,
                                color=self.tints[i], lw=0.9, alpha=0.85,
                                label=f"{rf['name']} (scaled)")
            self.spec_overlay_lines.append(ln)
        spec_ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
        spec_ax.set_ylabel("Intensity", fontsize=9)
        spec_ax.set_title("Mean spectrum across all pixels",
                          fontsize=10, fontweight="bold")
        spec_ax.grid(alpha=0.3)
        spec_ax.legend(loc="upper right", fontsize=8,
                       ncol=min(n_ref + 1, 4))
        spec_ax.tick_params(labelsize=8)
        spec_ax.set_xlim(wn[0], wn[-1])
        self.sel_lines = []
        for i in range(n_ref):
            v = spec_ax.axvline(self.selected_wn[i], color=self.tints[i],
                                 lw=1.2, alpha=0.7)
            self.sel_lines.append(v)
        self.spec_ax = spec_ax

        # Row layout for intensity + metric
        map_w = 0.78 / n_ref
        map_h_data = map_w * (fig_w / 11.6) * (ny / max(nx, 1))
        map_h = max(min(map_h_data, 0.21), 0.13)
        row1_y = 0.50
        row2_y = 0.23
        start_x = (1.0 - (map_w * n_ref + 0.02 * (n_ref - 1))) / 2

        self.intensity_axes = []
        self.intensity_ims = []
        self.intensity_titles = []
        self.prob_axes = []
        self.prob_ims = []
        self.prob_titles = []

        cmap_objs = [sd._resolve_cmap(c) for c in self.cmaps]

        for i in range(n_ref):
            ax_x = start_x + i * (map_w + 0.02)
            # Intensity
            ax = fig.add_axes([ax_x, row1_y, map_w, map_h])
            idx = int(np.argmin(np.abs(wn - self.selected_wn[i])))
            img = cube[:, :, idx]
            im = ax.imshow(img, cmap=cmap_objs[i],
                           extent=extent, origin="upper",
                           aspect="equal", interpolation="nearest")
            t = ax.set_title(
                f"Intensity: {r['refs'][i]['name']} @ "
                f"{wn[idx]:.1f} cm⁻¹",
                fontsize=10, fontweight="bold",
                color=self.tints[i])
            ax.set_xlabel("X (μm)", fontsize=9)
            ax.set_ylabel("Y (μm)", fontsize=9)
            ax.tick_params(labelsize=8)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
            self.intensity_axes.append(ax)
            self.intensity_ims.append(im)
            self.intensity_titles.append(t)

            # Metric
            ax2 = fig.add_axes([ax_x, row2_y, map_w, map_h])
            arr0, (vmin0, vmax0), lbl0 = self._metric_arrays(self.current_metric)
            im2 = ax2.imshow(arr0[i], cmap=cmap_objs[i],
                             extent=extent, origin="upper",
                             aspect="equal", interpolation="nearest",
                             vmin=vmin0, vmax=vmax0)
            t2 = ax2.set_title(
                f"{self.current_metric}: {r['refs'][i]['name']}",
                fontsize=10, fontweight="bold",
                color=self.tints[i])
            ax2.set_xlabel("X (μm)", fontsize=9)
            ax2.set_ylabel("Y (μm)", fontsize=9)
            ax2.tick_params(labelsize=8)
            fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.04)
            self.prob_axes.append(ax2)
            self.prob_ims.append(im2)
            self.prob_titles.append(t2)

        # Combined RGB at bottom
        rgb_w = 0.36
        rgb_h = 0.16
        rgb_x = (1.0 - rgb_w) / 2
        rgb_y = 0.03
        self.rgb_ax = fig.add_axes([rgb_x, rgb_y, rgb_w, rgb_h])
        rgb_init = np.zeros((ny, nx, 3))
        self.rgb_im = self.rgb_ax.imshow(rgb_init, extent=extent,
                                          origin="upper", aspect="equal",
                                          interpolation="nearest")
        self.rgb_ax.set_title("Combined RGB (current metric)",
                              fontsize=10, fontweight="bold")
        self.rgb_ax.set_xlabel("X (μm)", fontsize=9)
        self.rgb_ax.set_ylabel("Y (μm)", fontsize=9)
        self.rgb_ax.tick_params(labelsize=8)
        self.rgb_legend_box = self.rgb_ax.text(
            1.03, 0.98, "",
            transform=self.rgb_ax.transAxes,
            fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.85))

        # Mount canvas
        self.maps_canvas = FigureCanvasTkAgg(fig, master=card)
        self.maps_canvas.draw()
        self.maps_canvas.get_tk_widget().pack(fill="both", expand=True,
                                              padx=6, pady=6)
        self.maps_canvas.mpl_connect("button_press_event",
                                     self._on_canvas_click)

        # Initial paint of RGB + legend
        self._refresh_combined_rgb()
        # Populate the sidebar's intensity min/max textboxes with the
        # auto-detected data range, so the user can see + edit the numbers.
        self._sync_intensity_scale_widgets()

    def _on_canvas_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        # Spectrum panel click: do nothing (popup is for pixels)
        if event.inaxes is self.spec_ax:
            return
        # Did the user click a pixel on a map?
        all_map_axes = list(self.intensity_axes) + list(self.prob_axes)
        if event.inaxes in all_map_axes:
            r = self.result
            xi = int(np.argmin(np.abs(r["x_coords"] - event.xdata)))
            yi = int(np.argmin(np.abs(r["y_coords"] - event.ydata)))
            if 0 <= xi < r["test_cube"].shape[1] and 0 <= yi < r["test_cube"].shape[0]:
                spectrum = r["test_cube"][yi, xi, :]
                ref_names = [rf["name"] for rf in r["refs"]]
                # Pop dedicated spectrum window (Plaspector pattern)
                SpectrumPopup(
                    self, r["wn"], spectrum,
                    title=f"Pixel spectrum @ X={r['x_coords'][xi]:.1f}, "
                          f"Y={r['y_coords'][yi]:.1f}",
                    ref_specs=r["ref_specs"],
                    ref_names=ref_names,
                    ref_tints=self.tints,
                )
                print(f"[pixel-click spectrum] x={r['x_coords'][xi]:.1f} "
                      f"y={r['y_coords'][yi]:.1f}")

    def show_mean_spectrum(self):
        """Reset the top spectrum panel to the test mean."""
        if self.test_line is None:
            return
        self.test_line.set_ydata(self.test_mean)
        self.spec_state["mode"] = "mean"
        self.spec_ax.set_title("Mean spectrum across all pixels",
                                fontsize=10, fontweight="bold")
        y_min = float(min(self.test_mean.min(), 0.0))
        y_max = float(self.test_mean.max() * 1.08 + 1e-9)
        self.spec_ax.set_ylim(y_min, y_max)
        self.maps_canvas.draw_idle()

    def on_metric_change(self, value):
        self.current_metric = value
        if self.maps_fig is None:
            return
        arr, (vmin, vmax), lbl = self._metric_arrays(value)
        for i in range(len(self.prob_ims)):
            self.prob_ims[i].set_data(arr[i])
            self.prob_ims[i].set_clim(vmin, vmax)
            self.prob_titles[i].set_text(
                f"{value}: {self.result['refs'][i]['name']}"
            )
        self._refresh_combined_rgb()
        self.maps_canvas.draw_idle()

    # ---- Reliability / Validation / Stats panels (simple embeds) -----

    def _build_reliability_panel(self, parent):
        r = self.result
        n_ref = len(r["refs"])
        extent = [r["x_coords"].min(), r["x_coords"].max(),
                  r["y_coords"].min(), r["y_coords"].max()]
        ctk.CTkLabel(
            parent, text="📈  Reliability",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(14, 0))
        ctk.CTkLabel(
            parent, text=(
                "Pixel-level confidence (gap top1-top2 + entropy)  •  "
                "argmax labels per method  •  cross-method agreement and "
                "consensus."),
            font=ctk.CTkFont(size=11),
            text_color=("#6b7280", "#9ca3af"),
        ).pack(anchor="w", padx=18, pady=(0, 6))

        card = plot_card(parent)
        card.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        fig = Figure(figsize=(13, 7.5), dpi=100, facecolor="white")
        gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30)
        max_ent = float(np.log2(max(n_ref, 2)))

        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(r["confidence_gap_nnls"],
                  cmap=sd.PASTEL_CMAPS["PastelGreen"], vmin=0, vmax=1,
                  extent=extent, origin="upper", aspect="equal",
                  interpolation="nearest")
        ax.set_title("Confidence gap (NNLS)", fontsize=11, fontweight="bold")

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(r["entropy_nnls"],
                  cmap=sd.PASTEL_CMAPS["PastelOrange"], vmin=0, vmax=max_ent,
                  extent=extent, origin="upper", aspect="equal",
                  interpolation="nearest")
        ax.set_title("Entropy (NNLS, bits)", fontsize=11, fontweight="bold")

        ax = fig.add_subplot(gs[0, 2])
        # Agreement (categorical)
        palette = ["#dddddd", "#e8d18a", "#a3d39c"]
        cmap = mcolors.ListedColormap(palette)
        norm = mcolors.BoundaryNorm(np.arange(1, 5) - 0.5, cmap.N)
        ax.imshow(r["agreement"], cmap=cmap, norm=norm,
                  extent=extent, origin="upper", aspect="equal",
                  interpolation="nearest")
        ax.set_title("Method agreement\n(NNLS/MCR/CLS)",
                     fontsize=11, fontweight="bold")
        handles = [mpatches.Patch(facecolor=palette[k - 1],
                                  edgecolor="#888",
                                  label=f"{k}/{3}") for k in (1, 2, 3)]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

        # Consensus categorical
        ax = fig.add_subplot(gs[1, :])
        pal = [(0.82, 0.82, 0.82)] + [self.tints[i] for i in range(n_ref)]
        cmap2 = mcolors.ListedColormap(pal)
        bounds = np.arange(n_ref + 2) - 0.5
        norm2 = mcolors.BoundaryNorm(bounds, cmap2.N)
        remapped = np.where(r["consensus"] < 0, 0, r["consensus"] + 1).astype(int)
        ax.imshow(remapped, cmap=cmap2, norm=norm2, extent=extent,
                  origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title("Consensus label (majority across NNLS/MCR/CLS)",
                     fontsize=11, fontweight="bold")
        legend_handles = [mpatches.Patch(facecolor=pal[0], edgecolor="#888",
                                         label="tie")]
        for i, rf in enumerate(r["refs"]):
            legend_handles.append(mpatches.Patch(facecolor=pal[i + 1],
                                                  edgecolor="#888",
                                                  label=rf["name"]))
        ax.legend(handles=legend_handles, loc="center left",
                  bbox_to_anchor=(1.02, 0.5))

        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def _build_validation_panel(self, parent):
        r = self.result
        cv = r.get("cv_result")
        synth = r.get("synth_result")
        sweep = r.get("noise_sweep_result")

        ctk.CTkLabel(
            parent, text="🎯  Validation",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(14, 0))
        ctk.CTkLabel(
            parent, text=(
                "K-fold CV on reference pixels gives the classifier's upper "
                "bound (F1).  The noise sweep shows how unmixing degrades "
                "with measurement noise."),
            font=ctk.CTkFont(size=11),
            text_color=("#6b7280", "#9ca3af"),
            wraplength=900, justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 6))

        card = plot_card(parent)
        card.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        fig = Figure(figsize=(13, 7), dpi=100, facecolor="white")
        gs = fig.add_gridspec(1, 2, wspace=0.30)

        # Confusion matrix (left)
        if cv is not None:
            ax = fig.add_subplot(gs[0, 0])
            cm = cv["confusion"]
            names = cv["ref_names"]
            n = len(names)
            row_sums = cm.sum(axis=1, keepdims=True)
            norm = cm / np.maximum(row_sums, 1)
            im = ax.imshow(norm, cmap=sd.PASTEL_CMAPS["PastelGreen"],
                           vmin=0, vmax=1)
            for i in range(n):
                for j in range(n):
                    txt_c = "white" if norm[i, j] > 0.6 else "#222"
                    ax.text(j, i,
                            f"n={cm[i, j]}\n({norm[i, j] * 100:.1f}%)",
                            ha="center", va="center", color=txt_c,
                            fontsize=11)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(names, rotation=15, ha="right")
            ax.set_yticklabels(names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(
                f"Reference {cv['n_folds']}-fold CV\n"
                f"Accuracy {cv['accuracy']:.3f}  •  "
                f"Macro F1 {cv['macro_f1']:.3f}",
                fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label="Row-normalized")

        # Noise sweep (right)
        if sweep is not None:
            ax = fig.add_subplot(gs[0, 1])
            noise = np.array(sweep["noise_levels"]) * 100
            rmse_pr = sweep["rmse_per_ref"]
            overall = sweep["overall_rmse"]
            for i, rf in enumerate(r["refs"]):
                ax.plot(noise, rmse_pr[:, i], "o-",
                        color=self.tints[i % len(self.tints)],
                        lw=2, ms=7, label=rf["name"])
            ax.plot(noise, overall, "s--", color="#444", lw=2.2, ms=7,
                    label="overall")
            ax.set_xlabel("Noise (% of signal std)")
            ax.set_ylabel("NNLS unmixing RMSE")
            ax.set_title("Synthetic noise sweep — degradation curve",
                         fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=9)
            ax.set_xlim(0, noise.max() * 1.05)
            ax.set_ylim(0, max(rmse_pr.max(), overall.max()) * 1.15)

        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def _build_stats_panel(self, parent):
        r = self.result
        n_pix = r["consensus"].size
        summ = r["agreement_summary"]
        cv = r.get("cv_result")

        ctk.CTkLabel(
            parent, text="📋  Stats",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(14, 0))

        card = plot_card(parent)
        card.pack(fill="both", expand=True, padx=14, pady=(8, 14))

        txt = ctk.CTkTextbox(card, font=("Consolas", 12),
                             corner_radius=0, fg_color="white",
                             text_color="#111")
        txt.pack(fill="both", expand=True, padx=8, pady=8)

        lines = []
        lines.append("=" * 60)
        lines.append(f"  Test file: {r['test_path'].name}")
        lines.append(f"  Map: {r['test_cube'].shape[1]} × "
                     f"{r['test_cube'].shape[0]} = {n_pix} pixels")
        lines.append(f"  Wavenumber points: {len(r['wn'])} "
                     f"({r['wn'][0]:.1f} – {r['wn'][-1]:.1f} cm⁻¹)")
        lines.append("=" * 60)
        lines.append("")

        lines.append("RELIABILITY (test data)")
        lines.append("-" * 60)
        lines.append(f"  3/3 method agreement: {summ['full']:>6d} pixels "
                     f"({100*summ['full']/n_pix:.2f}%)")
        lines.append(f"  2/3 method agreement: {summ['two']:>6d} pixels "
                     f"({100*summ['two']/n_pix:.2f}%)")
        lines.append(f"  1/3 (no majority):    {summ['split']:>6d} pixels "
                     f"({100*summ['split']/n_pix:.2f}%)")
        lines.append(f"  Mean NNLS gap top1-top2: "
                     f"{float(r['confidence_gap_nnls'].mean()):.4f}")
        lines.append(f"  Mean NNLS entropy: "
                     f"{float(r['entropy_nnls'].mean()):.4f} bits")
        lines.append("")

        if cv is not None:
            lines.append("CLASSIFIER VALIDATION (reference data)")
            lines.append("-" * 60)
            lines.append(f"  {cv['n_folds']}-fold CV")
            lines.append(f"  Accuracy: {cv['accuracy']:.4f}")
            lines.append(f"  Macro F1: {cv['macro_f1']:.4f}")
            for i, name in enumerate(cv["ref_names"]):
                lines.append(
                    f"  {name:>30s}  "
                    f"prec={cv['precision'][i]:.3f}  "
                    f"rec={cv['recall'][i]:.3f}  "
                    f"F1={cv['f1'][i]:.3f}"
                )
            lines.append("")

        lines.append("PER-REFERENCE CONTRIBUTION SUMMARY")
        lines.append("-" * 60)
        ref_names = [rf["name"] for rf in r["refs"]]
        for i, name in enumerate(ref_names):
            lines.append(
                f"  {name:>30s}  "
                f"NNLS={r['nnls_norm'][i].mean():.3f}  "
                f"MCR={r['mcr_contrib'][i].mean():.3f}  "
                f"CLS={r['cls_norm'][i].mean():.3f}  "
                f"Cosine={r['cos_maps'][i].mean():.3f}"
            )

        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")

    # ---- Save handlers ----------------------------------------------

    def _sync_out_dir(self):
        """Push the sidebar's current Output folder into result['out_dir']
        so save_all writes to where the user picked. Returns the resolved
        Path or None if nothing valid is set."""
        out_dir = self.out_dir_var.get().strip()
        if not out_dir or out_dir.startswith("("):
            if self.test_file:
                out_dir = os.path.join(os.path.dirname(self.test_file),
                                        "discriminator_output")
                self.out_dir_var.set(out_dir)
            else:
                return None
        from pathlib import Path as _P
        self.result["out_dir"] = _P(out_dir)
        return self.result["out_dir"]

    def on_save_all(self):
        print("[Save All button clicked]")
        out = self._sync_out_dir()
        if out is None:
            messagebox.showerror(
                "No output folder",
                "Pick an output folder in STEP 5 before saving.")
            return
        try:
            sd.save_all(self.result, self.selected_wn, None,
                         self.test_mean, self.rgb_assignment,
                         cmaps_override=self.cmaps,
                         tints_override=self.tints)
            self.status_var.set(f"Saved to: {out}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Save failed", str(e))

    def on_save_selective(self):
        print("[Save... button clicked]")
        out = self._sync_out_dir()
        if out is None:
            messagebox.showerror(
                "No output folder",
                "Pick an output folder in STEP 5 before saving.")
            return
        selection = ask_save_selection_ctk(self, self.current_metric)
        if selection is None:
            print("[Save... cancelled]")
            return
        try:
            sd.save_all(self.result, self.selected_wn, None,
                         self.test_mean, self.rgb_assignment,
                         cmaps_override=self.cmaps,
                         tints_override=self.tints,
                         selection=selection)
            self.status_var.set(f"Saved to: {out}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Save failed", str(e))


def main():
    app = SERSDiscriminatorApp()
    app._own_root.mainloop()


if __name__ == "__main__":
    main()
