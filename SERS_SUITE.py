"""SERS Suite — instrument-console workbench.

A single dark "instrument" window for the SERS pesticide workflow. Instead of a
brochure of tool cards, the app is one workbench:

    left  : a workflow rail (Mixture / Discriminator / Report)
    center: the active tool, embedded and kept alive
    Report: a per-pixel readout — which pesticide, how many M, what ratio,
            and whether competitive adsorption is suppressing the signal
            (the numeric fields are PLACEHOLDERS until the report backend
            is wired to a classified pixel)

    python SERS_SUITE.py

Both tools expose an embeddable class — App(container, embedded=True) — so they
run standalone or here. Keep every tool file in the same folder.
"""
from __future__ import annotations

import os
import sys
import traceback

import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from sers_app import SERSApp
from sers_discriminator_ctk import SERSDiscriminatorApp

# One coherent dark instrument look for the whole shell.
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --------------------------------------------------------------------------
# identity + instrument palette (dark charcoal, teal/amber/blue accents)
# --------------------------------------------------------------------------
SUITE_NAME = "SERS pesticide console"
VERSION = "1.0"

PAGE     = "#0f1216"   # window background
PANEL    = "#171c22"   # panels / top bar
RAIL     = "#12161c"   # left rail
CARD     = "#1b222b"   # raised card inside dark
LINE     = "#262e38"   # hairline border
LINE_HI  = "#39424e"   # hover border

INK      = "#e6edf3"   # primary text
MUTE     = "#9aa3af"   # secondary text
FAINT    = "#5b636e"   # hints

TEAL     = "#1D9E75"   # THI / success / brand accent
TEAL_HI  = "#5DCAA5"
BLUE     = "#378ADD"   # DQ / neutral accent
BLUE_HI  = "#5aa0e6"
AMBER    = "#EF9F27"   # warning / competitive
CORAL    = "#D85A30"   # TBZ

# pesticide -> (label color, ratio, molar readout)  [placeholder values]
PESTICIDES = [
    ("THI", TEAL,  0.52, "1.2×10⁻⁶ M"),
    ("DQ",  BLUE,  0.34, "8.0×10⁻⁷ M"),
    ("TBZ", CORAL, 0.14, "3.3×10⁻⁷ M"),
]


def _hf():
    return "Segoe UI" if sys.platform == "win32" else "Arial"


def _mf():
    return "Consolas" if sys.platform == "win32" else "Menlo"


# key -> (rail letter, accent, title, subtitle, factory | None)
MODES = {
    "mixture": ("M", BLUE, "Mixture", "pure refs → detect + ratio",
                lambda c: SERSApp(c, embedded=True)),
    "pest": ("D", TEAL, "Discriminator", "map → per-pixel identify",
             lambda c: SERSDiscriminatorApp(c, embedded=True)),
    "report": ("R", AMBER, "Report", "pixel → M · ratio · competition",
               None),   # built by the shell (placeholder readout)
}
RAIL_ORDER = ["mixture", "pest", "report"]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _badge(parent, letter, color, size, font_size):
    box = ctk.CTkFrame(parent, width=size, height=size, corner_radius=9,
                       fg_color=color)
    box.pack_propagate(False)
    box.grid_propagate(False)
    ctk.CTkLabel(box, text=letter, text_color="#0b0f14",
                 font=ctk.CTkFont(family=_hf(), size=font_size, weight="bold")
                 ).place(relx=0.5, rely=0.5, anchor="center")
    return box


# --------------------------------------------------------------------------
# Left workflow rail
# --------------------------------------------------------------------------
class Rail(ctk.CTkFrame):
    WIDTH = 158

    def __init__(self, parent, on_select):
        super().__init__(parent, width=self.WIDTH, corner_radius=0,
                         fg_color=RAIL)
        self.pack_propagate(False)
        self._on_select = on_select
        self._rows = {}
        self._active = None

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(16, 6))
        _badge(head, "S", TEAL, 30, 15).pack(side="left")
        ctk.CTkLabel(head, text="SERS", text_color=INK,
                     font=ctk.CTkFont(family=_hf(), size=16, weight="bold")
                     ).pack(side="left", padx=8)

        ctk.CTkLabel(self, text="WORKFLOW", text_color=FAINT, anchor="w",
                     font=ctk.CTkFont(family=_mf(), size=10)
                     ).pack(fill="x", padx=16, pady=(12, 4))
        for key in RAIL_ORDER:
            self._row(key)

        ctk.CTkLabel(self, text=f"v{VERSION}", text_color=FAINT, anchor="w",
                     font=ctk.CTkFont(family=_mf(), size=10)
                     ).pack(side="bottom", fill="x", padx=16, pady=12)

    def _row(self, key):
        letter, color, title, _, _ = MODES[key]
        row = ctk.CTkFrame(self, fg_color="transparent", corner_radius=8,
                           height=40)
        row.pack(fill="x", padx=8, pady=2)
        row.pack_propagate(False)
        dot = ctk.CTkLabel(row, text="•", text_color=color, width=14,
                           font=ctk.CTkFont(size=18))
        dot.pack(side="left", padx=(8, 2))
        lab = ctk.CTkLabel(row, text=title, text_color=MUTE, anchor="w",
                           font=ctk.CTkFont(family=_hf(), size=14))
        lab.pack(side="left", fill="x", expand=True)
        self._rows[key] = (row, lab)
        for w in (row, dot, lab):
            w.bind("<Button-1>", lambda e, k=key: self._on_select(k))
            w.bind("<Enter>", lambda e, k=key: self._hover(k, True))
            w.bind("<Leave>", lambda e, k=key: self._hover(k, False))

    def _hover(self, key, on):
        if key == self._active:
            return
        self._rows[key][0].configure(fg_color=CARD if on else "transparent")

    def set_active(self, key):
        self._active = key
        for k, (row, lab) in self._rows.items():
            active = (k == key)
            row.configure(fg_color=PANEL if active else "transparent")
            lab.configure(text_color=INK if active else MUTE,
                          font=ctk.CTkFont(family=_hf(), size=14,
                                           weight="bold" if active else "normal"))


# --------------------------------------------------------------------------
# Thin dark top bar
# --------------------------------------------------------------------------
class TopBar(ctk.CTkFrame):
    def __init__(self, parent, height=52):
        super().__init__(parent, height=height, corner_radius=0, fg_color=PANEL)
        self.pack_propagate(False)
        self.title = ctk.CTkLabel(self, text="", text_color=INK, anchor="w",
                                  font=ctk.CTkFont(family=_hf(), size=16,
                                                   weight="bold"))
        self.title.pack(side="left", padx=18)
        self.sub = ctk.CTkLabel(self, text="", text_color=MUTE, anchor="w",
                                font=ctk.CTkFont(family=_hf(), size=13))
        self.sub.pack(side="left")
        self.status = ctk.CTkLabel(
            self, text="○  model: not trained", text_color=FAINT,
            font=ctk.CTkFont(family=_mf(), size=12))
        self.status.pack(side="right", padx=18)

    def set_mode(self, key):
        _, _, title, sub, _ = MODES[key]
        self.title.configure(text=title)
        self.sub.configure(text="   ·   " + sub)


# --------------------------------------------------------------------------
# Report view (placeholder readout — the console's hero panel)
# --------------------------------------------------------------------------
class ReportView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, corner_radius=0, fg_color=PAGE)
        wrap = ctk.CTkFrame(self, fg_color="transparent")
        wrap.place(relx=0.5, rely=0.02, anchor="n")

        ctk.CTkLabel(wrap, text="Per-pixel report", text_color=INK,
                     font=ctk.CTkFont(family=_hf(), size=22, weight="bold")
                     ).pack(anchor="w", pady=(8, 0))
        ctk.CTkLabel(
            wrap, text="Placeholder — classify a map, then click a pixel to "
                       "populate M · ratio · competition.",
            text_color=MUTE, font=ctk.CTkFont(family=_hf(), size=13)
        ).pack(anchor="w", pady=(2, 16))

        body = ctk.CTkFrame(wrap, fg_color="transparent")
        body.pack()
        self._map_card(body).grid(row=0, column=0, padx=(0, 14), sticky="n")
        self._report_card(body).grid(row=0, column=1, sticky="n")

    # ---- left: mini map + spectrum (dark canvas, self-contained) ----
    def _map_card(self, parent):
        card = ctk.CTkFrame(parent, corner_radius=12, fg_color=CARD,
                            border_width=1, border_color=LINE, width=340)
        card.grid_propagate(False)
        ctk.CTkLabel(card, text="banana target · 20×18 px · RGB unmix",
                     text_color=MUTE, anchor="w",
                     font=ctk.CTkFont(family=_hf(), size=12)
                     ).pack(fill="x", padx=14, pady=(12, 6))
        cv = tk.Canvas(card, width=300, height=200, bg=PAGE,
                       highlightthickness=0, bd=0)
        cv.pack(padx=14)
        cols = ["#12303f", "#0f5a4a", TEAL, "#2b4a6b", BLUE, "#5a2e1f", CORAL]
        import math
        nx, ny, cw = 20, 18, 15
        for j in range(ny):
            for i in range(nx):
                sel = (i == 14 and j == 9)
                c = TEAL if sel else cols[int(abs(math.sin((j * nx + i) * 12.9))
                                              * len(cols)) % len(cols)]
                x0, y0 = i * cw, j * (200 // ny)
                cv.create_rectangle(x0, y0, x0 + cw - 1, y0 + (200 // ny) - 1,
                                    fill=c, width=0)
                if sel:
                    cv.create_rectangle(x0 - 1, y0 - 1, x0 + cw, y0 + (200 // ny),
                                        outline=INK, width=2)
        ctk.CTkLabel(card, text="pixel spectrum", text_color=FAINT, anchor="w",
                     font=ctk.CTkFont(family=_mf(), size=10)
                     ).pack(fill="x", padx=14, pady=(10, 2))
        sp = tk.Canvas(card, width=300, height=54, bg=PAGE,
                       highlightthickness=0, bd=0)
        sp.pack(padx=14, pady=(0, 14))
        pts = [(0, 46), (30, 44), (55, 24), (70, 42), (110, 40), (130, 12),
               (150, 36), (185, 40), (210, 26), (235, 41), (270, 38), (300, 44)]
        sp.create_line(*[c for p in pts for c in p], fill=TEAL_HI, width=2,
                       smooth=True)
        return card

    # ---- right: the report hero ----
    def _report_card(self, parent):
        card = ctk.CTkFrame(parent, corner_radius=12, fg_color=CARD,
                            border_width=1, border_color=LINE, width=300)
        card.grid_propagate(False)
        pad = ctk.CTkFrame(card, fg_color="transparent")
        pad.pack(fill="both", expand=True, padx=18, pady=16)

        ctk.CTkLabel(pad, text="report · pixel (14, 9)", text_color=FAINT,
                     anchor="w", font=ctk.CTkFont(family=_mf(), size=11)
                     ).pack(anchor="w")
        ctk.CTkLabel(pad, text="THI dominant", text_color=INK, anchor="w",
                     font=ctk.CTkFont(family=_hf(), size=20, weight="bold")
                     ).pack(anchor="w", pady=(2, 0))
        ctk.CTkLabel(pad, text="●  hit · confidence 94%",
                     text_color=TEAL_HI, anchor="w",
                     font=ctk.CTkFont(family=_hf(), size=12)
                     ).pack(anchor="w", pady=(0, 14))

        for name, color, ratio, molar in PESTICIDES:
            self._pest_row(pad, name, color, ratio, molar)

        sep = ctk.CTkFrame(pad, height=1, fg_color=LINE)
        sep.pack(fill="x", pady=(14, 10))
        ctk.CTkLabel(pad, text="⇄  Competitive adsorption", text_color=AMBER,
                     anchor="w", font=ctk.CTkFont(family=_hf(), size=13,
                                                  weight="bold")
                     ).pack(anchor="w")
        ctk.CTkLabel(
            pad, text="THI out-competes DQ at the surface — measured ratio "
                      "is suppressed vs. true dose (Langmuir).",
            text_color=MUTE, anchor="w", justify="left", wraplength=250,
            font=ctk.CTkFont(family=_hf(), size=12)
        ).pack(anchor="w", pady=(4, 0))
        return card

    def _pest_row(self, parent, name, color, ratio, molar):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(0, 8))
        head = ctk.CTkFrame(row, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text=name, text_color=color, anchor="w",
                     font=ctk.CTkFont(family=_hf(), size=13, weight="bold")
                     ).pack(side="left")
        ctk.CTkLabel(head, text=molar, text_color=INK, anchor="e",
                     font=ctk.CTkFont(family=_mf(), size=12)
                     ).pack(side="right")
        bar = ctk.CTkProgressBar(row, height=6, corner_radius=3,
                                 progress_color=color, fg_color=PANEL)
        bar.set(ratio)
        bar.pack(fill="x", pady=(3, 1))
        ctk.CTkLabel(row, text=f"{ratio * 100:.0f}% ratio", text_color=FAINT,
                     anchor="w", font=ctk.CTkFont(family=_mf(), size=10)
                     ).pack(anchor="w")


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------
class SuiteShell:
    def __init__(self, root):
        self.root = root
        try:
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "SERS.Suite.1")
        except Exception:
            pass
        root.title(SUITE_NAME)
        root.geometry("1400x880")
        root.minsize(1160, 720)
        root.configure(fg_color=PAGE)

        self.rail = Rail(root, on_select=self.select)
        self.rail.pack(side="left", fill="y")

        right = ctk.CTkFrame(root, corner_radius=0, fg_color=PAGE)
        right.pack(side="left", fill="both", expand=True)
        self.topbar = TopBar(right)
        self.topbar.pack(fill="x", side="top")

        self.content = ctk.CTkFrame(right, corner_radius=0, fg_color=PAGE)
        self.content.pack(fill="both", expand=True)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.frames = {}       # key -> frame
        self.apps = {}         # key -> app instance (state kept)
        self.current = None
        self.select("pest")    # open straight into the map tool

    def _show(self, frame):
        if self.current is not None and self.current is not frame:
            self.current.grid_remove()
        frame.grid(row=0, column=0, sticky="nsew")
        self.current = frame
        frame.tkraise()

    def select(self, key):
        if key not in self.frames:
            frame = ctk.CTkFrame(self.content, corner_radius=0, fg_color=PAGE)
            frame.grid(row=0, column=0, sticky="nsew")
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            factory = MODES[key][4]
            try:
                if factory is None:                # shell-owned Report view
                    ReportView(frame).grid(row=0, column=0, sticky="nsew")
                else:
                    host = ctk.CTkFrame(frame, corner_radius=0,
                                        fg_color="transparent")
                    host.grid(row=0, column=0, sticky="nsew")
                    host.grid_rowconfigure(0, weight=1)
                    host.grid_columnconfigure(0, weight=1)
                    self.apps[key] = factory(host)
            except Exception as exc:
                traceback.print_exc()
                frame.destroy()
                messagebox.showerror(
                    "Could not open " + MODES[key][2],
                    f"{exc}\n\nMake sure every tool file from this release is in "
                    "the same folder, then restart.")
                return
            self.frames[key] = frame
        self.rail.set_active(key)
        self.topbar.set_mode(key)
        self._show(self.frames[key])


def _install_error_surface(root):
    """A windowed exe has no console; route uncaught callback errors to a dialog
    + a log file in the user's home folder instead of vanishing silently."""
    import traceback as _tb
    from tkinter import messagebox as _mb

    def _handler(exc_type, exc, tb):
        detail = "".join(_tb.format_exception(exc_type, exc, tb))
        try:
            with open(os.path.join(os.path.expanduser("~"),
                                   "sers_suite_error.log"), "a",
                      encoding="utf-8") as f:
                f.write(detail + "\n" + "-" * 60 + "\n")
        except Exception:
            pass
        try:
            _mb.showerror(SUITE_NAME + " error",
                          f"{exc_type.__name__}: {exc}")
        except Exception:
            pass

    root.report_callback_exception = _handler


def main():
    root = ctk.CTk()
    _install_error_surface(root)
    SuiteShell(root)
    root.mainloop()


if __name__ == "__main__":
    main()
