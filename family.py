"""family.py — shared UNMIXR visual family for the customtkinter tools.

Gives the Mixture tool and the Map tool the same light look as the PyQt app:
a white header bar with the "U" logo + wordmark, teal accent, white cards.
Import and call once in each tool.
"""
from __future__ import annotations

import sys
import customtkinter as ctk

# UNMIXR light palette
PAGE = "#f5f7fa"
PANEL = "#ffffff"
CARD = "#ffffff"
LINE = "#e3e8ee"
INK = "#1c2430"
MUTE = "#5b6673"
FAINT = "#98a1ac"
TEAL = "#0f9d6b"
TEAL_HOVER = "#0c855a"
BLUE = "#1a73e8"
BLUE_HOVER = "#155ec2"
AMBER = "#c98a15"
CORAL = "#d8542a"


def head_family():
    return "Segoe UI" if sys.platform == "win32" else "Arial"


def apply():
    ctk.set_appearance_mode("Light")
    ctk.set_default_color_theme("green")   # teal accent, matching UNMIXR


def make_header(parent, title, subtitle=""):
    """A UNMIXR-style header bar: 'U' logo badge + wordmark + subtitle + hairline.
    Returns a CTkFrame the caller grids/packs at the top."""
    wrap = ctk.CTkFrame(parent, corner_radius=0, fg_color=PANEL, height=57)
    wrap.pack_propagate(False)
    wrap.grid_propagate(False)
    ctk.CTkFrame(wrap, height=1, fg_color=LINE, corner_radius=0).pack(
        side="bottom", fill="x")
    row = ctk.CTkFrame(wrap, fg_color="transparent")
    row.pack(side="top", fill="both", expand=True)
    ctk.CTkLabel(row, text="U", width=30, height=30, corner_radius=8,
                 fg_color=TEAL, text_color="#ffffff",
                 font=ctk.CTkFont(family=head_family(), size=15, weight="bold")
                 ).pack(side="left", padx=(16, 8), pady=13)
    ctk.CTkLabel(row, text=title, text_color=INK,
                 font=ctk.CTkFont(family=head_family(), size=17, weight="bold")
                 ).pack(side="left")
    if subtitle:
        ctk.CTkLabel(row, text="  ·  " + subtitle, text_color=MUTE,
                     font=ctk.CTkFont(family=head_family(), size=13)
                     ).pack(side="left")
    return wrap


def card(parent, **kw):
    """A white rounded card matching the UNMIXR look."""
    opts = dict(corner_radius=12, fg_color=CARD, border_width=1, border_color=LINE)
    opts.update(kw)
    return ctk.CTkFrame(parent, **opts)
