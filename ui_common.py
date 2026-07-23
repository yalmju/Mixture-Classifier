"""ui_common.py — shared UI foundation for UNMIXR: palette, stylesheet, the
matplotlib Canvas, the KPI tile, the card helper and the figure-export helper.
Imported by every page module and by unmixr.py."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QSizePolicy


APP_NAME = "UNMIXR"
VERSION = "1.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# App icon: drop a PNG at assets/icon.png and it is picked up automatically.
ICON_PATH = os.path.join(BASE_DIR, "assets", "icon.png")

# ---- light palette -------------------------------------------------------
PAGE   = "#fafbfc"
PANEL  = "#ffffff"
CARD   = "#ffffff"
LINE   = "#e3e8ee"
INK    = "#1c2430"
MUTE   = "#5b6673"
FAINT  = "#98a1ac"
TEAL   = "#0f9d6b"
BLUE   = "#1a73e8"
AMBER  = "#c98a15"
CORAL  = "#d8542a"
PURPLE = "#6b5fd6"
PINK   = "#c85a8f"
GREEN  = "#4a9e2a"
RED    = "#d64545"   # detection "miss" (outcome grid)
TNGRAY = "#eef1f4"   # detection "correct absent"
# discrete label palette for scatter / bars
SERIES = [TEAL, BLUE, AMBER, CORAL, PURPLE, PINK, GREEN, "#546170"]
# single-hue teal ramp for the confusion matrix (matches the light theme):
# near-white for empty cells → teal for the strong diagonal
CM_CMAP = LinearSegmentedColormap.from_list(
    "unmixr_teal", ["#f1f7f4", "#a9ddc7", "#3fb488", TEAL, "#0a6b49"])

QSS = f"""
QMainWindow, QWidget {{ background: {PAGE}; color: {INK};
    font-family: 'Segoe UI', Arial; font-size: 14px; }}
#topbar {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
#wordmark {{ font-size: 18px; font-weight: 600; color: {INK}; }}
#logo {{ background: {TEAL}; color: #ffffff; font-size: 15px; font-weight: 700;
    border-radius: 8px; }}
#status {{ color: {FAINT}; font-family: 'Consolas', monospace; font-size: 12px; }}
QPushButton#nav {{ background: transparent; color: {MUTE}; border: none;
    padding: 8px 18px; border-radius: 8px; font-size: 14px; }}
QPushButton#nav:hover {{ background: {CARD}; color: {INK}; }}
QPushButton#nav:checked {{ background: {PAGE}; color: {INK}; font-weight: 600;
    border: 1px solid {LINE}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame#kpi {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; }}
#kpiLabel {{ color: {MUTE}; font-size: 12px; }}
#kpiValue {{ color: {INK}; font-size: 26px; font-weight: 600; }}
#cardTitle {{ color: {MUTE}; font-size: 12px; font-weight: 600; }}
#h1 {{ font-size: 22px; font-weight: 600; color: {INK}; }}
#sub {{ color: {MUTE}; font-size: 13px; }}
QPushButton#primary {{ background: {TEAL}; color: #ffffff; border: none;
    border-radius: 8px; padding: 9px 20px; font-size: 14px; font-weight: 600; }}
QPushButton#primary:hover {{ background: #0c855a; }}
QPushButton#primary:disabled {{ background: {LINE}; color: {FAINT}; }}
QPushButton#ghost {{ background: transparent; color: {INK}; border: 1px solid {LINE};
    border-radius: 8px; padding: 9px 20px; font-size: 14px; }}
QPushButton#ghost:hover {{ border-color: {TEAL}; }}
QSpinBox, QDoubleSpinBox {{ background: {PANEL}; color: {INK};
    border: 1px solid {LINE}; border-radius: 6px; padding: 4px 6px; min-width: 64px; }}
QComboBox {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
    border-radius: 6px; padding: 4px 8px; min-width: 128px; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; color: {INK};
    border: 1px solid {LINE}; selection-background-color: {PAGE};
    selection-color: {INK}; outline: none; }}
QLabel#field {{ color: {MUTE}; font-size: 13px; }}
QProgressBar {{ background: {PAGE}; border: 1px solid {LINE}; border-radius: 3px; }}
QProgressBar::chunk {{ background: {TEAL}; border-radius: 3px; }}
"""


def _save_figs(named_canvases, folder):
    """Save each (name, Canvas) to folder/<name>.png. Returns count."""
    n = 0
    for name, cv in named_canvases:
        cv.fig.savefig(os.path.join(folder, name + ".png"), dpi=300,
                       facecolor=CARD, bbox_inches="tight")
        n += 1
    return n


# --------------------------------------------------------------------------
# matplotlib canvas
# --------------------------------------------------------------------------
class Canvas(FigureCanvasQTAgg):
    def __init__(self, w=4.0, h=3.0):
        self.fig = Figure(figsize=(w, h), dpi=100)
        self.fig.patch.set_facecolor(CARD)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def style(self, ax):
        ax.set_facecolor(CARD)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.tick_params(colors=MUTE, labelsize=8)
        ax.xaxis.label.set_color(MUTE)
        ax.yaxis.label.set_color(MUTE)
        ax.title.set_color(INK)
        return ax

    def new_ax(self):
        self.fig.clear()
        return self.style(self.fig.add_subplot(111))

    def placeholder(self, text):
        ax = self.new_ax()
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color=FAINT,
                fontsize=10, transform=ax.transAxes)
        self.draw_idle()


# --------------------------------------------------------------------------
# KPI tile + card
# --------------------------------------------------------------------------
class Kpi(QFrame):
    def __init__(self, label):
        super().__init__()
        self.setObjectName("kpi")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        self._l = QLabel(label); self._l.setObjectName("kpiLabel")
        self._v = QLabel("—"); self._v.setObjectName("kpiValue")
        lay.addWidget(self._l); lay.addWidget(self._v)

    def set(self, value, color=INK):
        self._v.setText(value)
        self._v.setStyleSheet(f"color:{color};")


def _card(title):
    frame = QFrame(); frame.setObjectName("card")
    lay = QVBoxLayout(frame); lay.setContentsMargins(12, 10, 12, 12); lay.setSpacing(6)
    t = QLabel(title); t.setObjectName("cardTitle")
    lay.addWidget(t)
    return frame, lay


__all__ = [
    "APP_NAME", "VERSION", "BASE_DIR", "ICON_PATH", "QSS",
    "PAGE", "PANEL", "CARD", "LINE", "INK", "MUTE", "FAINT", "TEAL", "BLUE",
    "AMBER", "CORAL", "PURPLE", "PINK", "GREEN", "RED", "TNGRAY",
    "SERIES", "CM_CMAP", "Canvas", "Kpi", "_card", "_save_figs",
]
