"""ui_common.py — shared UI foundation for UNMIXR: palette, stylesheet, the
matplotlib Canvas, the KPI tile, the card helper and the figure-export helper.
Imported by every page module and by unmixr.py."""
from __future__ import annotations

import os
import sys

# Pin matplotlib's Qt binding to the one the app itself uses. Left unset,
# matplotlib picks whichever binding it finds first, so a machine that also has
# PyQt5 / PySide6 installed can load a SECOND Qt library into the process —
# which crashes at startup (most visibly on macOS). Must precede any matplotlib
# import; the app is PyQt6-only, so there is nothing else to negotiate.
os.environ.setdefault("QT_API", "PyQt6")


def _ensure_qt_plugin_path():
    """Point Qt at PyQt6's own plugins when nothing else has.

    Startup otherwise dies with `Could not find the Qt platform plugin "cocoa"
    in ""` on macOS (or "xcb" on Linux): the empty path means Qt was handed a
    blank QT_QPA_PLATFORM_PLUGIN_PATH — usually exported by a stale Homebrew /
    conda / PyQt5 Qt in the shell profile — and searched nowhere at all instead
    of falling back to the wheel's own plugins.

    A path that is set and actually usable is always left alone. One that is
    blank, or that points somewhere with no platforms/ directory (a Qt5 or
    Homebrew Qt left over in the shell profile — the reason this keeps coming
    back after a reinstall "fixes" it), is replaced with the wheel's own.

    Does nothing if the wheel has no plugins directory, which is the other cause
    of this error (a broken PyQt6-Qt6 install) and needs a reinstall, not a path.
    """
    key = "QT_QPA_PLATFORM_PLUGIN_PATH"
    current = os.environ.get(key)
    if current and os.path.isdir(os.path.join(current, "platforms")):
        return                               # set and usable: respect it
    try:
        import PyQt6
    except ImportError:                      # nothing to point at yet
        return
    plugins = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins")
    if os.path.isdir(os.path.join(plugins, "platforms")):
        if current:
            print(f"UNMIXR: ignoring unusable {key}={current!r} "
                  f"(no platforms/ there); using PyQt6's own plugins",
                  file=sys.stderr)
        os.environ[key] = plugins


_ensure_qt_plugin_path()

import matplotlib
matplotlib.use("QtAgg")
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QSizePolicy


APP_NAME = "UNMIXR"
VERSION = "1.0"

# ---- cross-platform UI font stacks ---------------------------------------
# Segoe UI / Consolas are Windows-only; naming them alone leaves macOS and Linux
# on an arbitrary Qt fallback. Each platform leads with its own system UI font
# and keeps the others as fallbacks, so one stylesheet looks native everywhere.
if sys.platform == "darwin":
    UI_FONT = "'SF Pro Text', 'Helvetica Neue', Helvetica, Arial"
    MONO_FONT = "'SF Mono', Menlo, Monaco, monospace"
elif sys.platform.startswith("win"):
    UI_FONT = "'Segoe UI', Arial"
    MONO_FONT = "Consolas, 'Courier New', monospace"
else:
    UI_FONT = "'Ubuntu', 'Noto Sans', 'DejaVu Sans', Arial"
    MONO_FONT = "'Ubuntu Mono', 'DejaVu Sans Mono', monospace"
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

# ---- shared per-substance colour state (global across every page) ---------
# One source of truth for DQ/TBZ/THI/… colours: the top-bar picker writes here,
# every plot reads via substance_color(), and COLOR_BUS.changed tells all pages
# to recolour. Falls back to the legacy SERIES palette when no override is set.
class _ColorBus(QObject):
    changed = pyqtSignal()

COLOR_BUS = _ColorBus()
_SUB_COLORS = {}                       # substance name -> "#hex" override


# a single trained composition model shared across tabs: the Model tab (Step 2)
# publishes it here, and Recovery / Real-data adopt it so they APPLY the trained
# model instead of retraining ("train once, reuse downstream").
class _ModelBus(QObject):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model = None
        self.origin = ""               # short label of where it came from

    def set(self, model, origin=""):
        self.model = model; self.origin = origin
        self.changed.emit()

MODEL_BUS = _ModelBus()


# the known-ratio mixtures are prepared once in Samples (Step 1) and shared: Model /
# Recovery reload from disk when this fires, so the list is entered in one place.
class _MixtureBus(QObject):
    changed = pyqtSignal()

MIXTURE_BUS = _MixtureBus()


def set_substance_colors(mapping):
    """Replace the global substance→colour overrides ({name: '#hex'})."""
    _SUB_COLORS.clear()
    _SUB_COLORS.update({str(k): str(v) for k, v in (mapping or {}).items()})


def substance_colors():
    """Current overrides as a plain dict (copy)."""
    return dict(_SUB_COLORS)


def substance_color(name, index=0):
    """Chosen colour for a substance, else the legacy SERIES default by index."""
    return _SUB_COLORS.get(name, SERIES[index % len(SERIES)])

# ---- cnsplots-style figure form (colours stay legacy; only geometry/type) --
# Ported from github.com/yalmju/cnsplots setup_matplotlib. Colours are left to
# the legacy palette above (prop_cycle = SERIES); this only fixes the *form*:
# thin despined axes, small Helvetica type, tight ticks — so every Canvas plot
# comes out in one consistent, publication-ready style.
matplotlib.rcParams.update({
    # fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    # axes / titles
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "center",
    "axes.titlepad": 4,
    "axes.labelsize": 10,
    "axes.labelpad": 2,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.xmargin": 0.05,
    "axes.ymargin": 0.05,
    "axes.prop_cycle": cycler(color=SERIES),   # legacy discrete palette
    # ticks
    "xtick.color": "black", "ytick.color": "black",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 1, "ytick.major.pad": 1,
    # legend
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.markerscale": 0.5,
    "legend.handlelength": 0.7,
    "legend.handleheight": 0.7,
    "legend.handletextpad": 0.3,
    # vector export stays editable in Illustrator
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

QSS = f"""
QMainWindow, QWidget {{ background: {PAGE}; color: {INK};
    font-family: {UI_FONT}; font-size: 14px; }}
#topbar {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
#wordmark {{ font-size: 18px; font-weight: 600; color: {INK}; }}
#logo {{ background: {TEAL}; color: #ffffff; font-size: 15px; font-weight: 700;
    border-radius: 8px; }}
#status {{ color: {FAINT}; font-family: {MONO_FONT}; font-size: 12px; }}
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
    border: 1px solid {LINE}; border-radius: 6px; padding: 4px 8px; min-width: 76px; }}
/* no steppers: the arrows are a tiny target and invite clicking a value up one
   step at a time. Type the number instead — arrow keys and the wheel still work. */
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0px; border: none; }}
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


def _grid_shape(fig):
    """(nrows, ncols) of the figure's subplot grid, ignoring colour-bar axes."""
    nr = nc = 1
    for ax in fig.axes:
        ss = ax.get_subplotspec()
        if ss is None:
            continue
        r, c = ss.get_gridspec().get_geometry()
        nr, nc = max(nr, r), max(nc, c)
    return nr, nc


# per-panel export size (inches ≈ 170×145 px @72) — one panel per subplot cell,
# so a 1×N grid exports N-times wider. Keeps the exported aspect deterministic.
_PANEL_W, _PANEL_H = 3.4, 2.8   # roomy enough that 9–10pt text never crowds
EXPORT_DPI = 600      # print-quality; every exported PNG is ≥300 dpi


def _save_figs(named_canvases, folder):
    """Save each (name, Canvas) to folder/<name>.png at a *deterministic* size
    derived from its subplot grid — not the arbitrary on-screen (stretched)
    window size. Returns count.

    Exports are publication-style: NO background (the figure and every axes patch
    go transparent, so the PNG drops onto any journal/slide background) and the
    layout is squeezed to the ink — tight_layout at minimal pad plus a hairline
    bbox margin. The on-screen canvas keeps its card background; only the saved
    file is stripped."""
    n = 0
    for name, cv in named_canvases:
        fig = cv.fig
        prev = fig.get_size_inches().copy()
        nr, nc = _grid_shape(fig)
        pw, ph = getattr(cv, "export_panel", (_PANEL_W, _PANEL_H))
        axes_bg = [(ax, ax.get_facecolor()) for ax in fig.get_axes()]
        try:
            fig.set_size_inches(nc * pw, nr * ph, forward=False)
            for ax, _bg in axes_bg:
                ax.set_facecolor("none")          # no card behind the panels either
            try:
                fig.tight_layout(pad=0.15, w_pad=0.25, h_pad=0.25)
            except Exception:
                pass
            fig.savefig(os.path.join(folder, name + ".png"), dpi=EXPORT_DPI,
                        transparent=True, bbox_inches="tight", pad_inches=0.01)
        finally:
            for ax, bg in axes_bg:
                ax.set_facecolor(bg)
            fig.set_size_inches(prev, forward=True)
            cv.draw_idle()
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

    def wheelEvent(self, event):
        # don't eat the wheel — let the enclosing QScrollArea scroll the page
        # (the app uses clicks, not matplotlib wheel-zoom, on these canvases)
        event.ignore()

    def style(self, ax):
        # cnsplots form: despine + thin (0.5pt) black spines + tight black ticks.
        ax.set_facecolor(CARD)
        for name, s in ax.spines.items():
            if name in ("top", "right"):
                s.set_visible(False)
            else:
                s.set_color("black")
                s.set_linewidth(0.5)
        ax.tick_params(colors="black", labelsize=9, length=2, width=0.6, pad=1)
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
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


# --------------------------------------------------------------------------
# background worker lifecycle
# --------------------------------------------------------------------------
# Every long job (train / unmix / calibrate / validate) runs on its own QThread.
# Qt aborts the whole process — exit code 0xC0000409 on Windows — if a QThread
# is destroyed while it is still running, so the thread must outlive the job and
# be joined before the app quits. These two helpers are the only place that
# lifecycle is spelled out; pages call them instead of hand-rolling it.
def type_in_spinboxes(root):
    """Drop the stepper arrows from every spin box under ``root`` and let the value be
    typed. Purely presentational — range and step are untouched, so the wheel and the
    arrow keys still step by the same amount."""
    from PyQt6.QtWidgets import QAbstractSpinBox
    for sb in root.findChildren(QAbstractSpinBox):
        sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        sb.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        sb.setKeyboardTracking(False)      # act on the finished number, not each keypress


def start_worker(owner, worker, *, done=None, fail=None, progress=None):
    """Run `worker.run()` on a fresh QThread owned by `owner`.

    Keeps strong references on `owner` (`_thread` / `_worker`) for as long as the
    job runs, so neither object can be garbage-collected mid-flight. Returns
    False without starting anything when a job is already in flight — otherwise a
    second click would rebind `owner._thread` and drop the last reference to the
    live thread, which is what makes Qt abort.
    """
    old = getattr(owner, "_thread", None)
    if old is not None:
        try:
            if old.isRunning():
                return False
        except RuntimeError:            # C++ side already gone; safe to replace
            pass

    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if progress is not None and hasattr(worker, "progress"):
        worker.progress.connect(progress)
    if done is not None:
        worker.done.connect(done)
    if fail is not None:
        worker.fail.connect(fail)
    # stop the thread's event loop once the job reports either way
    worker.done.connect(thread.quit)
    worker.fail.connect(thread.quit)
    # ...and only tear the objects down after that loop has actually stopped
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(lambda: _clear_worker(owner))

    owner._thread = thread
    owner._worker = worker
    thread.start()
    return True


def _clear_worker(owner):
    """Drop the finished thread/worker refs so the next run starts clean."""
    owner._thread = None
    owner._worker = None


def worker_busy(owner):
    """True while `owner`'s background job is still running.

    Pages check this before touching their UI so a click during a run is a clean
    no-op rather than a half-applied "Working…" state.
    """
    thread = getattr(owner, "_thread", None)
    if thread is None:
        return False
    try:
        return thread.isRunning()
    except RuntimeError:                # C++ side already gone
        return False


def stop_worker(owner, timeout_ms=15000):
    """Join `owner`'s worker thread if one is running.

    Call this from closeEvent: quitting the app while a job is still running
    destroys a live QThread, which aborts the process instead of exiting.
    """
    thread = getattr(owner, "_thread", None)
    if thread is None:
        return
    try:
        running = thread.isRunning()
    except RuntimeError:                # already deleted — nothing to join
        owner._thread = None
        return
    if not running:
        return
    thread.quit()
    if not thread.wait(timeout_ms):     # job ignores quit() (tight numeric loop)
        thread.terminate()
        thread.wait(2000)


__all__ = [
    "APP_NAME", "VERSION", "BASE_DIR", "ICON_PATH", "QSS",
    "PAGE", "PANEL", "CARD", "LINE", "INK", "MUTE", "FAINT", "TEAL", "BLUE",
    "AMBER", "CORAL", "PURPLE", "PINK", "GREEN", "RED", "TNGRAY",
    "SERIES", "CM_CMAP", "Canvas", "Kpi", "_card", "_save_figs", "EXPORT_DPI",
    "COLOR_BUS", "MODEL_BUS", "MIXTURE_BUS", "set_substance_colors", "substance_colors",
    "substance_color", "start_worker", "stop_worker", "worker_busy", "type_in_spinboxes",
]
