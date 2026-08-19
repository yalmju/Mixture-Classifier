"""UNMIXR — SERS mixture analysis suite (PyQt6).

One native PyQt6 window; each tool is a tab. The pages live in page_*.py and the
shared UI foundation (palette, stylesheet, Canvas, KPI tile) in ui_common.py:

    Samples    group raw maps into substance classes (batches / train-test role)
    Model      train a classifier on the reference maps (learning curve, F1, …)
    Recovery   known-ratio mixtures -> response factors + composition recovery
    Quantify   ratio -> M calibration + Langmuir competition
    Real data  map analysis: composition / mixtures / per-pixel µM / calibration

    python unmixr.py
"""
from __future__ import annotations

import os
import sys

# macOS: torch (libomp) and numpy/sklearn (MKL libiomp5) can each load their own OpenMP
# runtime; the duplicate abort crashes the app the moment a model trains. Allow it.
# Set before numpy/torch import below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase, QIcon, QPixmap, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QColorDialog,
)

from ui_common import (APP_NAME, VERSION, ICON_PATH, QSS, INK, COLOR_BUS,
                       set_substance_colors, substance_colors, substance_color,
                       stop_worker, type_in_spinboxes)
from dataset import discover_dataset, load_colors, save_colors
from page_samples import SamplingPage
from page_model import ModelPage
from page_quantify import QuantifyPage
from page_validate import ValidatePage
from page_real import RealDataPage


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    # native analysis pages — clicking switches the view in-place
    PAGES = [
        ("Samples",   "samples", "Group your maps into substance classes (batches)"),
        ("Model",     "model", "Train a classifier on your reference maps"),
        ("Recovery",  "valid", "Known-ratio mixtures → response factors + composition "
                               "recovery (predicted vs real · colour blend · drift)"),
        ("Quantify",  "quant", "Ratio → concentration + adsorption competition"),
        ("Real data", "real",  "Analyze real maps: composition · mixtures · µM · calibration"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1360, 880)

        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        # top command bar
        bar = QFrame(); bar.setObjectName("topbar"); bar.setFixedHeight(58)
        bl = QHBoxLayout(bar); bl.setContentsMargins(18, 0, 18, 0); bl.setSpacing(8)
        logo = QLabel()
        logo.setFixedSize(30, 30); logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.exists(ICON_PATH):                       # use the app icon
            logo.setPixmap(QPixmap(ICON_PATH).scaled(
                28, 28, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:                                               # fallback: teal "U" badge
            logo.setObjectName("logo"); logo.setText("U")
        word = QLabel(APP_NAME); word.setObjectName("wordmark")
        bl.addWidget(logo); bl.addWidget(word); bl.addSpacing(18)

        # native page tabs — everything lives in this one window now
        self._nav_btns = {}
        for label, key, desc in self.PAGES:
            b = QPushButton(label); b.setObjectName("nav"); b.setCheckable(True)
            b.setToolTip(desc)
            b.clicked.connect(lambda _=False, k=key: self.select(k))
            bl.addWidget(b); self._nav_btns[key] = b

        # per-substance colour picker — one swatch per class; click to recolour
        # every plot in the app at once (persisted to the dataset's colors.json).
        bl.addSpacing(14)
        self._color_bar = QHBoxLayout(); self._color_bar.setSpacing(6)
        bl.addLayout(self._color_bar)
        self._color_folder = None
        self._sub_index = {}

        bl.addStretch(1)
        self.status = QLabel(f"{APP_NAME} v{VERSION}"); self.status.setObjectName("status")
        bl.addWidget(self.status)
        outer.addWidget(bar)

        # stacked content
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.pages = {
            "samples": SamplingPage(),
            "model": ModelPage(),
            "quant": QuantifyPage(),
            "valid": ValidatePage(),
            "real": RealDataPage(),
        }
        for key in ("samples", "model", "valid", "quant", "real"):
            self.stack.addWidget(self.pages[key])
        for page in self.pages.values():          # type numbers instead of clicking arrows
            type_in_spinboxes(page)

        self.select("samples")

    def select(self, key):
        for k, b in self._nav_btns.items():
            b.setChecked(k == key)
        # the dataset folder is chosen once in Samples; downstream tabs adopt it
        folder = getattr(self.pages["samples"], "data_dir", None)
        page = self.pages[key]
        if folder and hasattr(page, "set_data_dir"):
            page.set_data_dir(folder)
        if folder != self._color_folder:                  # refresh swatches on new data
            self._color_folder = folder
            self._refresh_colors()
        self.stack.setCurrentWidget(page)

    # ---- top-bar per-substance colour picker -----------------------------
    def _refresh_colors(self):
        """Rebuild the swatch row from the current dataset's classes."""
        while self._color_bar.count():
            it = self._color_bar.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        folder = self._color_folder
        names = []
        if folder:
            try:
                names = [c for c, _ in discover_dataset(folder)]
            except Exception:
                names = []
        if not names:
            return
        set_substance_colors(load_colors(folder))         # adopt saved choices
        self._sub_index = {nm: i for i, nm in enumerate(names)}
        lbl = QLabel("colours:"); lbl.setObjectName("field")
        self._color_bar.addWidget(lbl)
        for i, nm in enumerate(names):
            self._color_bar.addWidget(self._color_swatch(nm, substance_color(nm, i)))

    def _color_swatch(self, name, color):
        b = QPushButton(name); b.setObjectName("ghost"); b.setFixedHeight(24)
        b.setStyleSheet(f"QPushButton{{border:2px solid {color};border-radius:6px;"
                        f"padding:2px 10px;color:{INK};}}")
        b.clicked.connect(lambda _=False, nm=name: self._pick_color(nm))
        return b

    def _pick_color(self, name):
        idx = self._sub_index.get(name, 0)
        c = QColorDialog.getColor(QColor(substance_color(name, idx)), self,
                                  f"Colour for {name}")
        if not c.isValid():
            return
        mapping = substance_colors(); mapping[name] = c.name()
        set_substance_colors(mapping)
        if self._color_folder:
            try:
                save_colors(self._color_folder, mapping)
            except Exception as exc:
                print("save colors:", exc, file=sys.stderr)
        COLOR_BUS.changed.emit()                           # recolour every page
        self._refresh_colors()

    # ---- shutdown --------------------------------------------------------
    def closeEvent(self, event):
        """Join every page's background job before the window goes away.

        Qt aborts the process (exit code 0xC0000409 on Windows) if a QThread is
        destroyed while still running, so closing the app mid-train must wait for
        the worker instead of tearing it down underneath itself.
        """
        for page in self.pages.values():
            stop_worker(page)
        super().closeEvent(event)


def main():
    # Point Qt at its bundled platform plugins. Fixes the macOS start-up error
    #   qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""
    # which happens when the plugin search path is empty (e.g. launched from Finder or a
    # venv where auto-discovery fails). QLibraryInfo works without a platform plugin.
    if "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ:
        try:
            from PyQt6.QtCore import QLibraryInfo
            pp = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
            if pp and os.path.isdir(pp):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pp
        except Exception:
            pass
    # Windows taskbar groups by AppUserModelID; without an explicit one the taskbar shows
    # the python(w).exe icon instead of our window icon. Set it before the QApplication.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("UNMIXR.SERS.app")
        except Exception:
            pass
    app = QApplication(sys.argv)
    # PyQt6 aborts the whole process (0xC0000409 on Windows) when a slot raises.
    # Show the traceback and keep running instead — a broken plot must not kill
    # an hour of loaded state.
    def _excepthook(t, v, tb):
        import traceback
        txt = "".join(traceback.format_exception(t, v, tb))
        print(txt, file=sys.stderr)
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "UNMIXR — unexpected error", txt[-1600:])
        except Exception:
            pass
    sys.excepthook = _excepthook
    app.setStyleSheet(QSS)
    # the real OS UI font: Segoe UI on Windows, SF Pro on macOS, the desktop's
    # font on Linux — hardcoding "Segoe UI" left mac/Linux on a Qt fallback
    ui_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    ui_font.setPointSize(10)
    app.setFont(ui_font)
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    win = MainWindow()
    if os.path.exists(ICON_PATH):
        win.setWindowIcon(QIcon(ICON_PATH))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
