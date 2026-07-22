"""UNMIXR — SERS mixture analysis suite (PyQt6).

One native PyQt6 window; each tool is a tab. The pages live in page_*.py and the
shared UI foundation (palette, stylesheet, Canvas, KPI tile) in ui_common.py:

    Samples    group raw maps into substance classes (batches / train-test role)
    Model      train a classifier on the reference maps (learning curve, F1, …)
    Predict    load one unknown sample -> its component ratio (per-pixel NNLS)
    Quantify   ratio -> M calibration + Langmuir competition
    Real data  map analysis: single-component / mixture / composition / calib

    python unmixr.py
"""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QHBoxLayout, QVBoxLayout, QStackedWidget,
)

from ui_common import APP_NAME, VERSION, ICON_PATH, QSS
from page_samples import SamplingPage
from page_model import ModelPage
from page_predict import PredictPage
from page_quantify import QuantifyPage
from page_real import RealDataPage


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    # native analysis pages — clicking switches the view in-place
    PAGES = [
        ("Samples",   "samples", "Group your maps into substance classes (batches)"),
        ("Model",     "model", "Train a classifier on your reference maps"),
        ("Quantify",  "quant", "Ratio → concentration + adsorption competition"),
        ("Predict",   "predict", "Load an unknown sample → read its component ratio"),
        ("Real data", "real",  "Analyze real maps: identify · mixtures · calibration"),
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
            "predict": PredictPage(),
            "quant": QuantifyPage(),
            "real": RealDataPage(),
        }
        for key in ("samples", "model", "quant", "predict", "real"):
            self.stack.addWidget(self.pages[key])

        self.select("samples")

    def select(self, key):
        for k, b in self._nav_btns.items():
            b.setChecked(k == key)
        self.stack.setCurrentWidget(self.pages[key])


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(QFont("Segoe UI", 10))
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    win = MainWindow()
    if os.path.exists(ICON_PATH):
        win.setWindowIcon(QIcon(ICON_PATH))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
