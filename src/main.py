"""
Entry point: QApplication bootstrap, dark theme.
"""

import os
import sys
from pathlib import Path

# PyInstaller frozen-bundle path injection
if getattr(sys, "frozen", False):
    bundle_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    sys.path.insert(0, str(bundle_dir))

# Retina / HiDPI support — must be set before QApplication creation
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from app import MainWindow


def _dark_palette() -> QPalette:
    palette = QPalette()
    dark = QColor(45, 45, 48)
    darker = QColor(30, 30, 30)
    mid = QColor(60, 60, 63)
    light_text = QColor(220, 220, 220)
    highlight = QColor(70, 130, 180)  # steel blue
    highlight_text = QColor(255, 255, 255)
    disabled_text = QColor(110, 110, 110)
    link = QColor(100, 160, 220)

    palette.setColor(QPalette.ColorRole.Window, dark)
    palette.setColor(QPalette.ColorRole.WindowText, light_text)
    palette.setColor(QPalette.ColorRole.Base, darker)
    palette.setColor(QPalette.ColorRole.AlternateBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase, dark)
    palette.setColor(QPalette.ColorRole.ToolTipText, light_text)
    palette.setColor(QPalette.ColorRole.Text, light_text)
    palette.setColor(QPalette.ColorRole.Button, mid)
    palette.setColor(QPalette.ColorRole.ButtonText, light_text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, highlight_text)
    palette.setColor(QPalette.ColorRole.Link, link)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text
    )
    return palette


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Stanley Kaleidoscope")
    app.setOrganizationName("Stanley")
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())

    # Extra stylesheet touches for sliders and group boxes
    app.setStyleSheet(
        """
        QSlider::groove:horizontal {
            height: 4px;
            background: #3c3c3c;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #4682b4;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::sub-page:horizontal {
            background: #4682b4;
            border-radius: 2px;
        }
        QPushButton {
            padding: 5px 10px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #505053;
        }
        QPushButton:pressed {
            background-color: #4682b4;
        }
        QStatusBar {
            color: #aaaaaa;
            font-size: 11px;
        }
        """
    )

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
