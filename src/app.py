"""
MainWindow: layout, sliders, signals, save, drag-and-drop.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QColor,
    QPaintEvent,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QFrame,
)

from worker import KaleidoscopeWorker
from kaleidoscope import apply_kaleidoscope

# Maximum dimension for live-preview source (keeps UI snappy)
PREVIEW_MAX_DIM = 4096
THUMBNAIL_SIZE = 240


class ThumbnailLabel(QLabel):
    """QLabel that shows the source image thumbnail with a draggable crosshair."""

    center_changed = None  # set to a callable by MainWindow

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._center_x_pct = 50.0
        self._center_y_pct = 50.0
        self._has_image = False
        self._show_placeholder()

    def _show_placeholder(self):
        pm = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        pm.fill(QColor(45, 45, 48))
        self.setPixmap(pm)
        self._has_image = False

    def set_image(self, pil_img: Image.Image):
        img = pil_img.copy()
        img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
        # Convert to RGB for display
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        ch = arr.shape[2]
        fmt = QImage.Format.Format_RGBA8888 if ch == 4 else QImage.Format.Format_RGB888
        bpl = w * ch
        qimg = QImage(np.ascontiguousarray(arr).tobytes(), w, h, bpl, fmt)
        # Center in the fixed 240×240 box with dark background
        canvas = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        canvas.fill(QColor(45, 45, 48))
        painter = QPainter(canvas)
        x_off = (THUMBNAIL_SIZE - w) // 2
        y_off = (THUMBNAIL_SIZE - h) // 2
        painter.drawImage(x_off, y_off, qimg)
        painter.end()
        self.setPixmap(canvas)
        self._has_image = True
        self._img_w = w
        self._img_h = h
        self._img_x_off = x_off
        self._img_y_off = y_off
        self.update()

    def set_center(self, x_pct: float, y_pct: float):
        self._center_x_pct = x_pct
        self._center_y_pct = y_pct
        self.update()

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        if not self._has_image:
            painter = QPainter(self)
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Drop image here\nor click Open",
            )
            painter.end()
            return
        # Draw crosshair at center position
        cx = self._img_x_off + int(self._center_x_pct / 100.0 * self._img_w)
        cy = self._img_y_off + int(self._center_y_pct / 100.0 * self._img_h)
        painter = QPainter(self)
        pen = QPen(QColor(255, 60, 60), 1)
        painter.setPen(pen)
        painter.drawLine(cx, self._img_y_off, cx, self._img_y_off + self._img_h)
        painter.drawLine(self._img_x_off, cy, self._img_x_off + self._img_w, cy)
        # Circle around crosshair
        pen2 = QPen(QColor(255, 60, 60), 1)
        painter.setPen(pen2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(cx - 6, cy - 6, 12, 12)
        painter.end()

    def _pixel_to_pct(self, px: int, py: int) -> tuple[float, float]:
        if not self._has_image:
            return 50.0, 50.0
        rx = px - self._img_x_off
        ry = py - self._img_y_off
        x_pct = max(0.0, min(100.0, rx / self._img_w * 100.0))
        y_pct = max(0.0, min(100.0, ry / self._img_h * 100.0))
        return x_pct, y_pct

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._has_image:
            x_pct, y_pct = self._pixel_to_pct(event.pos().x(), event.pos().y())
            self.set_center(x_pct, y_pct)
            if self.center_changed:
                self.center_changed(x_pct, y_pct)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton and self._has_image:
            x_pct, y_pct = self._pixel_to_pct(event.pos().x(), event.pos().y())
            self.set_center(x_pct, y_pct)
            if self.center_changed:
                self.center_changed(x_pct, y_pct)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stanley Kaleidoscope")
        self.setMinimumSize(900, 600)
        self.setAcceptDrops(True)

        self._pil_original: Image.Image | None = None  # full-resolution
        self._pil_preview: Image.Image | None = None   # downscaled for live render
        self._preview_arr: np.ndarray | None = None    # numpy of preview image
        self._worker: KaleidoscopeWorker | None = None
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(120)
        self._debounce_timer.timeout.connect(self._fire_worker)

        self._build_ui()
        self._status("Ready — open an image to start")

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        # Left: thumbnail panel
        left = self._build_left_panel()
        root_layout.addWidget(left)

        # Center: preview
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview_label.setMinimumSize(400, 400)
        self._preview_label.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3c3c3c;"
        )
        self._preview_label.setText("Open an image to see the kaleidoscope effect")
        self._preview_label.setWordWrap(True)
        root_layout.addWidget(self._preview_label, stretch=3)

        # Right: controls panel
        right = self._build_right_panel()
        root_layout.addWidget(right)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(THUMBNAIL_SIZE + 16)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        label = QLabel("Source Image")
        label.setStyleSheet("font-weight: bold; color: #cccccc;")
        layout.addWidget(label)

        self._thumbnail = ThumbnailLabel()
        self._thumbnail.center_changed = self._on_center_drag
        layout.addWidget(self._thumbnail)

        hint = QLabel("Click/drag to set\ncenter point")
        hint.setStyleSheet("color: #888888; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        layout.addStretch()
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Buttons
        self._btn_open = QPushButton("Open Image…")
        self._btn_open.clicked.connect(self._open_image)
        layout.addWidget(self._btn_open)

        self._btn_save = QPushButton("Save / Export…")
        self._btn_save.clicked.connect(self._save_image)
        self._btn_save.setEnabled(False)
        layout.addWidget(self._btn_save)

        self._btn_reset = QPushButton("Reset All")
        self._btn_reset.clicked.connect(self._reset_all)
        layout.addWidget(self._btn_reset)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #3c3c3c;")
        layout.addWidget(separator)

        # Sliders
        self._lbl_segments, self._sld_segments = self._add_slider(
            layout, "Segments", 2, 24, 8
        )
        self._lbl_rotation, self._sld_rotation = self._add_slider(
            layout, "Rotation", 0, 3600, 0
        )
        self._lbl_zoom, self._sld_zoom = self._add_slider(
            layout, "Zoom", 10, 300, 100
        )
        self._lbl_cx, self._sld_cx = self._add_slider(
            layout, "Center X", 0, 1000, 500
        )
        self._lbl_cy, self._sld_cy = self._add_slider(
            layout, "Center Y", 0, 1000, 500
        )

        self._update_slider_labels()

        layout.addStretch()
        return panel

    def _add_slider(
        self, layout: QVBoxLayout, name: str, lo: int, hi: int, default: int
    ) -> tuple[QLabel, QSlider]:
        lbl = QLabel()
        lbl.setStyleSheet("color: #cccccc; font-size: 12px;")
        layout.addWidget(lbl)

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(lo, hi)
        sld.setValue(default)
        sld.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(sld)
        return lbl, sld

    # ------------------------------------------------------------------ #
    # Slider helpers                                                       #
    # ------------------------------------------------------------------ #

    def _current_params(self) -> dict:
        return {
            "num_segments": self._sld_segments.value(),
            "rotation_deg": self._sld_rotation.value() / 10.0,
            "zoom": self._sld_zoom.value() / 100.0,
            "center_x_pct": self._sld_cx.value() / 10.0,
            "center_y_pct": self._sld_cy.value() / 10.0,
        }

    def _update_slider_labels(self):
        p = self._current_params()
        self._lbl_segments.setText(f"Segments: {p['num_segments']}")
        self._lbl_rotation.setText(f"Rotation: {p['rotation_deg']:.1f}°")
        self._lbl_zoom.setText(f"Zoom: {p['zoom']:.2f}×")
        self._lbl_cx.setText(f"Center X: {p['center_x_pct']:.1f}%")
        self._lbl_cy.setText(f"Center Y: {p['center_y_pct']:.1f}%")

    def _on_slider_changed(self):
        self._update_slider_labels()
        self._schedule_update()

    def _on_center_drag(self, x_pct: float, y_pct: float):
        self._sld_cx.blockSignals(True)
        self._sld_cy.blockSignals(True)
        self._sld_cx.setValue(round(x_pct * 10))
        self._sld_cy.setValue(round(y_pct * 10))
        self._sld_cx.blockSignals(False)
        self._sld_cy.blockSignals(False)
        self._update_slider_labels()
        self._schedule_update()

    def _reset_all(self):
        for sld, val in [
            (self._sld_segments, 8),
            (self._sld_rotation, 0),
            (self._sld_zoom, 100),
            (self._sld_cx, 500),
            (self._sld_cy, 500),
        ]:
            sld.blockSignals(True)
            sld.setValue(val)
            sld.blockSignals(False)
        self._thumbnail.set_center(50.0, 50.0)
        self._update_slider_labels()
        self._schedule_update()

    # ------------------------------------------------------------------ #
    # Worker / rendering                                                   #
    # ------------------------------------------------------------------ #

    def _schedule_update(self):
        if self._preview_arr is None:
            return
        self._debounce_timer.start()

    def _fire_worker(self):
        if self._preview_arr is None:
            return
        # Cancel any in-flight worker
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait(500)

        params = self._current_params()
        preview_w = self._preview_label.width()
        preview_h = self._preview_label.height()
        output_size = (max(preview_w, 100), max(preview_h, 100))

        self._worker = KaleidoscopeWorker(self)
        self._worker.configure(self._preview_arr, params, output_size)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()
        self._status("Rendering…")

    def _on_result(self, qimage: QImage):
        pm = QPixmap.fromImage(qimage)
        self._preview_label.setPixmap(
            pm.scaled(
                self._preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._status("Ready")

    def _on_worker_error(self, msg: str):
        self._status(f"Render error: {msg}")

    # ------------------------------------------------------------------ #
    # Image I/O                                                            #
    # ------------------------------------------------------------------ #

    def _load_image(self, path: str):
        try:
            img = Image.open(path)
            img.load()  # force decode to catch corruption early
        except (UnidentifiedImageError, Exception) as exc:
            QMessageBox.critical(self, "Cannot open image", str(exc))
            return

        self._pil_original = img

        # Downscale for live preview if needed
        if max(img.width, img.height) > PREVIEW_MAX_DIM:
            preview = img.copy()
            preview.thumbnail((PREVIEW_MAX_DIM, PREVIEW_MAX_DIM), Image.LANCZOS)
            self._pil_preview = preview
        else:
            self._pil_preview = img.copy()

        # Convert preview to RGB(A) numpy array
        mode = self._pil_preview.mode
        if mode not in ("RGB", "RGBA"):
            self._pil_preview = self._pil_preview.convert("RGB")
        self._preview_arr = np.asarray(self._pil_preview)

        # Update thumbnail
        self._thumbnail.set_image(img)

        # Enable save
        self._btn_save.setEnabled(True)
        self._status(f"Loaded: {Path(path).name}  ({img.width}×{img.height})")
        self._schedule_update()

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.webp *.bmp);;All Files (*)",
        )
        if path:
            self._load_image(path)

    def _save_image(self):
        if self._pil_original is None:
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Kaleidoscope Image",
            "kaleidoscope.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff *.tif);;WebP (*.webp);;BMP (*.bmp)",
        )
        if not path:
            return

        self._status("Exporting at full resolution…")
        try:
            # Work from original full-res image
            orig = self._pil_original
            mode = orig.mode
            if mode not in ("RGB", "RGBA"):
                orig = orig.convert("RGB")
            src_arr = np.asarray(orig)

            params = self._current_params()
            result_arr = apply_kaleidoscope(
                src_arr,
                num_segments=params["num_segments"],
                rotation_deg=params["rotation_deg"],
                zoom=params["zoom"],
                center_x_pct=params["center_x_pct"],
                center_y_pct=params["center_y_pct"],
            )

            result_img = Image.fromarray(result_arr)

            # JPEG requires RGB
            suffix = Path(path).suffix.lower()
            if suffix in (".jpg", ".jpeg") and result_img.mode == "RGBA":
                result_img = result_img.convert("RGB")

            result_img.save(path)
            self._status(f"Saved: {Path(path).name}")

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            self._status("Save failed")

    # ------------------------------------------------------------------ #
    # Drag and drop                                                        #
    # ------------------------------------------------------------------ #

    _ACCEPTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"}

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in self._ACCEPTED_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in self._ACCEPTED_EXTENSIONS:
                self._load_image(path)
                break

    # ------------------------------------------------------------------ #
    # Close                                                                #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.quit()
            self._worker.wait(2000)
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # Misc                                                                 #
    # ------------------------------------------------------------------ #

    def _status(self, msg: str):
        self._status_bar.showMessage(msg)
