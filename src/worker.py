"""
Background QThread worker for kaleidoscope rendering.
"""

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from kaleidoscope import apply_effect


class KaleidoscopeWorker(QThread):
    result_ready = pyqtSignal(QImage)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: np.ndarray | None = None
        self._params: dict = {}
        self._output_size: tuple[int, int] | None = None
        self._cancelled = False

    def configure(
        self,
        source: np.ndarray,
        params: dict,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        self._source = source
        self._params = params
        self._output_size = output_size
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        if self._source is None:
            return
        try:
            mode = self._params.get("mode", "radial")
            result = apply_effect(mode, self._source, self._params, self._output_size)

            if self._cancelled:
                return

            h, w = result.shape[:2]
            channels = result.shape[2] if result.ndim == 3 else 1

            if channels == 4:
                fmt = QImage.Format.Format_RGBA8888
                bytes_per_line = w * 4
            else:
                fmt = QImage.Format.Format_RGB888
                bytes_per_line = w * 3

            contiguous = np.ascontiguousarray(result)
            qimage = QImage(contiguous.tobytes(), w, h, bytes_per_line, fmt)
            self.result_ready.emit(qimage.copy())

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
