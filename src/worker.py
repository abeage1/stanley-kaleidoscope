"""
Background QThread worker for kaleidoscope rendering.
"""

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from kaleidoscope import apply_kaleidoscope


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
        """Set up the worker before calling start()."""
        self._source = source
        self._params = params
        self._output_size = output_size
        self._cancelled = False

    def cancel(self) -> None:
        """Signal the worker to stop after the current NumPy call returns."""
        self._cancelled = True

    def run(self) -> None:
        if self._source is None:
            return
        try:
            result = apply_kaleidoscope(
                self._source,
                num_segments=self._params["num_segments"],
                rotation_deg=self._params["rotation_deg"],
                zoom=self._params["zoom"],
                center_x_pct=self._params["center_x_pct"],
                center_y_pct=self._params["center_y_pct"],
                output_size=self._output_size,
            )

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

            # Make a contiguous copy so QImage owns its memory
            contiguous = np.ascontiguousarray(result)
            qimage = QImage(
                contiguous.tobytes(), w, h, bytes_per_line, fmt
            )
            # copy() detaches from the bytes buffer
            self.result_ready.emit(qimage.copy())

        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
