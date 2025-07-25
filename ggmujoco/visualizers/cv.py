# ggmujoco/visualizers/cv.py
from __future__ import annotations
import cv2, threading, time, numpy as np

class OpenCVViewer:
    """
    • Создаёт окна лениво при первом кадре.
    • Отдельный поток опрашивает события каждые 8 мс.
    • ESC в любом окне завершает viewer.
    """
    def __init__(self):
        self._frames: dict[str, np.ndarray] = {}
        self._created: set[str] = set()          # какие окна уже есть
        self._lock   = threading.Lock()
        self._running = True
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    # ----------- публичный вызов ---------------------------------
    def update(self, name: str, frame: np.ndarray):
        if frame is None:
            return
        with self._lock:
            self._frames[name] = frame.copy()

    def close(self):
        self._running = False
        self._th.join()
        cv2.destroyAllWindows()

    # ----------- фоновый поток -----------------------------------
    def _worker(self):
        while self._running:
            with self._lock:
                items = list(self._frames.items())

            for name, frm in items:
                if name not in self._created:
                    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
                    self._created.add(name)

                cv2.imshow(name, frm)

            # ESC закрывает все окна и останавливает поток
            if cv2.waitKey(1) & 0xFF == 27:
                self._running = False

            time.sleep(0.008)   # ~120 Гц опрос
