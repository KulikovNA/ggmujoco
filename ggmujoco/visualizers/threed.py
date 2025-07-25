import math, time, threading, queue, traceback
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


class Open3DViewer:
    """
    Асинхронное окно Open3D. Передавайте облако и захваты одной
    командой update_scene() — они никогда не «разъедутся».
    """

    def __init__(
        self,
        w: int = 640,
        h: int = 480,
        title: str = "PointCloud",
        apply_flip: bool = False,   # ← по умолчанию НЕ поворачиваем
    ):
        self.w, self.h, self.title = w, h, title
        self.apply_flip = apply_flip
        self.R = (
            o3d.geometry.get_rotation_matrix_from_xyz([math.pi, 0, 0])
            if apply_flip
            else None
        )

        self._q: "queue.Queue[Tuple[str, object]]" = queue.Queue(maxsize=8)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ─────────── Публичный API ───────────
    def show_async(self):
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self):
        self._running.clear()
        self._enqueue(("quit", None))
        if self._thread:
            self._thread.join()

    # лучше пользоваться одним вызовом ↓↓↓
    def update_scene(
        self,
        pts_xyz: np.ndarray,
        colors_rgb: np.ndarray,
        gg_array: Optional[np.ndarray] = None,
    ):
        assert pts_xyz.shape == colors_rgb.shape
        self._enqueue(
            (
                "scene",
                (
                    pts_xyz.copy(),
                    colors_rgb.copy(),
                    None if gg_array is None else gg_array.copy(),
                ),
            )
        )

    # совместимость со старым кодом
    def update_cloud(self, pts: np.ndarray, clr: np.ndarray):
        self._enqueue(("cloud", (pts.copy(), clr.copy())))

    def update_grasps(self, gg: Optional[np.ndarray]):
        self._enqueue(("grasp", None if gg is None else gg.copy()))

    # ─────────── Внутреннее ───────────
    def _enqueue(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def _loop(self):
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(self.title, self.w, self.h)
            pc = o3d.geometry.PointCloud()
            vis.add_geometry(pc)
            grasp_geoms = []

            pts = clr = gg_cached = None
            first_fit = True
            dt_target = 1 / 60.0

            while self._running.is_set():
                tic = time.time()

                # 1 — читаем все команды
                try:
                    while True:
                        cmd, payload = self._q.get_nowait()
                        if cmd == "scene":
                            pts, clr, gg_cached = payload
                        elif cmd == "cloud":
                            pts, clr = payload
                        elif cmd == "grasp":
                            gg_cached = payload
                        elif cmd == "quit":
                            self._running.clear()
                except queue.Empty:
                    pass

                # 2 — облако
                if pts is not None:
                    show_pts = pts @ self.R.T if self.R is not None else pts
                    pc.points = o3d.utility.Vector3dVector(show_pts)
                    pc.colors = o3d.utility.Vector3dVector(clr)
                    vis.update_geometry(pc)

                # 3 — захваты
                if gg_cached is not None:
                    for g in grasp_geoms:
                        vis.remove_geometry(g)
                    grasp_geoms.clear()

                    try:
                        from graspnetAPI import GraspGroup

                        gg = (
                            GraspGroup(gg_cached)
                            .nms()
                            .sort_by_score()
                        )
                        for g in gg[:50].to_open3d_geometry_list():
                            if self.R is not None:
                                g.rotate(self.R, center=(0, 0, 0))
                            vis.add_geometry(g)
                            grasp_geoms.append(g)
                    except ImportError:
                        print("[WARN] graspnetAPI not installed → grasps skipped")
                    gg_cached = None

                # 4 — первый авто‑fit
                if first_fit and pts is not None:
                    vis.reset_view_point(True)
                    first_fit = False

                # 5 — держим GUI живым
                vis.poll_events()
                vis.update_renderer()
                time.sleep(max(0.0, dt_target - (time.time() - tic)))

            vis.destroy_window()

        except Exception:
            traceback.print_exc()
