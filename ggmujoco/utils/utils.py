import numpy as np
from typing import Tuple
import json, pathlib
import re, cv2, random
from graspnetAPI import GraspGroup

def euler_to_quat(roll: float, pitch: float, yaw: float,
                  degrees: bool = True) -> Tuple[float, float, float, float]:
    """
    Конвертирует углы Эйлера (roll‑pitch‑yaw, ZYX) → кватернион w‑x‑y‑z,
    совместимый с MuJoCo.

    Parameters
    ----------
    roll, pitch, yaw : float
        Повороты вокруг X, Y, Z‑осей **в указанном порядке**.
    degrees : bool, default True
        Если True – углы задаются в градусах; иначе в радианах.

    Returns
    -------
    Tuple[w, x, y, z] : кватернион‑кортеж.
    """
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])

    cr = np.cos(roll * 0.5);  sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5);   sy = np.sin(yaw * 0.5)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return (float(w), float(x), float(y), float(z))


def intrinsics_to_fovy(fx: float, fy: float,
                       width: int, height: int) -> float:
    """
    Возвращает вертикальный FOV (deg) по фок. расстояниям и размеру кадра.
    При несquare пикселях берём fy + реальную высоту кадра.
    """
    return float(np.rad2deg(2 * np.arctan(height / (2 * fy))))

def load_intrinsics(path: pathlib.Path) -> Tuple[float, int, int]:
    """
    Читает JSON с fx, fy, width, height и возвращает (fovy, w, h).
    Если файл не найден → (58°, 640, 480).
    """
    try:
        data = json.loads(path.read_text())
        fx, fy = data["fx"], data["fy"]
        w, h   = data["width"], data["height"]
        fovy   = intrinsics_to_fovy(fx, fy, w, h)
        return fovy, w, h
    except Exception as exc:
        print(f"[WARN] intrinsics '{path}' not loaded → {exc}")
        return 58.0, 640, 480
    

# ═══════════════════════════════════════
# вспомогательные утилиты для текстур
# ═══════════════════════════════════════
def convert_to_png(jpg_path: pathlib.Path) -> pathlib.Path:
    png_path = jpg_path.with_suffix(".png")
    cv2.imwrite(str(png_path), cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED))
    return png_path


def pick_folder(txroot: pathlib.Path) -> pathlib.Path:
    subdirs = [d for d in txroot.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"no subfolders in {txroot}")
    return random.choice(subdirs)


def pick_color_map(folder: pathlib.Path) -> pathlib.Path:
    # приоритет по имени
    for f in sorted(folder.glob("*")):
        if re.search(r"(Color|BaseColor|Albedo)", f.stem, re.I) and \
           f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return convert_to_png(f) if f.suffix != ".png" else f
    # любой PNG/JPEG
    for f in sorted(folder.glob("*")):
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return convert_to_png(f) if f.suffix != ".png" else f
    raise FileNotFoundError(f"no image files in {folder}")

# ─────────────────────────────────────────────────────────────
# helper: заменить текущие меши grasp’ов на новые
# ─────────────────────────────────────────────────────────────
def update_grasps(vis, grasp_geoms, gg_array, R_flip, max_show=50):
    """Удалить предыдущие меши grasp’ов и добавить новые.
       grasp_geoms — mutable‑список, который хранит добавленные объекты."""
    # 1. убрать старые
    for g in grasp_geoms:
        vis.remove_geometry(g, reset_bounding_box=False)
    grasp_geoms.clear()

    # 2. добавить новые, если есть
    if gg_array is None or gg_array.size == 0:
        return

    gg = GraspGroup(gg_array);  gg.nms();  gg.sort_by_score()
    for g in gg[:max_show].to_open3d_geometry_list():
        g.rotate(R_flip, center=(0, 0, 0))
        vis.add_geometry(g, reset_bounding_box=False)
        grasp_geoms.append(g)