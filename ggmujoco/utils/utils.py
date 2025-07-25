# мусорка, надо будет распределить

import numpy as np
from typing import Tuple
import json, pathlib
import re, cv2, random
from graspnetAPI import GraspGroup

# Импорт пользовательских модулей из simlib
from ggmujoco.simlib import mujoco_io as mj     # утилиты MuJoCo I/O и матрицы трансформации
from ggmujoco.simlib import sbg_client as sbg   # клиент SBG для отправки/приёма облаков
from ggmujoco.simlib import transforms as tr    # геометрические преобразования
from ggmujoco.simlib import ik_driver as ikd    # драйвер обратной кинематики
from ggmujoco.simlib import gripper             # управление ЗУ
from ggmujoco.simlib import config as cfg       # конфигурация и параметры

from ggmujoco.scene import build_mjcf

def get_model_xml(textures_path, frac_paths, cam_fovy, img_w, img_h, resource_path, assets_path, manip_path, RGB_CAM, DEPTH_CAM, PROB_DROP):
    """
        Это все в конфиг надо засунуть
    """
    # ── floor texture ---------------------------------------------
    texture_cfg = {
        "file":            pick_color_map(pick_folder(textures_path)),
        "texrepeat_range": (5.0, 10.0),
        "metal_prob":      0.0,
        "rgba_jitter":     0.08,
    }

    # ── obj & materials ----------------------------------------
    # не будем сильно отсвечивать: specular_rng и shininess_rng в минималку 
    obj_mat_cfg = {"metal_prob": 0.2, 
                   "rgba_jitter": 0.25, 
                   "specular_rng": (0.01, 0.1), 
                   "shininess_rng": (0.0, 0.09),
                   }

    # ── lights & materials ----------------------------------------
    light_cfg = {
        "num":             3,#np.random.randint(1, 2),
        "xy_radius":       0.5,
        "z_range":         (1.0, 1.2),
        "kelvin_range":    (3000., 8000.),
        "intensity_range": (0.6, 0.7),
        "ambient":         (0.1, 0.1, 0.1),
        "directional":     False,
    }
    # ── cams & ... ----------------------------------------
    cam_quat = euler_to_quat(0, -70, 180)
    cam_cfg = {
        "body_pose": (0.0, 0.4, 0.7),
        "body_quat": cam_quat,
        "geom_box_size": (0.03, 0.05, 0.02),
        
        "name_rbg_cam": RGB_CAM,
        "rgb_cam_quat": (0.5, 0.5, 0.5, 0.5),
        "rgb_cam_pose": (0.0, 0.0, 0.0),
        "camera_fovy": cam_fovy,

        "name_depth_cam": DEPTH_CAM,
        "rgb_cam_quat": (0.5, 0.5, 0.5, 0.5),
        "rgb_cam_pose": (0.0, 0.0, 0.0)
    }
    # ── Render ----------------------------------------
    visual_cfg = {
        "offwidth": img_w, "offheight": img_h,
        "fovy": cam_fovy,
        "orthographic": False,
        "shadowsize":   8192,
        "offsamples":   1024,
        "znear": 0.02, "zfar": 20.0,
        "fogstart": 2.0, "fogend": 8.0,
        "haze": 0.2,
        "shadowclip": 2.0, "shadowscale": 0.9,
        "smoothing":    True
        }
    
    # ── build MJCF -------------------------------------------------
    # camera_quat = (0.5, 0.5, 0.5, 0.5), 
    model_xml = build_mjcf(
        texture_cfg = texture_cfg,
        resource      = resource_path,
        frac_paths  = frac_paths,
        assets = assets_path,
        obj_scale   = 0.0008, 
        prob_drop   = PROB_DROP,
        manip_path  = manip_path, 
        center_xy   = (0.3, 0.4),
        visual_cfg = visual_cfg,
        cam_cfg     = cam_cfg,
        light_cfg   = light_cfg,
        obj_mat_cfg = obj_mat_cfg,
    )

    return model_xml

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

def depth_rgb_to_pointcloud(
        depth,
        rgb,
        fx: float, fy: float, cx: float, cy: float,
        z_near: float = 0.0,
        z_far:  float = 1.0,
    ):
    """
    depth  – 2‑D float32/64 [m], shape (H, W)
    rgb    – 3‑D uint8/float, shape (H, W, 3), BGR *или* RGB – не важно
    fx, fy – фокусные пиксели
    cx, cy – координаты оптического центра
    z_near / z_far – диапазон глубины, точки вне диапазона отбрасываются

    Возвращает:
        pts  – (N, 3) XYZ в камерных координатах
        clr  – (N, 3) float64, 0‒1
    """
    assert depth.ndim == 2 and rgb.ndim == 3
    h, w = depth.shape
    assert rgb.shape[:2] == (h, w), "depth vs RGB size mismatch"

    # 1) маска валидных глубин
    z = depth.reshape(-1)
    msk = (z > z_near) & (z < z_far) & np.isfinite(z)
    if not np.any(msk):
        return np.empty((0, 3)), np.empty((0, 3))

    # 2) координатная сетка (u, v) в один проход
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    u = u.reshape(-1)[msk]
    v = v.reshape(-1)[msk]
    z = z[msk]

    # 3) декодируем в XYZ
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    pts = np.column_stack((X, Y, z))          # (N, 3)

    # 4) цвета → float 0‒1
    clr = rgb.reshape(-1, 3)[msk].astype(np.float64) / 255.0
    return pts, clr


# ─────────────────────────────────────────────────────────────
# Временно все что с симуляцией поместим сюда
# ─────────────────────────────────────────────────────────────
def simulation_falling_objects(mujoco, model, data, time_simulation: float, timestep: float, visual_on: bool, viewer = None):
    model.opt.timestep = timestep
    start_time = data.time
    while data.time - start_time < float(time_simulation):
        mujoco.mj_step(model, data)
        if visual_on and viewer is not None:
            viewer.sync()                      # показать текущий state

def _plan_gripper_widths(width_net_m: float, _PREOPEN_EXTRA_M, _CLOSE_MARGIN_M, _FALLBACK_PREOPEN, _FALLBACK_CLOSE, _F_THRESH):
    if not np.isfinite(width_net_m) or width_net_m <= 0.0:
        return _FALLBACK_PREOPEN, _FALLBACK_CLOSE
    w_pre   = min(width_net_m + _PREOPEN_EXTRA_M, cfg.GRIPPER_OPEN_M)
    w_close = max(width_net_m - _CLOSE_MARGIN_M, cfg.GRIPPER_CLOSE_M)
    return (w_pre, w_close) if w_close <= w_pre else (w_pre, w_pre)

def manipulation(viewer, gg, _tcp2tip, ctx, _PREOPEN_EXTRA_M, _CLOSE_MARGIN_M, _FALLBACK_PREOPEN, _FALLBACK_CLOSE, _F_THRESH):

     # открыть ЗУ в начале
    gripper.gripper_open(ctx, viewer=viewer)

    for i in range(len(gg)):

        g_row = gg[i]
        # разбор строки grasp (используем готовую функцию)
        t_cv, R_cv_raw, w, h, d, score, _ = sbg.parse_grasp_row(g_row)
        R_cv = tr.ortho_project(R_cv_raw)

        # преобразуем в систему базы робота
        G_net, G_tcp = tr.camcv2base(
            t_cv, R_cv, mj.T_b_c_gl(ctx), depth=d, tcp2tip=_tcp2tip
        )

        # -------- планирование захвата --------
        w_pre, _ = _plan_gripper_widths(w, _PREOPEN_EXTRA_M, _CLOSE_MARGIN_M, _FALLBACK_PREOPEN, _FALLBACK_CLOSE, _F_THRESH)
        gripper.gripper_set(ctx, w_pre, viewer=viewer)

        ok, _ = ikd.goto_arm(
            ctx,
            G_tcp.t,
            tr.safe_uq_from_R(G_tcp.R).vec,
            viewer=viewer,
        )

        if ok:
            gripper.gripper_close_until(
                ctx, f_thresh=_F_THRESH, step_ctrl=5e-4, viewer=viewer
            )

            # подъём детали на 0.1 м
            lift_target = G_tcp.t + np.array([0.0, 0.0, 0.10])
            ikd.goto_arm(
                ctx,
                lift_target,
                tr.safe_uq_from_R(G_tcp.R).vec,
                viewer=viewer,
            )

            gripper.gripper_open(ctx, viewer=viewer)
        else:
            print('[IK] не удалось дотянуться до grasp‑позы')

        grasp_done = True

        return grasp_done