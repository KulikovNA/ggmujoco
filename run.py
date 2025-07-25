#!/usr/bin/env python3
"""
MuJoCo 3.2.3 динамическая сцена + SBG‑инференс.
После заданного времени (SETTLE_T) объекты считаются «улёгшимися»,
и мы единожды запускаем SBGGraspDetector, получая:
  • gg_array  – точки захвата,
  • seg_vis   – RGB‑кадр с масками + центрами захватов,
  • mask_vis  – изображение только масок.
Отдельные окна OpenCV: «RGB», «Seg+Grasps», «Masks».
"""
import json, time, pathlib, math
from datetime import datetime
from typing import Optional, Tuple
from mujoco import viewer  

import cv2
import numpy as np
import open3d as o3d
import mujoco
from collections import namedtuple
from spatialmath import SE3

from ggmujoco.fracture import BlenderFractureManager
from ggmujoco.utils import (euler_to_quat, load_intrinsics, pick_folder, pick_color_map, depth_rgb_to_pointcloud, simulation_falling_objects, manipulation, get_model_xml)
from sbg_inference import SBGGraspDetector

# Импорт пользовательских модулей из simlib
from ggmujoco.simlib import config as cfg       # конфигурация и параметры
from ggmujoco.simlib import mujoco_io as mj     # утилиты MuJoCo I/O и матрицы трансформации
from ggmujoco.simlib import mujoco_render as mjr# рендеринг облака
from ggmujoco.simlib import sbg_client as sbg   # клиент SBG для отправки/приёма облаков
from ggmujoco.simlib import transforms as tr    # геометрические преобразования
from ggmujoco.simlib import ik_driver as ikd    # драйвер обратной кинематики
from ggmujoco.simlib import tcp_eval, logutil   # оценка ошибки TCP и логирование
from ggmujoco.simlib import gripper             # управление ЗУ

from ggmujoco.visualizers import OpenCVViewer, Open3DViewer

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
ROOT              = pathlib.Path(__file__).resolve().parent
RESOURCE_DIR      = (ROOT / "resource").resolve()
ASSETS_DIR        = (RESOURCE_DIR / "assets")
TEXTURES_DIR      = RESOURCE_DIR / "textures"
MODELS_DIR        = RESOURCE_DIR / "differBig/models"
DEFAULT_INTR_FILE = RESOURCE_DIR / "cam_d435" / "camera_435_640x480.json"
OUT_DIR           = (ROOT / "output").resolve()
MANIP_PATH         = RESOURCE_DIR / "panda_fixed.xml"
CHECKPOINT_SBG     = (ROOT / "weights/sbg_full_module/checkpoint.tar").resolve() 
YOLO_YAML          = (ROOT / "weights/yolo_module/merged_yolo_dataset2.yaml").resolve()
CHECKPOINT_YOLO    = (ROOT / "weights/yolo_module/best.onnx").resolve()

RGB_CAM, DEPTH_CAM = "rgb_cam", "depth_cam"
DEFAULT_SAVE_AFTER = 20.0      # секунд до сохранения point‑cloud
PROB_DROP          = 1.0       # вероятность «кучи» вместо рассеяния
SETTLE_T           = 1.0       # симуляционных секунд до инференса

_PREOPEN_EXTRA_M = 0.01
_CLOSE_MARGIN_M  = 0.002
_FALLBACK_PREOPEN = cfg.GRIPPER_OPEN_M
_FALLBACK_CLOSE   = cfg.GRIPPER_CLOSE_M
_F_THRESH = 150.0

R_FLIP = o3d.geometry.get_rotation_matrix_from_xyz([math.pi, 0, 0])


def _mk_tcp2tip():
    return SE3.Tz(cfg.TCP2TIP_Z) if cfg.USE_TCP_TIP else None

# ──────────────────────────────────────────────
# MAIN SCENE RUNNER
# ──────────────────────────────────────────────


def run_scene(*,
              intr_path: Optional[pathlib.Path] = None,
              resource_path: Optional[pathlib.Path] = None,
              assets_path:  Optional[pathlib.Path] = None,
              manip_path: Optional[pathlib.Path] = None,
              textures_path: Optional[pathlib.Path] = None,
              out_path: Optional[pathlib.Path] = None,
              models_path: Optional[pathlib.Path] = None,
              frac_out_dir: Optional[pathlib.Path] = None) -> None:

    # ── paths ------------------------------------------------------
    resource_path   = (resource_path   or RESOURCE_DIR).expanduser().resolve()
    assets_path     = (assets_path or ASSETS_DIR).expanduser().resolve()
    manip_path      = (manip_path or MANIP_PATH).expanduser().resolve()
    textures_path = (textures_path or TEXTURES_DIR).expanduser().resolve()
    models_path   = (models_path   or MODELS_DIR).expanduser().resolve()
    out_path      = (out_path      or OUT_DIR).expanduser().resolve()
    intr_path     = (intr_path     or DEFAULT_INTR_FILE).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if frac_out_dir:
        frac_out_dir = (OUT_DIR / pathlib.Path(frac_out_dir)).expanduser().resolve()
        frac_out_dir.mkdir(parents=True, exist_ok=True)

    # ── intrinsics -------------------------------------------------
    cam_fovy, img_w, img_h = load_intrinsics(intr_path)

    # ── fracture PLY → OBJ ----------------------------------------
    ply_files = sorted(models_path.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"no *.ply in {models_path}")

    mgr = BlenderFractureManager(permanent_dir=frac_out_dir)
    frac_paths = mgr.fracture(
        ply_files,
        chunks_range=(2, 6),
        noise=np.random.uniform(0, 0.007),
        seed=np.random.randint(1, 100),
        max_attempts=4,
        export_format="obj"
        #voxel = 1.5
        #scale=0.5
    )
    model_xml = get_model_xml(textures_path, frac_paths, cam_fovy, img_w, img_h, resource_path, assets_path, manip_path, RGB_CAM, DEPTH_CAM, PROB_DROP)

    # ── MuJoCo model & renderers ─────────────────────────────────────────
    model  = mujoco.MjModel.from_xml_string(model_xml)
    data   = mujoco.MjData(model)
    #model.opt.timestep   = 0.0003
    # ------------------------ создаём MjContext ----------------------------------
    # (cam_id, base_id и др. читаем из XML по именам в cfg)
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.CAM_NAME)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  cfg.BASE_BODY)
    try:
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.TCP_SITE)
    except Exception:
        tcp_site_id = -1
    obj_ids = [i for i in range(model.nbody)
            if 'fractured' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '')]

    ctx = mj.MjContext(model, data, cam_id, base_id, tcp_site_id, obj_ids)
    # калибровка TCP для RTB‑IK
    ikd.calibrate_rtb_tool_from_mj(ctx)
    _tcp2tip = _mk_tcp2tip()                   # смещение TCP→кончик инструмента

    rgb_r  = mujoco.Renderer(model, width=img_w, height=img_h, max_geom=50_000)
    depth_r = mujoco.Renderer(model, width=img_w, height=img_h); 
    depth_r.enable_depth_rendering()

    # ── пассивное окно MuJoCo (только для просмотра) --------------------
    viewer = mujoco.viewer.launch_passive(model, data)

    # ── Open3D окно -----------------------------------------------------
    o3d_viz = Open3DViewer(img_w, img_h)
    o3d_viz.show_async()
    # ── OpenCV окна -----------------------------------------------------
    cv_viz = OpenCVViewer()


    # ── предрасчёт сеток для point‑cloud --------------------------------
    intr_json = json.loads(intr_path.read_text())
    fx, fy, cx, cy = intr_json["fx"], intr_json["fy"], intr_json["cx"], intr_json["cy"]
    u, v = np.meshgrid(np.arange(img_w), np.arange(img_h)); u, v = u.ravel(), v.ravel()

    Intr = namedtuple("Intr", "fx fy ppx ppy");  intr = Intr(fx, fy, cx, cy)

    # ── SBG‑детектор (как раньше) ---------------------------------------
    det = SBGGraspDetector(
        checkpoint_path=CHECKPOINT_SBG,
        onnx_seg=CHECKPOINT_YOLO,
        seg_yaml=YOLO_YAML,
        num_view=300, collision_thresh=0.0, voxel_size=0.01,
        bbox_depth_pad=0.30, bbox_xy_pad=0.00,
        max_grasp_num=100, gripper_width_max=1.2,
        conf_threshold=0.6, iou_threshold=0.5
    )

    # ── цикл симуляции ---------------------------------------------------
    gg         = None       # захваты появятся позже
    loop_cnt   = 0          # счётчик кадров
    grasp_done = False      # выполнен ли уже захват
    num = 0 
    try:
        while True:
            
            loop_cnt += 1

            # ── 1. физика для падения объектов
            simulation_falling_objects(mujoco,
                                model, 
                                data, 
                                time_simulation = 1.0, 
                                timestep = 0.0006, 
                                visual_on = True,
                                viewer = viewer)
    
            print(f"[DBG] loop={loop_cnt:05d}  sim_t={data.time:6.3f}  "
                f"viewer_running={viewer.is_running()}")

            # ---------- рендер RGB / depth ----------
            rgb_r.update_scene(data, camera=RGB_CAM);     
            rgb_img   = rgb_r.render()[..., ::-1]
            depth_r.update_scene(data, camera=DEPTH_CAM); 
            depth_img = depth_r.render()

            # визуализация рендера изображения
            cv_viz.update("RGB", rgb_img)

            # перевод в облако точек 
            pts, clr = depth_rgb_to_pointcloud(depth_img, rgb_img, fx, fy, cx, cy)
            # визуализация облака точек
            o3d_viz.update_cloud(pts, clr)
            
            # ---------- однократный SBG ----------
            gg_array, gg, seg_vis, mask_vis = det.infer(rgb_img, depth_img, intr, depth_scale=1.0)
            print(f"[{data.time:6.2f}s] grasp‑кандидатов: {gg.shape[0]}")

            # визуализация захватов и масок
            o3d_viz.update_scene(pts, clr, gg)
            cv_viz.update("Seg+Grasps", seg_vis)
            cv_viz.update("Masks", mask_vis)

            # ---------- перемещение объекта/объектов манипулятором ----------
            if gg is not None and (not grasp_done) and gg.shape[0] > 0:
                manipulation(viewer, gg, _tcp2tip, ctx, _PREOPEN_EXTRA_M, _CLOSE_MARGIN_M, _FALLBACK_PREOPEN, _FALLBACK_CLOSE, _F_THRESH)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — пора сворачиваться")


    finally:
        # ===== финальное освобождение =====
        try:  viewer.close()
        except Exception: pass

        cv_viz.close()          # <-- гарантируем корректное завершение
        o3d_viz.close() 

        try:  rgb_r.close(); depth_r.close()
        except Exception: pass

        print("[INFO] Завершено корректно — можно выходить")

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def main() -> None:
    #run_scene(frac_out_dir='/home/nikita/diplom/out')
    run_scene()

if __name__ == "__main__":
    main()
