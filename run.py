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
from ggmujoco.scene import build_mjcf
from ggmujoco.utils import (euler_to_quat, load_intrinsics, pick_folder, pick_color_map, update_grasps)
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

RGB_CAM, DEPTH_CAM = "rgb_cam", "depth_cam"
DEFAULT_SAVE_AFTER = 20.0      # секунд до сохранения point‑cloud
PROB_DROP          = 1.0       # вероятность «кучи» вместо рассеяния
SETTLE_T           = 1.0       # симуляционных секунд до инференса

_PREOPEN_EXTRA_M = 0.01
_CLOSE_MARGIN_M  = 0.002
_FALLBACK_PREOPEN = cfg.GRIPPER_OPEN_M
_FALLBACK_CLOSE   = cfg.GRIPPER_CLOSE_M
_F_THRESH = 150.0

def _plan_gripper_widths(width_net_m: float):
    if not np.isfinite(width_net_m) or width_net_m <= 0.0:
        return _FALLBACK_PREOPEN, _FALLBACK_CLOSE
    w_pre   = min(width_net_m + _PREOPEN_EXTRA_M, cfg.GRIPPER_OPEN_M)
    w_close = max(width_net_m - _CLOSE_MARGIN_M, cfg.GRIPPER_CLOSE_M)
    return (w_pre, w_close) if w_close <= w_pre else (w_pre, w_pre)

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

    # ── floor texture ---------------------------------------------
    texture_cfg = {
        "file":            pick_color_map(pick_folder(textures_path)),
        "texrepeat_range": (5.0, 10.0),
        "metal_prob":      0.0,
        "rgba_jitter":     0.08,
    }

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
    cam_quat = euler_to_quat(0, -60, 180)
    cam_cfg = {
        "body_pose": (0.0, 0.4, 0.6),
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

    # ── MuJoCo model & renderers ─────────────────────────────────────────
    model  = mujoco.MjModel.from_xml_string(model_xml)
    data   = mujoco.MjData(model)

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
    depth_r = mujoco.Renderer(model, width=img_w, height=img_h); depth_r.enable_depth_rendering()

    # ── пассивное окно MuJoCo (только для просмотра) --------------------
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # открыть ЗУ в начале
    gripper.gripper_open(ctx, viewer=viewer)

    # ── Open3D окно -----------------------------------------------------
    vis = o3d.visualization.Visualizer(); vis.create_window("PointCloud", img_w, img_h)
    pc  = o3d.geometry.PointCloud();      vis.add_geometry(pc)
    grasp_geoms = []
    R_FLIP = o3d.geometry.get_rotation_matrix_from_xyz([math.pi, 0, 0])

    # ── OpenCV окна -----------------------------------------------------
    cv2.namedWindow("RGB", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Seg+Grasps", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Masks", cv2.WINDOW_AUTOSIZE)

    # ── предрасчёт сеток для point‑cloud --------------------------------
    intr_json = json.loads(intr_path.read_text())
    fx, fy, cx, cy = intr_json["fx"], intr_json["fy"], intr_json["cx"], intr_json["cy"]
    u, v = np.meshgrid(np.arange(img_w), np.arange(img_h)); u, v = u.ravel(), v.ravel()

    Intr = namedtuple("Intr", "fx fy ppx ppy");  intr = Intr(fx, fy, cx, cy)

    # ── SBG‑детектор (как раньше) ---------------------------------------
    det = SBGGraspDetector(
        checkpoint_path="/home/nikita/diplom/ggmujoco/weights/sbg_full_module/checkpoint.tar",
        onnx_seg="/home/nikita/diplom/ggmujoco/weights/yolo_module/best.onnx",
        seg_yaml="/home/nikita/diplom/ggmujoco/weights/yolo_module/merged_yolo_dataset2.yaml",
        num_view=300, collision_thresh=0.0, voxel_size=0.01,
        bbox_depth_pad=0.30, bbox_xy_pad=0.00,
        max_grasp_num=100, gripper_width_max=1.2,
        conf_threshold=0.6, iou_threshold=0.5
    )

    # ── цикл симуляции ---------------------------------------------------
    start_real, saved    = time.time(), False
    start_sim            = data.time
    gg                   = None                      # захваты появятся позже
    first_fit            = True

    loop_cnt  = 0                 # счётчик кадров
    dbg_every = 30                # печатать каждые N кадров (чтобы не спамить)
    
    grasp_done = False      # выполнен ли уже захват

    try:
        while True:
            
            loop_cnt += 1

            # физический шаг MuJoCo
            mujoco.mj_step(model, data)

            if loop_cnt % dbg_every == 0:
                print(f"[DBG] loop={loop_cnt:05d}  sim_t={data.time:6.3f}  "
                    f"viewer_running={viewer.is_running()}")

            # ---------- рендер RGB / depth ----------
            rgb_r.update_scene(data, camera=RGB_CAM);     rgb_img   = rgb_r.render()[..., ::-1]
            depth_r.update_scene(data, camera=DEPTH_CAM); depth_img = depth_r.render()

            # ---------- однократный SBG ----------
            if (not grasp_done) and (data.time - start_sim >= SETTLE_T):
                gg, seg_vis, mask_vis = det.infer(rgb_img, depth_img, intr, depth_scale=1.0)
                print(f"[{data.time:6.2f}s] grasp‑кандидатов: {gg.shape[0]}")
                update_grasps(vis, grasp_geoms, gg, R_FLIP, max_show=50)

            # ---------- управление манипулятором после инференса ----------
            if gg is not None and (not grasp_done) and gg.shape[0] > 0:
                g_row = gg[0]
                # разбор строки grasp (используем готовую функцию)
                t_cv, R_cv_raw, w, h, d, score, _ = sbg.parse_grasp_row(g_row)
                R_cv = tr.ortho_project(R_cv_raw)

                # преобразуем в систему базы робота
                G_net, G_tcp = tr.camcv2base(
                    t_cv, R_cv, mj.T_b_c_gl(ctx), depth=d, tcp2tip=_tcp2tip
                )

                # -------- планирование захвата --------
                w_pre, _ = _plan_gripper_widths(w)
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

            # ---------- OpenCV окна ----------
            cv2.imshow("RGB", rgb_img)
            if gg is not None:
                cv2.imshow("Seg+Grasps", seg_vis); cv2.imshow("Masks", mask_vis)

            # ---------- DEBUG ② ----------
            if loop_cnt % dbg_every == 0:
                print(f"[DBG] pc_pts={len(pc.points):6d}  "
                    f"grasp_geoms={len(grasp_geoms):3d}")
                
            # ---------- Point Cloud (статично) ----------
            z   = depth_img.astype(np.float32).ravel()
            msk = (z > 0) & (z < 1.0) & np.isfinite(z)

            X = (u[msk] - cx) * z[msk] / fx          # X вправо
            Y = (v[msk] - cy) * z[msk] / fy          # Y вниз
            pts_cam = np.column_stack((X, Y, z[msk]))  # (N,3)

            pts = pts_cam @ R_FLIP.T                   # (x,‑y,‑z) → Open3D
            clr = rgb_img.reshape(-1, 3)[msk] / 255.0

            # обновляем буферы без поворота геометрии
            pc.points = o3d.utility.Vector3dVector(pts)
            pc.colors = o3d.utility.Vector3dVector(clr)
            vis.update_geometry(pc)

            # ---------- рендер Open3D / MuJoCo viewer ----------
            if first_fit:
                vis.reset_view_point(True); first_fit = False
            vis.poll_events(); vis.update_renderer(); viewer.sync()

            # ---------- сохранение по таймеру ----------
            """if not saved and (time.time() - start_real >= save_after):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                o3d.io.write_point_cloud(str(out_path / f"cloud_{ts}.ply"), pc,
                                        write_ascii=False, compressed=True)
                saved = True; print("Point‑cloud сохранён")
            """
            # ---------- выход ----------
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or not viewer.is_running():
                print("[DBG] exit condition hit: "
                    f"key={key}  viewer_running={viewer.is_running()}")
                break
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — пора сворачиваться")


    finally:
        # ===== финальное освобождение =====
        try:  viewer.close()
        except Exception: pass

        try:  vis.destroy_window()
        except Exception: pass

        cv2.destroyAllWindows()

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
