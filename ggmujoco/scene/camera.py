import textwrap

def gen_body_cam(cfg: dict) -> str:
    """
    Формирует XML‑блок <body>…</body> для описания корпуса камеры и двух камер (RGB и глубины)
    на основе словаря *cfg*.

    Ключи *cfg* (все необязательные; в скобках приведены значения по умолчанию):

        body_pose         – Tuple[float, float, float] (0.1, 0.2, 0.3)  
                             Позиция тела камеры в мировых координатах.

        body_quat         – Tuple[float, float, float, float] (0, 0.38268343, 0, 0.92387953)  
                             Ориентация тела камеры в виде кватерниона (w, x, y, z).

        geom_box_size     – Tuple[float, float, float] (0.03, 0.05, 0.02)  
                             Размеры геометрии box для корпуса камеры.

        name_rbg_cam      – str ("rgb_cam")  
                             Имя RGB‑камеры.

        rgb_cam_pose      – Tuple[float, float, float] (0.0, 0.0, 0.0)  
                             Позиция RGB‑камеры внутри тела.

        rgb_cam_quat      – Tuple[float, float, float, float] (0.5, 0.5, 0.5, 0.5)  
                             Ориентация RGB‑камеры внутри тела (кватернион).

        name_depth_cam    – str ("depth_cam")  
                             Имя Depth‑камеры.

        depth_cam_pose    – Tuple[float, float, float] (0.0, 0.0, 0.0)  
                             Позиция Depth‑камеры внутри тела.

        depth_cam_quat    – Tuple[float, float, float, float] (0.5, 0.5, 0.5, 0.5)  
                             Ориентация Depth‑камеры внутри тела (кватернион).

        camera_fovy       – float (58)  
                             Угол обзора (field of view) камер в градусах.

    Возвращает:
        str – XML‑блок <body>…</body>, готовый для вставки в секцию <worldbody> MJCF.
    """
    
    body_pose = cfg.get("body_pose", (0.1, 0.2, 0.3))
    body_quat = cfg.get("body_quat", (0, 0.38268343, 0, 0.92387953))
    geom_box_size = cfg.get("geom_box_size", (0.03, 0.05, 0.02))
    
    name_rbg_cam = cfg.get("name_rbg_cam", "rgb_cam")
    rgb_cam_quat = cfg.get("rgb_cam_quat", (0.5, 0.5, 0.5, 0.5))
    rgb_cam_pose = cfg.get("rgb_cam_pose", (0.0, 0.0, 0.0))
    cam_fovy = cfg.get("camera_fovy", 58)

    name_depth_cam = cfg.get("name_depth_cam", "depth_cam")
    depth_cam_quat = cfg.get("rgb_cam_quat", (0.5, 0.5, 0.5, 0.5))
    depth_cam_pose = cfg.get("rgb_cam_pose", (0.0, 0.0, 0.0))

    # переводим в нужный вид
    body_pose_str = " ".join(f"{coord:.3f}" for coord in body_pose)
    body_quat_str = " ".join(f"{i:.3f}" for i in body_quat)
    geom_box_size_str = " ".join(f"{i:.3f}" for i in geom_box_size)
    rgb_cam_quat_str = " ".join(f"{i:.3f}" for i in rgb_cam_quat)
    rgb_cam_pose_str = " ".join(f"{i:.3f}" for i in rgb_cam_pose)

    depth_cam_quat_str = " ".join(f"{i:.3f}" for i in depth_cam_quat)
    depth_cam_pose_str = " ".join(f"{i:.3f}" for i in depth_cam_pose)

    body_blocks = textwrap.dedent(f"""
<body name="d435" pos="{body_pose_str}" quat="{body_quat_str}">
  <geom type="box" size="{geom_box_size_str}"
        rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0" group="2"/>
  <!-- RGB-камера -->
  <camera name="{name_rbg_cam}" pos="{rgb_cam_pose_str}" 
    quat="{rgb_cam_quat_str}" fovy="{cam_fovy}"/>
  <!-- RGB-камера -->
  <camera name="{name_depth_cam}" pos="{depth_cam_pose_str}" 
    quat="{depth_cam_quat_str}" fovy="{cam_fovy}"/>    
</body>
""")
    return body_blocks