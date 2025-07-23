# scene_builder.py
import textwrap
import pathlib
from typing import Sequence, Tuple, Optional, Dict, Any, List

#scripts_dir = pathlib.Path(__file__).parent.resolve()
#if str(scripts_dir) not in sys.path:
#    sys.path.insert(0, str(scripts_dir))
from ggmujoco.scene import  (gen_body_cam,
                              sample_floor_material,
                              gen_light_body,
                              gen_pose_body_mat,
                              gen_manip_block)

# ─────────────────────────────────────────
# MJCF‑генератор
# ─────────────────────────────────────────
def build_mjcf(texture_cfg: Dict[str, Any],
               resource: pathlib.Path,
               frac_paths: Sequence[pathlib.Path],
               *,
               obj_scale: float = 0.0005,
               prob_drop: float = 0.3,
               center_xy: Tuple[float, float] = (0.0, 0.0),
               manip_path: Optional[pathlib.Path] = None,
               assets: Optional[pathlib.Path] = None,
               visual_cfg: Optional[Dict[str, Any]] = None, 
               cam_cfg: Optional[Dict[str, Any]] = None,
               light_cfg: Optional[Dict[str, Any]] = None,
               obj_mat_cfg: Optional[Dict[str, Any]] = None
               ) -> str:
    """
    texture_cfg  – словарь параметров пола (см. sample_floor_material)
    obj_mat_cfg  – словарь параметров материалов объектов (см. sample_obj_material)
    light_cfg    – словарь параметров света (см. generate_lights)
    assets         – корень assets
    frac_paths     – OBJ‑шарды
    cam_name       – имя камеры
    prob_drop      – вероятность «кучи»
    center_xy      – смещение распределения объектов
    camera_pos     – позиция камеры  (x, y, z)
    camera_quat    – кватернион камеры (w, x, y, z)   MuJoCo‑порядок
    """
    assert manip_path is None or assets is not None, "нужен assets при manip_path"
    resource = resource.resolve()
    # ---------- блок floor texture + material -------------------------
    floor_asset = sample_floor_material(texture_cfg)

    # ---------- OBJ-камеры (mesh) --------------------------------------
    cam_meshes = "\n".join(
        f'  <mesh name="d435i_{i}" file="{resource/"cam_d435"/f"d435i_{i}.obj"}"/>'
        for i in range(9)
    )
    cam_body_blocks = gen_body_cam(cam_cfg)
    
    # ---------- Render --------------------------------------
    visual_block = visual_parameters(visual_cfg)

    # ---------- OBJ‑фрагменты (mesh) ----------------------------------
    frac_meshes = "\n".join(
          f'  <mesh name="{p.stem}" file="{p.resolve()}" '
          f'scale="{str(obj_scale)} {str(obj_scale)} {str(obj_scale)}"/>'
          for p in frac_paths
      )
    
    # ---------- тело, материалы объектов ------------------------------
    obj_body_blocks, obj_mat_blocks = gen_pose_body_mat(center_xy, 
                                              prob_drop, 
                                              frac_paths, 
                                              obj_mat_cfg)

    # ---------- освещение ---------------------------------------------
    lights_block = gen_light_body(center_xy, light_cfg)

    # ---------- сборка <asset> ----------------------------------------
    assets_block = (
        floor_asset +
        "\n".join(obj_mat_blocks) + "\n" +           # материалы объектов
        cam_meshes + "\n" +
        frac_meshes
    )

    # ---------- манипулятор ---------------------------------------------
    manip_block = "" if manip_path is None else gen_manip_block(manip_path, assets)

    # ---------- финальный MJCF‑текст ----------------------------------
    return textwrap.dedent(f"""
    <mujoco model="dynamic_scene">
      <option gravity="0 0 -9.81"/>
{visual_block}
      <default>
        <geom density="750"/>
      </default>

      <asset>
{assets_block}
      </asset>
      
{manip_block}
      
      <worldbody>
        <geom name="floor" type="plane" size="0 0 .1"
              material="floormat" contype="1" conaffinity="1"/>
{lights_block}
{"".join(obj_body_blocks)}
{"".join(cam_body_blocks)}
      </worldbody>
    </mujoco>
    """)


def visual_parameters(visual_cfg: dict) -> str:
    """
    Формирует XML‑блок <visual>…</visual> на основе словаря *visual_cfg*.

    Ключи *visual_cfg* (все необязательные; в скобках приведены значения по умолчанию):

        offwidth       – int (640)  
                         Ширина offscreen‑буфера (в пикселях).

        offheight      – int (480)  
                         Высота offscreen‑буфера (в пикселях).

        fovy           – float (58)  
                         Угол обзора камеры (field of view) в градусах.

        orthographic   – bool (False)  
                         Режим проекции: True — ортографическая, False — перспективная.

        shadowsize     – int (2048)  
                         Разрешение теневой карты (число пикселей по одной стороне).

        offsamples     – int (4)  
                         Число семплов на пиксель для мультисемплинга (MSAA).

        fogstart       – float (0.0)  
                         Расстояние до начала тумана (в единицах сцены).

        fogend         – float (0.0)  
                         Расстояние до конца тумана.

        znear          – float (0.01)  
                         Ближняя плоскость отсечения для камеры.

        zfar           – float (50.0)  
                         Дальняя плоскость отсечения для камеры.

        haze           – float (0.0)  
                         Интенсивность тумана (0 – без тумана).

        shadowclip     – float (1.0)  
                         Коэффициент зоны отсечения теней (множитель model.extent).

        shadowscale    – float (0.6)  
                         Угол конуса света для отбрасывания теней (множитель).

    Возвращает:
        str — готовый XML‑блок <visual>…, который можно вставить в начало MJCF.
    """
    vc = visual_cfg or {}
    offw       = vc.get("offwidth", 640)
    offh       = vc.get("offheight", 480)
    cam_fovy   = vc.get("fovy", 58)       # углы обзора
    ortho      = vc.get("orthographic", False)
    shadowsz   = vc.get("shadowsize", 2048)
    offsamples = vc.get("offsamples", 4)
    fogstart   = vc.get("fogstart", 0.0)
    fogend     = vc.get("fogend", 0.0)
    znear      = vc.get("znear", 0.01)
    zfar       = vc.get("zfar", 50.0)
    haze       = vc.get("haze", 0.0)
    shadowclip = vc.get("shadowclip", 1.0)
    shadowscale= vc.get("shadowscale", 0.6)

    visual_block = textwrap.dedent(f"""
  <visual>
    <global offwidth="{offw}" offheight="{offh}"
            fovy="{cam_fovy}"
            orthographic="{'true' if ortho else 'false'}"/>
    <quality shadowsize="{shadowsz}"
             offsamples="{offsamples}"/>
    <map fogstart="{fogstart}" fogend="{fogend}"
         znear="{znear}" zfar="{zfar}"
         haze="{haze}"
         shadowclip="{shadowclip}" shadowscale="{shadowscale}"/>
  </visual>
""")

    
    return visual_block