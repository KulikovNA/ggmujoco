import textwrap
import pathlib
import numpy as np
from typing import Tuple, Dict, Any

def gen_pose_body_mat(center_xy, prob_drop, frac_paths, obj_mat_cfg):
    # ---------- позиционирование шард‑тел -----------------------------
    pose_func = (lambda: rand_pose_drop(center_xy)
                 if np.random.rand() < prob_drop
                 else rand_pose_scatter(center_xy))
    
    # ---------- тело, материалы объектов ------------------------------
    obj_body_blocks, obj_mat_blocks = gen_body_mat_obj(frac_paths, 
                                               obj_mat_cfg, 
                                               pose_func)
    
    return obj_body_blocks, obj_mat_blocks

def gen_body_mat_obj(frac_paths, obj_mat_cfg, pose_func):
    """
    Формирует XML‑блоки для материалов объектов и соответствующих тел на основе списка фрагментов.

    Аргументы:
        frac_paths – Sequence[pathlib.Path]
            Список путей к OBJ‑файлам фрагментов (shard’ов).

        obj_mat_cfg – dict | None
            Конфигурация параметров материалов объектов.
            См. функцию sample_obj_material для списка поддерживаемых ключей.
            Если None или пустой словарь — используются значения по умолчанию.

        pose_func – Callable[[], Tuple[float, float, float]]
            Функция без аргументов, возвращающая случайную позицию (x, y, z)
            для каждого тела (например, rand_pose_drop или rand_pose_scatter).

    Возвращает:
        Tuple[List[str], List[str]]:
            mat_blocks  – список XML‑строк `<material>…</material>` для каждого фрагмента.
            body_blocks – список XML‑строк `<body>…</body>` с геометриями и телами.
    """
    mat_blocks   = []
    body_blocks  = []
    for idx, p in enumerate(frac_paths):
        mat_xml, mat_name = sample_obj_material(p.stem,
                                                obj_mat_cfg or {},
                                                idx)
        mat_blocks.append(mat_xml)

        x, y, z = pose_func()
        body_blocks.append(f"""
      <body name="{p.stem}_body" pos="{x:.3f} {y:.3f} {z:.3f}">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
        <geom type="mesh" mesh="{p.stem}" material="{mat_name}"
              group="1" contype="0" conaffinity="0"/>
        <geom type="mesh" mesh="{p.stem}"
              group="0" contype="1" conaffinity="1"/>
      </body>""")
        
    return body_blocks, mat_blocks
"""
<geom type="mesh" mesh="{p.stem}" material="{mat_name}"
              group="1" contype="0" conaffinity="0"/>
"""
# ─────────────────────────────────────────
# вспомогательные функции поз объектов
# ─────────────────────────────────────────
# --------------- позиция "кучей" -----------------
def rand_pose_drop(center_xy=(0.0, 0.0),
                   radius=0.1, z_min=0.2, z_max=0.3):
    theta = np.random.uniform(0, 2*np.pi)
    r     = np.random.uniform(0, radius)
    x, y  = r * np.cos(theta), r * np.sin(theta)
    x += center_xy[0];  y += center_xy[1]
    z = np.random.uniform(z_min, z_max)
    return x, y, z

# --------------- позиция "рассеивание" -----------
def rand_pose_scatter(center_xy=(0.0, 0.0),
                      box_min=(-0.3, -0.3, 0.1),
                      box_max=( 0.3,  0.3, 0.6)):
    x, y, z = np.random.uniform(box_min, box_max)
    x += center_xy[0];  y += center_xy[1]
    return x, y, z



# ─────────────────────────────────────────
# вспомогательные функции текстурирования объектов 
# ─────────────────────────────────────────
def _rand_rgb(hue_range=(0, 360),
              sat_range=(0.4, 1.0),
              val_range=(0.5, 1.0),
              jitter: float = 0.10) -> Tuple[float, float, float]:
    h  = np.random.uniform(*hue_range) / 60.0
    s  = np.random.uniform(*sat_range)
    v  = np.random.uniform(*val_range)
    c  = v * s
    x  = c * (1 - abs(h % 2 - 1))
    m  = v - c
    rgb = {
        0: (c, x, 0), 1: (x, c, 0), 2: (0, c, x),
        3: (0, x, c), 4: (x, 0, c), 5: (c, 0, x)
    }[int(h) % 6]
    rgb = np.array(rgb) + m
    rgb *= np.random.uniform(1-jitter, 1+jitter, 3)
    return tuple(np.clip(rgb, 0, 1))

def sample_obj_material(mesh_name: str,
                        cfg: Dict[str, Any],
                        idx: int) -> Tuple[str, str]:
    """
    Возвращает (xml_block, material_name) для OBJ-шарда.

    Параметры:
        mesh_name (str):
            Имя меша (OBJ-шарда), используется при формировании имени материала.
        cfg (dict, все поля опциональны):
            hue_range (tuple[float, float]):
                Диапазон оттенка (Hue) в HSV, в градусах от 0 до 360.
                Значения в градусах от 0 до 360°:
                0° и 360° → красный
                60° → жёлтый
                120° → зелёный
                180° → циан
                240° → синий
                300° → пурпурный
            sat_range (tuple[float, float]):
                Диапазон насыщенности (Saturation) в HSV, от 0 (серый) до 1 (насыщенный цвет).
                0.0 → полностью ненасыщенный (оттенки серого)
                1.0 → максимально насыщенный (чистый цвет)            
            val_range (tuple[float, float]):
                Диапазон значения (Value/яркость) в HSV, от 0 (черный) до 1 (максимальная яркость).
                0.0 → чёрный
                1.0 → максимально светлый
            rgba_jitter (float):
                ±дополнительный множитель для изменения яркости (коррекция V-канала) при конвертации HSV → RGB.
                Значение 0.10 означает, что итоговый V может сдвигаться на ±10%. По умолчанию 0.10.
            metal_prob (float):
                Вероятность того, что материал будет металлическим. Значение 0.1 (10%) по умолчанию.
            specular_rng (tuple[float, float]):
                Диапазон для specular (коэф. зеркального отражения) у неметаллического (матового) материала.
                По умолчанию (0.05, 0.25).
            shininess_rng (tuple[float, float]):
                Диапазон для shininess (экспонента блеска) у неметаллического материала.
                Чем выше значение, тем более «глянцевый» результат. По умолчанию (0.0, 0.30).
            reflect_metal_rng (tuple[float, float]):
                Диапазон для specular у металлического материала (обычно выше, чем для матового).
                По умолчанию (0.3, 0.6).    
        idx (int):
            Порядковый индекс, чтобы гарантировать уникальность имени материала для каждого шарда.

    Возвращает:
        xml_block (str):
            Строка с XML‑описанием <material> для MuJoCo, содержащая rgba, specular, shininess.
        material_name (str):
            Сгенерированное имя материала (например, "{mesh_name}_mat_{idx}").
    """
    # Насколько сильно мы будем «дрожать» V‑канал (значение/яркость) при преобразовании HSV → RGB
    jitter      = cfg.get("rgba_jitter",        0.10)
    # Вероятность выбрать металлический материал вместо матового
    metal_prob  = cfg.get("metal_prob",         0.1)
    # Диапазон specular для матового материала (низкое зеркальное отражение)
    spec_rng    = cfg.get("specular_rng",       (0.05, 0.15))
    # Диапазон shininess (блеска) для матового материала
    shin_rng    = cfg.get("shininess_rng",      (0.0, 0.20))
    # Диапазон specular для металлического материала (высокое зеркальное отражение)
    refl_rng    = cfg.get("reflect_metal_rng",  (0.1, 0.3))

    # Генерируем базовый цвет в HSV и конвертируем в RGB, с учётом jitter
    # по дефолту сделаем в градации серого (моделька лучше будет работать)
    rgb = _rand_rgb(
        cfg.get("hue_range",  (0,   360)), 
        cfg.get("sat_range",  (0.0, 0.3)), 
        cfg.get("val_range",  (0.1, 0.5)), 
        jitter
    )
    rgba_s = f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1"

    if np.random.rand() < metal_prob:
        spec = np.random.uniform(0.6, 1.0)
        shin = np.random.uniform(0.5, 1.0)
        refl = np.random.uniform(*refl_rng)
    else:
        spec = np.random.uniform(*spec_rng)
        shin = np.random.uniform(*shin_rng)
        refl = 0.0

    mat_name = f"{mesh_name}_mat{idx}"
    mat_block = (f'<material name="{mat_name}" '
                 f'rgba="{rgba_s}" '
                 f'specular="{spec:.3f}" shininess="{shin:.3f}" '
                 f'reflectance="{refl:.3f}"/>')

    return textwrap.indent(mat_block, "      "), mat_name