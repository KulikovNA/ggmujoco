import textwrap
import pathlib
import numpy as np

# ─────────────────────────────────────────
# вспомогательные функции текстурирования пола
# ─────────────────────────────────────────
def sample_floor_material(cfg: dict) -> str:
    """
    Формирует XML‑блок <texture> + <material> на основе словаря *cfg*.

    Ключи *cfg* (все необязательные; в скобках приведены значения
    по умолчанию):

        file               – pathlib.Path | str | None  
                             Путь к изображению текстуры пола.  
                             Если *None*, файл выбирается случайно.

        texrepeat_range    – (4.0, 8.0)  
                             Диапазон, из которого сэмплируется коэффициент
                             «плитки» — сколько раз изображение повторится
                             по X и Y.

        metal_prob         – 0.25  
                             Вероятность, что материал будет «металлическим»
                             (яркий, узкий блик + отражение).  
                             При 0 — всегда матовый, при 1 — всегда металл.

        rgba_jitter        – 0.10  
                             Случайный множитель ±10 % к каждому из каналов
                             базового цвета (лёгкий тёплый/холодный сдвиг).

        specular_range     – (0.05, 0.25)  
                             Диапазон *specular* для **матового** варианта
                             (чем выше — тем ярче блик).

        shininess_range    – (0.0, 0.3)  
                             Диапазон *shininess* для матового варианта
                             (0 — широкий блик, 1 — узкое «зеркальце»).

        reflect_metal_rng  – (0.2, 0.5)  
                             Диапазон *reflectance* для **металлического**
                             варианта: доля отражённого «зеркального» света.
    """
    # ------------- unpack cfg with defaults --------------------------
    tex_range   = cfg.get("texrepeat_range",  (4.0, 8.0))
    metal_prob  = cfg.get("metal_prob",       0.25)
    jitter      = cfg.get("rgba_jitter",      0.10)
    spec_mat_rng= cfg.get("specular_range",   (0.05, 0.25))
    shin_mat_rng= cfg.get("shininess_range",  (0.0,  0.30))
    refl_rng    = cfg.get("reflect_metal_rng",(0.1,  0.3))
    tex_path    = pathlib.Path(cfg["file"]).expanduser().resolve()

    # ------------- random sampling -----------------------------------
    texrep = np.random.uniform(*tex_range)
    warm   = np.random.uniform(1-jitter, 1+jitter, 3)

    if np.random.rand() < metal_prob:
        spec = np.random.uniform(0.6, 1.0)
        shin = np.random.uniform(0.5, 1.0)
        refl = np.random.uniform(*refl_rng)
    else:
        spec = np.random.uniform(*spec_mat_rng)
        shin = np.random.uniform(*shin_mat_rng)
        refl = 0.0

    rgba = np.clip(warm, 0, 1)
    rgba_s = f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} 1"

    # ------------- XML block -----------------------------------------
    return textwrap.dedent(f"""
      <texture name="floortex" type="2d" file="{tex_path}"
               width="0" height="0"/>
      <material name="floormat" texture="floortex"
                texrepeat="{texrep:.2f} {texrep:.2f}" texuniform="true"
                rgba="{rgba_s}"
                specular="{spec:.3f}"
                shininess="{shin:.3f}"
                reflectance="{refl:.3f}"/>
    """)