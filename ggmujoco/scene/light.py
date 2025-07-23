import numpy as np
from typing import Sequence, Tuple, Optional, Dict, Any


def gen_light_body(center_xy, light_cfg):
    light_specs  = generate_lights(center_xy, light_cfg)
    lights_block = "\n".join(light_to_xml(ls) for ls in light_specs)
    return lights_block
# ─────────────────────────────────────────
# вспомогательные функции освещения
# ─────────────────────────────────────────
# --- преобразование цветовой температуры (K) -> линейный RGB 0..1 ---
def kelvin_to_rgb(k: float) -> Tuple[float, float, float]:
    """
    Быстрая аппроксимация (см. Tanner Helland + адаптация под NumPy).
    k: температура в Кельвинах (1000..40000 разумно)
    Возвращает (r,g,b) в диапазоне [0,1].
    """
    k = np.clip(k, 1000.0, 40000.0) / 100.0
    # красный
    if k <= 66:
        r = 255
    else:
        r = 329.698727446 * ((k - 60) ** -0.1332047592)
    # зелёный
    if k <= 66:
        g = 99.4708025861 * np.log(k) - 161.1195681661
    else:
        g = 288.1221695283 * ((k - 60) ** -0.0755148492)
    # синий
    if k >= 66:
        b = 255
    elif k <= 19:
        b = 0
    else:
        b = 138.5177312231 * np.log(k - 10) - 305.0447927307
    rgb = np.clip([r, g, b], 0, 255) / 255.0
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


# --- сэмпл одного источника из диапазонов ---
def _sample_light(center_xy: Tuple[float, float],
                  cfg: Dict[str, Any],
                  idx: int) -> Dict[str, Any]:
    """
    Вернёт dict с полями: name,pos,dir,directional,diffuse,specular,ambient.
    cfg поля (все опциональны):
        num, xy_radius, z_range, kelvin_range, intensity_range,
        ambient, directional
    """
    cx, cy = center_xy
    r   = np.random.uniform(0, cfg["xy_radius"])
    ang = np.random.uniform(0, 2*np.pi)
    x   = cx + r * np.cos(ang)
    y   = cy + r * np.sin(ang)
    z   = np.random.uniform(*cfg["z_range"])

    # направление в центр (по умолчанию вниз к (cx,cy,0))
    dir_vec = np.array([cx - x, cy - y, 0.0 - z])
    norm = np.linalg.norm(dir_vec)
    if norm < 1e-6:
        dir_vec = np.array([0.0, 0.0, -1.0])
    else:
        dir_vec /= norm

    k   = np.random.uniform(*cfg["kelvin_range"])
    rgb = np.array(kelvin_to_rgb(k))
    inten = np.random.uniform(*cfg["intensity_range"])
    diff = np.clip(rgb * inten, 0, 1)
    spec = np.clip(rgb * inten * 0.5, 0, 1)
    amb  = cfg["ambient"]

    return dict(
        name=f"randlight_{idx}",
        pos=(x, y, z),
        dir=tuple(dir_vec),
        directional=cfg["directional"],
        diffuse=tuple(diff),
        specular=tuple(spec),
        ambient=amb,
    )


# --- генерация списка источников по cfg ---
def generate_lights(center_xy: Tuple[float, float],
                    cfg: Optional[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:

    if cfg is None:
        cfg = {}

    # Настройки по умолчанию
    cfg = {
        "num": int(cfg.get("num", 5 if cfg.get("directional", True) else 8)),
        "xy_radius": float(cfg.get("xy_radius", 2.0)),
        "z_range": tuple(cfg.get("z_range", (3.0, 4.0))),
        "kelvin_range": tuple(cfg.get("kelvin_range", (3500.0, 6000.0))),
        "intensity_range": tuple(cfg.get("intensity_range", (0.3, 0.7) if cfg.get("directional", True) else (0.5, 1.0))),
        "ambient": tuple(cfg.get("ambient", (0.1, 0.1, 0.1))),
        "directional": bool(cfg.get("directional", True)),
    }

    lights = []

    if cfg["directional"]:
        # Генерация нескольких близких направленных источников для мягких теней
        base_dir = np.array([0.0, 0.0, -1.0])
        for i in range(cfg["num"]):
            perturbation = np.random.normal(0, 0.1, size=3)
            dir_vec = base_dir + perturbation
            dir_vec /= np.linalg.norm(dir_vec)

            k = np.random.uniform(*cfg["kelvin_range"])
            rgb = np.array(kelvin_to_rgb(k))
            inten = np.random.uniform(*cfg["intensity_range"])

            diff = np.clip(rgb * inten, 0, 1)
            spec = np.clip(rgb * inten * 0.5, 0, 1)

            lights.append({
                "name": f"dirlight_{i}",
                "pos": (0, 0, np.random.uniform(*cfg["z_range"])),
                "dir": tuple(dir_vec),
                "directional": True,
                "diffuse": tuple(diff),
                "specular": tuple(spec),
                "ambient": cfg["ambient"],
            })
    else:
        # Равномерная расстановка точечных источников вокруг сцены
        cx, cy = center_xy
        for i in range(cfg["num"]):
            ang = 2 * np.pi * i / cfg["num"]
            x = cx + cfg["xy_radius"] * np.cos(ang)
            y = cy + cfg["xy_radius"] * np.sin(ang)
            z = np.random.uniform(*cfg["z_range"])

            k = np.random.uniform(*cfg["kelvin_range"])
            rgb = np.array(kelvin_to_rgb(k))
            inten = np.random.uniform(*cfg["intensity_range"])

            diff = np.clip(rgb * inten, 0, 1)
            spec = np.clip(rgb * inten * 0.5, 0, 1)

            lights.append({
                "name": f"pointlight_{i}",
                "pos": (x, y, z),
                "dir": (0.0, 0.0, -1.0),
                "directional": False,
                "diffuse": tuple(diff),
                "specular": tuple(spec),
                "ambient": cfg["ambient"],
            })

    return lights



# --- превратить dict источника в XML-строку Light ---
def light_to_xml(spec: Dict[str, Any]) -> str:
    d = spec
    diff = f'{d["diffuse"][0]:.3f} {d["diffuse"][1]:.3f} {d["diffuse"][2]:.3f}'
    specular = f'{d["specular"][0]:.3f} {d["specular"][1]:.3f} {d["specular"][2]:.3f}'
    amb = f'{d["ambient"][0]:.3f} {d["ambient"][1]:.3f} {d["ambient"][2]:.3f}'
    pos = f'{d["pos"][0]:.3f} {d["pos"][1]:.3f} {d["pos"][2]:.3f}'
    if d["directional"]:
        dir_str = f'{d["dir"][0]:.3f} {d["dir"][1]:.3f} {d["dir"][2]:.3f}'
        return (f'        <light name="{d["name"]}" pos="{pos}" dir="{dir_str}" '
                f'directional="true" diffuse="{diff}" specular="{specular}" ambient="{amb}"/>')
    else:
        return (f'        <light name="{d["name"]}" pos="{pos}" '
                f'diffuse="{diff}" specular="{specular}" ambient="{amb}"/>')
    
