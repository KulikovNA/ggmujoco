#!/usr/bin/env python
# blender_fracture.py
# Запуск:  blender -b --python blender_fracture.py -- --inputs a.stl b.obj ...

import argparse, json, os, sys, random, pathlib, bmesh
import bpy, addon_utils
import mathutils
import numpy as np

# ───────────────────── активируем нужные аддоны ───────────────────────
for mod in ("object_fracture_cell", "io_mesh_stl",
            "io_scene_obj", "io_mesh_ply"):
    try:
        addon_utils.enable(mod, default_set=True, persistent=True)
    except Exception as exc:
        print(f"[WARN] cannot enable addon {mod}: {exc}")

# ────────────────────────────── PARSER ─────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="blender_fracture.py",
        description="Head-less Cell Fracture + clean: импорт, очистка и дробление "
                    "3D-моделей без GUI Blender.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ──────────────────────────── I/O ────────────────────────────────
    parser.add_argument(
        "--inputs", nargs="+", metavar="FILE", required=True,
        help="Входные mesh-файлы (STL/OBJ/PLY). Допускается указать несколько "
             "файлов через пробел."
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR",
        help="Каталог, в который будут сохранены все полученные фрагменты."
    )
    parser.add_argument(
        "--json_out", required=True, metavar="FILE",
        help="Файл JSON, в который скрипт запишет список экспортированных "
             "файлов (имена без пути)."
    )

    # ────────────────────────── fracture ─────────────────────────────
    parser.add_argument(
        "--chunks_range", nargs=2, type=int, default=(2, 5),
        help="Количество seed-частиц для Cell Fracture, определяющее число "
             "фрагментов."
    )
    parser.add_argument(
        "--noise", type=float, default=0.001, metavar="F",
        help="Амплитуда шума при распределении seed-частиц. Большее значение "
             "даёт более нерегулярные фрагменты."
    )
    parser.add_argument(
        "--cellscale", nargs=3, type=float, default=(1, 1, 1),
        metavar=("SX", "SY", "SZ"),
        help="Анизотропный масштаб ячеек дробления по осям X, Y, Z."
    )
    parser.add_argument(
        "--margin", type=float, default=0.001, metavar="F",
        help="Зазор между фрагментами и исходной геометрией (избегает "
             "самопересечений)."
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="N",
        help="Начальное значение генератора случайных чисел для повторяемости."
    )
    parser.add_argument(
        "--max_attempts", type=int, default=4, metavar="N",
        help="Максимальное число попыток фрактурирования, если предыдущие "
             "дали вырожденные фрагменты."
    )

    # ─────────────────────────── cleanup ─────────────────────────────
    parser.add_argument(
        "--voxel", type=float, default=None, metavar="SIZE",
        help="Включить Voxel Remesh c указанным размером вокселя; если "
             "не задано, ремеш не выполняется."
    )

    # ────────────────────────── import/export ────────────────────────
    parser.add_argument(
        "--scale", type=float, default=None, metavar="F",
        help="Униформ‑масштаб, применяемый при импорте модели "
             "(STL игнорирует этот параметр, т.к. имеет собственный)."
    )
    parser.add_argument(
        "--format", choices=["obj", "stl"], default="stl",
        help="Формат файлов, в котором будут сохранены фрагменты."
    )

    # В Blender скрипты получают все параметры после ‘--’
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)



# ─────────────────────────────── MAIN ──────────────────────────────────


def main(args: argparse.Namespace) -> None:
    """Основная логика: импорт → очистка → фрактура для нескольких файлов."""

    json_path = pathlib.Path(args.json_out).resolve()
    out_dir   = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []

    # --- СБРОС Blender ОДИН РАЗ ---
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # --- включаем Cell‑Fracture и импортеры ---
    for mod in ("object_fracture_cell",
                "io_mesh_stl", "io_scene_obj", "io_mesh_ply"):
        try:
            addon_utils.enable(mod, default_set=True, persistent=True)
        except Exception as exc:
            print(f"[WARN] cannot enable addon {mod}: {exc}")
    
    
    chunks = tuple(args.chunks_range)
    # --- цикл по входным файлам ---
    for p in args.inputs:
        src = pathlib.Path(p).resolve()

        imp_obj = import_mesh(src, args.scale)
        # cleaning ...
        merge_by_distance(imp_obj, 0.0001)
        #ensure_outward_normals(imp_obj, thickness=0.2)
        select_and_fill_non_manifold(imp_obj)
        remove_interior_faces(imp_obj)
        recalc_normals(imp_obj)
        if args.voxel:
            voxel_remesh(imp_obj, args.voxel)
        # ensure_outward_normals без толщины
        #ensure_outward_normals(imp_obj, thickness=None)

        if is_non_manifold(imp_obj):
            print(f"[!] {src.name}: still non‑manifold after cleaning")
        
        saved_paths = fracture_with_retries(
            imp_obj,
            chunks=np.random.randint(chunks[0], chunks[1]),
            noise=args.noise,
            cell_scale=tuple(args.cellscale),
            margin=args.margin,
            seed=args.seed,
            max_attempts=args.max_attempts,
            out_dir=out_dir,
            fmt=args.format,
        )
        exported.extend([p.name for p in saved_paths])

    json_path.write_text(json.dumps(exported, ensure_ascii=False, indent=2))
    print(f"[✔] Экспортировано {len(exported)} файлов → {json_path}")


# ───────────────────────────── helpers ────────────────────────────────

def ensure_outward_normals(obj, thickness: float | None = None):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    recalc_normals(obj)

    bpy.ops.object.shade_flat()
    obj.data.use_auto_smooth = False

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.ops.object.mode_set(mode='OBJECT')

    if thickness is not None:
        mod = obj.modifiers.new("SolidifyFix", 'SOLIDIFY')
        mod.thickness = thickness
        mod.offset = 1.0
        mod.use_even_offset = True
        mod.use_quality_normals = True
        bpy.ops.object.modifier_apply(modifier=mod.name)

        voxel_remesh(obj, voxel_size=1.2)

def merge_by_distance(obj, dist=1e-4):
    """Blender 3.6+: mesh.merge_by_distance
       Blender ≤3.5: mesh.remove_doubles"""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    try:
        bpy.ops.mesh.merge_by_distance(threshold=dist)
    except AttributeError:
        bpy.ops.mesh.remove_doubles(threshold=dist)
    bpy.ops.object.mode_set(mode='OBJECT')

def select_and_fill_non_manifold(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    bm = bmesh.from_edit_mesh(obj.data)
    if any(e.select for e in bm.edges):
        try:
            bpy.ops.mesh.fill()
        except RuntimeError:
            pass
    bpy.ops.object.mode_set(mode='OBJECT')

def remove_interior_faces(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_interior_faces()
    bpy.ops.mesh.delete(type='FACE')
    bpy.ops.object.mode_set(mode='OBJECT')

def recalc_normals(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

def is_non_manifold(obj) -> bool:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    bm = bmesh.from_edit_mesh(obj.data)
    res = any(v.select for v in bm.verts)
    bpy.ops.object.mode_set(mode='OBJECT')
    return res

def voxel_remesh(obj, voxel_size=0.01):
    mod = obj.modifiers.new("RemeshFix", 'REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = voxel_size
    bpy.ops.object.modifier_apply(modifier=mod.name)

def import_mesh(src: pathlib.Path, scale: float | None):
    ext = src.suffix.lower()
    if ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(src), global_scale=scale or 1.0)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(src))
    else:
        bpy.ops.import_scene.obj(filepath=str(src))
    obj = bpy.context.selected_objects[0]
    if scale and ext != ".stl":
        obj.scale = (scale,)*3
    obj.name = src.stem
    return obj



def move_bbox_to_origin(obj: bpy.types.Object) -> None:
    """
    Сдвигает *объект* так, чтобы центр его мирового bounding‑box
    оказался в начале координат. Вершины не меняются.
    """
    # мировой центр bounding‑box
    bb = [obj.matrix_world @ mathutils.Vector(v) for v in obj.bound_box]
    center_world = sum(bb, mathutils.Vector()) / 8.0

    # переносим объект
    obj.location -= center_world


# ───────────────────────── fracture util ──────────────────────────────
def fracture_with_retries(
        obj,
        *,
        chunks: int,
        noise: float,
        cell_scale: tuple[float, float, float],
        margin: float,
        seed: int,
        max_attempts: int,
        out_dir: pathlib.Path,
        fmt: str
    ) -> list[pathlib.Path]:
    """
    Пытаемся сфрактурировать до max_attempts раз.
    Экспортируем ТОЛЬКО новые объекты, появившиеся после фрактуры.
    """

    # ─── гарантируем, что оператор доступен ──────────────────────────
    if not hasattr(bpy.ops.object, "add_fracture_cell_objects"):
        try:
            addon_utils.enable("object_fracture_cell",
                               default_set=True, persistent=True)
        except Exception as exc:                    # аддона реально нет
            raise RuntimeError("Cell‑Fracture addon not available") from exc

    paths: list[pathlib.Path] = []
    cur_chunks, cur_noise, cur_seed = chunks, noise, seed

    for _ in range(max_attempts):
        pre_existing = {o.name for o in bpy.context.scene.objects}

        # ― добавляем частицы‑семена (перезаписываем старый модификатор, если есть)
        if "FracSeeds" in obj.modifiers:
            obj.modifiers.remove(obj.modifiers["FracSeeds"])
        ps = obj.modifiers.new("FracSeeds", "PARTICLE_SYSTEM").particle_system
        ps.seed = cur_seed
        ps.settings.count = cur_chunks
        ps.settings.frame_end = 1
        ps.settings.emit_from = 'FACE'
        ps.settings.use_emit_random = True

        # ― запускаем Cell Fracture
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.add_fracture_cell_objects(
            source={'PARTICLE_OWN'},
            source_limit=cur_chunks,
            source_noise=cur_noise,
            cell_scale=cell_scale,
            margin=margin,
            use_remove_original=True,
            use_smooth_faces=False
        )

        # ― выбираем только новые меши
        shards = [o for o in bpy.context.scene.objects
                  if o.type == 'MESH' and o.name not in pre_existing]

        # ― проверяем на вырожденность
        if shards and all(sh.data.polygons and sh.data.vertices for sh in shards):
            for i, sh in enumerate(shards):
                # 1) Нормали
                #ensure_outward_normals(sh, thickness=0.05)
                voxel_remesh(sh, voxel_size=1.6)
                move_bbox_to_origin(sh)
                
                #mod = sh.modifiers.new(name="decimate", type='DECIMATE')
                #mod.ratio = 0.2    # оставить 20% от исходных полигонов 
                #bpy.context.view_layer.objects.active = sh
                #bpy.ops.object.modifier_apply(modifier=mod.name)

                fname = f"{obj.name}_shard{i}.{fmt}"
                fout  = out_dir / fname
                bpy.ops.object.select_all(action='DESELECT')
                sh.select_set(True)
                if fmt == "stl":
                    bpy.ops.export_mesh.stl(filepath=str(fout),
                                            use_selection=True,
                                            ascii=False,
                                            use_mesh_modifiers=True)
                else:
                    bpy.ops.export_scene.obj(filepath=str(fout),
                                             use_selection=True,
                                             use_materials=False,
                                             use_uvs=False,
                                             use_normals=True)
                paths.append(fout)
            break  # успех – выходим из цикла

        # ― если не получилось, готовим следующую попытку
        cur_noise  *= 0.5
        cur_chunks  = max(3, cur_chunks - 1)
        cur_seed   += 1

    return paths


# ────────────────────────────── ENTRY ──────────────────────────────────
if __name__ == "__main__":
    main(parse_args())