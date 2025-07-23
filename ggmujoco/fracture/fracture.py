# fracture_utils.py
# ──────────────────────────────────────────────────────────────────────
# Мини‑обёртка для blender_fracture.py в виде класса‑менеджера
# Два сценария хранения:
#   1) permanent_dir — файлы остаются навсегда;
#   2) tmp‑директория — удаляется автоматически (__exit__/__del__)
# В обоих случаях fracture() → list[Path]

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Union


class BlenderFractureManager:
    """
    Менеджер Cell‑Fracture для Blender (head‑less).

    Parameters
    ----------
    permanent_dir : str | Path | None
        • Path → писать фрагменты в указанную директорию (остаётся на диске).  
        • None → использовать временную директорию, которая будет удалена
          автоматически при выходе из контекста или уничтожении объекта.
    """

    # путь по‑умолчанию; перезаписываем $BLENDER при инициализации
    BLENDER_DEFAULT = "~/blender/blender-3.5.1-linux-x64/blender"

    SCRIPT_DIR: Path = Path(__file__).resolve().parent
    BLENDER_SCRIPT: Path = SCRIPT_DIR / "blender_fracture.py"

    # ────────────────────────── init ────────────────────────────────
    def __init__(self, permanent_dir: StrOrPath | None = None) -> None:
        os.environ["BLENDER"] = os.path.expanduser(self.BLENDER_DEFAULT)
        self.blender_bin: str = os.environ["BLENDER"]

        if permanent_dir is None:
            self._out_dir   = Path(tempfile.mkdtemp(prefix="fractured_"))
            self._tmp_owned = True
        else:
            self._out_dir   = Path(permanent_dir).expanduser().resolve()
            self._out_dir.mkdir(parents=True, exist_ok=True)
            self._tmp_owned = False

    # ───────────────────── публичный вызов ───────────────────────────
    def fracture(
        self,
        ply_paths: List[Union[str, Path]],
        *,
        chunks_range: Tuple[int, int] = (2, 5),
        noise: float = 0.001,
        cell_scale: Tuple[float, float, float] = (1, 1, 1),
        margin: float = 0.001,
        scale: float | None = None,
        seed: int = 1,
        max_attempts: int = 3,
        export_format: str = "obj",
        voxel: float | None = None
    ) -> List[Path]:
        """
        Запускает Blender, дробит модели и возвращает список Path к файлам.

        Все фрагменты сохраняются в self.out_dir (tmp или permanent).
        """
        ply_paths = [Path(p).expanduser().resolve() for p in ply_paths]
        for p in ply_paths:
            if not p.exists():
                raise FileNotFoundError(p)
        if export_format not in {"obj", "stl"}:
            raise ValueError("export_format must be 'obj' or 'stl'")

        # временный json c результатами
        json_tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", dir=self._out_dir
        )
        json_tmp.close()

        cmd = [
            self.blender_bin,
            "-b",
            "--python",
            str(self.BLENDER_SCRIPT),
            "--",
            "--inputs",
            *map(str, ply_paths),
            "--out",
            str(self._out_dir),
            "--json_out",
            json_tmp.name,
            "--chunks_range",
            *map(str, chunks_range),
            "--noise",
            str(noise),
            "--cellscale",
            *map(str, cell_scale),
            "--margin",
            str(margin),
            "--seed",
            str(seed),
            "--max_attempts",
            str(max_attempts),
            "--format",
            export_format,
        ]
        if scale is not None:
            cmd += ["--scale", str(scale)]
        if voxel is not None:
            cmd += ["--voxel", str(voxel)]

        res = subprocess.run(cmd)
        if res.returncode:
            raise RuntimeError(f"Blender exited with code {res.returncode}")

        try:
            exported_names = json.loads(Path(json_tmp.name).read_text())
        finally:
            Path(json_tmp.name).unlink(missing_ok=True)  # json удаляем всегда

        return [self._out_dir / n for n in exported_names]

    # ───────────────────────── helpers ──────────────────────────────
    @property
    def out_dir(self) -> Path:
        """Каталог, в котором лежат сгенерированные фрагменты."""
        return self._out_dir

    def cleanup(self) -> None:
        """
        Явно удаляет *все* файлы и каталог `out_dir`, если менеджер владеет им.

        • Для tmp‑каталога — всегда удаляет.  
        • Для permanent_dir — удаляет **только по вызову**.
        """
        if self._out_dir.exists():
            for f in self._out_dir.iterdir():
                f.unlink(missing_ok=True)
            self._out_dir.rmdir()
    
    def _prepare_out_dir(self) -> Path:
        return self._out_dir

    # ───────────────────── контекст‑менеджер ────────────────────────
    def __enter__(self) -> "BlenderFractureManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmp_owned:
            self.cleanup()

    def __del__(self) -> None:
        if self._tmp_owned:
            self.cleanup()
