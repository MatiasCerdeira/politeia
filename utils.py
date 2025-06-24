import json
import os
from typing import Any


def save_to_json(data: Any, filepath: str) -> None:
    """
    Guarda un objeto Python en un archivo JSON.

    Args:
        data: Objeto Python (serializable) a guardar.
        filepath: Ruta del archivo donde se guardará el JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Datos guardados en {filepath}")
