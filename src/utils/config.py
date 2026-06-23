from pathlib import Path
import json
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        return json.load(f)

def get_project_root() -> Path:
    return PROJECT_ROOT
