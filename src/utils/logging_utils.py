from pathlib import Path
import csv
from typing import Dict, Any, List


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_episode_logs(out_dir: str | Path, rows: List[Dict[str, Any]], filename: str = "episodes.csv") -> None:
    out_dir = ensure_dir(out_dir)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    csv_path = out_dir / filename
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
