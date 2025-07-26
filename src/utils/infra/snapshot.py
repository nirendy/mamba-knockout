import json
import shutil
from pathlib import Path
from typing import Any


def load_json_with_snapshot_recovery(path: Path, recover: bool = False) -> Any:
    """
    Attempts to read a JSON file from `path`. If it fails (e.g., due to corruption),
    it looks for a recovery file in the `.snapshot` directory of the same parent directory,
    trying them in descending order by modified time.

    Parameters:
        path (Path): Path to the target JSON file.
        recover (bool): Whether to copy the working snapshot back to the original file.

    Returns:
        The parsed JSON content.

    Raises:
        FileNotFoundError: If no valid JSON could be found.
    """

    def try_load_json(p: Path) -> Any:
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON from {p}: {e}")
            return None

    # First try the original path
    data = try_load_json(path)
    if data is not None:
        return data

    # If failed, check in .snapshot directory
    snapshot_dir = path.parent / ".snapshot"
    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        raise FileNotFoundError(f"No valid JSON found and no snapshot directory at {snapshot_dir}")

    # List and sort snapshot files by modified time (newest first)
    snapshots = sorted(snapshot_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    for snapshot in snapshots:
        snapshot_file = snapshot / path.name
        if not snapshot_file.is_file():
            continue
        snapshot_data = try_load_json(snapshot_file)
        if snapshot_data is not None:
            print(f"Recovered JSON from snapshot: {snapshot_file}")
            if recover:
                shutil.copy2(snapshot_file, path)
                print(f"Recovered snapshot copied to {path}")
            return snapshot_data

    raise FileNotFoundError(f"Could not recover a valid JSON from snapshots in {snapshot_dir}")
