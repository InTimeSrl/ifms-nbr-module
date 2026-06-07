"""
state.py -- Pipeline state tracking for a single AOI/tile.

Persists a JSON file at <data_dir>/pipeline_state.json with:
  - baseline          : build flag + metadata (n. scenes, coverage, ...)
  - last_processed_dt : ISO 8601 datetime of the last processed scene
                        (watermark for the STAC query: only scenes with
                        datetime > watermark are retrieved, without tracking
                        individual scene IDs)

The file is created on the first run and updated incrementally on each
execution; in continuous mode the system always starts from the last scene
seen and processes only new ones.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

STATE_FILENAME = "pipeline_state.json"

# ---------------------------------------------------------------------------
# Base I/O
# ---------------------------------------------------------------------------

def state_path(data_dir):
    """Return the Path of the state JSON file for a data_dir."""
    return Path(data_dir) / STATE_FILENAME


def load_state(data_dir):
    """Load state from the JSON file.

    If config.FORCE_REPROCESS is True, always returns an empty state,
    forcing a full recompute (baseline + all scenes).
    If the file does not exist, returns an empty state with a valid structure.
    Does not raise exceptions: a corrupted JSON is logged and ignored
    (returns empty state, forcing the pipeline to restart from scratch).
    """
    if config.FORCE_REPROCESS:
        logger.info("FORCE_REPROCESS=True: state JSON ignored, full reprocessing")
        return {"baseline": {"built": False}}
    p = state_path(data_dir)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                state = json.load(f)
            # Ensure required keys are present even on old files
            state.setdefault("baseline", {"built": False})
            # Migration from old format: processed_scenes -> last_processed_dt
            if "processed_scenes" in state and "last_processed_dt" not in state:
                scene_dates = [v.get("date", "") for v in state["processed_scenes"].values()
                               if v.get("date")]
                if scene_dates:
                    state["last_processed_dt"] = max(scene_dates)
                    logger.info("State migration: watermark derived from processed_scenes: %s",
                                state["last_processed_dt"])
                del state["processed_scenes"]
            return state
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("State JSON not readable (%s), restarting from scratch: %s", p, exc)
    return {"baseline": {"built": False}}


def save_state(state, data_dir):
    """Save state to JSON."""
    p = state_path(data_dir)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def mark_baseline_built(state, n_scenes, coverage_pct, scene_ids):
    """Record baseline construction in state (in-place).

    Parameters
    ----------
    n_scenes : int
    coverage_pct : float   Coverage 0-100.
    scene_ids : list[str]  IDs of the scenes used.
    """
    state["baseline"] = {
        "built": True,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_scenes": n_scenes,
        "coverage_pct": round(coverage_pct, 1),
        "scene_ids": list(scene_ids),
    }


# ---------------------------------------------------------------------------
# Monitoring watermark
# ---------------------------------------------------------------------------

def get_watermark(state):
    """ISO 8601 datetime of the last processed scene, or None on first run.

    Used as the exclusive lower bound for the STAC query: only scenes
    with datetime > watermark are retrieved.
    """
    return state.get("last_processed_dt")


def update_watermark(state, scene_datetime):
    """Update the watermark if scene_datetime is more recent (in-place).

    Parameters
    ----------
    scene_datetime : str
        ISO 8601 datetime of the just-processed scene
        (e.g. "2025-08-14T09:09:52.139000Z").
    """
    current = state.get("last_processed_dt")
    if scene_datetime and (current is None or scene_datetime > current):
        state["last_processed_dt"] = scene_datetime
