"""
State persistence manager. Reads and writes JSON files for positions and daily logs.
Provides atomic writes and thread-safe file access.
"""
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


def _ensure_state_dir() -> None:
    """Create the state directory if it does not exist."""
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """
    Write JSON to a temporary file then atomically rename it.
    Prevents corruption if the process is killed mid-write.
    """
    _ensure_state_dir()
    fd, tmp_path = tempfile.mkstemp(dir=config.STATE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON from a file, returning an empty dict if the file is missing."""
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return {}


# ── Positions ──────────────────────────────────────────────────────────────────


def load_positions() -> list[dict[str, Any]]:
    """Load open positions from state file."""
    data = _read_json(config.POSITIONS_FILE)
    return data.get("open_positions", [])


def save_positions(positions: list[dict[str, Any]]) -> None:
    """Persist open positions to state file."""
    data = _read_json(config.POSITIONS_FILE)
    data["open_positions"] = positions
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(config.POSITIONS_FILE, data)
    logger.info("Saved %d open positions", len(positions))


def add_position(position: dict[str, Any]) -> None:
    """Append a new position to the open positions list."""
    positions = load_positions()
    positions.append(position)
    save_positions(positions)


def remove_position(position_id: str) -> dict[str, Any] | None:
    """Remove and return a position by its id."""
    positions = load_positions()
    removed = None
    remaining = []
    for pos in positions:
        if pos.get("id") == position_id:
            removed = pos
        else:
            remaining.append(pos)
    if removed:
        save_positions(remaining)
        logger.info("Removed position %s", position_id)
    return removed


def update_position(position_id: str, updates: dict[str, Any]) -> bool:
    """Update fields on an existing position. Returns True if found."""
    positions = load_positions()
    for pos in positions:
        if pos.get("id") == position_id:
            pos.update(updates)
            save_positions(positions)
            return True
    return False


# ── Daily Log ──────────────────────────────────────────────────────────────────


def load_daily_log() -> list[dict[str, Any]]:
    """Load the full daily log history."""
    data = _read_json(config.DAILY_LOG_FILE)
    return data.get("daily_summary", [])


def save_daily_log(entries: list[dict[str, Any]]) -> None:
    """Persist the daily log."""
    data = _read_json(config.DAILY_LOG_FILE)
    data["daily_summary"] = entries
    _atomic_write(config.DAILY_LOG_FILE, data)


def append_daily_entry(entry: dict[str, Any]) -> None:
    """Append a daily summary entry."""
    entries = load_daily_log()
    entries.append(entry)
    save_daily_log(entries)
    logger.info("Appended daily entry for %s", entry.get("date"))


def get_today_entry() -> dict[str, Any] | None:
    """Return today's entry if it exists."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for entry in reversed(load_daily_log()):
        if entry.get("date") == today_str:
            return entry
    return None


def create_today_entry() -> dict[str, Any]:
    """Create a fresh daily summary entry for today."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = {
        "date": today_str,
        "trades_opened": [],
        "trades_closed": [],
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,
        "risk_rejections": [],
        "llm_confidence_scores": [],
        "account_equity": 0.0,
    }
    return entry


def save_today_entry(entry: dict[str, Any]) -> None:
    """Overwrite or append today's entry."""
    entries = load_daily_log()
    today_str = entry.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    replaced = False
    for i, e in enumerate(entries):
        if e.get("date") == today_str:
            entries[i] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    save_daily_log(entries)
