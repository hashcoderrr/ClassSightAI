"""Data-retention helpers (DPDP Act: keep personal data only as long as needed)."""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _older_than(p: Path, days: float) -> bool:
    return (time.time() - p.stat().st_mtime) > days * 86400


def cleanup_retention(cfg: Any) -> dict[str, int]:
    """Delete raw videos and session outputs older than the configured retention.

    Embeddings, attendance rows and the Excel/CSV/JSON reports' *data* remain in
    the database; session output folders (annotated video, snapshots, unknown
    thumbnails, report files) are removed after ``output_retention_days``.
    """
    pcfg = cfg.privacy
    removed = {"raw_videos": 0, "session_dirs": 0}
    vdir = cfg.path("videos_dir")
    if vdir.exists():
        days = float(pcfg.raw_video_retention_days) if pcfg.retain_raw_video else 0.0
        for p in vdir.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and _older_than(p, days):
                p.unlink(missing_ok=True)
                removed["raw_videos"] += 1
    odir = cfg.path("outputs_dir")
    if odir.exists() and float(pcfg.output_retention_days) > 0:
        for d in odir.iterdir():
            if d.is_dir() and _older_than(d, float(pcfg.output_retention_days)):
                shutil.rmtree(d, ignore_errors=True)
                removed["session_dirs"] += 1
    if any(removed.values()):
        log.info("Retention cleanup: %s", removed)
    return removed
