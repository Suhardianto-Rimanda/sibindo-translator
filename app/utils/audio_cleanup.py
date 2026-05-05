import os
import time
from pathlib import Path
from threading import Thread

from app.utils.logger import log


def cleanup_old_audio(directory: str, max_age_seconds: int = 600) -> int:
    """Remove mp3 files older than max_age_seconds. Returns count removed."""
    path = Path(directory)
    if not path.exists():
        return 0

    now = time.time()
    removed = 0
    for f in path.glob("*.mp3"):
        try:
            age = now - f.stat().st_mtime
            if age > max_age_seconds:
                f.unlink()
                removed += 1
        except OSError as exc:
            log.warning(f"failed to remove {f}: {exc}")
    return removed


def start_cleanup_thread(directory: str, max_age_seconds: int = 600, interval: int = 300):
    """Spawn daemon thread that periodically purges old TTS files."""
    def _loop():
        while True:
            try:
                count = cleanup_old_audio(directory, max_age_seconds)
                if count:
                    log.info(f"audio cleanup: removed {count} files from {directory}")
            except Exception as exc:
                log.error(f"audio cleanup error: {exc}")
            time.sleep(interval)

    t = Thread(target=_loop, daemon=True, name="audio-cleanup")
    t.start()
    return t
