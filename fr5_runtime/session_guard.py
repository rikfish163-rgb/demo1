"""Filesystem-only stop latch shared by calibration motion and acquisition."""
from pathlib import Path


def require_session_open(directory):
    directory = Path(directory).resolve()
    for ancestor in (directory, *directory.parents):
        if (ancestor / 'MOTION_HOLD').exists():
            raise RuntimeError(f'Calibration session on hold: {ancestor}')
