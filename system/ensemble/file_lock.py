"""Lightweight file-based locking for artifact directories.

Used by FedDES to prevent duplicate training when multiple parallel jobs
share the same base-classifier or graph config.  The lock is advisory
(``fcntl.flock``) and auto-releases on process exit — even on crashes —
so stale locks are not an issue.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def artifact_lock(lock_path: Path | str):
    """Acquire an exclusive file lock; blocks until available.

    Parameters
    ----------
    lock_path : Path | str
        Path to the ``.lock`` file (e.g. ``ckpt_root / "base_clf" / "base[abc].lock"``).
        Parent directories are created automatically.

    Usage
    -----
    ::

        with artifact_lock(ckpt_root / "base_clf" / f"base[{base_id}].lock"):
            if not all_artifacts_exist():
                train_base_classifiers()
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
