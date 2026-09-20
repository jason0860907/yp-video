"""Serialize editor saves and reviewed-feedback changes in the web process."""

from threading import RLock

annotation_write_lock = RLock()
