"""Tiny background-job runner (for model downloads and slow benchmarks). The UI polls /api/jobs/<id>."""
from __future__ import annotations

import threading
import traceback
import uuid


class Jobs:
    def __init__(self):
        self._jobs, self._lock = {}, threading.Lock()

    def start(self, fn, *args, **kwargs) -> str:
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "state": "running", "progress": 0.0, "message": "starting", "result": None, "error": None}
        with self._lock:
            self._jobs[jid] = job

        def report(progress=None, message=None):
            if progress is not None:
                job["progress"] = round(float(progress), 3)
            if message is not None:
                job["message"] = message

        def run():
            try:
                job["result"] = fn(*args, report=report, **kwargs)
                job["state"], job["progress"] = "done", 1.0
            except Exception as e:                       # shown to the user, never crashes the server
                job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"
                job["trace"] = traceback.format_exc()

        threading.Thread(target=run, daemon=True).start()
        return jid

    def get(self, jid):
        job = self._jobs.get(jid)
        return None if job is None else {k: v for k, v in job.items() if k != "trace"}
