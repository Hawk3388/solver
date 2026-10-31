"""Bounded single-worker queue with per-document state and expiring files."""

import atexit
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Full, Empty
import secrets
import shutil
import tempfile
from threading import Event, RLock, Thread
import time


class QueueFullError(ValueError):
    pass


@dataclass
class Job:
    id: str
    directory: Path
    options: dict
    items: list
    last_access: float = field(default_factory=time.monotonic)
    lock: RLock = field(default_factory=RLock)
    closed: bool = False


class JobManager:
    def __init__(self, processor, capacity=4, ttl=3600):
        self.processor = processor
        self.ttl = ttl
        self.temporary = tempfile.TemporaryDirectory(prefix="worksheet_jobs_")
        self.root = Path(self.temporary.name)
        self.jobs = {}
        self.lock = RLock()
        self.queue = Queue(maxsize=capacity)
        self.stopped = Event()
        Thread(target=self._work, daemon=True, name="worksheet-worker").start()
        Thread(target=self._reap, daemon=True, name="worksheet-cleanup").start()
        atexit.register(self.close)

    def create(self, uploads, options):
        if self.queue.full():
            raise QueueFullError("The queue is full. Please try again shortly.")
        job_id = secrets.token_urlsafe(24)
        directory = self.root / job_id
        directory.mkdir()
        items = []
        try:
            for index, upload in enumerate(uploads):
                item_directory = directory / str(index)
                item_directory.mkdir()
                extension = Path(upload.filename.replace('\\', '/')).suffix.lower()
                safe_extension = extension if extension in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".bin"
                source = item_directory / ("upload" + safe_extension)
                with source.open("wb") as destination:
                    shutil.copyfileobj(upload.file, destination)
                items.append({"id": str(index), "filename": Path(upload.filename.replace('\\', '/')).name,
                              "directory": item_directory, "source": source,
                              "extension": extension,
                              "state": "queued", "stage": "Waiting in queue", "error": None,
                              "document": None, "revision": 0, "geometry": None})
            job = Job(job_id, directory, dict(options), items)
            with self.lock:
                self.jobs[job_id] = job
            self._enqueue(job, list(range(len(items))))
            return job
        except Exception:
            with self.lock:
                self.jobs.pop(job_id, None)
            shutil.rmtree(directory)
            raise

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.closed:
                raise KeyError("Job expired or does not exist. Upload the images again.")
            job.last_access = time.monotonic()
            return job

    def retry(self, job, index, geometry=None):
        with job.lock:
            item = job.items[index]
            if item["state"] not in {"failed", "ready"}:
                raise ValueError("This worksheet is already being processed.")
            old = (item["state"], item["stage"], item["geometry"])
            item.update(state="queued", stage="Waiting in queue", geometry=geometry)
            try:
                self._enqueue(job, [index])
            except QueueFullError:
                item["state"], item["stage"], item["geometry"] = old
                raise

    def _enqueue(self, job, indices):
        try:
            self.queue.put_nowait((job, indices))
        except Full as error:
            raise QueueFullError("The queue is full. Please try again shortly.") from error

    def snapshot(self, job):
        with job.lock:
            items = []
            for item in job.items:
                info = {key: deepcopy(item[key]) for key in
                        ("id", "filename", "state", "stage", "error", "revision")}
                doc = item["document"]
                info["editable"] = doc is not None and item["state"] in {"ready", "failed"}
                info["warnings"] = deepcopy(doc.get("warnings", [])) if doc else []
                items.append(info)
            active = any(item["state"] in {"queued", "running"} for item in items)
            return {"id": job.id, "state": "processing" if active else "finished", "items": items,
                    "completed": sum(item["state"] in {"ready", "failed"} for item in items),
                    "total": len(items), "expires_in_seconds": self.ttl}

    def _work(self):
        while not self.stopped.is_set():
            try:
                job, indices = self.queue.get(timeout=1)
            except Empty:
                continue
            try:
                for index in indices:
                    item = job.items[index]
                    with job.lock:
                        item.update(state="running", stage="Processing worksheet", error=None)

                    def progress(stage):
                        with job.lock:
                            item["stage"] = stage if job.options.get("debug", False) else "Processing worksheet"

                    try:
                        self.processor(job, item, progress)
                        with job.lock:
                            notices = len((item.get("document") or {}).get("warnings", []))
                            stage = f"Ready to review · {notices} notice(s)" if notices else "Ready to review"
                            item.update(state="ready", stage=stage)
                    except Exception as error:
                        with job.lock:
                            item.update(state="failed", stage="Needs attention", error=str(error))
                    finally:
                        job.last_access = time.monotonic()
            finally:
                self.queue.task_done()

    def _reap(self):
        while not self.stopped.wait(30):
            self.cleanup_expired()

    def cleanup_expired(self):
        now = time.monotonic()
        with self.lock:
            for job_id, job in list(self.jobs.items()):
                with job.lock:
                    active = any(item["state"] in {"queued", "running"} for item in job.items)
                    if not active and now - job.last_access > self.ttl:
                        try:
                            shutil.rmtree(job.directory)
                        except OSError:
                            continue
                        job.closed = True
                        del self.jobs[job_id]

    def close(self):
        self.stopped.set()
        # A running model call may still own files. Its daemon exits with the process.
        try:
            self.temporary.cleanup()
        except OSError:
            pass
