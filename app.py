"""FastAPI server for queued worksheet solving, editing and feedback."""

from contextlib import asynccontextmanager
from copy import deepcopy
import logging
from pathlib import Path
import re
import socket
from threading import Lock
import unicodedata
from urllib.parse import urlsplit
import zipfile

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, UploadFile
from starlette.exceptions import HTTPException
import uvicorn

from worksheet_solver.config import (
    PROJECT_ROOT, MAX_BATCH_FILES, MAX_FILE_SIZE, MAX_PENDING_JOBS,
    JOB_TTL_SECONDS, FEEDBACK_DIRECTORY, DATASET_DIRECTORY, solver_defaults,
)
from worksheet_solver.documents import DocumentEdits, render_document, validate_edits
from worksheet_solver.feedback import FeedbackRequest, FeedbackStore
from worksheet_solver.jobs import JobManager, QueueFullError
from worksheet_solver.web_processing import process_worksheet

logger = logging.getLogger(__name__)
_runtime = None
_runtime_lock = Lock()


def detection_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            from main import WorksheetSolver
            _runtime = WorksheetSolver.preload_detection_model()
        return _runtime


def default_processor(job, item, progress):
    from main import WorksheetSolver
    return process_worksheet(job, item, progress, WorksheetSolver, detection_runtime)


class RequestLimitsMiddleware:
    """Enforce origin and byte limits, including streamed multipart bodies."""

    def __init__(self, app, max_body_size):
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        request = Request(scope)
        if scope["method"] in {"POST", "PUT", "DELETE"} and headers.get("origin"):
            origin = urlsplit(headers["origin"])
            if origin.netloc != request.url.netloc or origin.scheme != request.url.scheme:
                response = JSONResponse({"error": "Cross-origin changes are not allowed."}, status_code=403)
                return await response(scope, receive, send)
        message = f"The request exceeds the {self.max_body_size // (1024 * 1024)} MB batch limit."
        if "content-length" in headers:
            try:
                content_length = int(headers["content-length"])
                if content_length < 0:
                    raise ValueError
            except ValueError:
                response = JSONResponse({"error": "Invalid Content-Length header."}, status_code=400)
                return await response(scope, receive, send)
            if content_length > self.max_body_size:
                response = JSONResponse({"error": message}, status_code=413)
                return await response(scope, receive, send)
        received = 0

        async def receive_limited():
            nonlocal received
            event = await receive()
            if event["type"] == "http.request":
                received += len(event.get("body", b""))
                if received > self.max_body_size:
                    raise HTTPException(status_code=413, detail=message)
            return event

        await self.app(scope, receive_limited, send)


def download_stem(filename):
    """Create a portable ASCII name for downloads and archive entries."""
    stem = Path(filename.replace("\\", "/")).stem
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")[:160] or "worksheet"


def create_app(processor=None, feedback_store=None):
    @asynccontextmanager
    async def lifespan(application):
        manager = JobManager(processor or default_processor, MAX_PENDING_JOBS, JOB_TTL_SECONDS)
        application.state.worksheet_jobs = manager
        try:
            if processor is None:
                logger.info("Loading shared gap detection model...")
                await run_in_threadpool(detection_runtime)
            yield
        finally:
            await run_in_threadpool(manager.close)

    app = FastAPI(title="Worksheet Solver", version="1.0.0", lifespan=lifespan)
    app.state.worksheet_feedback = feedback_store or FeedbackStore(FEEDBACK_DIRECTORY, DATASET_DIRECTORY)
    app.add_middleware(RequestLimitsMiddleware, max_body_size=MAX_FILE_SIZE * MAX_BATCH_FILES)
    app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
    templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")

    @app.exception_handler(QueueFullError)
    async def queue_full(_request, error):
        return JSONResponse({"error": str(error)}, status_code=429)

    @app.exception_handler(KeyError)
    async def expired(_request, error):
        return JSONResponse({"error": str(error.args[0])}, status_code=404)

    @app.exception_handler(ValidationError)
    @app.exception_handler(RequestValidationError)
    async def invalid_schema(_request, error):
        messages = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()]
        return JSONResponse({"error": "; ".join(messages)}, status_code=400)

    @app.exception_handler(ValueError)
    async def invalid_value(_request, error):
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.exception_handler(HTTPException)
    async def http_error(_request, error):
        return JSONResponse({"error": str(error.detail)}, status_code=error.status_code, headers=error.headers)

    @app.exception_handler(Exception)
    async def unexpected_error(_request, error):
        logger.error("Worksheet operation failed", exc_info=error)
        return JSONResponse({"error": "The operation failed. Check the server log and try again."}, status_code=500)

    def manager():
        return app.state.worksheet_jobs

    def get_item(job_id, item_id):
        job = manager().get(job_id)
        if not item_id.isascii() or not item_id.isdigit() or int(item_id) >= len(job.items):
            raise KeyError("Worksheet does not exist.")
        return job, job.items[int(item_id)]

    def item_urls(job_id, item):
        base = f"/api/jobs/{job_id}/items/{item['id']}"
        revision = item["revision"]
        return {
            "preview_url": f"{base}/assets/result_preview?v={revision}",
            "source_preview_url": f"{base}/assets/page_preview?v={revision}",
            "source_url": f"{base}/assets/page?v={revision}",
            "original_url": f"{base}/assets/original?v={revision}",
            "image_url": f"{base}/assets/result?v={revision}",
            "download_url": f"{base}/download?v={revision}",
            "document_url": f"{base}/document",
        }

    def snapshot(job):
        data = manager().snapshot(job)
        for info in data["items"]:
            info.update(item_urls(job.id, info))
        data["archive_url"] = f"/api/jobs/{job.id}/download"
        return data

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    @app.get("/api/config")
    def config():
        return dict(defaults=solver_defaults(), max_files=MAX_BATCH_FILES,
                    max_file_size=MAX_FILE_SIZE, retention_seconds=JOB_TTL_SECONDS)

    @app.post("/solve", status_code=202, openapi_extra={"requestBody": {
        "required": True, "content": {"multipart/form-data": {"schema": {
            "type": "object", "required": ["files"], "properties": {
                "files": {"type": "array", "maxItems": MAX_BATCH_FILES, "items": {"type": "string", "format": "binary"}},
                "model_name": {"type": "string"},
                "local": {"type": "boolean"}, "think": {"type": "boolean"}, "debug": {"type": "boolean"},
            },
        }}},
    }})
    async def solve(request: Request):
        async with request.form(max_files=MAX_BATCH_FILES, max_fields=8, max_part_size=4096) as form:
            uploads = form.getlist("files") or form.getlist("file")
            if any(not isinstance(upload, UploadFile) for upload in uploads):
                raise ValueError("Worksheet images must be uploaded as files.")
            uploads = [upload for upload in uploads if upload.filename]
            if not uploads or len(uploads) > MAX_BATCH_FILES:
                raise ValueError(f"Select between 1 and {MAX_BATCH_FILES} images.")
            options = solver_defaults()
            model_name = form.get("model_name", options["llm_model_name"])
            if not isinstance(model_name, str) or not model_name.strip() or len(model_name.strip()) > 200:
                raise ValueError("Enter a valid model name (maximum 200 characters).")
            options["llm_model_name"] = model_name.strip()
            for field in ("local", "think", "debug"):
                if field in form:
                    if form[field] not in ("true", "false"):
                        raise ValueError(f"Invalid boolean setting: {field}.")
                    options[field] = form[field] == "true"
            job = await run_in_threadpool(manager().create, uploads, options)
        data = await run_in_threadpool(snapshot, job)
        return JSONResponse(data, status_code=202, headers={"Location": f"/api/jobs/{job.id}"})

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        return snapshot(manager().get(job_id))

    @app.post("/api/jobs/{job_id}/items/{item_id}/retry", status_code=202)
    def retry(job_id: str, item_id: str, payload: dict | None = Body(default=None)):
        job, item = get_item(job_id, item_id)
        geometry = (payload or {}).get("geometry", item.get("geometry"))
        if geometry is not None:
            if (not isinstance(geometry, dict) or set(geometry) != {"rotation", "perspective"}
                    or type(geometry["rotation"]) is not int or geometry["rotation"] not in {0, 90, 180, 270}
                    or type(geometry["perspective"]) is not bool):
                raise ValueError("Geometry requires rotation (0/90/180/270) and perspective (boolean).")
        manager().retry(job, int(item_id), geometry)
        return snapshot(job)

    def document_data(job_id, item_id, edits=None):
        job, item = get_item(job_id, item_id)
        with job.lock:
            if item["state"] not in {"ready", "failed"}:
                raise HTTPException(status_code=409, detail="Wait until this worksheet finishes processing.")
            if item["document"] is None:
                raise ValueError("No editable document is available for this image.")
            if edits is not None:
                validated = validate_edits(edits.model_dump(), item["document"])
                updated = deepcopy(item["document"])
                updated.update(validated)
                updated["version"] += 1
                updated["validation_warnings"] = []
                render_document(updated, item["directory"] / "page.png", item["directory"] / "result.png")
                item.update(document=updated, revision=updated["version"], state="ready", error=None, stage="Edited result")
            data = deepcopy(item["document"])
            data.update(item_urls(job_id, item))
            return data

    @app.get("/api/jobs/{job_id}/items/{item_id}/document")
    def get_document(job_id: str, item_id: str):
        return document_data(job_id, item_id)

    @app.put("/api/jobs/{job_id}/items/{item_id}/document")
    def update_document(job_id: str, item_id: str, edits: DocumentEdits):
        return document_data(job_id, item_id, edits)

    @app.get("/api/jobs/{job_id}/items/{item_id}/assets/{asset}")
    def asset(job_id: str, item_id: str, asset: str):
        job, item = get_item(job_id, item_id)
        if asset not in {"page", "original", "page_preview", "result", "result_preview"}:
            raise KeyError("Image does not exist.")
        with job.lock:
            path = item["directory"] / f"{asset}.png"
            if not path.is_file():
                raise KeyError("Image is not ready yet.")
            return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})

    @app.get("/api/jobs/{job_id}/items/{item_id}/download")
    def download(job_id: str, item_id: str):
        job, item = get_item(job_id, item_id)
        with job.lock:
            if item["state"] != "ready":
                raise ValueError("This image is not ready to download.")
            return FileResponse(item["directory"] / "result.png", media_type="image/png",
                                filename=f"{download_stem(item['filename'])}_solved.png",
                                headers={"Cache-Control": "private, no-store"})

    @app.get("/api/jobs/{job_id}/download")
    def download_archive(job_id: str):
        job = manager().get(job_id)
        with job.lock:
            ready = [item for item in job.items if item["state"] == "ready"]
            if len(ready) < 2:
                raise ValueError("Use the PNG download for a single solved worksheet.")
            revision = "_".join(f"{item['id']}-{item['revision']}" for item in ready)
            archive_path = job.directory / f"solutions_{revision}.zip"
            if not archive_path.exists():
                with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                    for index, item in enumerate(ready, start=1):
                        name = f"{index:02d}_{download_stem(item['filename'])}_solved.png"
                        archive.write(item["directory"] / "result.png", arcname=name)
            return FileResponse(archive_path, media_type="application/zip", filename="worksheet_solutions.zip",
                                headers={"Cache-Control": "private, no-store"})

    @app.post("/api/jobs/{job_id}/items/{item_id}/feedback", status_code=201)
    def report_feedback(job_id: str, item_id: str, payload: FeedbackRequest):
        job, item = get_item(job_id, item_id)
        with job.lock:
            if item["state"] in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Wait for processing before reporting this worksheet.")
            return app.state.worksheet_feedback.submit(payload.model_dump(), item["document"],
                                                       item["directory"] / "page.png", item["filename"])

    return app


app = create_app()

if __name__ == "__main__":
    host, port = "0.0.0.0", 5000
    print(f"Open http://127.0.0.1:{port}")
    print(f"Local network: http://{socket.gethostbyname(socket.gethostname())}:{port}")
    uvicorn.run(app, host=host, port=port, loop="asyncio", http="h11", ws="none", lifespan="on", workers=1)
