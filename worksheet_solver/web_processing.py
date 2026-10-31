"""Background worksheet processing and reusable editable result files."""

from PIL import Image

from .config import MAX_FILE_SIZE
from .documents import create_preview, document_from_solver, render_document
from .images import normalize_worksheet_image, validate_image_file


def process_worksheet(job, item, progress, solver_class, runtime_provider):
    directory = item["directory"]
    source = item["source"]
    if item["extension"] not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        raise ValueError("Unsupported file type. Use PNG, JPG, JPEG, WEBP or BMP.")
    if source.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("The image exceeds the per-file upload limit.")
    validate_image_file(source)
    progress("Validating and normalizing image")
    base = directory / "original.png"
    page = directory / "page.png"
    if not base.exists():
        normalize_worksheet_image(source, base, auto_rotate_page=False, correct_perspective=False)
    metadata = {}
    geometry = item.get("geometry")
    if geometry is None:
        normalize_worksheet_image(base, page, metadata=metadata)
    else:
        normalize_worksheet_image(base, page, auto_rotate_page=False,
                                  correct_perspective=geometry["perspective"], metadata=metadata)
        if geometry["rotation"]:
            with Image.open(page) as image:
                rotated = image.rotate(-geometry["rotation"], expand=True)
                try:
                    rotated.save(directory / "rotated.png", format="PNG")
                finally:
                    rotated.close()
            (directory / "rotated.png").replace(page)
        metadata["rotation"] = geometry["rotation"]
    create_preview(page, directory / "page_preview.png")
    for name in ("result.png", "result_preview.png"):
        (directory / name).unlink(missing_ok=True)
    with job.lock:
        item["document"] = None
        item["revision"] += 1
    progress("Loading detector")
    with solver_class(str(page), **job.options, detection_runtime=runtime_provider(),
                      auto_rotate_page=False, correct_perspective=False) as solver:
        solver.progress_callback = progress
        progress("Detecting and grouping answer areas")
        gaps, image = solver.detect_gaps()
        document = document_from_solver(solver, page)
        document["normalization"] = metadata
        document["version"] = item["revision"]
        with job.lock:
            item["document"] = document
        if not gaps:
            raise ValueError("No answer areas found. Open the editor to add regions manually or report the detection error.")
        marked = solver.mark_gaps(image, gaps)
        progress("Reading worksheet and solving exercises")
        solutions = solver.solve_all_gaps(marked)
        for answer in document["answers"]:
            answer["text"] = solutions.get(answer["id"] - 1, {}).get("solution", "")
        document["validation_warnings"] = list(getattr(solver, "answer_warnings", []))
        progress("Fitting text and rendering answers")
        render_document(document, page, directory / "result.png")

