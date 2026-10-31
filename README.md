# Worksheet Solver

> Detect answer areas, solve worksheet exercises with a vision-language model, and render the answers back onto the original page.

Worksheet Solver combines a custom YOLO detector with Google Gemini or a local Ollama vision model. It understands inline gaps, ruled answer lines, and larger free-response areas, keeps them in reading order, and produces a clean solved worksheet image.

The project includes a modern web interface, a command-line workflow, a Python API, batch processing, and utilities for maintaining and training the detection model.

> [!WARNING]
> AI-generated answers may be incomplete or incorrect. Always review the result before using it for teaching, assessment, or grading.

## Key Features

- Detects inline gaps, ruled lines, and free answer areas with YOLO.
- Groups related writing lines into one logical answer unit.
- Orders answers from top to bottom and left to right.
- Uses structured model responses to map answers back to the correct areas.
- Validates answer IDs and makes one bounded repair request for invalid or missing answers.
- Supports Google Gemini and local Ollama vision models.
- Fits, wraps, and aligns black answer text to the original worksheet layout.
- Locates printed underlines to compensate for imperfect detection boxes.
- Processes up to 10 worksheets per web request.
- Downloads one solved worksheet as PNG or multiple results as a ZIP archive.
- Lets you type directly into answers on the worksheet, with optional tools for adjusting detection areas.
- Processes uploads in a bounded background queue with per-file progress and retry.
- Stores error reports locally and adds reviewed detection labels to the training dataset only with explicit consent.
- Reuses one thread-safe YOLO runtime per server process.
- Normalizes and validates images before inference.
- Expires web job files after an inactivity window; CLI and API solvers clean up their own temporary files.
- Includes dataset preparation, annotation, training, and diagnostic tools.

## Processing Pipeline

```text
Input image(s) → Validate and normalize → Detect answer areas → Group and order
               → OCR and solve          → Map answers         → Render PNG
```

| Stage | What happens |
| --- | --- |
| Validation | The image header, dimensions, pixel count, aspect ratio, and file type are checked. |
| Normalization | EXIF orientation is applied, transparency is flattened, RGB is enforced, and clear page boundaries may be rectified. Optional Tesseract orientation detection rotates text only when confidence is sufficient. |
| Detection | YOLO finds candidate gaps and answer lines; overlapping detections are filtered by confidence. |
| Grouping | Related ruled lines are combined and all answer units are sorted in reading order. |
| Solving | OCR text, the normalized worksheet, and a numbered debug image are sent to the selected vision-language model. |
| Rendering | Answers are fitted to the available rows and written in black. Answers that do not fit are retained in the editor and explicitly flagged instead of silently truncated. |

## Requirements

- Python 3.10 or newer
- Windows, Linux, or macOS
- PNG, JPG, JPEG, WEBP, or BMP worksheet images
- One solving backend:
  - a Google Gemini API key, or
  - a running Ollama installation with a vision-capable model
- Internet access on first use if no detector model is installed locally

A CUDA-capable GPU is recommended for detector training and large local models. Normal detector inference can also run on CPU.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Hawk3388/solver.git
cd solver
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Or on Linux and macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure a solving backend

Local Ollama mode with the model name `qwen3.8` is selected by default. Start Ollama and make sure that exact model name is installed and supports image input. If your installed model has a different name, set `SOLVER_MODEL` accordingly.

For Google Gemini, copy `.env.example` to `.env` and configure:

```env
GOOGLE_API_KEY=your_api_key_here
SOLVER_MODEL=gemini-3-flash-preview
SOLVER_LOCAL=false
```

You can also change the model and backend in the web settings.

### 5. Start the application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The web application runs on FastAPI and Uvicorn. Interactive API documentation is available at [http://127.0.0.1:5000/docs](http://127.0.0.1:5000/docs), with the OpenAPI schema at `/openapi.json`.

Alternatively, launch the ASGI application directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 1
```

The detector is loaded once during server startup through the application's [lifespan handler](https://fastapi.tiangolo.com/advanced/events/). The handler also closes the job manager on shutdown. If no weights are available under `model/vX.Y.Z/gap_detection_model.pt`, the application attempts to download the newest compatible release automatically.

### Shared configuration

Web, CLI, Gradio, and Python constructor defaults are read from `worksheet_solver/config.py` and can be set through the environment or `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SOLVER_MODEL` | `qwen3.8` | Default vision model |
| `SOLVER_LOCAL` | `true` | Use Ollama by default |
| `SOLVER_THINK` | `false` | Enable model reasoning |
| `SOLVER_MAX_PENDING_JOBS` | `4` | Maximum waiting batches, in addition to the active batch |
| `SOLVER_JOB_TTL_SECONDS` | `3600` | Retention after the last access to a completed job |
| `SOLVER_REQUEST_TIMEOUT_SECONDS` | `180` | Backend timeout; for Ollama streams, maximum inactivity between received chunks |
| `SOLVER_OLLAMA_MAX_SECONDS` | `600` | Local generation time limit, checked between streamed chunks |
| `SOLVER_MAX_BATCH_FILES` | `10` | Upload count limit |
| `SOLVER_MAX_FILE_MB` | `10` | Per-file size limit |
| `SOLVER_FEEDBACK_DIR` | `feedback/` | Persistent error reports and consented images |
| `SOLVER_DATASET_DIR` | `dataset/` | Destination for reviewed YOLO samples |

Explicit Python arguments and web settings override the shared model defaults for the current operation. Restart the server after changing environment defaults.

## Web Interface

1. Select one or more worksheet images.
2. Optionally adjust the model or inference settings.
3. Click **Solve**.
4. Click a solution to open its worksheet and type directly into the answer fields.
5. Choose **Save changes** to update the result. Click an original on the left to select different images; **Enlarge** opens its full preview.
6. Download one result as a PNG or multiple successful results as a ZIP archive.

The upload immediately returns a job ID. One background worker processes queued batches sequentially; progress and completed previews are available while the remaining files are still running. A failed file does not discard successful results. Use its **Retry** button to resubmit only that worksheet.

Changing the selected upload is disabled during processing to prevent old results from appearing beside newly selected originals. Previews are resized images; full PNG files and ZIP archives are served separately without Base64 duplication.

### Correcting answers and detection areas

The editor opens with text fields positioned directly on the worksheet. Click a field and type; grouped answer lines share one complete editable answer. **Save changes** updates the result without another AI request. **Download image** saves pending changes before downloading that worksheet.

For detection corrections, choose **Adjust areas**. These optional controls support font sizes, text offsets, moving/resizing boxes, drawing missing areas, deleting false detections, selecting detector classes, and grouping several regions into one answer. **Back to typing** returns to the simple view.

Unresolved answers and text that exceeds the available space are shown as review notices. The full text stays editable; overflowing answers are omitted from the rendered page until corrected rather than being silently shortened.

Under **Adjust areas**, the editor also offers the original geometry, a rendered preview, manual 90°/180°/270° rotation, and an option to undo geometry correction. Geometry changes explicitly rerun detection and solving and replace the existing answers after confirmation.

### Feedback and dataset contributions

Use **Report error** on a result, in the editor, or beside a failed item. Reports are stored locally under `feedback/<report-id>/report.json`.

- Without image consent, only the error report and filename are stored persistently.
- Image-storage consent includes the normalized worksheet and its annotation snapshot.
- Training inclusion additionally requires agreement to use the image for training and confirmation that **all** detection areas and classes have been reviewed.

Approved contributions create an image and YOLO labels under `dataset/images/{train,val}/` and `dataset/labels/{train,val}/`. The existing dataset class mapping and directory layout must match. Duplicate feedback images are detected; conflicting annotations are rejected for manual review. A stable image hash keeps repeated contributions in the same split.

Text corrections are retained with consented reports, but detector training uses images and bounding boxes. Adding a sample does **not** automatically retrain the detector. Feedback and dataset files are persistent and are separate from expiring job files.

### Available Settings

| Setting | Description |
| --- | --- |
| LLM Model Name | Gemini model identifier or exact Ollama model name |
| Local Mode | Uses Ollama instead of the Gemini API |
| Thinking | Enables reasoning output when supported by the selected backend |
| Debug Mode | Shows detailed processing phases in the UI and prints timing/model diagnostics to the server console |

Thinking is disabled by default and can be enabled in Settings. When enabled, the selected model uses its default reasoning budget.

Conservative perspective correction and text-orientation checks run automatically. Automatic rotation uses the optional `tesseract` executable with its `osd` data on `PATH`. Without it, or when orientation confidence is low, the image orientation is preserved and a review notice is shown. Genuine landscape worksheets are not rotated solely because of their aspect ratio.

### Upload Limits and Image Safety

| Limit | Value |
| --- | ---: |
| Files per request | 10 |
| Size per file | 10 MB |
| Total request size | 100 MB |
| Maximum source pixels | 40 million |
| Maximum normalized output pixels | 16 million |
| Maximum aspect ratio | 6:1 |

These checks prevent highly compressed oversized images from bypassing the upload-size limit. Alpha channels are flattened onto white, all images are converted to RGB, and large images are proportionally downscaled.

> [!CAUTION]
> The server binds to `0.0.0.0` and is therefore reachable from the local network unless blocked by a firewall. Do not expose it to an untrusted network without authentication, TLS, and a reverse proxy.

## Solving Backends

### Google Gemini

Gemini is an optional cloud backend. Set `SOLVER_LOCAL=false` and `SOLVER_MODEL=gemini-3-flash-preview`, or select these settings in the web interface. The application reads `GOOGLE_API_KEY` from the environment or a local `.env` file.

Worksheet images and prompts are sent to Google's API when this backend is used. Review the applicable privacy and retention policies before processing sensitive documents.

### Ollama

Local Mode is enabled by default with the configured model name `qwen3.8`. It sends OCR and solving requests to the local Ollama service:

1. Start Ollama.
2. Install a vision-capable model.
3. Enter its exact name under **LLM Model Name**.
4. Enable **Local Mode**.

Verify installed model names with:

```bash
ollama list
```

Text-only models cannot inspect worksheet images.

## Command-Line Interface

Run the interactive CLI:

```bash
python main.py
```

It accepts:

- one image path;
- one directory; or
- multiple image and directory paths separated by semicolons.

A single image opens an interactive detection-and-solving workflow. Directory or multi-file input is processed sequentially and written to `solved_batch/`. Output names are collision-safe and existing files are never overwritten.

The CLI uses the same backend defaults as the web application. Set `SOLVER_MODEL` and `SOLVER_LOCAL` in `.env`, or configure the Python constructor explicitly.

## Python API

### Solve one worksheet

```python
from main import WorksheetSolver

with WorksheetSolver(
    path="worksheet.png",
    llm_model_name="gemini-3-flash-preview",
    local=False,
    think=False,
) as solver:
    gaps, image = solver.detect_gaps()
    marked_image = solver.mark_gaps(image, gaps)
    solutions = solver.solve_all_gaps(marked_image)
    solver.fill_gaps_in_image(
        solver.path,
        solutions,
        output_path="worksheet_solved.png",
    )
```

Using `WorksheetSolver` as a context manager guarantees that its private temporary directory is removed after success or failure.

### Solve a batch

```python
from main import solve_batch

results = solve_batch(
    ["worksheets", "extra_sheet.png"],
    output_directory="solved_batch",
    solver_kwargs={
        "llm_model_name": "gemini-3-flash-preview",
        "local": False,
        "think": False,
    },
)

for result in results:
    if result.success:
        print(f"Solved: {result.output_path}")
    else:
        print(f"Failed: {result.source_path}: {result.error}")
```

`solve_batch()` accepts files and directories, deduplicates resolved paths, ignores unsupported files inside directories, and continues after individual failures by default. Use `recursive=True` to include subdirectories or `continue_on_error=False` to stop at the first error.

### `WorksheetSolver` Options

| Parameter | Default | Description |
| --- | --- | --- |
| `path` | required | Source worksheet image |
| `gap_detection_model_path` | automatic | Optional explicit detector weights path |
| `llm_model_name` | `SOLVER_MODEL` | Gemini or Ollama model identifier |
| `think` | `SOLVER_THINK` | Enables reasoning where supported |
| `local` | `SOLVER_LOCAL` | Uses a local backend instead of Gemini |
| `debug` | `False` | Prints diagnostic timing information |
| `detection_runtime` | `None` | Reuses an explicitly supplied detector runtime |
| `max_output_pixels` | `16000000` | Downscales larger normalized images |
| `auto_rotate_page` | `True` | Checks text orientation with optional Tesseract and rotates only confident detections |
| `correct_perspective` | `True` | Applies conservative four-corner page rectification |

## Architecture

The public facade in `main.py` composes three focused pipeline mixins:

```text
main.py
└── WorksheetSolver
    ├── DetectionMixin  ── worksheet_solver/detection.py
    ├── SolvingMixin    ── worksheet_solver/solving.py
    └── RenderingMixin  ── worksheet_solver/rendering.py
```

| Path | Responsibility |
| --- | --- |
| `app.py` | FastAPI application factory and lifespan, job API, binary downloads, document edits, and feedback routes |
| `templates/index.html` | Semantic markup for the workspace, editor, and consent dialogs |
| `static/app.css` | Consolidated responsive styles, animation, and reduced-motion support |
| `static/app.js` | Upload state, progress polling, editor interaction, previews, and feedback |
| `worksheet_solver/config.py` | Shared environment configuration |
| `worksheet_solver/answers.py` | Duplicate/missing/unknown answer-ID validation |
| `worksheet_solver/jobs.py` | Bounded worker queue, item state, retention, and cleanup |
| `worksheet_solver/web_processing.py` | Background normalization, detection, solving, and document creation |
| `worksheet_solver/documents.py` | Editable document validation, previews, and rendering without AI |
| `worksheet_solver/feedback.py` | Consent records, class-mapping checks, and deduplicated dataset insertion |
| `main.py` | Public solver facade, model resolution, normalization orchestration, and CLI |
| `worksheet_solver/images.py` | Safe decoding, EXIF handling, RGB conversion, resizing, rotation, and perspective correction |
| `worksheet_solver/detection.py` | YOLO inference, overlap filtering, reading order, answer grouping, and debug marking |
| `worksheet_solver/solving.py` | OCR, prompt construction, Gemini/Ollama calls, and structured answer mapping |
| `worksheet_solver/rendering.py` | Font sizing, wrapping, underline alignment, and final PNG rendering |
| `worksheet_solver/runtime.py` | Process-wide detector cache and inference lock |
| `worksheet_solver/batch.py` | File discovery, collision-safe naming, cleanup, and fault-tolerant sequential processing |
| `worksheet_solver/schemas.py` | Pydantic schemas for structured model responses |
| `fonts/` | Liberation Sans font used for consistent answer rendering |

The detector is cached by its resolved model path. A process-wide lock serializes access to `YOLO.predict()`. The web queue and job registry are currently process-local: run a single application process, or provide sticky routing and separate storage before scaling across processes. Pending and completed jobs do not survive a server restart.

### Web API

| Endpoint | Purpose |
| --- | --- |
| `POST /solve` | Accept multipart `files`; return `202` with a job ID and status URL |
| `GET /api/config` | Read model defaults and upload limits |
| `GET /api/jobs/{job_id}` | Read per-file stages, warnings, preview URLs, and completion state |
| `POST /api/jobs/{job_id}/items/{item_id}/retry` | Retry a single item, optionally with explicit geometry |
| `GET/PUT /api/jobs/{job_id}/items/{item_id}/document` | Retrieve or save versioned regions and answers |
| `GET /api/jobs/{job_id}/items/{item_id}/assets/{asset}` | Read an original, normalized image, result, or thumbnail |
| `GET /api/jobs/{job_id}/items/{item_id}/download` | Download one completed PNG |
| `GET /api/jobs/{job_id}/download` | Download multiple completed PNGs as ZIP |
| `POST /api/jobs/{job_id}/items/{item_id}/feedback` | Submit a report and explicit storage/training consent |
| `GET /docs` | Interactive API documentation |
| `GET /openapi.json` | Machine-readable API schema |

Opaque job IDs are access tokens for their files; do not share them unintentionally. Completed jobs expire after the configured inactivity window.

Uploads are parsed with bounded file and field counts, and the total byte limit also applies to streamed requests without a `Content-Length` header. Multipart temporary files close after the queue has copied the uploads. File copying, rendering, feedback storage, and archive creation run outside the ASGI event loop. API errors retain the frontend's `{"error": "message"}` format, including validation errors (`400`), expired jobs (`404`), oversized requests (`413`), and a full queue (`429`).

## Optional Gradio Demo

`demo.py` contains a minimal single-image Gradio interface. Gradio is intentionally not part of the base dependencies:

```bash
pip install gradio
python demo.py
```

| Environment variable | Default | Description |
| --- | --- | --- |
| `PORT` | `7860` | Gradio server port |
| `GRADIO_SHARE` | `false` | Enables a public Gradio share link |

## Detector Development

Detector training is optional. It is only required when adapting the model to new worksheet styles or improving detection quality.

### Tooling Overview

| Script | Purpose |
| --- | --- |
| `prepare_dataset.py` | Creates an initial YOLO dataset using OpenCV heuristics or an existing model |
| `add_to_dataset.py` | Adds files or directories through model-assisted annotation and review |
| `edit_boxes.py` | Reviews and edits existing YOLO boxes interactively |
| `train_yolo.py` | Trains or resumes the worksheet detector |
| `yolo_test.py` | Runs standalone detection, grouping, and visualization diagnostics |
| `simple_boxes.py` | Provides a smaller detector visualization and ordering utility |

The paths and model constants near the top or bottom of these scripts are project-specific and should be reviewed before use.

### 1. Prepare source images

Place images in `raw_images/`, review the configuration at the bottom of `prepare_dataset.py`, and run:

```bash
python prepare_dataset.py
```

Automatically generated annotations are starting points and should be reviewed manually.

### 2. Review or extend annotations

```bash
python edit_boxes.py
python add_to_dataset.py image1.png image2.jpg
```

`add_to_dataset.py` also accepts directories and keeps the training/validation split close to its configured 80/20 target.

> [!IMPORTANT]
> Keep the class IDs in `dataset/data.yaml`, the annotation files, and the trained detector consistent. The runtime treats the exact class name `line` as a ruled answer line; other detected classes become independent answer units.

### 3. Train

```bash
python train_yolo.py
```

Current defaults:

| Parameter | Value |
| --- | --- |
| Base model | `yolo26l.pt` |
| Epochs | `1500` |
| Image size | `640` |
| Batch size | `16` |
| Device | GPU `0` |
| Output directory | `worksheet_yolo/transfer_learning/` |

The best checkpoint is written to:

```text
worksheet_yolo/transfer_learning/weights/best.pt
```

Adjust the model size, epochs, batch size, and device for the available hardware and dataset size.

### 4. Diagnose a trained model

Review `MODEL_PATH` and `IMAGE_PATH` in `yolo_test.py`, then run:

```bash
python yolo_test.py
```

## Build the Windows Executable

Install PyInstaller and build from the project root:

```bash
pip install pyinstaller
pyinstaller solver.spec
```

The executable is written to `dist/`. The specification bundles templates, static frontend assets, and the answer font. Keep `.env`, feedback, and dataset directories in a writable external location. Verify detector-model availability in the final distribution and never include a real API key in a release.

## Troubleshooting

### No detector model is available

- Confirm that internet access is available for the first-run release lookup.
- Or place weights at `model/vX.Y.Z/gap_detection_model.pt`.
- Or pass `gap_detection_model_path` explicitly through the Python API.

### No answer areas are detected

- Use a clear, upright image with sufficient contrast.
- Confirm that the worksheet style resembles the detector's training data.
- Run `yolo_test.py` and inspect the marked answer units.
- Add representative examples and hard negatives to the training dataset.

### Answers appear on the wrong lines

- Enable Debug Mode and inspect the detector output.
- Check whether tables, borders, or decorative lines resemble answer areas.
- Review the reading order and class labels in the annotations.

### Gemini fails to initialize

- Confirm that `.env` is in the project root.
- Verify that `GOOGLE_API_KEY` is present and valid.
- Confirm that the selected model is available to that API key.

### Ollama requests fail

- Confirm that the Ollama service is running.
- Verify the model name with `ollama list`.
- Use a vision-capable model rather than a text-only model.

Local OCR and solving responses are streamed. With **Debug Mode** enabled, the progress indicator distinguishes waiting for model/image processing, thinking, and answer output. Otherwise it shows a general processing message and batch progress. A long generation can continue while data arrives instead of failing because a complete reply took longer than 180 seconds. See [Ollama's streaming documentation](https://docs.ollama.com/api/streaming).

The inactivity limit is set by `SOLVER_REQUEST_TIMEOUT_SECONDS`. `SOLVER_OLLAMA_MAX_SECONDS` stops generations that continue streaming for too long; it is checked when a chunk arrives, so an idle network wait may extend beyond it up to the inactivity timeout. Both settings are read at server startup. For slow thinking models, try turning **Thinking** off in Settings; this reduces reasoning work when the model supports the switch. No automatic retry or model switch is performed for a timed-out generation.

### Rendered text does not fit

The renderer reduces the font size within a readable range and wraps text across detected rows. If the full answer still does not fit, the web editor retains the text and shows an overflow notice. Shorten the answer directly in its field, or use **Adjust areas** to adjust its font or regions, then choose **Save changes**.

## Known Limitations

- Accuracy depends on both detector quality and the selected language model.
- The solving prompt is currently tailored to German worksheets.
- Tables, colored backgrounds, handwriting, and unusual line styles may cause false or missed detections.
- Reliable automatic text orientation requires optional Tesseract OSD; uncertain orientations require review.
- PDF input is not supported; convert each page to an image first.
- Job metadata and queues are process-local and do not survive server restarts; image files live in expiring temporary directories.
- The web worker processes one worksheet at a time; a long-running backend call delays subsequent queued work.

## License

Worksheet Solver is licensed under the [GNU General Public License v3.0](LICENSE.md).
